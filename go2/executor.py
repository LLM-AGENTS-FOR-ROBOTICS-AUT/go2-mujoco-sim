"""Executes resolved tool commands on the attached robot adapter, one at a time.

Safety properties:
  * stop() pre-empts everything from the caller's thread: it sends StopMove,
    clears the queue and cancels the running task. Nothing resumes on its own.
  * Motion completion is judged from fresh odometry, never from the command.
  * Moves are checked against the LiDAR before and during motion; an obstacle
    inside the stopping margin stops the robot and the remaining queue.
  * Every task has a hard deadline; telemetry older than 0.5 s stops motion.
  * Acrobatics run only when confirmed in a later user turn than their first request.
"""
from __future__ import annotations

import math
import re
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import follow as F
from . import limits as L
from . import perception as P
from .adapters.base import AdapterError, LOW_POSTURES
from .object_memory import ObjectMemory
from .memory_visibility import clear_saved_location
from .places import NamedPlaces
from .path_planner import astar, simplify, turning_astar, keep_turns
from .room_map import OccupancyGrid
from .exploration import ExplorationMap, next_viewpoint, _pose, _safe_cells, recovery_route
from .room_regions import RoomRegions
from .room_names import RoomNamer, tells_room
from .exploration_navigation import (ExplorationNavigation,MOVES as EXPLORATION_STEPS,
                                     BODY_HALF_LENGTH_M,BODY_HALF_WIDTH_M)
from .tools import Command
from .visual_memory import crop

DECEL_LINEAR, DECEL_ANGULAR = 1.6, 4.0      # m/s², rad/s² used for braking lead
FINAL_DECEL_LINEAR = .8                   # conservative terminal speed profile for distance moves
# Stall: something the LiDAR can't see (glass, thin chair legs) is stopping the robot. After the ramp-up, a move
# or turn that makes less than a third of the progress its commanded speed should give over 0.4 s stops.
STALL_WINDOW_S, STALL_FRACTION, STALL_GRACE_S = .4, 1 / 3, .3
STALL_START_S = .8     # the latest the check starts if the robot never gets going (e.g. it starts against glass)
STALLED = "bumped into something the sensors missed"
BACKOFF_M, BACKOFF_SPEED = .2, .2           # after a stall, back away the way it came if the LiDAR says clear
BUMP_MEMORY_S = 180.                        # where it bumped counts as an obstacle for this long
BUMP_HALF_WIDTH = .3
STALE_S = .5
# go_to keeps waypoints this far from mapped obstacles: the body's turning circle (0.40 m) plus a cell, so it can
# turn at each waypoint; in a tighter space it tries the second value, where it can still walk but may not turn.
ROUTE_CLEARANCE_M = (.5, .42)
UNKNOWN_COST = 3.           # unmapped cells may be planned through, but known free space is preferred
MAX_REPLANS = 3             # go_to plans again this many times when blocked by something new
MAX_DETOUR = 2.5            # a route may be at most this many times the straight distance (+ 2 m)
TICK_S = .05
# A motion that can start right away is held back from the model for up to this long, so that the tool
# result can say "blocked" when the safety check refuses it instead of "queued".
START_CHECK_S = 1.0
# A thing this near fills enough of the picture for an ordinary resemblance to be trusted (semantic object memory).
CLOSE_M = 1.5
# What the detector (YOLO, COCO classes) can call things. Words that fit none of these labels are translated to the
# closest one by the semantic object memory's text side ("computer" -> laptop), when it is on.
DETECTOR_LABELS = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush")
# The closest label is taken only when it is this much closer to the words than the next one. Measured with SigLIP 2
# on 44 everyday words: right translations lead by 0.02-0.08 (computer -> laptop, fridge -> refrigerator, telly -> tv),
# words with no label of their own by under 0.01 (guitar, lamp, my keys); raw closeness alone doesn't separate them.
LABEL_MARGIN = .02
SYNONYMS = {"ball": "sports ball", "football": "sports ball", "basketball": "sports ball", "soccer": "sports ball",
            "table": "dining table", "desk": "dining table", "sofa": "couch", "plant": "potted plant",
            "flowerpot": "potted plant", "man": "person", "woman": "person", "human": "person", "guy": "person",
            "kid": "person", "people": "person", "someone": "person", "cardboard": "box", "crate": "box",
            "block": "box", "cube": "box", "carton": "box", "package": "box", "parcel": "box", "seat": "chair"}
STOP_WORDS = {"the", "a", "an", "my", "your", "our", "that", "this", "those", "these", "one", "ones", "over", "there",
              "here", "to", "of", "on", "in", "by", "near", "next", "thing", "please"}
# Words people add that detectors rarely output; they don't decide which object is meant.
ADJECTIVES = {"big", "small", "little", "large", "tiny", "huge", "tall", "short", "round", "square", "wooden", "plastic",
              "metal", "old", "new", "shiny", "soft", "squeaky", "empty", "full", "open", "closed", "folded", "striped",
              "dirty", "clean", "favourite", "favorite", "nearest", "closest", "far", "other", "left", "right"}
# COCO's generic labels stand for anything of their kind ("tennis ball" -> sports ball).
GENERIC = {"sports ball": "ball", "dining table": "table", "potted plant": "plant", "cell phone": "phone",
           "teddy bear": "bear", "wine glass": "glass", "hair drier": "dryer"}
# Colour names drift with lighting and viewpoint (a tan box reads "orange" from one side, "brown" from
# another), so neighbouring colours count as a weaker match.
COLOR_FAMILY = {"red": {"orange", "pink", "brown"}, "orange": {"red", "brown", "yellow"}, "brown": {"orange", "red", "black"},
                "yellow": {"orange", "green"}, "green": {"yellow", "cyan"}, "cyan": {"green", "blue"},
                "blue": {"cyan", "purple"}, "purple": {"blue", "pink"}, "pink": {"red", "purple"},
                "white": {"grey"}, "grey": {"white", "black"}, "gray": {"white", "black", "grey"}, "black": {"grey", "brown"}}
MEMORY_S = 180.
ROOMS_CACHE_S = 3.          # rooms() splits the map at most this often (it takes a few tenths of a second)
MIN_ROOM_M2 = 1.5           # smaller regions (a pocket behind furniture) aren't listed as rooms
ROOM_STAND_M = .45          # go_to_place into a room stops somewhere at least this far from anything in the way


def _room_key(name):
    """A room name to compare: "The living room" -> "living_room"."""
    return re.sub(r"^the_", "", "_".join(re.findall(r"[a-z0-9]+", str(name).casefold())))


class Cancelled(Exception):
    pass


def image_boxes(masks, shape):
    """Where each segmented object is in the camera image: label, score and [left, top, right, bottom] as fractions
    of the image size (for drawing on the frame)."""
    h, w = shape[:2]
    boxes = []
    for m in masks:
        rows, cols = np.nonzero(m.mask)
        if len(rows):
            boxes.append({"label": m.label, "score": round(float(m.score), 2),
                          "box": [round(cols.min() / w, 4), round(rows.min() / h, 4),
                                  round((cols.max() + 1) / w, 4), round((rows.max() + 1) / h, 4)]})
    return boxes


@dataclass
class Task:
    command: Command
    group: str
    generation: int
    id: str = field(default_factory=lambda: "t" + uuid.uuid4().hex[:6])
    status: str = "queued"          # queued, running, completed, stopped, blocked, failed, cancelled
    detail: dict = field(default_factory=dict)
    finished: float | None = None
    awaiting: bool = False          # the model's tool call is waiting to hear whether it started
    reported: bool = False          # its outcome went back in the tool result, so it isn't announced again
    user_turn: int = 0              # the user message it answers
    held: bool = False              # before_motion has had its say (it is asked once per task)

    def public(self):
        return {"id": self.id, "tool": self.command.tool, "summary": self.command.summary,
                "status": self.status, **self.detail}


class Executor:
    def __init__(self, on_event=None, memory_path=None, on_progress=None):
        self.adapter = None
        self.on_event = on_event or (lambda task: None)
        # f(task, text, alert): a running mission's progress (#18), as robot updates; alert is set for an alert
        # with a snapshot ({id, text, stamp}; its image from alert_image(id)).
        self.on_progress = on_progress or (lambda task, text, alert: None)
        self.alerts = deque(maxlen=20)      # mission alerts with their snapshots, newest last
        self._alert_count = 0
        self._missions = {}                 # mission key -> (state, when last run): for carrying on after a stop
        self.lock = threading.RLock()
        self.wake = threading.Condition(self.lock)
        self.queue = deque()
        self.current = None
        self.generation = 0
        self.cancel = threading.Event()
        self.history = deque(maxlen=40)
        self.activity = []          # everything that ran since the robot was attached (for get_status)
        self.activity_since = time.monotonic()
        self.confirmations = {}
        self.armed = False
        self.start_pose = None
        self.cloud = P.CloudBuffer()
        self._cloud_lock = threading.RLock()   # the follow camera thread reads the cloud too
        self.room_map = OccupancyGrid(resolution=.05, size=400, grow=True)
        self.room_regions = RoomRegions()
        self.room_sightings = deque(maxlen=2000)   # (x, y, label) of things that tell a room's type (#35)
        self._rooms_cache = None                   # (when, rooms()) for the live map's once-a-second polls
        self.last_detections = []
        self.object_memory = ObjectMemory(memory_path)
        self._places_path = (Path(memory_path).with_name(Path(memory_path).stem + ".places.json")
                             if memory_path is not None else None)
        self.places = NamedPlaces(self._places_path)
        # Semantic object memory (go2/visual_memory.py): what each thing looked like, as an image embedding, so a
        # remembered thing can be found by words its label doesn't contain. Off until use_visual_memory().
        self._visual_path = None if memory_path is None else (
            Path(memory_path).with_name(Path(memory_path).stem + "-visual.json"))
        self.visual = None
        self._look_vectors = {}     # detection id -> picture vector, from the last look
        self._label_vectors = {}    # detector label -> its text vector (the words-to-label translation)
        self.memory = []            # [(Detection, world_x, world_y, seen_at)] from recent looks
        self._next_id = 0
        self.detection_filter = None  # optional f(detections) -> detections (training-data noise)
        self.last_frame = None
        self.last_frame_meta = None
        self.closed = False
        self._observed = math.inf
        self._drop = self._obstacle = self._bump = math.inf
        self.bumps = []             # [(seen_at, Nx3 odom points)]: spots it bumped into something unseen
        # Optional f(task, cancel), asked once per task right before it first moves the robot (after its first
        # safety check). It may wait briefly: the console holds a spoken reply's first movement until Go2's voice
        # is ready, so the two start together. Stop cuts the wait short (cancel is set).
        self.before_motion = None
        threading.Thread(target=self._worker, daemon=True, name="executor").start()

    # ---- attachment ------------------------------------------------------------
    def attach(self, adapter):
        self.stop("robot changed")
        with self.lock:
            self.adapter = adapter
            # Real odometry resets between connections. Until its frame is
            # relocalised, names on hardware are connection-local, like memory.
            self.places = NamedPlaces(None if adapter.kind == "go2" else self._places_path)
            self.object_memory.reset_observation_evidence()
            self.armed = False
            self.cloud.clear()
            self.room_map = OccupancyGrid(resolution=.05, size=400, grow=True,
                                          detect_low=adapter.kind == "sim")
            self.room_regions = RoomRegions()
            self.room_sightings.clear()
            self._rooms_cache = None
            self._missions.clear()          # their rooms and spots belong to the old map
            self.history.clear()
            self.activity, self.activity_since = [], time.monotonic()
            self.last_detections = []
            self.memory = []
            self._next_id = 0
            try:
                s = adapter.state()
                self.start_pose = (s.x, s.y, s.yaw)
            except Exception:
                self.start_pose = None

    def detach(self):
        self.stop("robot disconnected")
        with self.lock:
            self.adapter = None
            self.armed = False

    def set_armed(self, armed):
        with self.lock:
            self.armed = bool(armed)
            if not armed:
                self.stop("motion disarmed")

    # ---- public API --------------------------------------------------------------
    def exploration_viewpoint(self, visited):
        """Refresh observations and choose a world-frame viewpoint for the mission.

        The mission queues it with submit_viewpoint, waits for its task outcome,
        then scans and records it as visited. No motion is started by this query.
        """
        with self.lock:
            self._require_adapter()
        for _ in range(8):
            self._refresh_cloud()
        s = self._fresh_state()
        forbidden = self._exploration_forbidden(s)
        with self.lock:
            bumps = [pts[:, :2] for t, pts in self.bumps if time.monotonic() - t <= BUMP_MEMORY_S]
        context = ExplorationMap(self.room_map, (s.x, s.y, s.yaw), clearance_m=.42,
                                 visited_radius_m=.5,
                                 extra_obstacles=np.concatenate(bumps) if bumps else None,
                                 turn_clearance_m=.65, heading_aware=True, forbidden_steps=forbidden,
                                 allow_recovery=False)
        for _ in range(3):
            goal = next_viewpoint(context, visited)
            if goal is not None:
                return goal
            # A few non-repetitive L1 scans can fill tiny unknown holes. Never
            # infer that exploration is complete from one sparse observation.
            for _ in range(5):
                self._refresh_cloud()
        # At the end of a sweep, closer viewpoints can reveal small occluded
        # pockets. Keep a finite exclusion radius instead of repeatedly choosing
        # the same pose, but don't suppress an entire doorway after one look.
        for resolution,turning in ((.1,.65),(.1,.6),(.05,.65),(.05,.55)):
            goal = next_viewpoint(ExplorationMap(self.room_map, (s.x, s.y, s.yaw), clearance_m=.42,
                                                 visited_radius_m=.25,
                                                 extra_obstacles=context.extra_obstacles,
                                                 turn_clearance_m=turning, heading_aware=True,
                                                 forbidden_steps=forbidden,
                                                 planning_resolution_m=resolution,
                                                 allow_recovery=False), visited)
            if goal is not None:
                return goal
        # A coarse recovery can bypass a valid fine-map route. Exhaust normal
        # plans first, then refresh observations and use the finest map for the
        # bounded retreat. The query itself still starts no movement.
        s = self._fresh_state()
        navigation = self._exploration_navigation(s, planning_resolution_m=self.room_map.resolution)
        return next_viewpoint(ExplorationMap(navigation.source_map, (s.x, s.y, s.yaw),
                              clearance_m=.42, visited_radius_m=.25, turn_clearance_m=.55,
                              heading_aware=True, forbidden_steps=navigation.forbidden,
                              planning_resolution_m=self.room_map.resolution), visited)

    def room_at(self, x, y):
        """Observed geometric region, or None: the user's name for it, else its name from what's been seen in it
        ("kitchen", #35), else room_N."""
        self.room_regions.update(self.room_map)
        region = self.room_regions.region_near(x, y, radius_m=0.)
        if region is None:
            return None
        return self.room_regions.names.get(region) or self._room_names().get(region) or f"room_{region}"

    def _room_names(self):
        """{region: name} from what's been seen where: each sighting votes for the region it's in now, so names
        follow the regions as the map grows. Run when asked, not on every look (updating the regions is slow)."""
        namer = RoomNamer()
        for x, y, label in list(self.room_sightings):
            namer.saw(self.room_regions.region_near(x, y), label)
        return namer.names()

    def rooms(self, fresh=False):
        """The observed rooms as [{name, region, x, y, area_m2}], named as room_at names them, (x, y) a spot near
        each one's middle. From a cache up to ROOMS_CACHE_S old unless fresh."""
        cached = self._rooms_cache
        if not fresh and cached is not None and time.monotonic() - cached[0] < ROOMS_CACHE_S:
            return cached[1]
        self.room_regions.update(self.room_map)
        guessed = self._room_names()
        rooms = []
        for region, area in sorted(self.room_regions.areas().items()):
            spot = self.room_regions.region_point(region) if area >= MIN_ROOM_M2 else None
            if spot is not None:
                name = self.room_regions.names.get(region) or guessed.get(region) or f"room_{region}"
                rooms.append({"name": name, "region": region, "x": spot[0], "y": spot[1], "area_m2": round(area, 1)})
        self._rooms_cache = (time.monotonic(), rooms)
        return rooms

    def _room_destination(self, name, x, y):
        """(x, y, room name) of a spot inside the room called `name` ("kitchen", "bedroom 2", "the living room"),
        or None. A room type ("bedroom") means the nearest room of that type to (x, y)."""
        wanted = _room_key(name)
        rooms = self.rooms(fresh=True)
        matches = ([r for r in rooms if _room_key(r["name"]) == wanted] or
                   [r for r in rooms if not r["name"].startswith("room_") and
                    re.sub(r"_\d+$", "", _room_key(r["name"])) == wanted])
        if not matches:
            return None
        room = min(matches, key=lambda r: math.hypot(r["x"] - x, r["y"] - y))
        spot = self.room_regions.region_point(room["region"], clearance_m=ROOM_STAND_M)
        return (*spot, room["name"]) if spot else None

    def _note_room_sightings(self, detections, pose):
        """Note where each thing in view that tells a room's type (a bed, a fridge) was seen."""
        c, sn = math.cos(pose[2]), math.sin(pose[2])
        for d in detections:
            if d.x_m is not None and tells_room(d.label):
                self.room_sightings.append((pose[0] + c * d.x_m - sn * d.y_m, pose[1] + sn * d.x_m + c * d.y_m,
                                            d.label))

    # ---- missions (#18, go2/missions.py) -----------------------------------------------------------------
    def now(self):
        """This executor's clock (offline evaluations run it on simulated time)."""
        return time.monotonic()

    def _idle(self, seconds, cancel):
        """Wait without moving, in this executor's time; Stop ends the wait (Cancelled)."""
        end = time.monotonic() + seconds
        while True:
            self._check(cancel)
            left = end - time.monotonic()
            if left <= 0:
                return
            time.sleep(min(.1, left))

    def _subtask(self, command, parent):
        """A step of a mission, run directly by the mission (in this worker), not through the queue. Its route and
        destination show on the mission task too, so the live map draws where the mission is going."""
        class Shown(dict):
            def __setitem__(self, key, value):
                super().__setitem__(key, value)
                if key in ("route", "destination"):
                    parent.detail[key] = value
        task = Task(command, parent.group, parent.generation, user_turn=parent.user_turn, held=True)
        task.detail = Shown()
        return task

    def _step_budget(self, cancel, seconds):
        return _MotionDeadline(cancel, seconds)

    def _mission_state(self, key, resume_s):
        """(state, resumed): the state of an unfinished mission asked for again within resume_s, else a new one."""
        with self.lock:
            state, when = self._missions.get(key, (None, None))
            resumed = state is not None and not state.get("finished") and time.monotonic() - when <= resume_s
            if not resumed:
                state = {}
            self._missions[key] = (state, time.monotonic())
            return state, resumed

    def _mission_progress(self, task, text, image=None):
        """A mission's progress line, or an alert when it comes with a camera image."""
        alert = None
        if image is not None:
            with self.lock:
                self._alert_count += 1
                alert = {"id": self._alert_count, "mission": task.command.tool, "text": text, "stamp": time.time()}
                self.alerts.append({**alert, "image": np.array(image, copy=True)})
        try:
            self.on_progress(task, text, alert)
        except Exception:
            pass

    def alert_image(self, alert_id):
        """The camera image of alert `alert_id`, or None once it's been dropped."""
        with self.lock:
            return next((a["image"] for a in self.alerts if a["id"] == alert_id), None)

    def _detections(self):
        with self.lock:
            return list(self.last_detections)

    def _sightings(self):
        """What the last look saw, as {label, color, clothing, x, y} in map coordinates."""
        out = []
        for d, (x, y, yaw) in self._detections():
            if d.x_m is not None and d.label != "wall":
                out.append({"label": d.label, "color": d.color, "clothing": getattr(d, "clothing", None),
                            "x": x + math.cos(yaw) * d.x_m - math.sin(yaw) * d.y_m,
                            "y": y + math.sin(yaw) * d.x_m + math.cos(yaw) * d.y_m})
        return out

    def _way_in(self, x, y):
        """The heading (radians) from (x, y) toward the nearest way into its room (where it meets another room),
        or None when the room has no known way in."""
        self.room_regions.update(self.room_map)
        door = self.room_regions.way_in(x, y)
        return None if door is None else math.atan2(door[1] - y, door[0] - x)

    def name_room(self, name):
        """Name the room the robot is in (the user's name wins over a guess). Lasts as long as this map."""
        with self.lock:
            self._require_adapter()
            state = self._fresh_state()
        self.room_regions.update(self.room_map)
        region = self.room_regions.region_near(state.x, state.y)
        if region is None:
            raise ValueError("I can't tell which room I'm in yet; look around first so I can map it")
        self.room_regions.name_room(*self.room_regions.region_point(region), name)
        self._rooms_cache = None
        return {"name": name, "area_m2": round(self.room_regions.areas().get(region, 0.), 1)}

    def remember_place(self, name):
        with self.lock:
            self._require_adapter()
            state = self._fresh_state()
            place = self.places.remember(name, state.x, state.y, state.yaw)
        return {"name": place.name, "x_m": place.x, "y_m": place.y,
                "heading_deg": math.degrees(place.yaw), "frame": "map",
                "persisted": self.places.path is not None}

    def memory_report(self, target="", limit=3, *, places_only=False):
        """Bounded saved information; never scan or move while answering."""
        words = re.findall(r"\w+", target.casefold())
        places = [p for p in self.places.all()
                  if not words or all(w in p.name.casefold() for w in words)]
        result = {"places": [{"name": p.name, "x_m": p.x, "y_m": p.y,
                              "heading_deg": math.degrees(p.yaw), "frame": "map"}
                             for p in places[:limit]], "places_total": len(places)}
        if self.places.load_error:
            result["warning"] = self.places.load_error
        if places_only:
            return result
        rooms = [r for r in self.rooms() if all(w in _room_key(r["name"]) for w in words)]
        result["rooms"] = [{"name": r["name"], "x_m": r["x"], "y_m": r["y"], "area_m2": r["area_m2"],
                            "frame": "map"} for r in rooms[:limit]]
        result["rooms_total"] = len(rooms)
        candidates = []
        for item in self.object_memory.snapshot():
            detection = SimpleNamespace(label=item.get("label", ""), color=item.get("color"))
            score, mismatch = self._score(detection, words) if words else (1, False)
            if score and not mismatch:
                candidates.append((score, float(item.get("last_seen", 0.)), item))
        candidates.sort(key=lambda row: (row[0], row[1]), reverse=True)
        unavailable = [row for row in candidates if row[2].get("state") == "missing"]
        candidates = [row for row in candidates if row[2].get("state") != "missing"]
        now = time.time()
        result["objects"] = [{"label": item["label"], "color": item.get("color"),
                              "x_m": item["x"], "y_m": item["y"], "frame": "map",
                              "seen_seconds_ago": max(0., now-item.get("last_seen", now)),
                              "confidence": item.get("confidence"), "state": "remembered"}
                             for _, _, item in candidates[:limit]]
        result["objects_total"] = len(candidates)
        result["unavailable_objects"] = [{"label": item["label"], "color": item.get("color"),
                    "x_m": item["x"], "y_m": item["y"], "frame": "map", "state": "missing",
                    "reason": item.get("missing_reason", "not found at its saved location")}
                    for _, _, item in unavailable[:limit]]
        result["unavailable_total"] = len(unavailable)
        warning = self.object_memory.save_error or self.object_memory.load_error
        if warning:
            result["warning"] = warning
        return result

    def submit_viewpoint(self, viewpoint, *, group="exploration", user_turn=0, timeout_s=90):
        """Queue one world-frame exploration stop through the normal safety worker.

        This internal mission API does not add a model tool or change its prompt.
        """
        x, y, yaw = _pose(viewpoint, "viewpoint")
        if not math.isfinite(timeout_s) or not 0 < timeout_s <= 300:
            raise ValueError("viewpoint timeout must be between 0 and 300 seconds")
        command = Command("viewpoint", "exploration", {},
                          {"x": x, "y": y, "yaw": yaw, "timeout_s": timeout_s},
                          summary=f"visit exploration viewpoint ({x:.2f}, {y:.2f})")
        return self.submit([command], group, user_turn)[0]

    def submit(self, commands, group, user_turn):
        """Queue resolved commands from one model reply; queries run immediately. Returns results.
        A motion that starts right away is reported as blocked or failed if its first safety check
        refuses it (waiting up to START_CHECK_S); otherwise motions are reported as queued."""
        results, tasks = [], []
        try:
            for command in commands:
                try:
                    result = self._submit_one(command, group, user_turn)
                except (AdapterError, ValueError) as error:
                    result = {"status": "rejected", "error": str(error)}
                tasks.append(result.pop("_task", None))
                results.append(result)
            first = next((t for t in tasks if t is not None and t.awaiting), None)
            if first is not None and self._await_start(first):
                for i, task in enumerate(tasks):
                    if task is first:
                        results[i] = {"status": task.status, "action": task.command.summary,
                                      "error": task.detail.get("error", "unknown problem")}
                    elif task is not None and task.status == "cancelled":
                        results[i] = {"status": "cancelled", "action": task.command.summary,
                                      "reason": task.detail.get("reason")}
        finally:
            with self.lock:
                for task in tasks:
                    if task is not None:
                        task.awaiting = False
        return results

    def _await_start(self, task):
        """Wait until `task` is under way or has ended; True if it ended blocked or failed (now reported)."""
        deadline = time.monotonic() + START_CHECK_S
        with self.lock:
            while task.awaiting and not self.closed:
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                self.wake.wait(left)
            task.awaiting = False
            return task.reported

    def _underway(self, task):
        """The task passed its first safety check and is moving: its tool result can say 'queued'."""
        if task.awaiting:
            with self.lock:
                task.awaiting = False
                self.wake.notify_all()

    def _hold(self, task, cancel):
        """Right before `task` first moves the robot: let before_motion wait if it wants to (once per task)."""
        if task.held:
            return
        task.held = True
        hold = self.before_motion
        if hold is not None:
            try:
                hold(task, cancel)
            except Exception:
                pass
        self._check(cancel)

    def _submit_one(self, command, group, user_turn):
        if command.kind == "stop":
            return self.stop("requested by the model")
        if command.kind == "status":
            self._after_group(group)
            return {"status": "done", **self.status_report()}
        if command.kind == "look":
            self._after_group(group)
            return {"status": "done", **self.look()}
        if command.kind in ("remember_place", "name_room", "places", "memory"):
            self._after_group(group)
            if command.kind == "remember_place":
                return {"status": "done", "place": self.remember_place(command.params["name"])}
            if command.kind == "name_room":
                return {"status": "done", "room": self.name_room(command.params["name"])}
            return {"status": "done", **self.memory_report(command.params["target"],
                    command.params["limit"], places_only=command.kind == "places")}
        with self.lock:
            self._require_adapter()
            if command.tool == "go_to_place":
                try:
                    place = self.places.get(command.params["name"])
                    target = (place.x, place.y, place.name)
                except ValueError as unknown:      # not a saved place: perhaps a room ("go to the kitchen", #35)
                    s = self._fresh_state()
                    target = self._room_destination(command.params["name"], s.x, s.y)
                    if target is None:
                        raise ValueError(f"I do not know a place or room named {command.params['name']!r}; remember "
                                         "the place first, or explore so I can find the room") from unknown
                command = replace(command, params={**command.params, "x": target[0], "y": target[1],
                                                   "name": target[2]})
            if command.kind == "sport":
                missing = self.adapter.supports(command.params["name"])
                if missing:
                    raise AdapterError(missing)
            if command.needs_confirmation:
                self._require_armed()
                name = command.params["name"]
                pending = self.confirmations.get(name)
                fresh = pending is not None and time.monotonic() - pending[1] <= L.CONFIRMATION_TTL_S
                if not (command.params.get("confirmed") and fresh and user_turn > pending[0]):
                    # First request, self-confirmation in the same turn, or an expired request: ask.
                    if not fresh:
                        self.confirmations[name] = (user_turn, time.monotonic())
                    return {"status": "needs_confirmation",
                            "instruction": f"Ask the user to confirm the {command.summary} now. If they agree in "
                                           f"their next message, call acrobatic again with confirmed=true."}
                del self.confirmations[name]
            self._require_armed()
            note = {}
            if command.kind == "approach":
                meant = self._label_for(command.params["target"])
                if meant:
                    # Said in the action, so the model tells the user and a wrong translation is caught at once.
                    command.params["meant"] = meant
                    command.summary = f"approach the {meant} (what I take '{command.params['target']}' to mean)"
                    note = {"looking_for": meant}
            task = Task(command, group, self.generation, user_turn=user_turn)
            self.queue.append(task)
            position = len(self.queue) + (1 if self.current else 0)
            task.awaiting = position == 1 and command.kind in ("velocity", "goto", "approach", "follow", "viewpoint")
            self.wake.notify_all()
            return {"status": "queued", "task_id": task.id, "position": position, "action": command.summary,
                    **note, "_task": task}

    def _after_group(self, group, timeout=90.):
        """Queries run in order with the motions of the same request: 'turn around and look' looks after
        the turn ('look' would otherwise see the old view, since motions are only queued)."""
        deadline = time.monotonic() + timeout
        with self.lock:
            while not self.closed and time.monotonic() < deadline:
                if not ((self.current is not None and self.current.group == group) or any(t.group == group for t in self.queue)):
                    return
                self.wake.wait(timeout=.1)

    def stop(self, reason="user"):
        """Immediate halt. Safe to call from any thread, at any time."""
        adapter = self.adapter
        sent, error = False, None
        if adapter is not None:
            try:
                adapter.stop_move()
                sent = True
            except Exception as exc:
                error = str(exc)
        with self.lock:
            self.generation += 1
            self.cancel.set()
            self.cancel = threading.Event()
            dropped = list(self.queue)
            self.queue.clear()
            self.confirmations.clear()
            for task in dropped:
                task.status, task.finished = "cancelled", time.monotonic()
                task.detail["reason"] = reason
                self.history.append(task)
            self.wake.notify_all()
        result = {"status": "stopped", "stop_sent": sent, "cancelled_queued": len(dropped), "reason": reason}
        if error:
            result["error"] = error
        return result

    def snapshot(self):
        with self.lock:
            return {"current": self.current.public() if self.current else None,
                    "queue": [t.public() for t in self.queue],
                    "recent": [t.public() for t in list(self.history)[-8:]],
                    "armed": self.armed, "requires_arming": bool(self.adapter and self.adapter.requires_arming),
                    "pending_confirmations": sorted(self.confirmations)}

    def last_frame_info(self):
        with self.lock:
            return dict(self.last_frame_meta) if self.last_frame_meta else None

    def tasks(self):
        """Every task still known (running, queued, or among the last 40 finished), by id, as public dicts."""
        with self.lock:
            known = [*self.history, *([self.current] if self.current else []), *self.queue]
            return {t.id: t.public() for t in known}

    def close(self):
        self.stop("app closing")
        with self.lock:
            self.closed = True
            self.wake.notify_all()
        if self.visual is not None:
            self.visual.save(force=True)

    # ---- queries -------------------------------------------------------------------
    def status_report(self):
        report = {"robot": None}
        with self.lock:
            adapter = self.adapter
            report["doing"] = self.current.command.summary if self.current else "nothing"
            if self.queue:
                report["queued"] = [t.command.summary for t in self.queue]
            if adapter and adapter.requires_arming:
                report["motion_armed"] = self.armed
            if self.history:
                last = self.history[-1]
                report["last_action"] = f"{last.command.summary}: {last.status}"
            if adapter is not None:
                report["done_so_far"] = _activity(self.activity, self.activity_since)
        if adapter is None:
            report["robot"] = "not connected"
            return report
        try:
            s = adapter.state()
        except AdapterError as error:
            report["robot"] = f"{adapter.label}: no telemetry ({error})"
            return report
        report["robot"] = adapter.label
        report["posture"] = s.posture
        if s.fault:
            report["fault"] = s.fault
        if self.start_pose:
            x0, y0, yaw0 = self.start_pose
            c, sn = math.cos(yaw0), math.sin(yaw0)
            dx, dy = s.x - x0, s.y - y0
            fwd, left = c * dx + sn * dy, -sn * dx + c * dy
            heading = round(math.degrees(_wrap(s.yaw - yaw0)))
            report["position"] = "right where it started" if math.hypot(dx, dy) < .15 else (
                f"{math.hypot(dx, dy):.1f} m from the start: {abs(fwd):.1f} m "
                                  f"{'ahead of' if fwd >= 0 else 'behind'} it and {abs(left):.1f} m to its "
                                  f"{'left' if left >= 0 else 'right'}, facing {abs(heading)} degrees "
                                  f"{'left' if heading >= 0 else 'right'} of the starting direction")
        if s.battery_pct is not None:
            report["battery_pct"] = round(s.battery_pct)
        return report

    def _refresh_cloud(self):
        points, state = self.adapter.lidar()
        level = P.base_to_level(points, state.roll, state.pitch)
        with self._cloud_lock:
            # Following compares scan times with this executor's monotonic
            # clock. Simulator uptime is a different epoch. Offline evaluators
            # supply this module's clock explicitly, including scan expiry.
            self.cloud.add(time.monotonic(), level, (state.x, state.y, state.yaw), state.height)
        localization = state.extra.get("localization")
        if len(level) and (localization is None or localization.get("accepted")):
            # The room map is in odometry coordinates, so a box stays where it is when the robot turns.
            world = P.level_to_odom(level, (state.x, state.y, state.yaw))
            # Flat simulator rooms have a fixed world floor at z=0. The local
            # ground ray can hit a book under the belly; using that as the map
            # floor would make the book disappear and shift all floor heights.
            world_z = state.extra.get("world_z")
            flat = getattr(self.adapter, "scene", None) in ("room", "empty", "house", "procedural")
            world[:, 2] += world_z if flat and world_z is not None else state.height
            self.room_map.mark_scan((state.x, state.y), world)
        return state

    def look(self):
        with self.lock:
            adapter = self._require_adapter()
        result = {}
        memory_warning = None
        try:
            for _ in range(3):  # a few scans so the non-repetitive L1 pattern fills in
                state = self._refresh_cloud()
                time.sleep(.05)
            with self._cloud_lock:
                floor_cloud = self.cloud.level((state.x, state.y, state.yaw))
            free = P.sector_free_space(self._with_bumps(self._with_low_obstacles(floor_cloud, state), state))
            result["free_space_m"] = {side: ("clear" if v is None else v) for side, v in free.items()}
        except AdapterError as error:
            result["lidar_error"] = str(error)
            floor_cloud, state = None, None
        try:
            frame = adapter.camera()
            s = frame.state
            cloud_base = np.zeros((0, 3))
            camera_floor_cloud = None
            if floor_cloud is not None and len(floor_cloud):
                with self._cloud_lock:
                    camera_floor_cloud = self.cloud.level((s.x, s.y, s.yaw))
                    local = camera_floor_cloud.copy()
                local[:, 2] -= s.height
                cloud_base = local @ P.rotation_rp(s.roll, s.pitch)
            detections = P.fuse(frame.masks, frame.rgb, cloud_base, frame.intrinsics, frame.camera_pos,
                                frame.camera_rot, s.roll, s.pitch, s.height)
            if self.detection_filter is not None:
                detections = self.detection_filter(detections)
            with self.lock:
                self._stable_ids(detections, (s.x, s.y, s.yaw))
                self.last_detections = [(d, (s.x, s.y, s.yaw)) for d in detections]
                seen_ids = self._remember(detections, (s.x, s.y, s.yaw), save=False)
                self._note_room_sightings(detections, (s.x, s.y, s.yaw))
                if adapter.kind == "sim":
                    self._verify_saved_objects(frame, camera_floor_cloud, detections, seen_ids)
                try:
                    self.object_memory.save()
                except (OSError, ValueError):
                    memory_warning = self.object_memory.save_error
                self.last_frame = frame.rgb
                self.last_frame_meta = {"stamp": time.time(), "objects": len(detections),
                                        "boxes": image_boxes(frame.masks, frame.rgb.shape)}
            self._memorise(frame, detections, (s.x, s.y, s.yaw))     # what each thing looked like (not in the result)
            result["objects"] = [{**d.public(), "direction": P.describe_bearing(d.bearing_deg)} for d in detections]
        except AdapterError as error:
            result["camera_error"] = str(error)
        result["note"] = "objects come from the front camera (about 120° wide); free_space_m is the free distance from the body on each side, 'clear' meaning nothing within 5 m"
        if memory_warning:
            result["note"] += "; " + memory_warning
        return result

    def _verify_saved_objects(self, frame, floor_cloud, detections, seen_ids):
        clear = []
        s = frame.state
        for item in self.object_memory.snapshot(include_missing=False):
            if item["id"] in seen_ids:
                continue
            bearing = math.atan2(item["y"]-s.y, item["x"]-s.x)-s.yaw
            distance = math.hypot(item["x"]-s.x, item["y"]-s.y)
            # A visible but unranged detection still protects its memory.
            visible = any(d.label == item["label"] and
                          (d.color is None or item.get("color") is None or d.color == item["color"]) and
                          abs(_wrap(math.radians(d.bearing_deg)-bearing)) < math.radians(20) and
                          (d.distance_m is None or abs(d.distance_m-distance) < .6)
                          for d in detections)
            if not visible and clear_saved_location(item, frame, floor_cloud):
                clear.append(item["id"])
        missing = self.object_memory.observe_empty(clear, observed_at=time.monotonic(),
                                                   seen_ids=seen_ids, save=False)
        if missing:
            self.memory = [row for row in self.memory if not any(
                row[0].label == item["label"] and row[0].color == item.get("color") and
                math.hypot(row[1]-item["x"], row[2]-item["y"]) < .6 for item in missing)]

    # ---- worker ----------------------------------------------------------------------
    def _require_adapter(self):
        if self.adapter is None or not self.adapter.connected:
            raise AdapterError("No robot is connected. Open the simulator or connect the Go2 first.")
        return self.adapter

    def _require_armed(self):
        if self.adapter.requires_arming and not self.armed:
            raise AdapterError("Motion is disarmed. Ask the user to switch on Arm Motion in the app.")

    def _worker(self):
        while True:
            with self.lock:
                while not self.queue and not self.closed:
                    self.wake.wait()
                if self.closed:
                    return
                task = self.queue.popleft()
                if task.generation != self.generation:
                    continue
                self.current = task
                task.status = "running"
                cancel = self.cancel
            try:
                self._run(task, cancel)
                if task.status == "running":
                    task.status = "completed"
            except Cancelled:
                task.status = "stopped"
            except AdapterError as error:
                if task.status != "blocked":
                    task.status = "failed"
                task.detail["error"] = str(error)
                self._safe_stop()
            except Exception as error:   # never leave the robot moving on a bug
                task.status, task.detail["error"] = "failed", f"internal error: {error}"
                self._safe_stop()
            task.finished = time.monotonic()
            with self.lock:
                self.current = None
                self.history.append(task)
                self.activity = self.activity[-499:] + [task]
                if task.status in ("blocked", "failed") and self.queue:
                    # Later steps assumed this one succeeded; don't run them blindly.
                    dropped = [t for t in self.queue if t.group == task.group]
                    for t in dropped:
                        self.queue.remove(t)
                        t.status, t.finished = "cancelled", time.monotonic()
                        t.detail["reason"] = f"previous step {task.status}"
                        self.history.append(t)
                    task.detail["cancelled_after"] = len(dropped)
                remaining = sum(1 for t in self.queue if t.group == task.group)
                if task.awaiting:                # its tool call is still waiting: answer it there
                    task.awaiting = False
                    task.reported = task.status in ("blocked", "failed")
                    self.wake.notify_all()
            task.detail["group_remaining"] = remaining
            try:
                self.on_event(task)
            except Exception:
                pass

    def _safe_stop(self):
        try:
            if self.adapter is not None:
                self.adapter.stop_move()
        except Exception:
            pass

    def _check(self, cancel):
        if cancel.is_set():
            raise Cancelled()

    def _sleep(self, seconds, cancel):
        if cancel.wait(seconds):
            raise Cancelled()

    def _fresh_state(self):
        s = self.adapter.state()
        if s.age > STALE_S:
            raise AdapterError(f"robot telemetry is {s.age:.1f} s old; stopped for safety")
        if s.posture == "fallen":
            raise AdapterError("the robot has fallen" + (f" ({s.fault})" if s.fault else "") + "; it needs a recovery stand")
        if s.extra.get("localization", {}).get("state") == "lost":
            raise AdapterError("scan localisation lost; stop and obtain clear LiDAR scans before navigating")
        return s

    def _run(self, task, cancel):
        kind = task.command.kind
        if kind == "velocity":
            self._ensure_standing(task, cancel)
            self._velocity(task, task.command.params, cancel)
        elif kind == "sport":
            self._sport(task, cancel)
        elif kind == "wait":
            self._sleep(task.command.params["duration"], cancel)
        elif kind == "goto":
            budget = _MotionDeadline(cancel, 180) if task.command.params["target"] == "place" else cancel
            self._ensure_standing(task, budget)
            self._goto(task, budget)
        elif kind == "approach":
            budget = _MotionDeadline(cancel, 180)
            self._ensure_standing(task, budget)
            self._approach(task, budget)
        elif kind == "viewpoint":
            if self.adapter.kind == "go2":
                raise AdapterError("exploration viewpoints are currently validated only in simulation")
            budget = _MotionDeadline(cancel, task.command.params["timeout_s"])
            self._ensure_standing(task, budget)
            self._visit_viewpoint(task, budget)
        elif kind == "follow":
            if self.adapter.kind == "go2":
                raise AdapterError("following people only works in the simulator for now; it hasn't been tried "
                                   "on the real Go2")
            self._ensure_standing(task, cancel)
            self._follow(task, cancel)
        elif kind == "mission":
            if self.adapter.kind == "go2":
                raise AdapterError("missions only run in the simulator for now: unattended runs on the real Go2 "
                                   "need its safety work first (#34)")
            from . import missions          # here, not at the top: missions builds on this module
            self._ensure_standing(task, cancel)
            self._hold(task, cancel)
            missions.run(self, task, cancel)
        else:
            raise AdapterError(f"unsupported command {kind}")

    # ---- postures, tricks, body ----------------------------------------------------------
    def _ensure_standing(self, task, cancel):
        s = self._fresh_state()
        if s.posture in ("standing", "walking"):
            return
        if s.posture == "busy":
            self._wait_idle(cancel, 8.0)
            s = self._fresh_state()
            if s.posture in ("standing", "walking"):
                return
        if s.posture in LOW_POSTURES or s.posture in ("busy", "unknown"):
            # Standing up takes longer than START_CHECK_S, so the tool call hears 'queued' now rather than after a
            # second's wait; the path is checked once standing, and a block then is reported as a robot update.
            self._underway(task)
            self._hold(task, cancel)
            note = self.adapter.sport("stand")
            self._wait_idle(cancel, 6.0, want_ready=True)
            task.detail["stood_up_first"] = True
            if note and note != "already standing":
                task.detail["note"] = note

    def _wait_idle(self, cancel, timeout, want_ready=False):
        deadline = time.monotonic() + timeout
        time.sleep(.15)
        while time.monotonic() < deadline:
            self._check(cancel)
            s = self.adapter.state()
            if s.posture == "fallen":
                raise AdapterError("the robot fell" + (f" ({s.fault})" if s.fault else ""))
            if s.posture != "busy" and (not want_ready or s.posture in ("standing", "walking")):
                return s
            time.sleep(.1)
        raise AdapterError("the robot did not finish its posture change in time")

    def _sport(self, task, cancel):
        p = task.command.params
        name = p["name"]
        args = {k: v for k, v in p.items() if k not in ("name", "expected_s")}
        if task.command.tool in ("trick", "body", "acrobatic") and name not in ("speed_level",):
            self._ensure_standing(task, cancel)
        elif self._fresh_state().posture in ("busy", "walking"):
            self._wait_idle(cancel, 8.0)
        self._hold(task, cancel)
        note = self.adapter.sport(name, **args)
        if note:
            task.detail["note"] = note
        s = self._wait_idle(cancel, p.get("expected_s", 2.) * 2 + 3)
        task.detail["posture"] = s.posture

    # ---- motion ------------------------------------------------------------------------------
    def _with_bumps(self, cloud, s):
        """The level-frame cloud plus the spots it recently bumped into (as obstacle points)."""
        now = time.monotonic()
        with self.lock:
            self.bumps = [b for b in self.bumps if now - b[0] <= BUMP_MEMORY_S]
            spots = [pts for _, pts in self.bumps]
        if not spots:
            return cloud
        return np.concatenate([cloud, P.odom_to_level(np.concatenate(spots), (s.x, s.y, s.yaw))])

    def _with_low_obstacles(self, cloud, s):
        """Add confirmed low surfaces to safety checks, preserving raw camera depth."""
        low = self.room_map.low_obstacles()
        if not len(low):
            return cloud
        return np.concatenate([cloud, P.odom_to_level(low, (s.x, s.y, s.yaw))])

    def _remember_bump(self, s, ux, uy):
        """Mark a short wall across the travel direction just past the body edge, in odom coordinates."""
        extent = abs(ux) * P.HALF_LENGTH + abs(uy) * P.HALF_WIDTH + .05
        c, sn = math.cos(s.yaw), math.sin(s.yaw)
        wx, wy = c * ux - sn * uy, sn * ux + c * uy          # travel direction in odom
        px, py = -wy, wx                                     # across it
        offsets = np.linspace(-BUMP_HALF_WIDTH, BUMP_HALF_WIDTH, 13)
        pts = np.array([(s.x + wx * extent + px * o, s.y + wy * extent + py * o, z)
                        for o in offsets for z in (.15, .3)])
        with self.lock:
            self.bumps.append((time.monotonic(), pts))

    def _clearance(self, ux, uy, turning, *, footprint=None):
        """Free distance along the travel direction, limited to what the LiDAR can actually see. Drops (a
        ledge, stairs going down) and spots it recently bumped into count as obstacles."""
        self._refresh_cloud()
        # Mapping can take time while the robot keeps moving. Reproject the
        # captured returns at the current pose before judging stopping space.
        s = self._fresh_state()
        with self._cloud_lock:
            cloud = self.cloud.level((s.x, s.y, s.yaw))
        cloud = self._with_low_obstacles(cloud, s)
        dimensions={} if footprint is None else {'footprint':footprint}
        if turning:
            return P.turn_clearance(self._with_bumps(cloud,s),**dimensions)
        self._observed=P.observed_along(cloud,ux,uy,**dimensions)
        self._drop=P.drop_along(cloud,ux,uy,**dimensions)
        self._obstacle=P.obstacle_along(cloud,ux,uy,**dimensions)
        bumped = self._with_bumps(cloud, s)
        self._bump=P.obstacle_along(bumped,ux,uy,**dimensions) if len(bumped)>len(cloud) else math.inf
        return min(self._obstacle, self._drop, self._bump, self._observed)

    def _why_blocked(self, clear, toward):
        """Which limit stopped a straight move: a drop, a remembered bump, an obstacle, or what it can't see."""
        if self._drop != math.inf and self._drop <= clear + 1e-9:
            return f"drop {toward}"
        if self._bump < self._obstacle and self._bump <= clear + 1e-9:
            return f"something I bumped into earlier, {toward}"
        if self._obstacle < self._observed - 1e-9 or self._bump < self._observed - 1e-9:
            return f"obstacle {toward}"
        return UNSEEN

    @staticmethod
    def _margin(speed):
        """Stopping margin: fixed buffer + conservative braking distance + sensing/command latency."""
        return .22 + speed * speed / (2 * 1.2) + speed * .25

    def _velocity(self, task, p, cancel, leg=None):
        """Run one velocity segment; returns measured progress (m or rad)."""
        adapter = self.adapter
        turning = p.get("angle") is not None or (p["vx"] == 0 and p["vy"] == 0)
        tracking = (task.command.kind == "viewpoint" or p.get("track_line", False)) and not turning
        if tracking:
            p = dict(p)
        start = self._fresh_state()
        x0, y0, yaw0 = start.x, start.y, start.yaw
        c0, s0 = math.cos(yaw0), math.sin(yaw0)
        wx, wy = c0 * p["ux"] - s0 * p["uy"], s0 * p["ux"] + c0 * p["uy"]   # travel direction in odom
        speed = p["speed"]
        dimensions=({'footprint':(BODY_HALF_LENGTH_M,BODY_HALF_WIDTH_M)}
                    if task.command.kind=='viewpoint' else {})
        try:
            clear=self._clearance(p['ux'],p['uy'],turning,**dimensions)
        except AdapterError as error:
            raise AdapterError(f"no LiDAR data, so I can't check the path ({error})") from None
        toward = None if turning else _toward(p["ux"], p["uy"])
        if turning and clear < .03:
            raise _blocked(task, "too close to an obstacle or a drop to turn safely", clear, 0.)
        if not turning and clear < self._margin(speed):
            task._observed = self._observed
            why = self._why_blocked(clear, toward)
            if why.startswith("obstacle") or why == UNSEEN:
                why = "path is blocked"     # the wording go2-ft-04 was trained on (_blocked adds UNSEEN)
            raise _blocked(task, why, clear, 0., toward)
        self._underway(task)
        self._hold(task, cancel)            # the loop below checks the path again before the first step
        distance, angle, duration = p.get("distance"), p.get("angle"), p.get("duration")
        nominal = duration if duration is not None else (
            distance / speed if distance is not None else angle / p["turn_rate"])
        deadline = time.monotonic() + nominal + 2.5
        began = time.monotonic()
        last_send = last_scan = 0.
        progress = 0.
        yaw_prev, turned = yaw0, 0.
        outcome = "completed"
        # Stall check: expected progress at the commanded speed. Legs take a moment to get going (a side-step
        # in the simulator barely moves for the first 0.3 s), so the check starts once the robot has first
        # reached half the commanded speed, after the ramp-up and a short grace period. If it never gets going
        # (it started pressed against glass), it starts anyway STALL_START_S later. Only samples from then on
        # are judged, so a slow start can't count against it.
        rate = p["turn_rate"] if turning else speed
        commanded_turn_rate = rate
        commanded_speed = speed
        earliest = began + rate / (DECEL_ANGULAR if turning else DECEL_LINEAR) + STALL_GRACE_S
        watch_from = None
        recent = deque()

        def stalled(state, stamp, at):
            nonlocal watch_from
            moving = abs(state.wz) if turning else state.vx*p["ux"] + state.vy*p["uy"]
            if watch_from is None and stamp >= earliest and (moving >= rate/2 or stamp >= earliest+STALL_START_S):
                watch_from = stamp
            if watch_from is None:
                return False
            recent.append((stamp, at))
            while len(recent) > 1 and stamp-recent[1][0] >= STALL_WINDOW_S:
                recent.popleft()
            old_t, old_progress = recent[0]
            expected = (commanded_turn_rate if turning else commanded_speed)*STALL_FRACTION
            # Confirm poor progress over the whole window and low current
            # velocity, so delayed position updates alone do not cause a stop.
            return (stamp-old_t >= STALL_WINDOW_S and
                    at-old_progress < expected*(stamp-old_t) and moving < expected)

        try:
            while True:
                self._check(cancel)
                now = time.monotonic()
                s = None
                if not turning:
                    s = self._fresh_state()
                    progress = (s.x-x0)*wx + (s.y-y0)*wy
                    # Check both sides of potentially slow scan processing.
                    # A confirmed contact must not wait for another map update.
                    if stalled(s, now, progress):
                        outcome = "stalled"
                        break
                send_due = now - last_send >= .1
                if tracking and send_due:
                    s = self._fresh_state()
                    # Follow the planned line in odometry coordinates. Small
                    # gait/yaw errors otherwise accumulate against a side wall
                    # even while the requested body velocity stays constant.
                    cross = -(s.x-x0)*wy + (s.y-y0)*wx
                    correction = max(-.4*speed, min(.4*speed, -2.*cross))
                    forward = math.sqrt(max(0., speed*speed-correction*correction))
                    vxw, vyw = forward*wx-correction*wy, forward*wy+correction*wx
                    co, sn = math.cos(s.yaw), math.sin(s.yaw)
                    p["vx"], p["vy"] = co*vxw+sn*vyw, -sn*vxw+co*vyw
                    p["ux"], p["uy"] = p["vx"]/speed, p["vy"]/speed
                    # Keep direction updates and their safety check together.
                    # Mapping can shift the scan/send timers apart; no corrected
                    # velocity may be sent using a check for the previous one.
                    clear = self._clearance(p["ux"], p["uy"], False, **dimensions)
                    s = self._fresh_state()
                    now = time.monotonic()
                    last_scan = now
                    if clear < self._margin(max(s.speed, .1)):
                        outcome = "blocked"
                        at = (s.x-x0)*wx + (s.y-y0)*wy
                        trigger = (clear, self._observed, at, self._why_blocked(clear, toward))
                        break
                if send_due:
                    wz = p["wz"]
                    if not turning and distance is not None:
                        current = self._fresh_state()
                        remaining = max(0., distance-((current.x-x0)*wx+(current.y-y0)*wy))
                        # Slow before the final stop rather than relying on one
                        # nominal braking impulse across physics runtimes.
                        commanded_speed = min(speed, max(L.DEFAULT_SPEED_MPS,
                            math.sqrt(2*FINAL_DECEL_LINEAR*max(0., remaining-.05))))
                    if turning and angle is not None:
                        current = self._fresh_state()
                        remaining = max(0., angle-abs(turned+_wrap(current.yaw-yaw_prev)))
                        # Close the last part of a measured turn slowly. The
                        # walking policy may brake less sharply than odometry's
                        # nominal model, particularly across physics runtimes.
                        commanded_turn_rate = min(rate, max(.15, 2.*remaining))
                        wz = math.copysign(commanded_turn_rate, wz)
                    scale = commanded_speed/speed if not turning else 1.
                    adapter.move(p["vx"]*scale, p["vy"]*scale, wz)
                    last_send = now
                if s is None:
                    s = self._fresh_state()
                turned += _wrap(s.yaw - yaw_prev)
                yaw_prev = s.yaw
                progress = abs(turned) if turning else (s.x - x0) * wx + (s.y - y0) * wy
                if not turning and not tracking and now - last_scan >= .1:
                    last_scan = now
                    clear=self._clearance(p['ux'],p['uy'],False,**dimensions)
                    # Sensing/mapping may span the instant contact stops the
                    # robot. Use the post-scan sample and clock for stall timing.
                    s = self._fresh_state()
                    now = time.monotonic()
                    progress = (s.x-x0)*wx + (s.y-y0)*wy
                    if clear < self._margin(max(s.speed, .1)):
                        outcome = "blocked"
                        trigger = (clear, self._observed, progress, self._why_blocked(clear, toward))
                        break
                if stalled(s, now, progress):
                    outcome = "stalled"
                    break
                if distance is not None and progress >= distance - s.speed ** 2 / (2 * DECEL_LINEAR) - .01:
                    break
                if angle is not None and progress >= angle - s.wz ** 2 / (2 * DECEL_ANGULAR) - .01:
                    break
                if duration is not None and now - began >= duration:
                    break
                if now > deadline:
                    outcome = "timeout"
                    break
                time.sleep(TICK_S)
        finally:
            adapter.stop_move()
        settle = time.monotonic() + 2.0
        while time.monotonic() < settle:
            s = adapter.state()
            if s.speed < .05 and abs(s.wz) < .1:
                break
            time.sleep(.05)
        s = adapter.state()
        turned += _wrap(s.yaw - yaw_prev)
        progress = abs(turned) if turning else (s.x - x0) * wx + (s.y - y0) * wy
        record = task.detail if leg is None else task.detail.setdefault("legs", [{}])[-1]
        if turning:
            record.update(turned_deg=round(math.degrees(progress)))
            if angle is not None:
                record["target_deg"] = round(math.degrees(angle))
        else:
            record.update(moved_m=round(progress, 2))
            if distance is not None:
                record["target_m"] = round(distance, 2)
        if p.get("note"):
            task.detail["note"] = p["note"]
        if outcome == "blocked":
            # Judge by what the LiDAR showed when deciding to stop (a re-scan after braking can miss an
            # obstacle at the corridor's edge); the gap shrinks by the distance covered while braking.
            clear, observed, at, why = trigger
            task._observed = observed
            raise _blocked(task, why, max(0., clear - max(0., progress - at)), progress, toward)
        if outcome == "stalled":
            backed = 0.
            if not turning:
                self._remember_bump(self._fresh_state(), p["ux"], p["uy"])
                backed = self._back_off(p["ux"], p["uy"], cancel)
            error = _blocked(task, STALLED, 0., progress, toward)
            task.detail["blocked"]["stall"] = True
            if backed:
                task.detail["blocked"]["backed_off_m"] = round(float(backed), 2)
            raise error
        if outcome == "timeout":
            task.status = "failed"
            task.detail["error"] = "did not reach the target in time"
        return progress

    def _back_off(self, ux, uy, cancel):
        """After a stall: walk BACKOFF_M back the way it came, only if the LiDAR says that way is clear, so it
        isn't left leaning on the obstacle. Returns the distance it moved back."""
        try:
            if self._clearance(-ux, -uy, False) < self._margin(BACKOFF_SPEED) + BACKOFF_M:
                return 0.
        except AdapterError:
            return 0.
        adapter = self.adapter
        s0 = self._fresh_state()
        c, sn = math.cos(s0.yaw), math.sin(s0.yaw)
        wx, wy = -(c * ux - sn * uy), -(sn * ux + c * uy)
        deadline = time.monotonic() + BACKOFF_M / BACKOFF_SPEED + 1.5
        moved, last_send = 0., 0.
        try:
            while True:
                self._check(cancel)
                now = time.monotonic()
                if now - last_send >= .1:
                    adapter.move(-ux * BACKOFF_SPEED, -uy * BACKOFF_SPEED, 0.)
                    last_send = now
                s = self._fresh_state()
                moved = (s.x - s0.x) * wx + (s.y - s0.y) * wy
                if moved >= BACKOFF_M - s.speed ** 2 / (2 * DECEL_LINEAR) - .01 or now > deadline:
                    break
                time.sleep(TICK_S)
        finally:
            adapter.stop_move()
        return max(0., moved)

    def _goto_leg(self, task, cancel, tx, ty, *, max_speed=None, arrival_tolerance=.08):
        """Turn toward (tx, ty) in odom, then walk there in legs within the per-call limits."""
        s = self._fresh_state()
        dx, dy = tx - s.x, ty - s.y
        distance = math.hypot(dx, dy)
        if distance < arrival_tolerance:
            return 0.
        bearing = _wrap(math.atan2(dy, dx) - s.yaw)
        if math.cos(bearing) < -.7 and distance <= 2.0:   # short way back: reverse instead of turning around
            ux, uy = math.cos(bearing), math.sin(bearing)
            speed = L.default_speed(distance)
            if max_speed is not None:
                speed = min(speed, max_speed)
            self._velocity(task, {"vx": ux * speed, "vy": uy * speed, "wz": 0., "ux": ux, "uy": uy, "speed": speed,
                                  "distance": min(distance, L.MAX_DISTANCE_M), "track_line": True}, cancel, leg=True)
            return distance
        if distance < .35:   # short hop: side-step instead of turning on the spot
            ux, uy = math.cos(bearing), math.sin(bearing)
            speed = L.DEFAULT_SPEED_MPS
            if max_speed is not None:
                speed = min(speed, max_speed)
            self._velocity(task, {"vx": ux * speed, "vy": uy * speed, "wz": 0., "ux": ux, "uy": uy, "speed": speed,
                                  "distance": distance, "track_line": True}, cancel, leg=True)
            return distance
        if abs(bearing) > math.radians(8):
            rate = L.default_turn_rate(abs(bearing))
            self._velocity(task, {"vx": 0., "vy": 0., "wz": math.copysign(rate, bearing), "ux": 0., "uy": 0.,
                                  "speed": 0., "turn_rate": rate, "angle": abs(bearing)}, cancel, leg=True)
        s = self._fresh_state()
        dx, dy = tx-s.x, ty-s.y
        distance = math.hypot(dx, dy)
        if distance < arrival_tolerance:
            return 0.
        leg = min(distance, L.MAX_GOTO_LEG_M)
        speed = L.default_speed(leg)
        if max_speed is not None:
            speed = min(speed, max_speed)
        # Turns have a measured tolerance. Aim the translation at the waypoint
        # from the actual post-turn pose, rather than extending a heading error
        # into the obstacle beside the planned line.
        co, sn = math.cos(s.yaw), math.sin(s.yaw)
        ux, uy = (co*dx+sn*dy)/distance, (-sn*dx+co*dy)/distance
        self._velocity(task, {"vx": ux*speed, "vy": uy*speed, "wz": 0., "ux": ux, "uy": uy, "speed": speed,
                              "distance": leg, "track_line": True}, cancel, leg=True)
        return leg

    def _goto(self, task, cancel):
        p = task.command.params
        s = self._fresh_state()
        if p["target"] == "start":
            if self.start_pose is None:
                raise AdapterError("the start position is unknown")
            tx, ty = self.start_pose[:2]
        elif p["target"] == "place":
            tx, ty = p["x"], p["y"]
            if math.hypot(tx-s.x, ty-s.y) > 20.:
                raise AdapterError("the named place is over the 20 m go_to limit; use intermediate places")
            task.detail["destination"] = {"name": p["name"], "x_m": tx, "y_m": ty, "frame": "map"}
        else:
            c, sn = math.cos(s.yaw), math.sin(s.yaw)
            tx, ty = s.x + c * p["x"] - sn * p["y"], s.y + sn * p["x"] + c * p["y"]
        task.detail["legs"] = []
        for _ in range(3):                  # a few scans first, so the map shows what's around
            self._refresh_cloud()
        replans = 0
        waypoints = None
        for _ in range(12):
            s = self._fresh_state()
            remaining = math.hypot(tx - s.x, ty - s.y)
            if remaining < .12:
                break
            # Keep the remaining checked waypoints between successful legs.
            # A new live obstruction invalidates the route below; a temporary
            # gap in map coverage alone must not discard a valid detour.
            if waypoints is None:
                route = self._route(s, tx, ty)
                waypoints = list(route) if route else [(tx, ty)]
                if route:
                    task.detail["route"] = [[round(x, 2), round(y, 2)] for x, y in route]
            wx, wy = waypoints[0]
            task.detail["legs"].append({})
            try:
                self._goto_leg(task, cancel, wx, wy, arrival_tolerance=.02 if len(waypoints)>1 else .08)
                if task.status == "failed":
                    raise AdapterError(task.detail.get("error", "navigation motion failed"))
                arrived = self._fresh_state()
                if math.hypot(wx-arrived.x, wy-arrived.y) < .08:
                    waypoints.pop(0)
                    if not waypoints:
                        waypoints = None
            except _BlockedError:
                # Something the map didn't show yet is in the way: the scans have now added it, so plan again.
                if replans >= MAX_REPLANS:
                    raise
                replans += 1
                waypoints = None
                task.status = "running"
                task.detail["replanned"] = replans
                task.detail.pop("blocked", None)
                for _ in range(3):
                    self._refresh_cloud()
        s = self._fresh_state()
        task.detail["remaining_m"] = round(math.hypot(tx - s.x, ty - s.y), 2)
        if math.hypot(tx - s.x, ty - s.y) >= .12:
            raise AdapterError("navigation reached its movement limit before arriving")

    def _route(self, s, tx, ty, *, known_only=False, cancel=None):
        """Waypoints in odometry coordinates from the robot to (tx, ty) around obstacles on the room map, ending
        at the target; [] when the straight line is already clear; None when the map has no route (the target is
        off the map, inside an obstacle, or walled off)."""
        m = self.room_map.snapshot()
        m.grow = False
        start, goal = m.world_to_cell(s.x, s.y), m.world_to_cell(tx, ty)
        if start is None or goal is None:
            return None
        initial_forbidden = set()
        co, sn = math.cos(s.yaw), math.sin(s.yaw)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            if cancel is not None:
                self._check(cancel)
            nx, ny = m.cell_to_world(start[0]+dr, start[1]+dc)
            dx, dy = nx-s.x, ny-s.y
            length = max(math.hypot(dx, dy), 1e-9)
            ux, uy = (co*dx+sn*dy)/length, (-sn*dx+co*dy)/length
            if self._clearance(ux, uy, False) < self._margin(.2):
                initial_forbidden.add((dr, dc))
        # Each live probe also observes floor/obstacles. Plan on that completed
        # evidence rather than the snapshot taken before the scans arrived.
        m = self.room_map.snapshot()
        m.grow = False
        start, goal = m.world_to_cell(s.x, s.y), m.world_to_cell(tx, ty)
        if start is None or goal is None:
            return None
        now = time.monotonic()
        with self.lock:                                     # spots it bumped into: the LiDAR can't see them
            bumps = [pts for t, pts in self.bumps if now - t <= BUMP_MEMORY_S]
        extra = np.concatenate(bumps) if bumps else None
        for clearance in ROUTE_CLEARANCE_M:
            if known_only:
                if extra is not None:
                    m.mark_obstacles(extra[:, :2])
                blocked = ~_safe_cells(m, clearance)
                if blocked[start] or blocked[goal]:
                    continue
            else:
                blocked = m.blocked(clearance, extra)
            k = int(math.ceil(clearance / m.resolution))         # it may already stand closer than that
            if not known_only:
                blocked[max(0, start[0] - k):start[0] + k + 1, max(0, start[1] - k):start[1] + k + 1] = False
            if m.line_clear(blocked, start, goal):
                distance = math.hypot(tx - s.x, ty - s.y)
                co, sn = math.cos(s.yaw), math.sin(s.yaw)
                dx, dy = tx - s.x, ty - s.y
                ux, uy = (co * dx + sn * dy) / max(distance, 1e-9), (-sn * dx + co * dy) / max(distance, 1e-9)
                if distance < .01 or self._clearance(ux, uy, False) >= self._margin(.2):
                    return []
            path = (turning_astar(m.grid, start, goal, blocked, _safe_cells(m, .65),
                                  cancelled=None if cancel is None else cancel.is_set,
                                  forbidden_first_steps=initial_forbidden or ()) if known_only else
                    turning_astar(m.grid, start, goal, blocked, _safe_cells(m, .6) & ~m.blocked(.6, extra),
                                  cancelled=None if cancel is None else cancel.is_set,
                                  forbidden_first_steps=initial_forbidden or (),
                                  unknown_cost=UNKNOWN_COST))
            if path is not None:
                cells = keep_turns(path)
                if not known_only:
                    # Bends need observed turning space. Unknown cells may be
                    # traversed in straight legs, with live LiDAR still in charge.
                    # Keep the first planned bend: simplifying from the carved
                    # start area could restore a direction the live probe just
                    # refused, even when the rest of that line looks map-free.
                    cells = cells[:1] + simplify(cells[1:], lambda a, b: m.line_clear(blocked, a, b))
                route = [m.cell_to_world(*c) for c in cells[1:-1]] + [(tx, ty)]
                points = [(s.x, s.y)] + route
                length = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
                if not known_only and length > MAX_DETOUR * math.hypot(tx - s.x, ty - s.y) + 2.:
                    return None               # a long way round, probably through unmapped space: don't wander
                return route
        return None

    def _exploration_forbidden(self, s, cancel=None):
        """Probe the same world directions as the navigation graph.

        A tiny step to a cell centre can point diagonally when the robot is off
        centre. Do not forbid an entire straight corridor on that basis; the
        velocity worker still checks the exact executed segment independently.
        """
        m = self.room_map
        cell = m.world_to_cell(s.x, s.y)
        if cell is None:
            return tuple(EXPLORATION_STEPS)
        co, sn = math.cos(s.yaw), math.sin(s.yaw)
        forbidden = []
        for dr, dc in EXPLORATION_STEPS:
            if cancel is not None:
                self._check(cancel)
            distance = math.hypot(dr,dc)
            ux, uy = (co*dc+sn*dr)/distance, (-sn*dc+co*dr)/distance
            if self._clearance(ux,uy,False,footprint=(BODY_HALF_LENGTH_M,BODY_HALF_WIDTH_M))<self._margin(.15):
                forbidden.append((dr, dc))
        return tuple(forbidden)

    def _exploration_navigation(self, s, cancel=None, planning_resolution_m=.1):
        forbidden = self._exploration_forbidden(s, cancel)
        m = self.room_map.snapshot()
        with self.lock:
            bumps = [pts[:, :2] for t, pts in self.bumps if time.monotonic()-t <= BUMP_MEMORY_S]
        if bumps:
            m.mark_obstacles(np.concatenate(bumps))
        return ExplorationNavigation(m, (s.x, s.y, s.yaw), forbidden_steps=forbidden,
                                     cancelled=None if cancel is None else cancel.is_set,
                                     planning_resolution_m=planning_resolution_m)

    def _viewpoint_recovery_leg(self, task, s, navigation):
        m = navigation.source_map
        path = recovery_route(m, (s.x, s.y, s.yaw), clearance=navigation.turning_radius,
                              forbidden_steps=navigation.forbidden)
        if path is None or len(path) < 2:
            raise _blocked(task, "no observed translation route to turning space", math.inf, 0.)
        first = path[1]
        delta = first[0]-path[0][0], first[1]-path[0][1]
        last = first
        for a, b in zip(path[1:4], path[2:5]):
            if (b[0]-a[0], b[1]-a[1]) != delta:
                break
            last = b
        # Recovery retains the measured subcell offset and fixed body heading.
        wx = s.x + (last[1] - path[0][1]) * m.resolution
        wy = s.y + (last[0] - path[0][0]) * m.resolution
        task.detail["recovering"] = "translating to observed turning space"
        return wx, wy, s.yaw

    def _visit_viewpoint(self, task, cancel):
        p = task.command.params
        task.detail["legs"] = []
        replans, recovered, no_progress, prefer_fine = 0, False, 0, False
        for _ in range(8):
            self._check(cancel)
            self._refresh_cloud()
        for _ in range(400):
            self._check(cancel)
            for _ in range(3):
                self._refresh_cloud()
            s = self._fresh_state()
            navigation = self._exploration_navigation(s,cancel,.05 if prefer_fine else .1)
            if not navigation.start_fits and navigation.can_refine:
                refined=navigation.refined()
                if refined.start_fits:
                    navigation=refined
                    prefer_fine=True
            m = navigation.map
            if not navigation.start_fits:
                wx, wy, heading = self._viewpoint_recovery_leg(task, s, navigation)
                recovered = True
            else:
                remaining = math.hypot(p["x"]-s.x, p["y"]-s.y)
                if recovered and remaining > .12 and navigation.turning[navigation.start_cell]:
                    task.detail["recovered_pose"] = [s.x, s.y, s.yaw]
                    raise _blocked(task, "recovered turning space; request a fresh viewpoint", math.inf, 0.)
                angle = _wrap(p["yaw"]-s.yaw)
                if remaining<.12 and abs(angle)>math.radians(6) and not navigation.turning[navigation.start_cell] and navigation.can_refine:
                    prefer_fine=True
                    continue
                if remaining < .12 and (abs(angle) < math.radians(6) or navigation.turning[navigation.start_cell]):
                    if abs(angle) > math.radians(6):
                        task.detail["legs"].append({})
                        self._turn_by(task, cancel, angle)
                    task.detail["remaining_m"] = round(remaining, 2)
                    task.detail["room"] = self.room_at(s.x, s.y)
                    return
                path = navigation.route(p["x"], p["y"])
                self._check(cancel)
                if path is None or len(path) < 2:
                    if navigation.can_refine:
                        prefer_fine=True
                        continue
                    if navigation.turning[navigation.start_cell]:
                        raise _blocked(task, "no safe route through observed free space", math.inf, 0.)
                    wx, wy, heading = self._viewpoint_recovery_leg(task, s, navigation)
                    recovered = True
                else:
                    wx, wy, heading = navigation.first_leg(path)
                    task.detail["route"] = [[*m.cell_to_world(*cell[:2]), navigation.headings[cell[2]]]
                                            for cell in path[::max(1, len(path)//50)]]
                    if path[0][:2] == path[1][:2]:
                        task.detail["legs"].append({})
                        self._turn_by(task, cancel, _wrap(heading-s.yaw))
                        if task.status == "failed":
                            raise AdapterError(task.detail.get("error", "viewpoint turn failed"))
                        continue
            dx, dy = wx-s.x, wy-s.y
            distance = math.hypot(dx, dy)
            if distance < .005:
                raise _blocked(task, "route has no remaining translation", math.inf, 0.)
            co, sn = math.cos(s.yaw), math.sin(s.yaw)
            ux, uy = (co*dx+sn*dy)/distance, (-sn*dx+co*dy)/distance
            task.detail["legs"].append({})
            try:
                moved = self._velocity(task, {"vx": ux*.15, "vy": uy*.15, "wz": 0., "ux": ux, "uy": uy,
                                             "speed": .15, "distance": distance}, cancel, leg=True)
                if task.status == "failed":
                    raise AdapterError(task.detail.get("error", "viewpoint motion failed"))
                no_progress = no_progress+1 if moved < .005 else 0
                if no_progress >= 3:
                    raise _blocked(task, "viewpoint route tracking made no progress", math.inf, 0.)
            except _BlockedError:
                if navigation.can_refine:
                    prefer_fine=True
                if replans >= MAX_REPLANS:
                    raise
                replans += 1
                task.status = "running"
                task.detail.pop("blocked", None)
                task.detail["replanned"] = replans
        raise AdapterError("exploration viewpoint exceeded its movement limit")

    def _stable_ids(self, detections, pose):
        """Keep an object's id across looks (same label within 0.6 m of where it was seen), so an id the
        model got from one look still means that object when approach looks again."""
        c, sn = math.cos(pose[2]), math.sin(pose[2])
        taken = set()
        for d in detections:
            d.id = None
            if d.x_m is not None:
                wx, wy = pose[0] + c * d.x_m - sn * d.y_m, pose[1] + sn * d.x_m + c * d.y_m
                near = [(math.hypot(m[1] - wx, m[2] - wy), m[0].id) for m in self.memory
                        if m[0].label == d.label and m[0].id not in taken]
                near = [n for n in near if n[0] < .6]
                if near:
                    d.id = min(near)[1]
                    taken.add(d.id)
        for d in detections:
            if d.id is None:
                self._next_id += 1
                d.id = f"obj{self._next_id}"

    def _remember(self, detections, pose, *, save=True):
        now = time.monotonic()
        c, sn = math.cos(pose[2]), math.sin(pose[2])
        kept = [m for m in self.memory if now - m[3] <= MEMORY_S]
        seen_ids = []
        for d in detections:
            if d.x_m is None:
                continue
            wx, wy = pose[0] + c * d.x_m - sn * d.y_m, pose[1] + sn * d.x_m + c * d.y_m
            kept = [m for m in kept if not (m[0].label == d.label and math.hypot(m[1] - wx, m[2] - wy) < .6)]
            kept.append((d, wx, wy, now))
            item = self.object_memory.remember(label=d.label, x=wx, y=wy, color=d.color, confidence=d.confidence,
                                               save=False)
            seen_ids.append(item["id"])
        self.memory = kept[-60:]
        if save:
            self.object_memory.save()
        return seen_ids

    @staticmethod
    def _score(det, words):
        """(score, mismatch) for how well a detection fits the user's words; score 0 = not this kind.
        Labels can be anything the detector says ("crate", "blue water bottle"). A word fits the label
        itself, its singular, or a synonym ("sofa" -> couch). Every naming word has to fit for a real
        match; otherwise ("coffee table" vs a side table) it is only the closest match, like a wrong
        colour, so the user is asked instead of being taken to a different thing."""
        label = det.label.lower()
        label_words = set(re.findall(r"[a-z]+", label))

        def fits(word):
            singular = word[:-1] if word.endswith("s") and not word.endswith("ss") else word
            forms = {word, singular, word[:-2] if word.endswith("es") else word}
            alt = SYNONYMS.get(word) or SYNONYMS.get(singular)
            if alt:
                forms.add(alt)
            return any(f == label or f in label_words or (" " in f and set(f.split()) <= label_words) for f in forms)
        naming = [w for w in words if w not in STOP_WORDS and w not in ADJECTIVES and w not in P.COLOR_WORDS]
        hits = [w for w in naming if fits(w)]
        if not hits:
            return 0, False
        head = naming[-1][:-1] if naming[-1].endswith("s") else naming[-1]
        partial = len(hits) < len(naming) and GENERIC.get(label) != head
        score = 2 * len(hits)
        asked = [w for w in words if w in P.COLOR_WORDS]
        if not asked or any(w in label_words for w in asked):
            return score, partial
        if det.color in asked:
            return score + 2, partial
        if det.color and any(det.color in COLOR_FAMILY.get(w, ()) for w in asked):
            return score + 1, partial
        return score, True

    def _words(self, target):
        return re.findall(r"[a-z]+", target.lower())

    def _track(self, locked):
        """The visible detection of the locked object: same label, close to where it was. Colour names
        drift with viewpoint, so once chosen an object is followed by position, not re-matched by colour."""
        label, wx, wy = locked
        with self.lock:
            detections = list(self.last_detections)
        best = None
        for det, pose in detections:
            if det.label != label or det.x_m is None:
                continue
            c, sn = math.cos(pose[2]), math.sin(pose[2])
            dx, dy = pose[0] + c * det.x_m - sn * det.y_m - wx, pose[1] + sn * det.x_m + c * det.y_m - wy
            d = math.hypot(dx, dy)
            if d < .8 and (best is None or d < best[0]):
                best = (d, det, pose)
        return (best[1], best[2]) if best else None

    def _match(self, target):
        """Best visible detection for an id from look or words like 'red ball' -> ((det, pose), closest_other)."""
        with self.lock:
            detections = list(self.last_detections)
        for det, pose in detections:
            if det.id == target:
                return (det, pose), None
        words = self._words(target)
        best, best_key, closest = None, None, None
        for det, pose in detections:
            score, mismatch = self._score(det, words)
            if not score:
                continue
            if mismatch:
                closest = closest or det
                continue
            key = (score, -(det.distance_m or 99))
            if best_key is None or key > best_key:
                best, best_key = (det, pose), key
        if best is None:
            # No label fits. With the semantic object memory, the thing in view that looks most like the words
            # will do ("can" when the detector said "bottle").
            alike = self._looks_like(target, detections)
            if alike:
                best = alike[0][:2]
        return best, closest



    def _recall(self, target):
        """An object seen before that fits the words, by its remembered position in the room: from this run's
        recent looks or the saved memory (which survives a restart). The best fit wins; between equal fits, the
        most recently seen, so an object that has moved is looked for where it was last seen."""
        now, wall = time.monotonic(), time.time()
        for d, wx, wy, t in reversed(self.memory):
            if d.id == target and now - t <= MEMORY_S:
                return d, wx, wy
        words = self._words(target)
        candidates = []                       # (score, seen at (wall clock), detection-like, x, y)
        for d, wx, wy, t in self.memory:
            if now - t <= MEMORY_S:
                candidates.append((self._score(d, words), wall - (now - t), d, wx, wy))
        for item in self.object_memory.snapshot(include_missing=False):
            d = SimpleNamespace(id=item.get("id", item.get("label", "")), label=item.get("label", ""),
                                color=item.get("color"))
            candidates.append((self._score(d, words), item.get("last_seen", 0.), d, item["x"], item["y"]))
        candidates = [c for c in candidates if c[0][0] and not c[0][1]]
        if not candidates:
            # Nothing remembered under those words: something that looked like them, if the semantic object
            # memory has one ("bottle" brings back the can). As with labels, the best fit decides what kind of
            # thing it is, and of those the latest sighting wins, so a thing that has moved is looked for where
            # it was last seen.
            alike = [entry for entry, _ in self._remembered_like(target, 5)]
            if not alike:
                return None
            entry = max((e for e in alike if e.get("label") == alike[0].get("label")), key=lambda e: e.get("t", 0.))
            thing = SimpleNamespace(id=entry["id"], label=entry.get("label") or self._phrase(target),
                                    color=entry.get("color"), by_look=bool(entry.get("label")))
            return thing, entry["x"], entry["y"]
        # Reversed, so a tie goes to the later sighting: Windows' clock can give two sightings the same time.
        _, _, det, wx, wy = max(reversed(candidates), key=lambda c: (c[0][0], c[1]))
        return det, wx, wy

    # ---- semantic object memory --------------------------------------------------------------------
    def _label_for(self, target):
        """The detector label the words mean, when none fits them as said and the semantic object memory can tell:
        'computer' -> 'laptop', 'the red mug' -> 'red cup'. None when a label already fits, the memory is off, or
        no label is clearly closest (words for things the detector has no name for)."""
        visual = self.visual
        if visual is None or re.fullmatch(r"obj\d+", str(target)):
            return None
        words = self._words(target)
        with self.lock:
            seen = {d.label for d, *_ in self.memory}
        labels = sorted((set(DETECTOR_LABELS) | seen | {o.get("label") or "" for o in self.object_memory.objects}) - {""})
        if any(self._score(SimpleNamespace(label=label, color=None), words)[0] for label in labels):
            return None                         # a label fits the words as they are
        naming = [w for w in words if w not in STOP_WORDS and w not in ADJECTIVES and w not in P.COLOR_WORDS]
        if not naming:
            return None
        try:
            cache = self._label_vectors
            if cache.get("embedder") is not visual.embedder:
                cache.clear()
                cache["embedder"] = visual.embedder
            missing = [label for label in labels if label not in cache]
            if missing:
                cache.update(zip(missing, np.asarray(visual.embedder.texts(missing), dtype=np.float32)))
            closeness = np.stack([cache[label] for label in labels]) @ visual.text(" ".join(naming))
        except Exception:
            return None
        first, second = np.argsort(-closeness)[:2]
        if closeness[first] - closeness[second] < LABEL_MARGIN:
            return None
        colours = [w for w in words if w in P.COLOR_WORDS]
        return " ".join(colours + [labels[first]])

    def use_visual_memory(self, embedder):
        """Switch the semantic object memory on with this embedding model (saved beside the object memory), or
        off with None: what it remembered stays saved, and is there again when it is switched back on."""
        old, self.visual, self._look_vectors = self.visual, None, {}
        if old is not None:
            old.save(force=True)
        if embedder is not None:
            from .visual_memory import VisualMemory
            self.visual = VisualMemory(self._visual_path, embedder)
        return self.visual

    def _memorise(self, frame, detections, pose):
        """After a look: turn a crop of each thing in view into a vector, keep it for matching against what
        is being looked for (_looks_like), and remember the ones with a known position. People are left out."""
        self._look_vectors = {}
        visual = self.visual                # one reference: the console's switch may turn it off meanwhile
        if visual is None:
            return
        things = [(d, crop(frame.rgb, d.box)) for d in detections if d.label != "person" and d.box is not None]
        things = [(d, c) for d, c in things if c is not None]
        if not things:
            return
        try:
            vectors = visual.embed([c for _, c in things])
        except Exception:
            return                          # a model that fails costs the memory, not the look
        c, sn = math.cos(pose[2]), math.sin(pose[2])
        for (det, picture_of), vector in zip(things, vectors):
            self._look_vectors[det.id] = vector
            if det.x_m is not None:
                visual.add(vector, pose[0] + c * det.x_m - sn * det.y_m, pose[1] + sn * det.x_m + c * det.y_m,
                                label=det.label, color=det.color, pixels=picture_of.shape[0] * picture_of.shape[1])
        visual.save()

    @staticmethod
    def _phrase(target):
        """What to ask the semantic object memory for: the thing's own words, without a leading 'the' or 'my'.
        The rest is kept as said: 'something to drink from' means more to the model whole than as its nouns."""
        words = re.findall(r"[a-z0-9']+", str(target).lower())
        while words and words[0] in ("the", "a", "an", "my", "your", "our", "that", "this", "some"):
            words = words[1:]
        return " ".join(words)

    def _looks_like(self, target, detections):
        """The detections in view whose pictures fit the words well enough to act on, best first:
        [(detection, pose, score)]. Used only when no label fits, so 'can' finds what the detector called a
        bottle or a cup. A strong fit counts from anywhere, an ordinary one only from close up (a thing across
        the room is a few dozen pixels, and what such a picture gets wrong it gets wrong both ways)."""
        visual = self.visual
        phrase = self._phrase(target) if visual is not None and not re.fullmatch(r"obj\d+", str(target)) else ""
        seen = [(d, pose) for d, pose in detections if d.id in self._look_vectors]
        if not phrase or not seen:
            return []
        try:
            scores = visual.scores(phrase, np.stack([self._look_vectors[d.id] for d, _ in seen]))
        except Exception:
            return []
        strict, fair = visual.strict_threshold, visual.threshold
        fits = [(d, pose, float(score)) for (d, pose), score in zip(seen, scores)
                if score >= strict or (score >= fair and d.distance_m is not None and d.distance_m <= CLOSE_M)]
        return sorted(fits, key=lambda fit: -fit[2])

    def _remembered_like(self, target, limit=3):
        """Remembered things that fit the words by resemblance: [(entry, score)], best first."""
        visual = self.visual
        phrase = self._phrase(target) if visual is not None and not re.fullmatch(r"obj\d+", str(target)) else ""
        if not phrase:
            return []
        try:
            return visual.search(phrase, limit)
        except Exception:
            return []

    def _turn_by(self, task, cancel, angle):
        rate = L.default_turn_rate(abs(angle))
        self._velocity(task, {"vx": 0., "vy": 0., "wz": math.copysign(rate, angle), "ux": 0., "uy": 0.,
                              "speed": 0., "turn_rate": rate, "angle": abs(angle)}, cancel, leg=True)

    def _approach(self, task, cancel):
        p = dict(task.command.params)
        if p.get("meant"):
            p["target"] = p["meant"]            # the detector's word for what was asked for
        stop_at = p["stop_distance"]
        searched, recalled, closest, locked, unranged = 0, False, None, None, 0
        for attempt in range(40):
            self._check(cancel)
            self.look()
            match, near_miss = (self._track(locked), None) if locked else self._match(p["target"])
            if locked and match is None:
                match, near_miss = self._match(p["target"])
            closest = closest or near_miss
            if match is None:
                if locked is not None:
                    s = self._fresh_state()
                    if math.hypot(locked[1] - s.x, locked[2] - s.y) <= stop_at + P.HALF_LENGTH + .2:
                        bearing = _wrap(math.atan2(locked[2]-s.y, locked[1]-s.x)-s.yaw)
                        if abs(bearing) > math.radians(6):
                            self._turn_by(task, cancel, bearing)
                        for _ in range(2):
                            self._check(cancel)
                            self.look()
                            if self._track(locked) or self._match(p["target"])[0]:
                                break
                        else:
                            missing = any(item["label"] == locked[0] and item.get("state") == "missing" and
                                          math.hypot(item["x"]-locked[1], item["y"]-locked[2]) < .6
                                          for item in self.object_memory.snapshot())
                            if missing:
                                raise AdapterError(f"the {locked[0]} is no longer at its remembered position; it may have moved")
                            raise AdapterError(f"the {locked[0]} is not visible at its last-seen position; I could not verify whether it moved")
                        continue
                    task.detail["searching"] = f"following a route to the last-seen {locked[0]}"
                    self._approach_route(task, cancel, s, locked[1], locked[2], stop_at, force=True)
                    continue
                memory = None if recalled else self._recall(p["target"])
                s = self._fresh_state()
                if memory is not None:
                    # Seen recently but not in view now: turn toward where it was, then look again.
                    recalled = True
                    # Keep the recalled detector label and map position locked
                    # through detours, including semantic appearance matches.
                    locked = (memory[0].label, memory[1], memory[2])
                    bearing = _wrap(math.atan2(memory[2] - s.y, memory[1] - s.x) - s.yaw)
                    task.detail["searching"] = f"turned toward where I last saw the {memory[0].label}"
                    if abs(bearing) > math.radians(10):
                        self._turn_by(task, cancel, bearing)
                    continue
                if searched < 3:
                    # Never seen: the camera covers ~120°, so look around in quarter turns.
                    searched += 1
                    task.detail["searching"] = f"looked around ({searched * 90}° so far)"
                    self._turn_by(task, cancel, math.radians(90))
                    continue
                with self.lock:
                    names = sorted({_name(d) for d, *_ in self.memory})[:4]
                text = f"I looked all the way around but can't find a {p['target']}"
                if p.get("meant"):
                    text += f" (what I took '{task.command.params['target']}' to mean)"
                if closest is not None:
                    text += f"; the closest match is the {_name(closest)}"
                text += f"; I can see: {', '.join(names)}" if names else "; I don't see any objects"
                raise AdapterError(text)
            det, pose = match
            task.detail["target"] = det.public()
            task.detail.pop("searching", None)
            if det.x_m is not None:
                c0, s0 = math.cos(pose[2]), math.sin(pose[2])
                locked = (det.label, pose[0] + c0 * det.x_m - s0 * det.y_m, pose[1] + s0 * det.x_m + c0 * det.y_m)
            if det.distance_m is None:
                # Seen but not ranged (no LiDAR return yet, or too far to judge): head toward it a step at
                # a time and look again; close up the LiDAR will see it.
                unranged += 1
                if unranged > 4:
                    raise AdapterError(f"I can see the {p['target']} but can't tell how far away it is")
                task.detail["searching"] = f"walking toward the {det.label} to see how far it is"
                if abs(det.bearing_deg) > 6:
                    self._turn_by(task, cancel, math.radians(det.bearing_deg))
                try:
                    self._velocity(task, {"vx": .3, "vy": 0., "wz": 0., "ux": 1., "uy": 0., "speed": .3,
                                          "distance": 1.0}, cancel, leg=True)
                    if task.status == "failed":
                        raise AdapterError(task.detail.get("error", "approach motion failed"))
                except _BlockedError:
                    task.detail.pop("blocked", None)
                    task.status = "running"       # look again: maybe that was the object itself
                continue
            gap = det.distance_m - P.HALF_LENGTH - stop_at
            if gap <= .1 and abs(det.bearing_deg) < 10:
                break
            s = self._fresh_state()
            # Aim at the object, then cover at most 1.5 m before looking again.
            x, y = pose[0], pose[1]
            c, sn = math.cos(pose[2]), math.sin(pose[2])
            ox, oy = x + c * det.x_m - sn * det.y_m, y + sn * det.x_m + c * det.y_m
            if gap > .1 and self._approach_route(task, cancel, s, ox, oy, stop_at):
                continue
            bearing = _wrap(math.atan2(oy - s.y, ox - s.x) - s.yaw)
            if abs(bearing) > math.radians(6):
                self._turn_by(task, cancel, bearing)
            if gap <= .1:
                continue
            leg = min(gap, 1.5)
            speed = .2 if gap < .8 else L.default_speed(leg)
            try:
                self._velocity(task, {"vx": speed, "vy": 0., "wz": 0., "ux": 1., "uy": 0., "speed": speed,
                                      "distance": leg}, cancel, leg=True)
                if task.status == "failed":
                    raise AdapterError(task.detail.get("error", "approach motion failed"))
            except _BlockedError:
                # The object itself is in the corridor: being blocked close to it means we arrived.
                self.look()
                again = self._track(locked) if locked else self._match(p["target"])[0]
                if again and again[0].distance_m is not None and again[0].distance_m - P.HALF_LENGTH <= stop_at + .3:
                    task.detail["target"] = again[0].public()
                    task.detail.pop("blocked", None)
                    task.status = "running"
                    break
                raise
        else:
            raise AdapterError("approach did not reach and confirm its target within the movement limit")
        final = self._track(locked) if locked else self._match(p["target"])[0]
        if final and final[0].distance_m is not None:
            task.detail["final_gap_m"] = round(final[0].distance_m - P.HALF_LENGTH, 2)

    def _approach_route(self, task, cancel, s, ox, oy, stop_at, *, force=False):
        """Follow one planned leg to a free stand-off point, never into the object.

        On a clear direct approach return False so the existing slow final
        visual servo controls the requested gap. On detours, retain the locked
        world position even when the target temporarily leaves the camera.
        """
        distance = math.hypot(ox - s.x, oy - s.y)
        stand_off = max(stop_at + P.HALF_LENGTH, .46)
        if distance <= stand_off + .08:
            return False
        bearing = math.atan2(s.y - oy, s.x - ox)
        selected = None
        for offset in (0., math.pi / 6, -math.pi / 6, math.pi / 3, -math.pi / 3, math.pi / 2, -math.pi / 2):
            gx = ox + stand_off * math.cos(bearing + offset)
            gy = oy + stand_off * math.sin(bearing + offset)
            route = self._route(s, gx, gy)
            if route is not None:
                selected = route, gx, gy
                break
        if selected is None:
            raise _blocked(task, "no route to a free stand-off position near the object", math.inf, 0.)
        route, gx, gy = selected
        if not route and not force:
            return False
        task.detail["route"] = [[round(x, 2), round(y, 2)] for x, y in (route or [(gx, gy)])]
        wx, wy = route[0] if route else (gx, gy)
        leg = math.hypot(wx - s.x, wy - s.y)
        if leg > 1.5:
            wx, wy = s.x + (wx - s.x) * 1.5 / leg, s.y + (wy - s.y) * 1.5 / leg
        task.detail.setdefault("legs", []).append({})
        try:
            # A nearby intermediate bend still needs to be reached. Otherwise
            # replanning can choose the same bend forever inside the ordinary
            # destination tolerance without issuing any movement.
            self._goto_leg(task, cancel, wx, wy, arrival_tolerance=.02 if route else .08)
            if task.status == "failed":
                raise AdapterError(task.detail.get("error", "approach motion failed"))
        except _BlockedError:
            replans = task.detail.get("replanned", 0)
            if replans >= MAX_REPLANS:
                raise
            task.detail["replanned"] = replans + 1
            task.detail.pop("blocked", None)
            task.status = "running"
            for _ in range(3):
                self._refresh_cloud()
        return True


    # ---- follow-me (#15) -----------------------------------------------------------------
    def _follow(self, task, cancel):
        p = task.command.params
        feed = F.PersonFeed(self, self.adapter)
        feed.start()
        try:
            tracker = self._follow_acquire(task, cancel, feed, p)
            self._underway(task)
            self._hold(task, cancel)
            self._follow_loop(task, cancel, feed, tracker, p)
        finally:
            feed.stop()
            self._safe_stop()
            task.detail.pop("following", None)
            feed.join(1.)

    def _follow_acquire(self, task, cancel, feed, p):
        """Pick the person to follow; look around in quarter turns if nobody is in view. Refuses rather than
        guessing when it can't tell which person is meant."""
        wanted, who = p.get("color"), p["target"]
        for turn in range(4):
            if not feed.wait_frames(2, 6.):
                self._check(cancel)
                raise AdapterError(f"I need the camera to follow someone ({feed.error or 'no camera frames'})")
            self._check(cancel)
            s = self._fresh_state()
            with feed.lock:
                people = list(feed.people)
            ranked = []
            for x, y, clothing, det in people:
                if wanted and (clothing is None or not F.colour_fits(clothing, wanted)):
                    continue
                ranked.append((math.hypot(x - s.x, y - s.y), x, y, clothing, det))
            ranked.sort(key=lambda r: r[0])
            if ranked:
                break
            if turn < 3:
                task.detail["searching"] = f"looked around ({(turn + 1) * 90}° so far)"
                self._turn_by(task, cancel, math.radians(90))
        else:
            seen = f"anyone in {wanted}" if wanted else "anyone"
            raise AdapterError(f"I looked all the way around but can't see {seen} to follow")
        task.detail.pop("searching", None)
        if len(ranked) > 1 and ranked[1][0] - ranked[0][0] < .7:
            options = "; ".join(F.describe_person(r[3], r[4].distance_m or r[0], r[4].bearing_deg) for r in ranked[:3])
            raise AdapterError(f"I can see {len(ranked)} people and can't tell which one is {who}: {options}. "
                               "Ask the user which one, for example by shirt colour")
        _, x, y, clothing, det = ranked[0]
        tracker = F.PersonTracker(x, y, time.monotonic(), clothing or wanted)
        with feed.lock:
            feed.tracker = tracker
        task.detail["target"] = {"clothing": clothing, "distance_m": round(det.distance_m, 2) if det.distance_m else None,
                                 "direction": P.describe_bearing(det.bearing_deg)}
        return tracker

    def _follow_loop(self, task, cancel, feed, tracker, p):
        adapter, set_gap = self.adapter, p["distance"]
        who = "you" if p["target"].lower() in ("me", "") else p["target"]
        began = time.monotonic()
        s = self._fresh_state()
        last_xy, walked, gaps = (s.x, s.y), 0., []
        blocked_since, lost = None, 0
        was_seen = True
        stall = F.StallWatch()
        sent_vx = 0.
        while True:
            self._check(cancel)
            tick = time.monotonic()
            if tick - began >= p["duration"]:
                break
            s = self._refresh_cloud()
            s = self._fresh_state()
            walked += math.hypot(s.x - last_xy[0], s.y - last_xy[1])
            last_xy = (s.x, s.y)
            with self._cloud_lock:
                cloud = self.cloud.level((s.x, s.y, s.yaw))
            cloud = self._with_bumps(cloud, s)       # spots it bumped into recently count as obstacles
            if stall.update(tick, sent_vx, s.vx, s.x, s.y):
                self._safe_stop()
                self._remember_bump(s, 1., 0.)
                backed = self._back_off(1., 0., cancel)
                task.detail.update(followed_s=round(tick - began), walked_m=round(walked, 1))
                error = _blocked(task, STALLED, 0., walked, "ahead")
                task.detail["blocked"]["stall"] = True
                if backed:
                    task.detail["blocked"]["backed_off_m"] = round(backed, 2)
                raise error
            with feed.lock:
                tx, ty = tracker.predict(tick)
                unseen = tick - tracker.seen
                ambiguous, tvx, tvy = tracker.ambiguous, tracker.vx, tracker.vy
            c, sn = math.cos(s.yaw), math.sin(s.yaw)
            dx, dy = c * (tx - s.x) + sn * (ty - s.y), -sn * (tx - s.x) + c * (ty - s.y)
            gap, bearing = math.hypot(dx, dy) - P.HALF_LENGTH, math.atan2(dy, dx)
            seen = unseen <= F.SEEN_S
            if was_seen and not seen:
                lost += 1
            was_seen = seen
            if unseen > F.GIVE_UP_S:
                raise AdapterError(f"I lost sight of {who}{self._near_landmark(tx, ty)} and stopped")
            vx = wz = 0.
            if seen and not ambiguous:
                state = "following"
                vx, wz = F.follow_step(gap, bearing, set_gap, tvx * c + tvy * sn)
                gaps.append(gap)
            elif ambiguous:
                state = "waiting: two people are close together and I won't follow the wrong one"
            elif unseen <= F.LOST_STOP_S:
                state = f"lost sight of {who}, turning to where they were"
                wz = max(-.8, min(.8, 1.5 * bearing))
            else:
                state = f"lost sight of {who}, looking where I last saw them"
                wz = max(-.6, min(.6, 1.5 * bearing)) if abs(bearing) > math.radians(10) else 0.
            clear = math.inf
            if vx > 0:
                clear = min(P.clearance_along(cloud, 1., 0.), P.observed_along(cloud, 1., 0.))
                cap = F.max_safe_speed(clear)
                if cap < .1 and vx > .15 and clear < gap - .35:
                    blocked_since = blocked_since or tick       # something other than the person is in the way
                    state = "held up by an obstacle"
                else:
                    blocked_since = None
                vx = min(vx, cap) if cap >= .05 else 0.
            else:
                blocked_since = None
                behind = min(P.clearance_along(cloud, -1., 0.), P.observed_along(cloud, -1., 0.))
                if vx < 0 and behind < self._margin(-vx):
                    vx = 0.
            if wz and P.turn_clearance(cloud) < .03:
                wz = 0.
            if blocked_since is not None and tick - blocked_since > F.BLOCKED_GIVE_UP_S:
                task.detail.update(followed_s=round(tick - began), walked_m=round(walked, 1))
                why = "drop ahead" if P.drop_along(cloud, 1., 0.) <= clear + 1e-9 else "path is blocked"
                raise _blocked(task, why, clear, walked, "ahead")
            adapter.move(vx, 0., wz)
            sent_vx = vx
            task.detail["following"] = {"target": who, "set_m": set_gap, "state": state,
                                        "gap_m": round(gap, 2) if seen else None}
            time.sleep(max(0., F.TICK_S - (time.monotonic() - tick)))
        adapter.stop_move()
        task.detail.pop("following", None)
        task.detail.update(followed_s=round(time.monotonic() - began), walked_m=round(walked, 1),
                           lost_sight=lost)
        if gaps:
            task.detail["final_gap_m"] = round(gaps[-1], 2)

    def _near_landmark(self, x, y):
        with self.lock:
            near = [(math.hypot(wx - x, wy - y), d) for d, wx, wy, _ in self.memory if d.label != "person"]
        near = [n for n in near if n[0] < 2.5]
        return f" near the {_name(min(near, key=lambda n: n[0])[1])}" if near else ""


class _MotionDeadline:
    """An event-compatible budget checked inside every motion tick."""
    def __init__(self, event, timeout):
        self.event, self.deadline = event, time.monotonic() + timeout

    def is_set(self):
        if self.event.is_set():
            return True
        if time.monotonic() >= self.deadline:
            raise AdapterError("motion timed out")
        return False

    def wait(self, seconds):
        if self.is_set():
            return True
        result = self.event.wait(min(seconds, max(0., self.deadline - time.monotonic())))
        return result or self.is_set()


class _BlockedError(AdapterError):
    pass


UNSEEN = "I can't see far enough in that direction to be sure it's clear"


ACTIVITY_LINES = 12


def _activity(tasks, since):
    """What the robot has done since it was connected, from measured progress, so 'what have you done?'
    is answered from facts: totals plus the most recent actions (oldest first) with how they ended."""
    def total(key):
        return sum(abs(t.detail.get(key, 0)) + sum(abs(leg.get(key, 0)) for leg in t.detail.get("legs", []))
                   for t in tasks)
    out = {"minutes": round((time.monotonic() - since) / 60, 1), "actions": len(tasks),
           "walked_m": round(total("moved_m"), 1), "turned_deg": round(total("turned_deg"))}
    if len(tasks) > ACTIVITY_LINES:
        out["earlier_actions"] = len(tasks) - ACTIVITY_LINES
    lines = []
    for t in tasks[-ACTIVITY_LINES:]:
        line, d = f"{t.command.summary}: {t.status}", t.detail
        if t.status != "completed":
            if "moved_m" in d and "target_m" in d:
                line += f" after {d['moved_m']} of {d['target_m']} m"
            why = (d.get("blocked") or {}).get("reason") or d.get("error")
            if why:
                line += f" ({why})"
        lines.append(line)
    out["recent"] = lines
    return out


def _toward(ux, uy):
    """Where a move's travel direction (body frame) points, in words."""
    if abs(ux) >= abs(uy):
        return "ahead" if ux > 0 else "behind"
    return "to the left" if uy > 0 else "to the right"


def _blocked(task, why, clearance, progress, toward=None):
    if why == "path is blocked" and clearance >= getattr(task, "_observed", math.inf) - 1e-9:
        why = UNSEEN
    task.status = "blocked"
    task.detail["blocked"] = {"reason": why, "clearance_m": None if clearance == math.inf else round(clearance, 2)}
    if toward:
        task.detail["blocked"]["toward"] = toward
    return _BlockedError(f"{why}: {clearance:.2f} m of free space" if clearance != math.inf else why)


def _name(det):
    """'red ball', without repeating a colour the label already has ('blue water bottle')."""
    if det.color and det.color not in det.label.split():
        return f"{det.color} {det.label}"
    return det.label


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))

