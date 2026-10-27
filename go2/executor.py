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
from dataclasses import dataclass, field

import numpy as np

from . import limits as L
from . import perception as P
from .adapters.base import AdapterError, LOW_POSTURES
from .tools import Command

DECEL_LINEAR, DECEL_ANGULAR = 1.6, 4.0      # m/s², rad/s² used for braking lead
# Stall: something the LiDAR can't see (glass, thin chair legs) is stopping the robot. After the ramp-up, a move
# or turn that makes less than a third of the progress its commanded speed should give over 0.4 s stops.
STALL_WINDOW_S, STALL_FRACTION, STALL_GRACE_S = .4, 1 / 3, .3
STALL_START_S = .8     # the latest the check starts if the robot never gets going (e.g. it starts against glass)
STALLED = "bumped into something the sensors missed"
BACKOFF_M, BACKOFF_SPEED = .2, .2           # after a stall, back away the way it came if the LiDAR says clear
BUMP_MEMORY_S = 180.                        # where it bumped counts as an obstacle for this long
BUMP_HALF_WIDTH = .3
STALE_S = .5
TICK_S = .05
# A motion that can start right away is held back from the model for up to this long, so that the tool
# result can say "blocked" when the safety check refuses it instead of "queued".
START_CHECK_S = 1.0
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
    def __init__(self, on_event=None):
        self.adapter = None
        self.on_event = on_event or (lambda task: None)
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
        self.last_detections = []
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
            self.armed = False
            self.cloud.clear()
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
        with self.lock:
            self._require_adapter()
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
            task = Task(command, group, self.generation, user_turn=user_turn)
            self.queue.append(task)
            position = len(self.queue) + (1 if self.current else 0)
            task.awaiting = position == 1 and command.kind in ("velocity", "goto", "approach")
            self.wake.notify_all()
            return {"status": "queued", "task_id": task.id, "position": position, "action": command.summary,
                    "_task": task}

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
        self.cloud.add(state.stamp, level, (state.x, state.y, state.yaw), state.height)
        return state

    def look(self):
        with self.lock:
            adapter = self._require_adapter()
        result = {}
        try:
            for _ in range(3):  # a few scans so the non-repetitive L1 pattern fills in
                state = self._refresh_cloud()
                time.sleep(.05)
            floor_cloud = self.cloud.level((state.x, state.y, state.yaw))
            free = P.sector_free_space(self._with_bumps(floor_cloud, state))
            result["free_space_m"] = {side: ("clear" if v is None else v) for side, v in free.items()}
        except AdapterError as error:
            result["lidar_error"] = str(error)
            floor_cloud, state = None, None
        try:
            frame = adapter.camera()
            s = frame.state
            cloud_base = np.zeros((0, 3))
            if floor_cloud is not None and len(floor_cloud):
                local = self.cloud.level((s.x, s.y, s.yaw))
                local[:, 2] -= s.height
                cloud_base = local @ P.rotation_rp(s.roll, s.pitch)
            detections = P.fuse(frame.masks, frame.rgb, cloud_base, frame.intrinsics, frame.camera_pos,
                                frame.camera_rot, s.roll, s.pitch, s.height)
            if self.detection_filter is not None:
                detections = self.detection_filter(detections)
            with self.lock:
                self._stable_ids(detections, (s.x, s.y, s.yaw))
                self.last_detections = [(d, (s.x, s.y, s.yaw)) for d in detections]
                self._remember(detections, (s.x, s.y, s.yaw))
                self.last_frame = frame.rgb
                self.last_frame_meta = {"stamp": time.time(), "objects": len(detections),
                                        "boxes": image_boxes(frame.masks, frame.rgb.shape)}
            result["objects"] = [{**d.public(), "direction": P.describe_bearing(d.bearing_deg)} for d in detections]
        except AdapterError as error:
            result["camera_error"] = str(error)
        result["note"] = "objects come from the front camera (about 120° wide); free_space_m is the free distance from the body on each side, 'clear' meaning nothing within 5 m"
        return result

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
            self._ensure_standing(task, cancel)
            self._goto(task, cancel)
        elif kind == "approach":
            self._ensure_standing(task, cancel)
            self._approach(task, cancel)
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

    def _clearance(self, ux, uy, turning):
        """Free distance along the travel direction, limited to what the LiDAR can actually see. Drops (a
        ledge, stairs going down) and spots it recently bumped into count as obstacles."""
        s = self._refresh_cloud()
        cloud = self.cloud.level((s.x, s.y, s.yaw))
        if turning:
            return P.turn_clearance(self._with_bumps(cloud, s))
        self._observed = P.observed_along(cloud, ux, uy)
        self._drop = P.drop_along(cloud, ux, uy)
        self._obstacle = P.obstacle_along(cloud, ux, uy)
        bumped = self._with_bumps(cloud, s)
        self._bump = P.obstacle_along(bumped, ux, uy) if len(bumped) > len(cloud) else math.inf
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
        start = self._fresh_state()
        x0, y0, yaw0 = start.x, start.y, start.yaw
        c0, s0 = math.cos(yaw0), math.sin(yaw0)
        wx, wy = c0 * p["ux"] - s0 * p["uy"], s0 * p["ux"] + c0 * p["uy"]   # travel direction in odom
        speed = p["speed"]
        try:
            clear = self._clearance(p["ux"], p["uy"], turning)
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
        earliest = began + rate / (DECEL_ANGULAR if turning else DECEL_LINEAR) + STALL_GRACE_S
        watch_from = None
        recent = deque()
        try:
            while True:
                self._check(cancel)
                now = time.monotonic()
                if now - last_send >= .1:
                    adapter.move(p["vx"], p["vy"], p["wz"])
                    last_send = now
                s = self._fresh_state()
                turned += _wrap(s.yaw - yaw_prev)
                yaw_prev = s.yaw
                progress = abs(turned) if turning else (s.x - x0) * wx + (s.y - y0) * wy
                if not turning and now - last_scan >= .1:
                    last_scan = now
                    clear = self._clearance(p["ux"], p["uy"], False)
                    if clear < self._margin(max(s.speed, .1)):
                        outcome = "blocked"
                        trigger = (clear, self._observed, progress, self._why_blocked(clear, toward))
                        break
                # A stall needs both: little progress over the window (wall clock) and a low measured body
                # speed. The robot's own speed reading doesn't drop when a busy computer briefly slows the
                # simulator or delays telemetry, so those hiccups aren't mistaken for touching something.
                moving = abs(s.wz) if turning else s.vx * p["ux"] + s.vy * p["uy"]
                if watch_from is None and now >= earliest and (moving >= rate / 2 or now >= earliest + STALL_START_S):
                    watch_from = now
                if watch_from is not None:
                    recent.append((now, progress, moving))
                    while len(recent) > 1 and now - recent[1][0] >= STALL_WINDOW_S:
                        recent.popleft()        # keep the newest sample at least a window old
                old_t, old_progress, _ = recent[0] if recent else (now, progress, 0.)
                if recent and now - old_t >= STALL_WINDOW_S:
                    expected = rate * STALL_FRACTION
                    measured = sum(r[2] for r in recent) / len(recent)
                    if progress - old_progress < expected * (now - old_t) and measured < expected:
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
                task.detail["blocked"]["backed_off_m"] = round(backed, 2)
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

    def _goto_leg(self, task, cancel, tx, ty):
        """Turn toward (tx, ty) in odom, then walk there in legs within the per-call limits."""
        s = self._fresh_state()
        dx, dy = tx - s.x, ty - s.y
        distance = math.hypot(dx, dy)
        if distance < .08:
            return 0.
        bearing = _wrap(math.atan2(dy, dx) - s.yaw)
        if math.cos(bearing) < -.7 and distance <= 2.0:   # short way back: reverse instead of turning around
            ux, uy = math.cos(bearing), math.sin(bearing)
            speed = L.default_speed(distance)
            self._velocity(task, {"vx": ux * speed, "vy": uy * speed, "wz": 0., "ux": ux, "uy": uy, "speed": speed,
                                  "distance": min(distance, L.MAX_DISTANCE_M)}, cancel, leg=True)
            return distance
        if distance < .35:   # short hop: side-step instead of turning on the spot
            ux, uy = math.cos(bearing), math.sin(bearing)
            speed = L.DEFAULT_SPEED_MPS
            self._velocity(task, {"vx": ux * speed, "vy": uy * speed, "wz": 0., "ux": ux, "uy": uy, "speed": speed,
                                  "distance": distance}, cancel, leg=True)
            return distance
        if abs(bearing) > math.radians(8):
            rate = L.default_turn_rate(abs(bearing))
            self._velocity(task, {"vx": 0., "vy": 0., "wz": math.copysign(rate, bearing), "ux": 0., "uy": 0.,
                                  "speed": 0., "turn_rate": rate, "angle": abs(bearing)}, cancel, leg=True)
        s = self._fresh_state()
        distance = math.hypot(tx - s.x, ty - s.y)
        leg = min(distance, L.MAX_GOTO_LEG_M)
        speed = L.default_speed(leg)
        self._velocity(task, {"vx": speed, "vy": 0., "wz": 0., "ux": 1., "uy": 0., "speed": speed,
                              "distance": leg}, cancel, leg=True)
        return leg

    def _goto(self, task, cancel):
        p = task.command.params
        s = self._fresh_state()
        if p["target"] == "start":
            if self.start_pose is None:
                raise AdapterError("the start position is unknown")
            tx, ty = self.start_pose[:2]
        else:
            c, sn = math.cos(s.yaw), math.sin(s.yaw)
            tx, ty = s.x + c * p["x"] - sn * p["y"], s.y + sn * p["x"] + c * p["y"]
        task.detail["legs"] = []
        for _ in range(8):
            s = self._fresh_state()
            remaining = math.hypot(tx - s.x, ty - s.y)
            if remaining < .12:
                break
            task.detail["legs"].append({})
            self._goto_leg(task, cancel, tx, ty)
        s = self._fresh_state()
        task.detail["remaining_m"] = round(math.hypot(tx - s.x, ty - s.y), 2)

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

    def _remember(self, detections, pose):
        now = time.monotonic()
        c, sn = math.cos(pose[2]), math.sin(pose[2])
        kept = [m for m in self.memory if now - m[3] <= MEMORY_S]
        for d in detections:
            if d.x_m is None:
                continue
            wx, wy = pose[0] + c * d.x_m - sn * d.y_m, pose[1] + sn * d.x_m + c * d.y_m
            kept = [m for m in kept if not (m[0].label == d.label and math.hypot(m[1] - wx, m[2] - wy) < .6)]
            kept.append((d, wx, wy, now))
        self.memory = kept[-60:]

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
        return best, closest

    def _recall(self, target):
        """A recently seen object that fits the words, by its remembered position in the room."""
        now = time.monotonic()
        for d, wx, wy, t in reversed(self.memory):
            if d.id == target and now - t <= MEMORY_S:
                return d, wx, wy
        words = self._words(target)
        candidates = [(self._score(d, words), -abs(now - t), d, wx, wy) for d, wx, wy, t in self.memory if now - t <= MEMORY_S]
        candidates = [c for c in candidates if c[0][0] and not c[0][1]]
        if not candidates:
            return None
        _, _, det, wx, wy = max(candidates, key=lambda c: (c[0][0], c[1]))
        return det, wx, wy

    def _turn_by(self, task, cancel, angle):
        rate = L.default_turn_rate(abs(angle))
        self._velocity(task, {"vx": 0., "vy": 0., "wz": math.copysign(rate, angle), "ux": 0., "uy": 0.,
                              "speed": 0., "turn_rate": rate, "angle": abs(angle)}, cancel, leg=True)

    def _approach(self, task, cancel):
        p = task.command.params
        stop_at = p["stop_distance"]
        searched, recalled, closest, locked, unranged = 0, False, None, None, 0
        for attempt in range(12):
            self._check(cancel)
            self.look()
            match, near_miss = (self._track(locked), None) if locked else self._match(p["target"])
            if locked and match is None:
                match, near_miss = self._match(p["target"])
            closest = closest or near_miss
            if match is None:
                memory = None if recalled else self._recall(p["target"])
                s = self._fresh_state()
                if memory is not None:
                    # Seen recently but not in view now: turn toward where it was, then look again.
                    recalled = True
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
        final = self._track(locked) if locked else self._match(p["target"])[0]
        if final and final[0].distance_m is not None:
            task.detail["final_gap_m"] = round(final[0].distance_m - P.HALF_LENGTH, 2)


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

