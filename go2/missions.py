"""Missions (#18): long tasks Go2 carries out on its own in a house.

  search(target, where)  what's remembered first, then room by room, the likeliest rooms first (a teddy bear in a
                         bedroom before the kitchen), looking all round at each stop and exploring where the map has
                         no rooms yet; walks up to the target when it's seen and says which room it was in.
  guard(place)           goes to a room or saved place, sits facing the way in and watches the camera for people,
                         alerting with a snapshot when someone comes in. It watches and reports only: once there it
                         never moves, let alone toward anyone.
  patrol(rooms, every)   walks through the rooms on a schedule, looking round each one, and reports what changed
                         since the last round: things moved, new or gone, and people seen.
  explore()              visits every reachable part of the house (#14's exploration), mapping it and naming its
                         rooms (#35), then says which rooms there are.

A mission runs as one executor task, so Stop (the button, Esc, saying "stop", the stop tool) cancels it as it does any
motion, and every movement is an ordinary executor step with its limits and LiDAR checks: missions only decide where to
go next. Each has a time limit and ends with a plain result (mission["outcome"], mission["result"]). Progress and
alerts go out through executor.on_progress, as robot updates in the chat. A stopped or timed-out search, patrol or
explore keeps its progress: asking for the same mission again within RESUME_S carries on where it left off.
"""
from __future__ import annotations

import math
import re
from types import SimpleNamespace

from .adapters.base import AdapterError
from .tools import Command, resolve

RESUME_S = 600.            # a stopped mission asked for again this soon carries on where it left off
STEP_S = 180.              # the most one step (a walk, an approach) may take
LOOK_TURNS = 4             # quarter turns to look all round (the camera covers about 120 degrees)
GUARD_LOOK_S = 1.          # guard looks this often, so an alert comes within about 3 s of someone appearing
NEW_PERSON_S = 10.         # someone seen after nobody was in view for this long is a new arrival
SAME_PLACE_M = .8          # patrol: an object this close to where it was hasn't moved
STAND_CLEAR_M = .45        # a room is walked into at a spot at least this clear of anything in the way
LOG_LINES = 30

# Where things usually are: detector label -> room types, likeliest first (as in the houses of #17, and most homes).
LIKELY = {
    "teddy bear": ("bedroom", "living_room"), "laptop": ("office", "living_room", "bedroom"),
    "keyboard": ("office",), "cup": ("kitchen", "living_room", "office"), "bowl": ("kitchen",),
    "book": ("bedroom", "living_room", "office"), "backpack": ("bedroom", "office"), "clock": ("bedroom", "office"),
    "sports ball": ("living_room", "bedroom"), "dog": ("living_room",), "elephant": ("living_room",),
    "bed": ("bedroom",), "couch": ("living_room",), "tv": ("living_room",), "refrigerator": ("kitchen",),
    "oven": ("kitchen",), "sink": ("kitchen", "bathroom"), "toilet": ("bathroom",), "desk": ("office",),
    "dining table": ("kitchen", "living_room"), "chair": ("kitchen", "office", "living_room"),
}


class TimeUp(Exception):
    """The mission's time limit has passed."""


def run(ex, task, cancel):
    """Run the mission in task.command.params on executor `ex` (called from its worker)."""
    mission = Mission(ex, task, cancel)
    try:
        {"search": search, "guard": guard, "patrol": patrol, "explore": explore}[mission.kind](mission)
    except TimeUp:
        mission.timed_out()


class Mission:
    def __init__(self, ex, task, cancel):
        self.ex, self.task, self.cancel = ex, task, cancel
        p = task.command.params
        self.kind = p["mission"]
        self.began = ex.now()
        self.deadline = self.began + p["time_limit"]
        self.key = (self.kind, p.get("target"), p.get("where"), tuple(p.get("rooms") or ()))
        self.state, self.resumed = ex._mission_state(self.key, RESUME_S)
        task.detail["mission"] = {"kind": self.kind, "goal": task.command.summary, "log": [], "outcome": None}

    # ---- bookkeeping ---------------------------------------------------------------------------------------
    def left(self):
        return self.deadline - self.ex.now()

    def check(self):
        """Stop on Stop (Cancelled), or when time is up (TimeUp)."""
        self.ex._check(self.cancel)
        if self.left() <= 0:
            raise TimeUp()

    def say(self, text, image=None):
        """Report progress: in the mission's log, and to the user as a robot update (an alert with an image)."""
        log = self.task.detail["mission"]["log"]
        log.append(text)
        del log[:-LOG_LINES]
        self.ex._mission_progress(self.task, text, image)

    def finish(self, outcome, result, done=True, **extra):
        self.task.detail["mission"].update(outcome=outcome, result=result, seconds=round(self.ex.now() - self.began),
                                           **extra)
        self.state["finished"] = done

    def timed_out(self):
        if self.kind == "patrol":            # its time limit is how long to patrol for: the normal end
            rounds = self.state.get("rounds", 0)
            self.finish("done", f"I patrolled {rounds} round{'s' if rounds != 1 else ''}", rounds=rounds)
            return
        what = {"search": f"I didn't find the {self.task.command.params.get('target')}",
                "explore": "I hadn't been everywhere yet", "guard": "I didn't get to my guarding spot"}[self.kind]
        extra = {"checked": sorted(self.state.get("checked_names", set()))} if self.kind == "search" else {}
        self.finish("timed_out", f"{what} before the time limit ({self.task.command.params['time_limit']:g} s); "
                    "ask again to carry on", done=False, **extra)

    # ---- steps: each an ordinary executor step, with its limits and LiDAR checks -----------------------------
    def budget(self):
        return self.ex._step_budget(self.cancel, max(1., min(STEP_S, self.left())))

    def step(self, fn, command):
        """Run one executor step (fn(subtask, budget)); False if it couldn't finish (blocked, no route). Stop and
        the time limit end the mission; a fall or lost telemetry ends it with that error."""
        self.check()
        sub = self.ex._subtask(command, self.task)
        try:
            fn(sub, self.budget())
            return sub.status not in ("failed", "blocked")
        except AdapterError:
            self.check()                     # a step cut short by the time limit ends the mission as timed out
            self.ex._fresh_state()           # a fall or lost telemetry ends it (raises); anything else: carry on
            return False

    def walk_to(self, x, y, name):
        return self.step(self.ex._goto, Command("goto", "go_to_place", {}, {"target": "place", "name": name,
                                                                         "x": x, "y": y}, summary=f"go to the {name}"))

    def visit(self, viewpoint):
        x, y, yaw = viewpoint
        return self.step(self.ex._visit_viewpoint, Command("viewpoint", "exploration", {}, {
            "x": x, "y": y, "yaw": yaw, "timeout_s": max(1., min(90., self.left()))}, summary="explore"))

    def turn(self, angle):
        return self.step(lambda sub, budget: self.ex._turn_by(sub, budget, angle),
                         Command("velocity", "turn", {}, {}, summary="turn"))

    def look_around(self, wanted=None):
        """Look all round in quarter turns; stops early and returns wanted() once it gives something."""
        seen = []
        for quarter in range(LOOK_TURNS):
            self.check()
            self.ex.look()
            seen += self.ex._sightings()
            if wanted is not None:
                found = wanted()
                if found:
                    return found
            if quarter < LOOK_TURNS - 1:
                self.turn(math.pi / 2)
        return None if wanted is not None else seen

    def approach(self, target):
        """Walk up to the target; its room, or None if it couldn't be reached."""
        if not self.step(self.ex._approach, resolve("approach", {"target": target})[0]):
            return None
        room = None
        match = self.ex._match(target)[0]
        if match and match[0].x_m is not None:
            det, (x, y, yaw) = match
            room = self.ex.room_at(x + math.cos(yaw) * det.x_m - math.sin(yaw) * det.y_m,
                                   y + math.sin(yaw) * det.x_m + math.cos(yaw) * det.y_m)
        if room is None:                     # the target's own spot is occupied on the map: the room Go2 is in
            s = self.ex._fresh_state()
            room = self.ex.room_at(s.x, s.y)
        if room is not None and room.startswith("room_"):
            # Its room has no name yet (a room is named once what's in it has been seen twice): look round it first.
            self.look_around()
            s = self.ex._fresh_state()
            room = self.ex.room_at(s.x, s.y) or room
        return (room or "this room").replace("_", " ")

    def toward(self, target):
        """Walk to about a metre short of where the target was last seen: go_to plans through unmapped space,
        which approach won't, so a target seen through a doorway into an unmapped room can be reached. False if
        there's nowhere to go."""
        spot = self.ex._recall(target)
        if spot is None:
            return False
        _, ox, oy = spot
        x, y, _ = self.here()
        distance = math.hypot(ox - x, oy - y)
        if distance < 1.5:
            return False
        k = (distance - 1.2) / distance
        return self.walk_to(x + (ox - x) * k, y + (oy - y) * k, target)

    def here(self):
        s = self.ex._fresh_state()
        return s.x, s.y, s.yaw

    def rooms(self):
        return self.ex.rooms(fresh=True)

    def walk_into(self, room):
        """Walk to a roomy spot in the room: near its middle, clear of furniture (the middle itself can be wedged
        between a bed and a wardrobe)."""
        spot = self.ex.room_regions.region_point(room["region"], clearance_m=STAND_CLEAR_M)
        x, y = spot if spot is not None else (room["x"], room["y"])
        return self.walk_to(x, y, room["name"].replace("_", " "))

    def explore_step(self):
        """Go to the next unexplored spot and look all round there: True, or False once nowhere is left."""
        st = self.state.setdefault("explored", {"visited": [], "blocked": []})
        goal = self.ex.exploration_viewpoint(st["visited"] + st["blocked"])
        if goal is None and st["blocked"]:
            st["blocked"].clear()           # try once more the spots that couldn't be reached
            goal = self.ex.exploration_viewpoint(st["visited"])
        if goal is None:
            return False
        (st["visited"] if self.visit(goal) else st["blocked"]).append(goal)
        return goal


# ---- search -----------------------------------------------------------------------------------------------------
def _label(ex, target):
    words = ex._words(target)
    return next((label for label in LIKELY if ex._score(SimpleNamespace(label=label, color=None), words)[0]), None)


def _kind(name):
    return re.sub(r"_\d+$", "", name.replace(" ", "_").casefold())


def _the(room):
    """'the kitchen', but 'bedroom 2'."""
    room = room.replace("_", " ")
    return room if re.search(r"\d$", room) or room.startswith("this ") else f"the {room}"


def search(m):
    ex, p = m.ex, m.task.command.params
    target, where = p["target"], p["where"]
    st = m.state
    st.setdefault("checked", set())                     # room regions looked round in
    st.setdefault("checked_names", set())
    likely = LIKELY.get(_label(ex, target), ())
    if m.resumed:
        m.say(f"carrying on the search for the {target}" +
              (f": already checked {', '.join(sorted(st['checked_names']))}" if st["checked_names"] else ""))

    def seen():
        return ex._match(target)[0]

    def found(room):
        m.say(f"found the {target} in {_the(room)}")
        m.finish("found", f"I found the {target} in {_the(room)} and I'm next to it", room=room,
                 checked=sorted(st["checked_names"]))

    def reach():
        """Walk up to the target in view: True once found. If there's no way to it from here, the search goes on
        (more of the house mapped can open a way); it ends 'blocked' only when nowhere is left to look."""
        room = m.approach(target)
        if not room and m.toward(target):       # seen into a room not on the map yet: get closer, then try again
            room = m.approach(target)
        if room:
            found(room)
            return True
        st["seen_unreached"] = True
        return False

    place = None
    if where:
        try:
            saved = ex.places.get(where)
            place = (saved.x, saved.y, saved.name)
        except ValueError:
            pass
    if place:                                           # a saved place: look round there, and that's all
        m.walk_to(*place)
        if m.look_around(seen) and reach():
            return
        m.finish("not_found", f"I looked round {_the(place[2])} and the {target} isn't there")
        return

    if not st.get("memory_tried"):                      # remembered, or in view where Go2 stands: go straight there
        st["memory_tried"] = True
        if ex._recall(target) is not None:
            m.say(f"I remember seeing a {target}; checking there first")
            if reach():
                return
        elif m.look_around(seen) and reach():
            return

    while True:
        m.check()
        x, y, _ = m.here()
        rooms = [r for r in m.rooms() if r["region"] not in st["checked"]]
        if where:
            wanted = _kind(where)
            rooms = [r for r in rooms if _kind(r["name"]) in (wanted, wanted.removeprefix("the_"))]
        rank = {kind: i for i, kind in enumerate(likely)}
        likely_rooms = rooms if where else [r for r in rooms if _kind(r["name"]) in rank]
        # Likely rooms first; then explore for more; then whatever rooms are left (unknown kinds, hallways).
        pick = likely_rooms or ([] if not st.get("explored_all") else rooms)
        if pick:
            room = min(pick, key=lambda r: (rank.get(_kind(r["name"]), len(rank)), math.hypot(r["x"] - x, r["y"] - y)))
            name = room["name"].replace("_", " ")
            m.walk_into(room)
            if m.look_around(seen) and reach():
                return
            st["checked"].add(room["region"])        # only once looked round: a stop on the way leaves it to do
            st["checked_names"].add(name)
            m.say(f"checked {_the(name)}: no {target}")
            continue
        if not st.get("explored_all"):
            if m.explore_step():
                if m.look_around(seen) and reach():
                    return
                continue
            st["explored_all"] = True
            continue
        break
    if st.get("seen_unreached"):
        if m.toward(target) and m.look_around(seen) and reach():   # one last try with the whole map known
            return
        m.finish("blocked", f"I can see the {target} but can't find a way to it; it may be boxed in by furniture",
                 done=False, checked=sorted(st["checked_names"]))
        return
    scope = f" in {_the(where)}" if where else " anywhere I could reach"
    m.finish("not_found", f"I searched{scope} and couldn't find the {target}" +
             (f" (checked {', '.join(sorted(st['checked_names']))})" if st["checked_names"] else ""),
             checked=sorted(st["checked_names"]))


# ---- guard -------------------------------------------------------------------------------------------------------
def guard(m):
    ex, p = m.ex, m.task.command.params
    where = p["place"]
    if where:
        destination = None
        try:
            saved = ex.places.get(where)
            destination = (saved.x, saved.y, saved.name, saved.yaw)
        except ValueError:
            s = ex._fresh_state()
            room = ex._room_destination(where, s.x, s.y)
            if room is not None:
                destination = (*room, None)
        if destination is None:
            raise AdapterError(f"I don't know a room or saved place called {where!r}; save it or explore first")
        x, y, name, yaw = destination
        if not m.walk_to(x, y, name):
            raise AdapterError(f"I couldn't get to the {name}")
        facing = yaw if yaw is not None else ex._way_in(x, y)
        if facing is not None:
            m.turn(_wrap(facing - m.here()[2]))
    m.step(ex._sport, resolve("posture", {"name": "sit"})[0])
    what = _the(where) if where else "this spot"
    m.say(f"guarding {what}; I'll tell you if anyone comes in")
    alerts, last_seen = 0, None
    while m.left() > 0:
        ex._check(m.cancel)
        ex.look()
        people = [d for d, _ in ex._detections() if d.label == "person"]
        now = ex.now()
        if people and (last_seen is None or now - last_seen > NEW_PERSON_S):
            alerts += 1
            person = min(people, key=lambda d: d.distance_m if d.distance_m is not None else 99)
            who = f"a person in {person.clothing}" if person.clothing else "a person"
            where_seen = f"{person.distance_m:.1f} m away, " if person.distance_m is not None else ""
            m.say(f"someone came in: {who}, {where_seen}{_bearing(person.bearing_deg)}", image=ex.last_frame)
        if people:
            last_seen = now
        ex._idle(min(GUARD_LOOK_S, max(0., m.left())), m.cancel)
    m.finish("done", f"I finished guarding {what}: " +
             (f"{alerts} alert{'s' if alerts != 1 else ''}" if alerts else "nobody came in"), alerts=alerts)


def _bearing(degrees):
    if abs(degrees) < 15:
        return "straight ahead"
    return f"{'left' if degrees > 0 else 'right'} of straight ahead"


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


# ---- patrol ------------------------------------------------------------------------------------------------------
def patrol(m):
    ex, p = m.ex, m.task.command.params
    st = m.state
    rounds = st.setdefault("rounds", 0)
    if not m.rooms():
        m.say("I don't know any rooms yet, so I'm exploring first")
        while m.explore_step():
            m.look_around()
    while True:
        m.check()
        started = ex.now()
        rooms = m.rooms()
        if p["rooms"]:
            wanted = {_kind(r) for r in p["rooms"]}
            rooms = [r for r in rooms if _kind(r["name"]) in wanted or r["name"].casefold() in
                     {w.casefold() for w in p["rooms"]}]
            if not rooms:
                raise AdapterError(f"I don't know rooms called {', '.join(p['rooms'])}; explore or name them first")
        seen = []
        for room in _route(rooms, m.here()):
            m.walk_into(room)
            seen += [{**thing, "room": ex.room_at(thing["x"], thing["y"]) or room["name"]}
                     for thing in m.look_around()]
        rounds += 1
        st["rounds"] = rounds
        changes = _changes(st.get("last"), _merge(seen), {r["name"] for r in rooms})
        st["last"] = _merge(seen)
        m.say(f"patrol round {rounds}: " + ("; ".join(changes) if changes else
                                            "nothing has changed" if rounds > 1 else "noted where everything is"))
        wait = p["every"] - (ex.now() - started)
        if wait >= m.left():
            break
        ex._idle(max(0., wait), m.cancel)
    m.finish("done", f"I patrolled {rounds} round{'s' if rounds != 1 else ''}", rounds=rounds)


def _route(rooms, here):
    """Rooms in a nearest-next order from here."""
    x, y, _ = here
    left, route = list(rooms), []
    while left:
        room = min(left, key=lambda r: math.hypot(r["x"] - x, r["y"] - y))
        left.remove(room)
        route.append(room)
        x, y = room["x"], room["y"]
    return route


def _merge(seen):
    """One entry per thing seen in a round: sightings of the same label and colour within SAME_PLACE_M are one."""
    things = []
    for s in seen:
        if any(t["label"] == s["label"] and t["color"] == s["color"] and
               math.hypot(t["x"] - s["x"], t["y"] - s["y"]) < SAME_PLACE_M for t in things):
            continue
        things.append(s)
    return things


def _changes(before, now, rooms):
    """Plain sentences for what changed between two rounds' sightings, in the patrolled rooms."""
    def name(t):
        return " ".join(filter(None, (t["color"], t["label"])))

    def room(t):
        return _the(t["room"])
    out = [f"a person{' in ' + t['clothing'] if t.get('clothing') else ''} in {room(t)}"
           for t in now if t["label"] == "person"]
    if before is None:
        return out
    before = [t for t in before if t["label"] != "person" and t["room"] in rooms]
    now = [t for t in now if t["label"] != "person"]
    unmatched = list(now)
    for t in before:
        same = [n for n in unmatched if n["label"] == t["label"] and n["color"] == t["color"]]
        here = [n for n in same if math.hypot(n["x"] - t["x"], n["y"] - t["y"]) < SAME_PLACE_M]
        if here:
            unmatched.remove(here[0])
        elif same:
            moved = min(same, key=lambda n: math.hypot(n["x"] - t["x"], n["y"] - t["y"]))
            unmatched.remove(moved)
            out.append(f"the {name(t)} moved" + (f" from {room(t)} to {room(moved)}"
                                                 if room(moved) != room(t) else f" within {room(t)}"))
        else:
            out.append(f"the {name(t)} is gone from {room(t)}")
    out += [f"there's a new {name(n)} in {room(n)}" for n in unmatched]
    return out


# ---- explore -----------------------------------------------------------------------------------------------------
def explore(m):
    ex = m.ex
    st = m.state
    announced = st.setdefault("announced", set())
    m.say("carrying on exploring" if m.resumed else "exploring the house")
    while True:
        m.check()
        if not m.explore_step():
            break
        m.look_around()
        for room in m.rooms():
            if not room["name"].startswith("room_") and room["name"] not in announced:
                announced.add(room["name"])
                m.say(f"found {_the(room['name'])}")
    rooms = m.rooms()
    named = [r["name"].replace("_", " ") for r in rooms if not r["name"].startswith("room_")]
    unnamed = len(rooms) - len(named)
    listing = ", ".join(named) if named else "none I could name"
    extra = f" and {unnamed} I couldn't name" if unnamed else ""
    m.finish("explored", f"I've been everywhere I could reach: {len(rooms)} room{'s' if len(rooms) != 1 else ''}"
             f" ({listing}{extra})", rooms=[r["name"] for r in rooms])
