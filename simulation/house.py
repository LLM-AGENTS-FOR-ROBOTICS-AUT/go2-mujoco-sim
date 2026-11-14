"""Multi-room houses for the simulation: rooms, doorways and furniture that fits each room.

A house is a plain, JSON-friendly description (like random_world's output), so it can be written to a
file for the simulator process and replayed later:

  {"scene": "house",
   "rooms":   [{"name": "bedroom_2", "type": "bedroom", "rect": [x0, y0, x1, y1]}, ...],
   "doors":   [{"name": "door_0", "rooms": ["hallway", "bedroom_2"], "pos": [x, y], "axis": "x" | "y",
                "width": 1.0, "open": true}, ...],
   "objects": [{"name": "teddy_bear", "label": "teddy bear", "kind": "teddy", "room": "bedroom_2",
                "pos": [x, y], "yaw": 0.8, "rgba": [r, g, b, 1], "gso": "teddy_bear"},
               {"name": "laptop", ..., "on": "desk_1"}, ...],     # "on": standing on that furniture
   "start":   [x, y, yaw]}

Rooms are axis-aligned rectangles that tile the house. A door sits on a wall line: axis "x" means the
wall runs along x (a horizontal wall at y = pos[1]) and the gap spans x; axis "y" is a vertical wall at
x = pos[0]. An exterior door has a single room in "rooms". The description is the ground truth used
to score missions: which room each object is in and where each doorway is.

This module only describes houses; building one into MuJoCo (build_house) imports mujoco when called,
so the console can make descriptions without loading the simulator.
"""
from __future__ import annotations

import colorsys
import json
import math
import random
from pathlib import Path

ROOM_TYPES = ("bedroom", "kitchen", "living_room", "bathroom", "hallway", "office")
WALL_THICK = .05                          # half thickness, as in the single room
# Walls reach the ceiling (unlike the single room's 2 m walls under a 2.4 m ceiling), so the camera never
# sees sky over them: in YOLO mode that strip was taken for birds and people. Doorways are 2 m tall, with
# wall above them.
CEILING, WALL_HEIGHT, DOOR_HEIGHT = 2.4, 2.4, 2.0
DOOR_MIN_WIDTH = .9                       # Go2 fits through with the executor's clearance margin
# No furniture this close to a doorway's centre, either side. All of Go2 is in a room once its centre is
# 0.5 m past the wall (0.05 m wall + 0.36 m half length); its nose is then 0.86 m in, and the executor
# keeps about 0.4 m clear ahead at walking pace (Executor._margin): 1.25 m, plus room to turn around.
DOOR_CLEAR = 1.4
START_CLEAR = .9                          # ... or to the robot's start

# kind -> (detector label, footprint half extents (x, y) before yaw, big furniture, Google Scanned Object)
# Big furniture stands against a wall with its back (local +y) to it; small things go anywhere free.
KINDS = {
    "bed": ("bed", (.7, 1.0), True, None),
    "refrigerator": ("refrigerator", (.35, .35), True, None),
    "oven": ("oven", (.32, .32), True, None),
    "sink": ("sink", (.5, .3), True, None),
    "tv": ("tv", (.6, .2), True, None),
    "couch": ("couch", (.9, .45), True, None),
    "dining_table": ("dining table", (.6, .4), True, None),
    "desk": ("desk", (.6, .35), True, None),
    "toilet": ("toilet", (.25, .35), True, None),
    "chair": ("chair", (.25, .25), False, None),
    "plant": ("potted plant", (.28, .28), False, None),
    "teddy": ("teddy bear", (.18, .18), False, "teddy_bear"),
    "backpack": ("backpack", (.18, .14), False, "backpack"),
    "cup": ("cup", (.06, .06), False, "mug"),
    "bowl": ("bowl", (.11, .11), False, "dog_bowl"),
    "laptop": ("laptop", (.18, .13), False, "laptop"),
    "keyboard": ("keyboard", (.22, .08), False, "keyboard"),
    "book": ("book", (.12, .09), False, "book"),
    "clock": ("clock", (.08, .08), False, "alarm_clock"),
    "ball": ("sports ball", (.12, .12), False, None),
    "toy_dog": ("dog", (.12, .2), False, "toy_dog"),
    "toy_elephant": ("elephant", (.13, .16), False, "toy_elephant"),
}
# What each room type usually has: (kind, chance). The teddy bear is placed separately, once per house.
FURNISHING = {
    "bedroom": [("bed", 1.), ("chair", .4), ("plant", .3), ("backpack", .4), ("book", .5), ("clock", .4), ("ball", .2)],
    "kitchen": [("refrigerator", 1.), ("oven", 1.), ("sink", 1.), ("dining_table", .8), ("chair", .8), ("chair", .6),
                ("cup", .7), ("bowl", .7), ("plant", .3)],
    "living_room": [("couch", 1.), ("tv", 1.), ("dining_table", .3), ("chair", .4), ("plant", .6), ("ball", .5),
                    ("toy_dog", .4), ("toy_elephant", .3), ("book", .4), ("laptop", .3)],
    "bathroom": [("toilet", 1.), ("sink", 1.), ("plant", .2)],
    "office": [("desk", 1.), ("chair", 1.), ("laptop", .8), ("keyboard", .6), ("book", .6), ("backpack", .4),
               ("plant", .4), ("clock", .4)],
    "hallway": [],                         # kept clear: every route between rooms runs through it
}
PORTABLE = {"teddy", "ball", "backpack", "book", "cup", "toy_dog", "toy_elephant"}
LENGTHWISE = {"bed", "dining_table", "desk"}   # may also stand side-on to the wall, when head-on won't fit
UNLIKELY_SHARE = .1                        # portable things sometimes end up in an unlikely room
TEDDY_IN_BEDROOM = .85
COLOURS = {"bed": (.9, .9, .92, 1), "refrigerator": (.88, .88, .9, 1), "oven": (.92, .92, .9, 1),
           "sink": (.9, .9, .88, 1), "tv": (.15, .15, .17, 1), "couch": (.5, .5, .55, 1), "dining_table": (.65, .5, .35, 1),
           "desk": (.6, .45, .3, 1), "toilet": (.95, .95, .95, 1), "chair": (.45, .3, .2, 1), "plant": (.2, .6, .25, 1),
           "teddy": (.6, .42, .25, 1), "backpack": (.15, .3, .85, 1), "cup": (.95, .95, .92, 1), "bowl": (.15, .3, .85, 1),
           "laptop": (.5, .5, .55, 1), "keyboard": (.12, .12, .14, 1), "book": (.85, .1, .1, 1), "clock": (.85, .1, .1, 1),
           "toy_dog": (.55, .38, .22, 1), "toy_elephant": (.55, .55, .6, 1)}
BALL_COLOURS = [(.85, .1, .1, 1), (.15, .3, .85, 1), (.2, .6, .25, 1), (.95, .8, .1, 1)]
# Random houses pick each thing's colour from its kind's palette and name it in the description ("colour"), so
# a mission can ask for "the blue cup" (#21). Each word is what go2.perception.color_name says when Go2's camera
# sees that colour in a house (tests/test_house.py renders them all). Kinds with a realistic model (beds, couches,
# chairs, plants, TVs) or a scanned one (its first teddy, mug, laptop...) keep their fixed colour and get no
# word: the model's own texture is what shows. Laptops have none either: from the front Go2 sees the black screen.
RED, BLUE, GREEN, YELLOW = (.8, .12, .1, 1), (.15, .3, .8, 1), (.2, .6, .25, 1), (.95, .8, .1, 1)
ORANGE, PINK, PURPLE = (.95, .45, .1, 1), (.9, .4, .65, 1), (.5, .25, .7, 1)
WHITE, GREY, BLACK, BROWN = (.95, .95, .93, 1), (.5, .5, .52, 1), (.1, .1, .11, 1), (.45, .3, .17, 1)
_BRIGHT = [("red", RED), ("blue", BLUE), ("green", GREEN), ("yellow", YELLOW), ("orange", ORANGE)]
PALETTES = {
    "cup": _BRIGHT + [("white", WHITE), ("black", BLACK), ("pink", PINK)],
    "bowl": _BRIGHT + [("white", WHITE), ("black", BLACK)],
    "ball": _BRIGHT + [("white", WHITE), ("pink", PINK), ("purple", PURPLE)],
    "backpack": _BRIGHT + [("black", BLACK), ("grey", GREY), ("pink", PINK), ("purple", PURPLE)],
    "book": _BRIGHT + [("white", WHITE), ("black", BLACK), ("purple", PURPLE)],
    "clock": [("red", RED), ("blue", BLUE), ("white", WHITE), ("black", BLACK), ("yellow", YELLOW)],
    "keyboard": [("black", BLACK), ("white", WHITE), ("grey", GREY)],
    "dining_table": [("brown", BROWN), ("white", WHITE), ("black", BLACK), ("grey", GREY)],
    "desk": [("brown", BROWN), ("black", BLACK), ("grey", GREY), ("red", RED)],      # its shaded front: no white
    "refrigerator": [("white", WHITE), ("grey", GREY), ("black", BLACK), ("red", RED)],
    "oven": [("white", WHITE), ("black", BLACK), ("grey", GREY)],
    "sink": [("white", WHITE), ("grey", GREY), ("black", BLACK)],
    "toilet": [("white", WHITE)],
}
# Realistic furniture from simulation/fetch_furniture.py: kind -> [(model, lift, shown, turn)]. A model is
# turned by `turn` quarter turns (anticlockwise) so its back faces local +y like the plain shapes, scaled to
# its kind's footprint and drawn over the kind's plain shapes, which stay its physical body, faded so the
# camera doesn't see them but the LiDAR still does: collisions and obstacle checks are as before. The first
# `shown` plain shapes stay visible (the TV's stand) and `lift` raises the model onto them. Tables, sinks and
# fridges keep plain shapes: from Go2's low camera YOLO named no table or sink model tried, and the best
# fridge needs a JPEG texture, which MuJoCo can't load.
FURNITURE_DIR = Path(__file__).resolve().parents[1] / "_deps/furniture"
MODELS = {"chair": [("dining_chair_02", 0., 0, 0), ("painted_wooden_chair_01", 0., 0, 0)],
          "couch": [("couch_starstarbird", 0., 0, 0)],
          "plant": [("potted_plant_02", 0., 0, 0), ("potted_plant_01", 0., 0, 0)],
          "tv": [("Television_01", .5, 1, 0)],
          "bed": [("bed_tharadelamo", 0., 0, -1)],        # its headboard is at -x as downloaded
          "oven": [("electric_stove", 0., 0, 0)]}
FADED = .01                                # opacity of the shapes under a model (0 would hide them from the LiDAR)

# Things on furniture (#21): a teddy bear on the bed, a laptop on the desk, cups on the table. Random houses only;
# the demo house keeps everything on the floor. SURFACES: kind -> (height of its top, usable top in its own
# frame (x0, y0, x1, y1)), from the plain shapes in _geoms (the bed's top stops short of the pillow). In MuJoCo
# each thing is set down on whatever is drawn there (rest_on_furniture), so on a realistic model too.
SURFACES = {"bed": (.43, (-.62, -.91, .62, .55)),
            "desk": (.76, (-.58, -.33, .58, .33)),
            "dining_table": (.78, (-.58, -.38, .58, .38))}
# What goes on each surface, with the chance that a thing of that kind in the room ends up there rather than on
# the floor. Keyboards, bowls, clocks and books are left off desks and tables: from Go2's camera, about 0.36 m
# up, something that flat is hidden behind the table's edge wherever it stands (see sight_share).
ON_TOP = {"bed": {"teddy": .5, "book": .4, "backpack": .3, "laptop": .4, "toy_dog": .4, "toy_elephant": .4},
          "desk": {"laptop": .85, "cup": .4},
          "dining_table": {"cup": .7, "laptop": .4}}
# Height of each thing that can go on furniture: the lower of its plain shapes and its scanned model (which is
# only used when downloaded), so a sight-line check never counts on the taller one.
HEIGHTS = {"teddy": .37, "book": .04, "backpack": .4, "laptop": .19, "cup": .1, "toy_dog": .2, "toy_elephant": .27}
# Go2's camera, for the sight-line check (views): height when standing, the top of its view (half of the 88.5
# degree vertical field of view), how far back it stands to look at most, and how far ahead of the body's
# centre the camera is (world.CAMERA_POS).
CAMERA_EYE, VIEW_UP_DEG, STANDOFF, CAMERA_AHEAD = .36, 44., 1.5, .33
SCREEN_BACK = .15                          # a laptop faces Go2 (_place_on): its screen, the part that shows, is
                                           # this much further back than its middle
VISIBLE_SHARE = .5                         # at least half of a thing on furniture can be seen from the floor
# Things taller than the sight line, on the floor or beside the thing on the furniture: they hide what is behind.
BLOCKS_VIEW = {"bed", "refrigerator", "oven", "sink", "tv", "couch", "dining_table", "desk", "toilet", "chair",
               "plant", "backpack", "teddy", "laptop", "toy_elephant", "toy_dog"}


def room_at(house, x, y):
    """The name of the room containing (x, y), or None outside the house."""
    for room in house["rooms"]:
        x0, y0, x1, y1 = room["rect"]
        if x0 <= x <= x1 and y0 <= y <= y1:
            return room["name"]
    return None


def objects_in(house, room):
    """Names of the objects the description places in `room`."""
    return [item["name"] for item in house["objects"] if item["room"] == room]


def footprint(item):
    """An object's floor rectangle (x0, y0, x1, y1) in the world, from its kind, position and yaw."""
    hx, hy = KINDS[item["kind"]][1]
    quarter = abs(math.sin(item["yaw"])) > .7            # turned a quarter: the extents swap
    if KINDS[item["kind"]][2]:
        ex, ey = (hy, hx) if quarter else (hx, hy)
    else:
        ex = ey = math.hypot(hx, hy)                        # small things can face any way
    x, y = item["pos"]
    return x - ex, y - ey, x + ex, y + ey


def _gap(box, x, y):
    """Distance from point (x, y) to rectangle `box` (0 inside)."""
    return math.hypot(max(box[0] - x, 0, x - box[2]), max(box[1] - y, 0, y - box[3]))


def _local(furniture, x, y):
    """(x, y) in `furniture`'s own frame."""
    fx, fy = furniture["pos"]
    c, s = math.cos(furniture["yaw"]), math.sin(furniture["yaw"])
    return c * (x - fx) + s * (y - fy), -s * (x - fx) + c * (y - fy)


def _crosses(box, a, b):
    """True if the segment from a to b passes through rectangle box (x0, y0, x1, y1)."""
    t0, t1 = 0., 1.
    for p, d, lo, hi in ((a[0], b[0] - a[0], box[0], box[2]), (a[1], b[1] - a[1], box[1], box[3])):
        if abs(d) < 1e-9:
            if not lo <= p <= hi:
                return False
            continue
        ta, tb = sorted(((lo - p) / d, (hi - p) / d))
        t0, t1 = max(t0, ta), min(t1, tb)
        if t0 > t1:
            return False
    return True


def views(house, item):
    """Where Go2 can stand to see `item` on a piece of furniture, best first, as [(share, (x, y, yaw))]: the
    share of the thing (0 to 1) in view over the furniture's edge, and the pose of Go2's body (its centre and
    heading). Empty when there is nowhere to see it from.

    Side view of one spot: the camera is CAMERA_EYE up, e metres back from the top's edge, and the thing is s
    metres in from that edge. The line from the camera grazing the edge rises to CAMERA_EYE + (top -
    CAMERA_EYE) * (e + s) / e above the thing; only the part above that line is in view, and below the top of
    the camera's view. Spots are tried in front of each open side (not a bed's head end, where the headboard
    is), straight on and half a metre to either side, up to STANDOFF back. A spot needs room for Go2's body
    inside the room, and a clear line from the camera to the thing past other furniture taller than the
    camera. Simplified: the thing's own shape is ignored, and a chair or plant either blocks the line or not."""
    furniture = next(o for o in house["objects"] if o["name"] == item["on"])
    top, (x0, y0, x1, y1) = SURFACES[furniture["kind"]]
    height = HEIGHTS[item["kind"]]
    fx, fy = furniture["pos"]
    c, sn = math.cos(furniture["yaw"]), math.sin(furniture["yaw"])
    u, v = _local(furniture, *item["pos"])
    tx, ty = item["pos"]
    rx0, ry0, rx1, ry1 = next(r["rect"] for r in house["rooms"] if r["name"] == furniture["room"])
    floor = [o for o in house["objects"] if not o.get("on")]
    blockers = [footprint(o) for o in floor if o is not furniture and o["kind"] in BLOCKS_VIEW]
    blockers += [footprint(o) for o in house["objects"]                # a teddy bear in front of a book
                 if o.get("on") == furniture["name"] and o["name"] != item.get("name") and o["kind"] in BLOCKS_VIEW]
    room_for_body = ROBOT_RADIUS + WALL_THICK
    out = []
    sides = [((0, -1), v - y0, (u, y0)), ((-1, 0), u - x0, (x0, v)), ((1, 0), x1 - u, (x1, v))]
    if furniture["kind"] != "bed":
        sides.append(((0, 1), y1 - v, (u, y1)))
    deeper = SCREEN_BACK if item["kind"] == "laptop" else 0.
    for (su, sv), s, (eu, ev) in sides:
        s += deeper
        nx, ny = c * su - sn * sv, sn * su + c * sv                   # outward, in the world
        px, py = fx + c * eu - sn * ev, fy + sn * eu + c * ev         # the edge, level with the thing
        for e in (STANDOFF, 1., .6):
            lowest = max(top, CAMERA_EYE + (top - CAMERA_EYE) * (e + s) / e)
            for side_step in (0., -.5, .5):
                cx, cy = px + nx * e - ny * side_step, py + ny * e + nx * side_step       # the camera
                yaw = math.atan2(ty - cy, tx - cx)
                bx, by = cx - CAMERA_AHEAD * math.cos(yaw), cy - CAMERA_AHEAD * math.sin(yaw)
                if not all(rx0 + room_for_body <= x <= rx1 - room_for_body and
                           ry0 + room_for_body <= y <= ry1 - room_for_body for x, y in ((bx, by), (cx, cy))):
                    continue
                if any(_gap(footprint(o), bx, by) < ROBOT_RADIUS or _gap(footprint(o), cx, cy) < .05 for o in floor):
                    continue                                          # no room for Go2 there
                if any(_crosses(box, (cx, cy), (tx, ty)) for box in blockers):
                    continue
                d = math.hypot(tx - cx, ty - cy)
                highest = min(top + height, CAMERA_EYE + math.tan(math.radians(VIEW_UP_DEG)) * d)
                share = min(max((highest - lowest) / height, 0.), 1.)
                out.append((round(share, 2), (round(bx, 2), round(by, 2), round(yaw, 2))))
    out.sort(key=lambda view: -view[0])                               # stable: further back first on ties
    return out


def sight_share(house, item):
    """How much of `item` (0 to 1) Go2 can see from the best spot on the floor (views). 1 for things on the
    floor, which this doesn't check."""
    if not item.get("on"):
        return 1.
    found = views(house, item)
    return found[0][0] if found else 0.


def _place_on(rng, house, furniture, kind, on_top):
    """A spot for `kind` on top of `furniture` where Go2 can see it from the floor, as (x, y, yaw), or None.
    on_top: what is already on it, as (x, y, radius)."""
    _, (x0, y0, x1, y1) = SURFACES[furniture["kind"]]
    r = math.hypot(*KINDS[kind][1])
    if kind == "laptop":         # turned square to the side it's seen from, so it needs less margin
        r = max(KINDS[kind][1]) + .02
    if x1 - x0 < 2 * r or y1 - y0 < 2 * r:
        return None
    c, s = math.cos(furniture["yaw"]), math.sin(furniture["yaw"])
    for _ in range(30):
        u, v = rng.uniform(x0 + r, x1 - r), rng.uniform(y0 + r, y1 - r)
        x = round(furniture["pos"][0] + c * u - s * v, 2)
        y = round(furniture["pos"][1] + s * u + c * v, 2)
        if any(math.hypot(x - ox, y - oy) < r + orad + .03 for ox, oy, orad in on_top):
            continue
        yaw = round(furniture["yaw"] + rng.uniform(-.3, .3), 2)      # facing out, like the furniture
        found = views(house, {"kind": kind, "pos": [x, y], "on": furniture["name"]})
        if not found or found[0][0] < VISIBLE_SHARE:
            continue
        if kind == "laptop":     # opened towards where it's seen from: edge on, a laptop is a thin line
            bx, by, look = found[0][1]
            dx, dy = bx + CAMERA_AHEAD * math.cos(look) - x, by + CAMERA_AHEAD * math.sin(look) - y
            yaw = round(math.atan2(dx, -dy), 2)                       # its keyboard side (-y) towards Go2
        return x, y, yaw
    return None


def _place(rng, house, room, kind, taken):
    """A free spot for `kind` in `room`, as (x, y, yaw), or None. Big furniture backs onto a wall."""
    x0, y0, x1, y1 = room["rect"]
    hx, hy = KINDS[kind][1]
    big = KINDS[kind][2]
    margin = WALL_THICK + .05
    open_sides = "".join(side for door in house["doors"] if door.get("opening") and room["name"] in door["rooms"]
                         for side, axis, line in (("N", "x", y1), ("S", "x", y0), ("E", "y", x1), ("W", "y", x0))
                         if door["axis"] == axis and abs((door["pos"][1] if axis == "x" else door["pos"][0]) - line) < 1e-6)
    sides = "".join(side for side in "NSEW" if side not in open_sides)    # nothing to stand against there
    for _ in range(60):
        if big:
            side = rng.choice(sides)
            yaw = {"N": 0., "S": math.pi, "E": -math.pi / 2, "W": math.pi / 2}[side]
            turned = kind in LENGTHWISE and rng.random() < .5
            if turned:
                yaw += math.pi / 2
            ex, ey = (hy, hx) if (side in "EW") != turned else (hx, hy)
            if x1 - x0 < 2 * (ex + margin) or y1 - y0 < 2 * (ey + margin):
                return None
            x = rng.uniform(x0 + margin + ex, x1 - margin - ex)
            y = rng.uniform(y0 + margin + ey, y1 - margin - ey)
            if side in "EW":
                x = x1 - margin - ex if side == "E" else x0 + margin + ex
            else:
                y = y1 - margin - ey if side == "N" else y0 + margin + ey
        else:
            yaw = rng.uniform(-math.pi, math.pi)
            r = math.hypot(hx, hy)
            if x1 - x0 < 2 * (r + margin) or y1 - y0 < 2 * (r + margin):
                return None
            x, y = rng.uniform(x0 + margin + r, x1 - margin - r), rng.uniform(y0 + margin + r, y1 - margin - r)
        box = footprint({"kind": kind, "pos": (x, y), "yaw": yaw})
        if any(box[0] < t[2] + .15 and t[0] < box[2] + .15 and box[1] < t[3] + .15 and t[1] < box[3] + .15 for t in taken):
            continue
        if any(_gap(box, *door["pos"]) < DOOR_CLEAR for door in house["doors"]):
            continue
        if _gap(box, *house["start"][:2]) < START_CLEAR:
            continue
        return round(x, 2), round(y, 2), round(yaw, 2)
    return None


def furnish(rng, house, teddy_room=None, on_furniture=False, colours=False):
    """Fill house["objects"] with furniture that fits each room type, mostly in likely places, plus one
    teddy bear (in `teddy_room` when given). Each scanned object is used once; repeats are plain shapes.
    Kinds with realistic models name one, the styles taking turns. With on_furniture, some small things go
    on a bed, desk or table (ON_TOP), always where Go2 can see them from the floor; otherwise all stay on the
    floor. With colours, plain-shape things get a colour from PALETTES, named in "colour". The demo house uses
    neither, so it never changes."""
    objects, taken, used, counts, on_top = [], [], set(), {}, {}
    house["objects"] = objects                     # sight_share looks the furniture up as things are added

    def surface_for(kind, room):
        """Furniture in `room` that `kind` could go on this time, or None."""
        if not on_furniture:
            return None
        tops = [o for o in objects if o["room"] == room["name"] and kind in ON_TOP.get(o["kind"], {})]
        if not tops:
            return None
        furniture = rng.choice(tops)
        return furniture if rng.random() < ON_TOP[furniture["kind"]][kind] else None

    def add(kind, room, furniture=None):
        if furniture is not None:
            spot = _place_on(rng, house, furniture, kind, on_top.setdefault(furniture["name"], []))
            if spot is None:
                return add(kind, room)             # no spot in view up there: on the floor instead
        else:
            spot = _place(rng, house, room, kind, taken)
        if spot is None:
            return False
        x, y, yaw = spot
        gso = KINDS[kind][3] if KINDS[kind][3] not in used else None
        used.add(gso)
        counts[kind] = counts.get(kind, 0) + 1
        word = None
        if colours and kind in PALETTES and not gso:
            word, rgba = rng.choice(PALETTES[kind])
        else:
            rgba = rng.choice(BALL_COLOURS) if kind == "ball" else COLOURS[kind]
        item = {"name": gso or f"{kind}_{counts[kind]}", "label": KINDS[kind][0], "kind": kind, "room": room["name"],
                "pos": [x, y], "yaw": yaw, "rgba": list(rgba)}
        if word:
            item["colour"] = word
        if gso:
            item["gso"] = gso
        if kind in MODELS:          # in turn rather than at random, so the houses themselves don't change
            item["model"] = MODELS[kind][(counts[kind] - 1) % len(MODELS[kind])][0]
        if furniture is not None:
            item["on"] = furniture["name"]
            on_top[furniture["name"]].append((x, y, math.hypot(*KINDS[kind][1])))
        else:
            taken.append(footprint(item))          # things on furniture leave the floor free
        objects.append(item)
        return True

    rooms = [r for r in house["rooms"] if r["type"] != "hallway"]      # hallways stay clear
    wishes = [(kind, room) for room in rooms for kind, chance in FURNISHING[room["type"]] if rng.random() < chance]
    for kind, room in [w for w in wishes if KINDS[w[0]][2]]:          # big furniture first, while walls are free
        add(kind, room)
    if teddy_room is not None:                     # then the teddy bear, before small things fill the floor
        homes = [r for r in rooms if r["name"] == teddy_room]
    else:
        bedrooms = [r for r in rooms if r["type"] == "bedroom"]
        homes = bedrooms if bedrooms and rng.random() < TEDDY_IN_BEDROOM else list(rooms)
        rng.shuffle(homes)
    for room in homes + [r for r in rooms if r not in homes]:   # the next room if its first home is full
        if add("teddy", room, surface_for("teddy", room)):
            break
    for kind, room in [w for w in wishes if not KINDS[w[0]][2]]:
        if kind in PORTABLE and len(rooms) > 1 and rng.random() < UNLIKELY_SHARE:
            room = rng.choice([r for r in rooms if r is not room])
        add(kind, room, surface_for(kind, room))
    # Floor things added after something went up on a table can hide it, or stand where Go2 would look from:
    # move it to another spot up there that is in view, or else down to the floor.
    for item in [o for o in objects if o.get("on")] * 2:              # twice: a move can hide another
        if item not in objects or sight_share(house, item) >= VISIBLE_SHARE:
            continue
        mine = on_top[item["on"]]
        mine[:] = [t for t in mine if [t[0], t[1]] != item["pos"]]
        furniture = next(o for o in objects if o["name"] == item["on"])
        spot = _place_on(rng, house, furniture, item["kind"], mine)
        if spot is not None:
            mine.append((spot[0], spot[1], math.hypot(*KINDS[item["kind"]][1])))
        else:
            del item["on"]
            spot = _place(rng, house, next(r for r in rooms if r["name"] == item["room"]), item["kind"], taken)
            if spot is None:
                objects.remove(item)
                continue
        item["pos"], item["yaw"] = [spot[0], spot[1]], spot[2]
        if not item.get("on"):
            taken.append(footprint(item))
    return house


def _door(name, rooms, pos, axis, width=1.0, open_=True):
    return {"name": name, "rooms": list(rooms), "pos": [round(pos[0], 2), round(pos[1], 2)], "axis": axis,
            "width": width, "open": open_}


def house_demo():
    """A fixed, hand-designed house for demos and repeatable tests (12 x 10 m).

         y=5  +-----------------+-----------------+
              |   living_room   |     kitchen     |
              |                 D                 |
         y=1  +--------D--------+--------D--------+
       front  D             hallway        start  |
         y=-1 +-----D-----+------D-----+----D-----+
              | bedroom_1 |  bathroom  | bedroom_2 |   (teddy bear in bedroom_2)
         y=-5 +-----------+------------+-----------+
             x=-6        x=-2         x=1         x=6
    """
    rooms = [{"name": "hallway", "type": "hallway", "rect": [-6., -1., 6., 1.]},
             {"name": "living_room", "type": "living_room", "rect": [-6., 1., 0., 5.]},
             {"name": "kitchen", "type": "kitchen", "rect": [0., 1., 6., 5.]},
             {"name": "bedroom_1", "type": "bedroom", "rect": [-6., -5., -2., -1.]},
             {"name": "bathroom", "type": "bathroom", "rect": [-2., -5., 1., -1.]},
             {"name": "bedroom_2", "type": "bedroom", "rect": [1., -5., 6., -1.]}]
    doors = [_door("door_living", ("hallway", "living_room"), (-3., 1.), "x"),
             _door("door_kitchen", ("hallway", "kitchen"), (3., 1.), "x"),
             _door("door_living_kitchen", ("living_room", "kitchen"), (0., 3.), "y"),
             _door("door_bedroom_1", ("hallway", "bedroom_1"), (-4., -1.), "x"),
             _door("door_bathroom", ("hallway", "bathroom"), (-.5, -1.), "x"),
             _door("door_bedroom_2", ("hallway", "bedroom_2"), (3.5, -1.), "x"),
             _door("front_door", ("hallway",), (-6., 0.), "y", 1.0, False)]
    house = {"scene": "house", "rooms": rooms, "doors": doors, "objects": [], "start": [-4.5, 0., 0.]}
    return furnish(random.Random(17), house, teddy_room="bedroom_2")     # fixed seed: the same every time


# ---- random houses (training data and tests) -------------------------------------------------------------
MIN_ROOM = 2.6                             # a room's narrowest side: space to turn around and to furnish
CORNER = .4                                # wall kept either side of a doorway, so it never runs into a corner
EXTRA_DOOR = .25                           # chance of a second route between rooms that already connect
ROOM_ORDER = ["living_room", "bedroom", "kitchen", "bathroom", "bedroom", "office", "bedroom"]
# Shapes beyond a plain rectangle (#21). CARVE: chance a split layout loses one or two corner rooms (an L, T or
# Z-shaped house). L_HALL: chance a house of five rooms or more has two hallways at right angles. OPEN_PLAN:
# chance a living room and kitchen that share at least OPEN_PLAN_MIN of wall have no wall between them, and are
# still two rooms in the description (an "opening" in its doors).
CARVE, L_HALL, OPEN_PLAN, OPEN_PLAN_MIN = .5, .35, .5, 2.4


def _snap(v):
    return round(round(v / GRID) * GRID, 2)


def _split(rng, length, parts):
    """[0, length] cut into `parts` spans of at least MIN_ROOM, at 10 cm steps."""
    cuts = sorted(rng.uniform(0, length - parts * MIN_ROOM) for _ in range(parts - 1))
    edges = [0.] + [_snap(c + (i + 1) * MIN_ROOM) for i, c in enumerate(cuts)] + [length]
    return list(zip(edges, edges[1:]))


def _corridor_layout(rng, n):
    """A hallway through the middle with the other n - 1 rooms either side of it."""
    north = rng.randint(1, n - 2)
    length = _snap(max(north, n - 1 - north) * rng.uniform(3., 4.2))
    half = _snap(rng.uniform(.8, 1.1))                     # a 1.6-2.2 m hallway
    rects = [("hallway", (0., -half, length, half))]
    for count, sign in ((north, 1), (n - 1 - north, -1)):
        depth = _snap(rng.uniform(3., 4.5))
        for a, b in _split(rng, length, count):
            near, far = sign * half, sign * (half + depth)
            rects.append((None, (a, min(near, far), b, max(near, far))))
    return rects


def _split_layout(rng, n):
    """A rectangle split again and again across its longer side, until there are n rooms."""
    rows = 1 if n <= 2 else 2
    cols = -(-n // rows)
    w, h = _snap(rng.uniform(3.2, 4.2) * cols), _snap(rng.uniform(3.2, 4.2) * rows)
    if rng.random() < .5:
        w, h = h, w
    rects = [(0., 0., w, h)]
    while len(rects) < n:
        rng.shuffle(rects)
        for k, (x0, y0, x1, y1) in enumerate(rects):
            sizes = [(axis, size) for axis, size in (("x", x1 - x0), ("y", y1 - y0)) if size >= 2 * MIN_ROOM]
            if not sizes:
                continue
            axis, size = max(sizes, key=lambda s: s[1])
            if axis == "x":
                cut = _snap(x0 + rng.uniform(MIN_ROOM, size - MIN_ROOM))
                rects[k:k + 1] = [(x0, y0, cut, y1), (cut, y0, x1, y1)]
            else:
                cut = _snap(y0 + rng.uniform(MIN_ROOM, size - MIN_ROOM))
                rects[k:k + 1] = [(x0, y0, x1, cut), (x0, cut, x1, y1)]
            break
        else:
            return None                                    # nothing left to split: too small for n rooms
    return [(None, r) for r in rects]


def _l_hall_layout(rng, n):
    """Two hallways at right angles, an L, with rooms along both: south of the first, north of it up to the
    second, and east of the second. The south-east corner stays outside, so the house is L-shaped too."""
    k1 = rng.randint(1, n - 4)
    k2 = rng.randint(1, n - 3 - k1)
    k3 = n - 2 - k1 - k2
    half = _snap(rng.uniform(.8, 1.1))
    up = _snap(max(rng.uniform(3., 4.5), k3 * rng.uniform(3., 3.6)))         # the second hallway's length
    length = _snap(max(k1 * rng.uniform(3., 4.2), k2 * rng.uniform(3., 4.2) + 2 * half))
    south, east = _snap(rng.uniform(3., 4.5)), _snap(rng.uniform(3., 4.5))
    rects = [("hallway", (0., -half, length, half)), ("hallway", (length - 2 * half, half, length, half + up))]
    rects += [(None, (a, -half - south, b, -half)) for a, b in _split(rng, length, k1)]
    rects += [(None, (a, half, b, half + up)) for a, b in _split(rng, length - 2 * half, k2)]
    rects += [(None, (length, half + a, length + east, half + b)) for a, b in _split(rng, up, k3)]
    return rects


def _carve(rng, rects, count):
    """Take `count` rooms off corners of the layout, for an L, T or Z-shaped house."""
    rects = list(rects)
    for _ in range(count):
        xs = [v for _, r in rects for v in r[0::2]]
        ys = [v for _, r in rects for v in r[1::2]]
        corners = [(x, y) for x in (min(xs), max(xs)) for y in (min(ys), max(ys))]
        at_corner = [i for i, (_, (x0, y0, x1, y1)) in enumerate(rects)
                     if any(abs(cx - x) < 1e-6 and abs(cy - y) < 1e-6 for cx, cy in corners
                            for x in (x0, x1) for y in (y0, y1))]
        rects.pop(rng.choice(at_corner))
    return rects


def room_count(house):
    """Rooms as random_house counts them: two hallways at right angles count as one, so a house of n rooms
    can list n + 1."""
    halls = sum(room["type"] == "hallway" for room in house["rooms"])
    return len(house["rooms"]) - max(halls - 1, 0)


def rectangular(house):
    """True if the rooms fill their bounding rectangle: a plain rectangular house."""
    xs = [v for room in house["rooms"] for v in room["rect"][0::2]]
    ys = [v for room in house["rooms"] for v in room["rect"][1::2]]
    area = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in (room["rect"] for room in house["rooms"]))
    return abs(area - (max(xs) - min(xs)) * (max(ys) - min(ys))) < 1e-3


def _name_rooms(rng, rects):
    """Room types: the biggest room is the living room and the smallest a bathroom (from four rooms up)."""
    others = sorted((r for t, r in rects if t is None), key=lambda r: -(r[2] - r[0]) * (r[3] - r[1]))
    types = ROOM_ORDER[:len(others)]
    middle = [t for t in types[1:] if t != "bathroom"]
    rng.shuffle(middle)
    types = types[:1] + middle + (["bathroom"] if "bathroom" in types else [])
    rooms, bedrooms, halls = [], 0, 0
    for kind, rect in [(t, r) for t, r in rects if t == "hallway"] + list(zip(types, others)):
        if kind == "bedroom":
            bedrooms += 1
        halls += kind == "hallway"
        name = f"bedroom_{bedrooms}" if kind == "bedroom" else "hallway_2" if kind == "hallway" and halls > 1 else kind
        rooms.append({"name": name, "type": kind, "rect": [round(v, 2) for v in rect]})
    return rooms


def _shared_wall(a, b):
    """Where rectangles a and b touch: (axis, line, lo, hi) of the wall they share, or None."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    if abs(ax1 - bx0) < 1e-6 or abs(bx1 - ax0) < 1e-6:
        return "y", ax1 if abs(ax1 - bx0) < 1e-6 else ax0, max(ay0, by0), min(ay1, by1)
    if abs(ay1 - by0) < 1e-6 or abs(by1 - ay0) < 1e-6:
        return "x", ay1 if abs(ay1 - by0) < 1e-6 else ay0, max(ax0, bx0), min(ax1, bx1)
    return None


def _doorway(rng, axis, line, lo, hi):
    """A door position and width on the wall [lo, hi] along `axis`, or None if the wall is too short."""
    room = hi - lo - 2 * CORNER + 1e-6            # (1e-6: 1.7 m must not come out as 1.6999...)
    if room < DOOR_MIN_WIDTH:
        return None
    width = min(_snap(rng.uniform(DOOR_MIN_WIDTH, 1.2)), math.floor(room * 10) / 10)
    along = _snap(rng.uniform(lo + CORNER + width / 2, hi - CORNER - width / 2))
    return ((along, line) if axis == "x" else (line, along)), width


def _connect(rng, rooms, openings=()):
    """Doors: a random spanning tree of open doors from the first room (every room gets a route), a few
    extra doors open or closed, and a closed front door on an outside wall. None if a room can't connect.
    openings: pairs of room indices with no wall between them at all (open plan, or hallways meeting): an
    open "door" the length of the wall they share, marked "opening"."""
    walls = {}
    for i, a in enumerate(rooms):
        for j in range(i + 1, len(rooms)):
            shared = _shared_wall(a["rect"], rooms[j]["rect"])
            if shared and (shared[3] - shared[2] + 1e-6 >= DOOR_MIN_WIDTH + 2 * CORNER or (i, j) in openings):
                walls[i, j] = walls[j, i] = shared
    tree, seen, queue = set(openings), {0}, [0]
    while queue:
        i = queue.pop(0)
        nearby = [j for j in range(len(rooms)) if (i, j) in walls]
        rng.shuffle(nearby)
        nearby.sort(key=lambda j: (min(i, j), max(i, j)) not in openings)       # through an opening if there is one
        for j in nearby:
            if j not in seen:
                seen.add(j)
                tree.add((min(i, j), max(i, j)))
                queue.append(j)
    if len(seen) < len(rooms):
        return None
    doors = []
    for (i, j) in sorted({(min(p), max(p)) for p in walls}):
        if (i, j) in openings:
            axis, line, lo, hi = walls[i, j]
            middle = round((lo + hi) / 2, 2)
            door = _door(f"opening_{rooms[i]['name']}_{rooms[j]['name']}", (rooms[i]["name"], rooms[j]["name"]),
                         (middle, line) if axis == "x" else (line, middle), axis, round(hi - lo, 2), True)
            door["opening"] = True
            doors.append(door)
            continue
        if (i, j) not in tree and rng.random() >= EXTRA_DOOR:
            continue
        axis, line, lo, hi = walls[i, j]
        pos, width = _doorway(rng, axis, line, lo, hi)
        doors.append(_door(f"door_{rooms[i]['name']}_{rooms[j]['name']}", (rooms[i]["name"], rooms[j]["name"]),
                           pos, axis, width, (i, j) in tree or rng.random() < .5))
    xs = [v for r in rooms for v in r["rect"][0::2]]
    ys = [v for r in rooms for v in r["rect"][1::2]]

    def on_edge(v, values):
        return abs(v - min(values)) < 1e-6 or abs(v - max(values)) < 1e-6
    for room in rooms:                    # the front door: the start room's outside wall if it has room for one
        x0, y0, x1, y1 = room["rect"]
        outside = [("x", y, x0, x1) for y in (y0, y1) if on_edge(y, ys)] + \
                  [("y", x, y0, y1) for x in (x0, x1) if on_edge(x, xs)]
        rng.shuffle(outside)
        spots = [(axis, _doorway(rng, axis, line, lo, hi)) for axis, line, lo, hi in outside]
        spots = [(axis, spot) for axis, spot in spots if spot]
        if spots:
            axis, (pos, width) = spots[0]
            doors.append(_door("front_door", (room["name"],), pos, axis, width, False))
            break
    return doors


def furnished(house):
    """Every room has its main piece: a bed in each bedroom, a couch in the living room, a fridge in the
    kitchen, a toilet in the bathroom, a desk in the office (the first entry of its FURNISHING list)."""
    return all(any(o["room"] == room["name"] and o["kind"] == FURNISHING[room["type"]][0][0]
                   for o in house["objects"])
               for room in house["rooms"] if FURNISHING[room["type"]])


def random_house(rng, n_rooms=None):
    """A random, furnished house of 2-7 rooms (a hallway, or two at right angles, counts as one), drawn from `rng` (random.Random):
    the same seed always gives the same house, so a failing case can be replayed. Go2 starts in the hallway
    or the living room, and every room is reachable from there."""
    n = n_rooms or rng.randint(2, 7)
    for _ in range(100):
        shape = rng.random()
        if n >= 5 and shape < L_HALL:
            rects = _l_hall_layout(rng, n + 1)            # its two hallways count as one, like a single hallway
        elif n >= 4 and shape < L_HALL + .3:
            rects = _corridor_layout(rng, n)
        else:
            carve = rng.randint(1, 2 if n >= 3 else 1) if rng.random() < CARVE else 0
            rects = _split_layout(rng, n + carve)
            if rects is not None and carve:
                rects = _carve(rng, rects, carve)
        if rects is None:
            continue
        rooms = _name_rooms(rng, rects)
        rooms.sort(key=lambda r: r["type"] not in ("hallway", "living_room"))     # the start room first
        openings = set()
        for i, a in enumerate(rooms):
            for j in range(i + 1, len(rooms)):
                pair = {a["type"], rooms[j]["type"]}
                shared = _shared_wall(a["rect"], rooms[j]["rect"])
                if not shared or shared[3] - shared[2] < 1.:
                    continue
                if pair == {"hallway"} or (pair == {"living_room", "kitchen"} and shared[3] - shared[2] >= OPEN_PLAN_MIN
                                           and rng.random() < OPEN_PLAN):
                    openings.add((i, j))
        doors = _connect(rng, rooms, openings)
        if doors is None:
            continue
        xs = [v for r in rooms for v in r["rect"][0::2]]
        ys = [v for r in rooms for v in r["rect"][1::2]]
        cx, cy = _snap((min(xs) + max(xs)) / 2), _snap((min(ys) + max(ys)) / 2)   # centre the house on 0, 0
        for room in rooms:
            x0, y0, x1, y1 = room["rect"]
            room["rect"] = [round(x0 - cx, 2), round(y0 - cy, 2), round(x1 - cx, 2), round(y1 - cy, 2)]
        for door in doors:
            door["pos"] = [round(door["pos"][0] - cx, 2), round(door["pos"][1] - cy, 2)]
        x0, y0, x1, y1 = rooms[0]["rect"]
        if rooms[0]["type"] == "hallway":                  # along the hallway, facing down it
            along_x = x1 - x0 >= y1 - y0
            start = [_snap(rng.uniform(x0 + 1, x1 - 1)), _snap((y0 + y1) / 2)] if along_x else \
                    [_snap((x0 + x1) / 2), _snap(rng.uniform(y0 + 1, y1 - 1))]
            yaw = rng.choice([0., math.pi]) if along_x else rng.choice([math.pi / 2, -math.pi / 2])
        else:
            start, yaw = [_snap((x0 + x1) / 2), _snap((y0 + y1) / 2)], _snap(rng.uniform(-math.pi, math.pi))
        house = {"scene": "house", "rooms": rooms, "doors": doors, "objects": [], "start": start + [round(yaw, 2)]}
        furnish(rng, house, on_furniture=True, colours=True)
        teddies = sum(o["kind"] == "teddy" for o in house["objects"])
        if furnished(house) and teddies == 1 and reachability(house)[0]:
            return house
    raise RuntimeError(f"no furnished {n}-room house with every room reachable in 100 tries")


# ---- reachability ----------------------------------------------------------------------------------------
ROBOT_RADIUS = .3                          # Go2's half width (0.2 m) plus a margin, for passing between things
GRID = .1                                  # metres per cell


def reachability(house):
    """Check that Go2 can get from the start into every room and through every open doorway.

    Floor plan on squared paper: walls, closed doors and furniture are marked on a 10 cm grid, grown by
    ROBOT_RADIUS so a free cell is one the robot's centre can stand on, then a flood fill from the start
    finds every cell Go2 can walk to. Returns (ok, problems), problems being plain sentences."""
    from collections import deque

    import numpy as np

    problems = []
    names = {room["name"] for room in house["rooms"]}
    for door in house["doors"]:
        if door["open"] and door["width"] < DOOR_MIN_WIDTH:
            problems.append(f"{door['name']} is {door['width']:.2f} m wide; Go2 needs {DOOR_MIN_WIDTH} m")
        if not set(door["rooms"]) <= names:
            problems.append(f"{door['name']} joins a room that isn't in the house: {door['rooms']}")
    xs = [v for room in house["rooms"] for v in room["rect"][0::2]]
    ys = [v for room in house["rooms"] for v in room["rect"][1::2]]
    x0, y0 = min(xs), min(ys)
    nx, ny = int(round((max(xs) - x0) / GRID)) + 1, int(round((max(ys) - y0) / GRID)) + 1

    def cell(x, y):
        return int(round((x - x0) / GRID)), int(round((y - y0) / GRID))

    blocked = np.zeros((nx, ny), bool)

    def mark(bx0, by0, bx1, by1):
        i0, j0 = cell(bx0, by0)
        i1, j1 = cell(bx1, by1)
        blocked[max(i0, 0):max(i1 + 1, 0), max(j0, 0):max(j1 + 1, 0)] = True

    for axis, line, a, b, *_ in wall_pieces(house):
        if axis == "x":
            mark(a, line - WALL_THICK, b, line + WALL_THICK)
        else:
            mark(line - WALL_THICK, a, line + WALL_THICK, b)
    for door in house["doors"]:
        if door["open"]:
            continue
        (dx, dy), half = door["pos"], door["width"] / 2
        if door["axis"] == "x":
            mark(dx - half, dy - WALL_THICK, dx + half, dy + WALL_THICK)
        else:
            mark(dx - WALL_THICK, dy - half, dx + WALL_THICK, dy + half)
    for item in house["objects"]:
        mark(*footprint(item))
    r = int(round(ROBOT_RADIUS / GRID))        # grow every obstacle by the robot's radius (a disc)
    padded = np.pad(blocked, r)
    grown = np.zeros_like(blocked)
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            if di * di + dj * dj <= r * r:
                grown |= padded[r + di:r + di + nx, r + dj:r + dj + ny]

    start = cell(*house["start"][:2])
    seen = np.zeros_like(grown)
    if grown[start]:
        problems.append("the start is too close to a wall or an object")
    else:
        seen[start] = True
        queue = deque([start])
        while queue:
            i, j = queue.popleft()
            for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if 0 <= a < nx and 0 <= b < ny and not grown[a, b] and not seen[a, b]:
                    seen[a, b] = True
                    queue.append((a, b))
    for room in house["rooms"]:
        rx0, ry0, rx1, ry1 = room["rect"]
        i0, j0 = cell(rx0, ry0)
        i1, j1 = cell(rx1, ry1)
        if not seen[i0:i1 + 1, j0:j1 + 1].any():
            problems.append(f"Go2 can't get into {room['name']}")
    for door in house["doors"]:
        if door["open"] and not seen[cell(*door["pos"])]:
            problems.append(f"Go2 can't get through {door['name']}")
    return not problems, problems


# ---- building a house into MuJoCo ----------------------------------------------------------------------
# Matte floor per room type, so camera frames (and YOLO) see a real-looking floor. Wood boards, textured like
# the single room's: type -> (colour 1, colour 2, joint colour, texture repeats: 2 gives the room's 0.5 m checks).
# Rooms differ in tone, though Go2 shouldn't rely on it.
FLOOR_LOOK = {"bedroom": ((.62, .5, .38), (.58, .46, .35), (.5, .4, .3), 2.),
              "living_room": ((.55, .42, .3), (.51, .39, .28), (.44, .33, .24), 2.),
              "hallway": ((.66, .58, .46), (.62, .55, .43), (.54, .47, .37), 2.),
              "office": ((.5, .44, .38), (.47, .41, .35), (.4, .35, .3), 2.)}
# Tiles in kitchens and bathrooms (#21): square tiles with dark grout lines, a little uneven in shade, not so
# bright that the camera washes them out. type -> (tile colour, grout colour, tile size in metres)
TILE_LOOK = {"kitchen": ((.66, .64, .6), (.33, .32, .3), .3),
             "bathroom": ((.6, .68, .72), (.32, .36, .38), .2)}
TILE_PIXELS, GROUT_SHARE = 64, .03         # texture pixels per tile; grout width as a share of a tile (each side)


def tile_pixels(tile, grout, tiles=4, seed=0):
    """A square texture of tiles x tiles tiles, as an (H, W, 3) uint8 array: each tile a slightly different
    shade, with grout lines between."""
    import numpy as np
    rng = np.random.default_rng(seed)
    size = tiles * TILE_PIXELS
    shade = 1 + rng.uniform(-.04, .04, (tiles, tiles))
    image = np.empty((size, size, 3))
    image[:] = np.asarray(tile) * np.kron(shade, np.ones((TILE_PIXELS, TILE_PIXELS)))[..., None]
    edge = max(1, round(GROUT_SHARE * TILE_PIXELS))
    inside = np.arange(size) % TILE_PIXELS
    line = (inside < edge) | (inside >= TILE_PIXELS - edge)
    image[line, :] = grout
    image[:, line] = grout
    return np.clip(image * 255, 0, 255).astype(np.uint8)


WALL_RGBA, DOOR_RGBA = (.86, .84, .8, 1), (.5, .34, .2, 1)


def wall_pieces(house):
    """The wall boxes as (axis, line, a, b, gap_before, gap_after): axis "x" is a wall along x at y = line
    from x = a to b, "y" a wall along y at x = line. Shared walls appear once; door gaps are cut out, and
    the flags say whether a door gap is next to either end."""
    lines = {}
    for room in house["rooms"]:
        x0, y0, x1, y1 = room["rect"]
        for key, a, b in ((("x", y0), x0, x1), (("x", y1), x0, x1), (("y", x0), y0, y1), (("y", x1), y0, y1)):
            lines.setdefault((key[0], round(key[1], 2)), []).append((a, b))
    gaps = {}
    for door in house["doors"]:
        along, across = (door["pos"][0], door["pos"][1]) if door["axis"] == "x" else (door["pos"][1], door["pos"][0])
        gaps.setdefault((door["axis"], round(across, 2)), []).append((along - door["width"] / 2, along + door["width"] / 2))
    pieces = []
    for (axis, line), spans in sorted(lines.items()):
        merged = []
        for a, b in sorted(spans):                      # union of the room edges on this line
            if merged and a <= merged[-1][1] + 1e-6:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        for a, b in merged:
            after_gap = False
            for g0, g1 in sorted(g for g in gaps.get((axis, line), []) if g[1] > a and g[0] < b):
                if g0 - a > 1e-3:
                    pieces.append((axis, line, a, g0, after_gap, True))
                a, after_gap = g1, True
            if b - a > 1e-3:
                pieces.append((axis, line, a, b, after_gap, False))
    return pieces


def _geoms(kind, rgba, W):
    """Plain shapes for an object, as world.ROOM_OBJECTS lists them: (type, size, local pos, rgba).
    Also stands in for a scanned object whose files aren't downloaded."""
    dark, white, wood = (.12, .12, .14, 1), (.95, .95, .95, 1), (.6, .45, .3, 1)
    shapes = {
        "bed": [("box", (.7, 1.0, .12), (0, 0, .12), wood), ("box", (.66, .95, .1), (0, 0, .33), rgba),
                ("box", (.5, .14, .06), (0, .75, .48), white), ("box", (.7, .04, .35), (0, .98, .35), wood)],
        "refrigerator": [("box", (.35, .33, .9), (0, 0, .9), rgba), ("box", (.02, .01, .3), (.25, -.34, 1.1), dark)],
        "oven": [("box", (.32, .3, .45), (0, 0, .45), rgba), ("box", (.22, .01, .14), (0, -.31, .4), dark)],
        "sink": [("box", (.5, .3, .43), (0, 0, .43), rgba), ("box", (.25, .17, .01), (0, -.03, .865), (.6, .62, .65, 1))],
        "tv": [("box", (.55, .2, .25), (0, 0, .25), wood), ("box", (.6, .03, .34), (0, .05, .85), rgba)],
        "couch": [("box", (.9, .4, .2), (0, 0, .2), rgba), ("box", (.9, .1, .25), (0, .3, .55), rgba)],
        "dining_table": W._table(rgba),
        "desk": [("box", (.6, .35, .02), (0, 0, .74), rgba), ("box", (.02, .33, .36), (-.56, 0, .36), rgba),
                 ("box", (.02, .33, .36), (.56, 0, .36), rgba)],
        "toilet": [("cylinder", (.2, .2), (0, -.08, .2), rgba), ("box", (.2, .1, .25), (0, .22, .45), rgba)],
        "chair": W._chair(rgba),
        "plant": [("cylinder", (.15, .15), (0, 0, .15), (.75, .4, .25, 1)), ("sphere", (.28,), (0, 0, .5), rgba)],
        "teddy": [("sphere", (.13,), (0, 0, .13), rgba), ("sphere", (.09,), (0, 0, .33), rgba),
                  ("sphere", (.035,), (.06, 0, .41), rgba), ("sphere", (.035,), (-.06, 0, .41), rgba)],
        "backpack": [("box", (.15, .1, .2), (0, 0, .2), rgba)],
        "cup": [("cylinder", (.045, .05), (0, 0, .05), rgba)],
        "bowl": [("cylinder", (.1, .03), (0, 0, .03), rgba)],
        "laptop": [("box", (.17, .12, .012), (0, 0, .012), rgba),                    # open: base and screen
                   ("box", (.17, .008, .11), (0, .115, .13), (.1, .1, .12, 1))],
        "keyboard": [("box", (.21, .07, .01), (0, 0, .01), rgba)],
        "book": [("box", (.11, .08, .02), (0, 0, .02), rgba)],
        "clock": [("cylinder", (.07, .03), (0, 0, .03), rgba)],
        "ball": [("sphere", (.12,), (0, 0, .12), rgba)],
        "toy_dog": [("box", (.06, .14, .06), (0, 0, .12), rgba), ("sphere", (.06,), (0, .16, .2), rgba)],
        "toy_elephant": [("sphere", (.13,), (0, 0, .15), rgba), ("sphere", (.08,), (0, .15, .22), rgba)],
    }
    return shapes[kind]


def _add_model(spec, body, item, added):
    """Draw item's realistic model on `body`, scaled to its kind's footprint. Returns how many of the kind's
    plain shapes stay visible, or None when the model isn't downloaded (then plain shapes are used)."""
    import mujoco
    name = item.get("model")
    entry = next((e for e in MODELS.get(item["kind"], []) if e[0] == name), None)
    try:
        info = json.loads((FURNITURE_DIR / name / "model.json").read_text())
    except (OSError, TypeError, ValueError):
        return None
    if entry is None:
        return None
    _, lift, shown, turn = entry
    hx, hy = KINDS[item["kind"]][1]
    sx, sy = info["size"][:2] if turn % 2 == 0 else info["size"][1::-1]    # a quarter turn swaps its sides
    scale = min(2 * hx / sx, 2 * hy / sy)
    angle = turn * math.pi / 2
    for i, part in enumerate(info["parts"]):
        asset = f"{name}_{i}"
        if asset not in added:                        # a model used twice shares its mesh, texture and material
            added.add(asset)
            mesh = spec.add_mesh()
            mesh.name, mesh.file, mesh.scale = f"{asset}_mesh", str(FURNITURE_DIR / name / part["obj"]), (scale,) * 3
            material = spec.add_material()
            material.name, material.specular, material.shininess = f"{asset}_mat", .2, .2
            if "texture" in part:
                texture = spec.add_texture()
                texture.name, texture.type = f"{asset}_tex", mujoco.mjtTexture.mjTEXTURE_2D
                texture.file = str(FURNITURE_DIR / name / part["texture"])
                material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texture.name
            else:
                material.rgba = part["rgba"]
        geom = body.add_geom()
        geom.type, geom.meshname, geom.material = mujoco.mjtGeom.mjGEOM_MESH, f"{asset}_mesh", f"{asset}_mat"
        geom.pos, geom.group = (0, 0, lift), 0
        geom.quat = (math.cos(angle / 2), 0, 0, math.sin(angle / 2))
        geom.contype = geom.conaffinity = 0           # only looks: the plain shapes are the body
    return shown


def rest_on_furniture(model, house):
    """Set each thing that is "on" furniture down on what is drawn under it, once the house is compiled: the
    plain top, or the realistic model's (a bed model's mattress isn't at the plain shapes' height). A ray
    straight down through the thing's centre, against the furniture's visible shapes only."""
    import mujoco
    import numpy as np
    items = [o for o in house["objects"] if o.get("on")]
    if not items:
        return
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    down = np.array([0., 0., -1.])
    for item in items:
        furniture = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, item["on"])
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, item["name"])
        start = np.array([*item["pos"], CEILING])
        hits = []
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != furniture:
                continue
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                d = mujoco.mj_rayMesh(model, data, g, start, down)
            elif model.geom_rgba[g][3] > FADED:
                d = mujoco.mju_rayGeom(data.geom_xpos[g], data.geom_xmat[g], model.geom_size[g], start, down,
                                       model.geom_type[g])
            else:
                continue                                  # a faded shape under a model: not what is seen
            if d >= 0:
                hits.append(d)
        kind = next(o["kind"] for o in house["objects"] if o["name"] == item["on"])
        top = CEILING - min(hits) if hits else SURFACES[kind][0]
        model.body_pos[body][2] += top


def build_house(spec, house, objects):
    """Add the walls, doors, floors and ceiling of `house` to a MuJoCo spec; append (body name, WorldObject)
    pairs to `objects` like world.build does for the room."""
    import mujoco
    import world as W

    spec.visual.headlight.ambient = (.45, .45, .45)     # same indoor light as the single room
    spec.visual.headlight.diffuse = (.5, .5, .5)
    materials = {}
    for kind, (tile, grout, size) in TILE_LOOK.items():
        pixels = tile_pixels(tile, grout)
        texture = spec.add_texture()
        texture.name, texture.type = f"floor_{kind}", mujoco.mjtTexture.mjTEXTURE_2D
        texture.width, texture.height, texture.nchannel = pixels.shape[1], pixels.shape[0], 3
        texture.data = pixels.tobytes()
        material = spec.add_material()
        material.name, material.specular, material.shininess, material.reflectance = f"floor_{kind}", .05, .05, 0.
        repeat = 2 / (4 * size)                # texuniform: repeats per 2 m; the texture is 4 tiles across
        material.texuniform, material.texrepeat = True, (repeat, repeat)
        material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texture.name
        materials[kind] = material.name
    for kind, (rgb1, rgb2, joints, repeat) in FLOOR_LOOK.items():
        texture = spec.add_texture()
        texture.name, texture.type, texture.builtin = f"floor_{kind}", mujoco.mjtTexture.mjTEXTURE_2D, mujoco.mjtBuiltin.mjBUILTIN_CHECKER
        texture.rgb1, texture.rgb2, texture.width, texture.height = rgb1, rgb2, 512, 512
        texture.mark, texture.markrgb = mujoco.mjtMark.mjMARK_EDGE, joints
        material = spec.add_material()
        material.name, material.specular, material.shininess, material.reflectance = f"floor_{kind}", .05, .05, 0.
        material.texuniform, material.texrepeat = True, (repeat, repeat)
        material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texture.name
        materials[kind] = material.name
    xs = [v for room in house["rooms"] for v in room["rect"][0::2]]
    ys = [v for room in house["rooms"] for v in room["rect"][1::2]]
    for room in house["rooms"]:          # floors: seen by the camera and viewer, never touched by the feet
        x0, y0, x1, y1 = room["rect"]
        body = spec.worldbody.add_body()
        body.name, body.pos = f"floor_{room['name']}", ((x0 + x1) / 2, (y0 + y1) / 2, .001)
        geom = body.add_geom()
        geom.type, geom.size = mujoco.mjtGeom.mjGEOM_PLANE, ((x1 - x0) / 2, (y1 - y0) / 2, .1)
        geom.material, geom.contype, geom.conaffinity = materials[room["type"]], 0, 0
    ceiling = spec.worldbody.add_body()
    ceiling.name, ceiling.pos = "ceiling", ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, CEILING)
    geom = ceiling.add_geom()
    geom.type, geom.rgba = mujoco.mjtGeom.mjGEOM_BOX, (.93, .93, .9, 1)
    geom.size = ((max(xs) - min(xs)) / 2 + .1, (max(ys) - min(ys)) / 2 + .1, .02)
    geom.group = 5                       # only the robot's camera renders it; the viewer and LiDAR ignore it
    geom.contype = geom.conaffinity = 0
    for i, (axis, line, a, b, gap_before, gap_after) in enumerate(wall_pieces(house)):
        # Walls along x overlap the corners by a thickness so rooms close up; never into a door gap.
        lo = a - (0 if gap_before or axis == "y" else WALL_THICK)
        hi = b + (0 if gap_after or axis == "y" else WALL_THICK)
        body = spec.worldbody.add_body()
        body.name = f"wall_{i}"
        if axis == "x":
            body.pos, size = ((lo + hi) / 2, line, WALL_HEIGHT / 2), ((hi - lo) / 2, WALL_THICK, WALL_HEIGHT / 2)
        else:
            body.pos, size = (line, (lo + hi) / 2, WALL_HEIGHT / 2), (WALL_THICK, (hi - lo) / 2, WALL_HEIGHT / 2)
        W._box(body, size, (0, 0, 0), WALL_RGBA)
        objects.append((body.name, W.WorldObject(body.name, "wall", "white")))
    for door in house["doors"]:
        if door.get("opening"):
            continue                            # open plan: no wall at all, so nothing over it either
        (dx, dy), half_w = door["pos"], door["width"] / 2
        along_x = door["axis"] == "x"
        above = spec.worldbody.add_body()       # the wall over the doorway, up to the ceiling
        above.name, above.pos = f"{door['name']}_lintel", (dx, dy, (DOOR_HEIGHT + WALL_HEIGHT) / 2)
        lintel = (WALL_HEIGHT - DOOR_HEIGHT) / 2
        W._box(above, (half_w, WALL_THICK, lintel) if along_x else (WALL_THICK, half_w, lintel), (0, 0, 0), WALL_RGBA)
        objects.append((above.name, W.WorldObject(above.name, "wall", "white")))
        if door["open"]:
            continue
        body = spec.worldbody.add_body()  # a closed door fills its gap (Go2 can't open doors)
        body.name, body.pos = door["name"], (dx, dy, DOOR_HEIGHT / 2)
        W._box(body, (half_w, .03, DOOR_HEIGHT / 2) if along_x else (.03, half_w, DOOR_HEIGHT / 2), (0, 0, 0), DOOR_RGBA)
        objects.append((body.name, W.WorldObject(body.name, "door", "brown")))
    added = set()
    for item in house["objects"]:
        (x, y), yaw = item["pos"], item["yaw"]
        if item.get("gso") and W._add_scanned(spec, item["gso"], item["name"], x, y, yaw):
            colour = ""                                    # the scanned texture shows the real colours
        else:
            body = spec.worldbody.add_body()
            body.name, body.pos = item["name"], (x, y, 0)
            body.quat = (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
            shown = _add_model(spec, body, item, added)
            for i, (kind, size, pos, rgba) in enumerate(_geoms(item["kind"], tuple(item["rgba"]), W)):
                geom = body.add_geom()
                geom.type, geom.size, geom.pos, geom.rgba = W.GEOM_TYPES[kind], W._size3(size), pos, rgba
                geom.friction = (.8, .02, .01)
                if shown is not None and i >= shown:
                    geom.rgba = list(rgba[:3]) + [FADED]
            r, g, b = item["rgba"][:3]
            colour = "" if shown is not None else W.colour_word(*colorsys.rgb_to_hsv(r, g, b))
        objects.append((item["name"], W.WorldObject(item["name"], item["label"], colour)))
