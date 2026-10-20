"""Multi-room houses for the simulation: rooms, doorways and furniture that fits each room.

A house is a plain, JSON-friendly description (like random_world's output), so it can be written to a
file for the simulator process and replayed later:

  {"scene": "house",
   "rooms":   [{"name": "bedroom_2", "type": "bedroom", "rect": [x0, y0, x1, y1]}, ...],
   "doors":   [{"name": "door_0", "rooms": ["hallway", "bedroom_2"], "pos": [x, y], "axis": "x" | "y",
                "width": 1.0, "open": true}, ...],
   "objects": [{"name": "teddy_bear", "label": "teddy bear", "kind": "teddy", "room": "bedroom_2",
                "pos": [x, y], "yaw": 0.8, "rgba": [r, g, b, 1], "gso": "teddy_bear"}, ...],
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
import math
import random

ROOM_TYPES = ("bedroom", "kitchen", "living_room", "bathroom", "hallway", "office")
WALL_HEIGHT, WALL_THICK = 2.0, .05        # half thickness, as in the single room
DOOR_MIN_WIDTH = .9                       # Go2 fits through with the executor's clearance margin
DOOR_CLEAR = 1.2                          # no furniture this close to a doorway's centre, either side: room
                                          # for Go2 (0.7 m long) to step through and turn around
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
UNLIKELY_SHARE = .1                        # portable things sometimes end up in an unlikely room
TEDDY_IN_BEDROOM = .85
COLOURS = {"bed": (.9, .9, .92, 1), "refrigerator": (.88, .88, .9, 1), "oven": (.92, .92, .9, 1),
           "sink": (.9, .9, .88, 1), "tv": (.15, .15, .17, 1), "couch": (.5, .5, .55, 1), "dining_table": (.65, .5, .35, 1),
           "desk": (.6, .45, .3, 1), "toilet": (.95, .95, .95, 1), "chair": (.45, .3, .2, 1), "plant": (.2, .6, .25, 1),
           "teddy": (.6, .42, .25, 1), "backpack": (.15, .3, .85, 1), "cup": (.95, .95, .92, 1), "bowl": (.15, .3, .85, 1),
           "laptop": (.5, .5, .55, 1), "keyboard": (.12, .12, .14, 1), "book": (.85, .1, .1, 1), "clock": (.85, .1, .1, 1),
           "toy_dog": (.55, .38, .22, 1), "toy_elephant": (.55, .55, .6, 1)}
BALL_COLOURS = [(.85, .1, .1, 1), (.15, .3, .85, 1), (.2, .6, .25, 1), (.95, .8, .1, 1)]


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


def _place(rng, house, room, kind, taken):
    """A free spot for `kind` in `room`, as (x, y, yaw), or None. Big furniture backs onto a wall."""
    x0, y0, x1, y1 = room["rect"]
    hx, hy = KINDS[kind][1]
    big = KINDS[kind][2]
    margin = WALL_THICK + .05
    for _ in range(60):
        if big:
            side = rng.choice("NSEW")
            yaw = {"N": 0., "S": math.pi, "E": -math.pi / 2, "W": math.pi / 2}[side]
            ex, ey = (hy, hx) if side in "EW" else (hx, hy)
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


def furnish(rng, house, teddy_room=None):
    """Fill house["objects"] with furniture that fits each room type, mostly in likely places, plus one
    teddy bear (in `teddy_room` when given). Each scanned object is used once; repeats are plain shapes."""
    objects, taken, used, counts = [], [], set(), {}

    def add(kind, room):
        spot = _place(rng, house, room, kind, taken)
        if spot is None:
            return False
        x, y, yaw = spot
        gso = KINDS[kind][3] if KINDS[kind][3] not in used else None
        used.add(gso)
        counts[kind] = counts.get(kind, 0) + 1
        rgba = rng.choice(BALL_COLOURS) if kind == "ball" else COLOURS[kind]
        item = {"name": gso or f"{kind}_{counts[kind]}", "label": KINDS[kind][0], "kind": kind, "room": room["name"],
                "pos": [x, y], "yaw": yaw, "rgba": list(rgba)}
        if gso:
            item["gso"] = gso
        objects.append(item)
        taken.append(footprint(item))
        return True

    rooms = [r for r in house["rooms"] if r["type"] != "hallway"]      # hallways stay clear
    wishes = [(kind, room) for room in rooms for kind, chance in FURNISHING[room["type"]] if rng.random() < chance]
    wishes.sort(key=lambda w: not KINDS[w[0]][2])       # big furniture first, while the walls are free
    for kind, room in wishes:
        if kind in PORTABLE and len(rooms) > 1 and rng.random() < UNLIKELY_SHARE:
            room = rng.choice([r for r in rooms if r is not room])
        add(kind, room)
    if teddy_room is not None:
        homes = [r for r in rooms if r["name"] == teddy_room]
    else:
        bedrooms = [r for r in rooms if r["type"] == "bedroom"]
        homes = bedrooms if bedrooms and rng.random() < TEDDY_IN_BEDROOM else list(rooms)
        rng.shuffle(homes)
    for room in homes + [r for r in rooms if r not in homes]:   # the next room if its first home is full
        if add("teddy", room):
            break
    house["objects"] = objects
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


# ---- building a house into MuJoCo ----------------------------------------------------------------------
# Matte floor colour per room type: the camera can tell rooms apart, though Go2 shouldn't rely on it.
FLOOR_RGBA = {"bedroom": (.6, .48, .36, 1), "living_room": (.55, .42, .3, 1), "hallway": (.66, .6, .5, 1),
              "office": (.5, .45, .4, 1), "kitchen": (.78, .78, .74, 1), "bathroom": (.72, .8, .82, 1)}
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
        "laptop": [("box", (.17, .12, .012), (0, 0, .012), rgba)],
        "keyboard": [("box", (.21, .07, .01), (0, 0, .01), rgba)],
        "book": [("box", (.11, .08, .02), (0, 0, .02), rgba)],
        "clock": [("cylinder", (.07, .03), (0, 0, .03), rgba)],
        "ball": [("sphere", (.12,), (0, 0, .12), rgba)],
        "toy_dog": [("box", (.06, .14, .06), (0, 0, .12), rgba), ("sphere", (.06,), (0, .16, .2), rgba)],
        "toy_elephant": [("sphere", (.13,), (0, 0, .15), rgba), ("sphere", (.08,), (0, .15, .22), rgba)],
    }
    return shapes[kind]


def build_house(spec, house, objects):
    """Add the walls, doors, floors and ceiling of `house` to a MuJoCo spec; append (body name, WorldObject)
    pairs to `objects` like world.build does for the room."""
    import mujoco
    import world as W

    spec.visual.headlight.ambient = (.45, .45, .45)     # same indoor light as the single room
    spec.visual.headlight.diffuse = (.5, .5, .5)
    materials = {}
    for kind, rgba in FLOOR_RGBA.items():
        material = spec.add_material()
        material.name, material.rgba, material.specular, material.shininess = f"floor_{kind}", rgba, .05, .05
        material.reflectance = 0.
        materials[kind] = material.name
    xs = [v for room in house["rooms"] for v in room["rect"][0::2]]
    ys = [v for room in house["rooms"] for v in room["rect"][1::2]]
    for room in house["rooms"]:          # floors: seen by the camera and viewer, never touched by the feet
        x0, y0, x1, y1 = room["rect"]
        body = spec.worldbody.add_body()
        body.name, body.pos = f"floor_{room['name']}", ((x0 + x1) / 2, (y0 + y1) / 2, .001)
        geom = body.add_geom()
        geom.type, geom.size = mujoco.mjtGeom.mjGEOM_BOX, ((x1 - x0) / 2, (y1 - y0) / 2, .001)
        geom.material, geom.contype, geom.conaffinity = materials[room["type"]], 0, 0
    ceiling = spec.worldbody.add_body()
    ceiling.name, ceiling.pos = "ceiling", ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, 2.4)
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
        if door["open"]:
            continue
        body = spec.worldbody.add_body()  # a closed door fills its gap (Go2 can't open doors)
        body.name, body.pos = door["name"], (door["pos"][0], door["pos"][1], WALL_HEIGHT / 2)
        half = (door["width"] / 2, .03, WALL_HEIGHT / 2) if door["axis"] == "x" else (.03, door["width"] / 2, WALL_HEIGHT / 2)
        W._box(body, half, (0, 0, 0), DOOR_RGBA)
        objects.append((body.name, W.WorldObject(body.name, "door", "brown")))
    for item in house["objects"]:
        (x, y), yaw = item["pos"], item["yaw"]
        if item.get("gso") and W._add_scanned(spec, item["gso"], item["name"], x, y, yaw):
            colour = ""                                    # the scanned texture shows the real colours
        else:
            body = spec.worldbody.add_body()
            body.name, body.pos = item["name"], (x, y, 0)
            body.quat = (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
            for kind, size, pos, rgba in _geoms(item["kind"], tuple(item["rgba"]), W):
                geom = body.add_geom()
                geom.type, geom.size, geom.pos, geom.rgba = W.GEOM_TYPES[kind], W._size3(size), pos, rgba
                geom.friction = (.8, .02, .01)
            r, g, b = item["rgba"][:3]
            colour = W.colour_word(*colorsys.rgb_to_hsv(r, g, b))
        objects.append((item["name"], W.WorldObject(item["name"], item["label"], colour)))
