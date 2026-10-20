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

ROOM_TYPES = ("bedroom", "kitchen", "living_room", "bathroom", "hallway", "office")
WALL_HEIGHT, WALL_THICK = 2.0, .05        # half thickness, as in the single room
DOOR_MIN_WIDTH = .9                       # Go2 fits through with the executor's clearance margin


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
    return {"scene": "house", "rooms": rooms, "doors": doors, "objects": [], "start": [-4.5, 0., 0.]}


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
