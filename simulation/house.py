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
