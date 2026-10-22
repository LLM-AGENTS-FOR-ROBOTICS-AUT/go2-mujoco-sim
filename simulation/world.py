"""Build the Go2 MuJoCo world: robot, sensor mounts and an optional indoor room.

Sensor mounts follow Unitree's go2_description URDF:
  front_camera_joint  base -> front_camera  xyz (0.32715, -0.00003, 0.04297), rpy 0
  radar_joint         base -> radar (L1)    xyz (0.28945, 0, -0.046825), rpy (0, 2.8782, 0)
The L1 dome orientation is configurable (GO2_SIM_LIDAR_PITCH, radians). Taken literally, the URDF
pitch 2.8782 tilts the dome 15° forward, which would leave the rear blind at body height; the real
robot is reported to see behind it, so the default points the dome straight down (all-around
coverage) until a capture from the real robot (scripts/capture_lidar.py) pins the true geometry.
Robot geoms live in groups 2 (visual) and 3 (collision); everything else in the
world uses groups 0/1 so ray casts and segmentation can exclude the robot.
"""
from __future__ import annotations

import colorsys
import os
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

DEPS = Path(__file__).resolve().parents[1] / "_deps"      # filled by macos/setup.sh or windows/setup.ps1
SCENE = DEPS / "unitree_mujoco/unitree_robots/go2/scene.xml"

CAMERA_POS = (0.32715, -0.00003, 0.04297)
CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_HFOV_DEG = 1280, 720, 120.0
LIDAR_POS = (0.28945, 0.0, -0.046825)
# Dome pitch: straight down by default (sees all around); Unitree's URDF value 2.8782 leaves the rear blind.
LIDAR_PITCH = float(os.environ.get("GO2_SIM_LIDAR_PITCH", np.pi))
START_POSE = (-2.5, 0.0, 0.0)  # x, y, yaw in the world frame


def camera_fovy(width=CAMERA_WIDTH, height=CAMERA_HEIGHT, hfov_deg=CAMERA_HFOV_DEG):
    return float(np.degrees(2 * np.arctan(np.tan(np.radians(hfov_deg) / 2) * height / width)))


@dataclass(frozen=True)
class WorldObject:
    name: str      # unique body name, e.g. "ball_red"
    label: str     # detector-style class label (COCO names where they exist)
    color: str     # human colour word used in descriptions


# Each entry: object, world xy, yaw, list of geoms (type, size, local pos, rgba)
RED, BLUE, BROWN, WOOD, GREEN, SKIN, GREY, CARDBOARD, TERRACOTTA, NAVY = (
    (.85, .1, .1, 1), (.15, .3, .85, 1), (.45, .3, .2, 1), (.65, .5, .35, 1), (.2, .6, .25, 1),
    (.85, .7, .55, 1), (.5, .5, .55, 1), (.75, .6, .4, 1), (.75, .4, .25, 1), (.15, .2, .4, 1))


def _chair(rgba):
    legs = [("box", (.02, .02, .22), (sx * .2, sy * .2, .22), rgba) for sx in (-1, 1) for sy in (-1, 1)]
    return legs + [("box", (.23, .23, .025), (0, 0, .46), rgba), ("box", (.025, .23, .25), (-.21, 0, .73), rgba)]


def _table(rgba):
    legs = [("box", (.03, .03, .36), (sx * .55, sy * .35, .36), rgba) for sx in (-1, 1) for sy in (-1, 1)]
    return legs + [("box", (.6, .4, .03), (0, 0, .75), rgba)]


def _person(shirt, skin):
    return [("capsule", (.07, .4), (0, .1, .45), NAVY), ("capsule", (.07, .4), (0, -.1, .45), NAVY),
            ("box", (.12, .2, .3), (0, 0, 1.15), shirt), ("sphere", (.11,), (0, 0, 1.6), skin)]


ROOM_OBJECTS = [
    (WorldObject("ball_red", "sports ball", "red"), (0.6, 1.3), 0, [("sphere", (.12,), (0, 0, .12), RED)]),
    (WorldObject("ball_blue", "sports ball", "blue"), (-0.8, -2.6), 0, [("sphere", (.12,), (0, 0, .12), BLUE)]),
    (WorldObject("box", "box", "brown"), (0.2, -1.1), .3, [("box", (.2, .2, .2), (0, 0, .2), CARDBOARD)]),
    (WorldObject("chair", "chair", "brown"), (1.8, -1.8), 2.4, _chair(BROWN)),
    (WorldObject("table", "dining table", "wooden"), (2.6, 1.7), 0, _table(WOOD)),
    (WorldObject("person", "person", "green"), (3.1, -0.4), 3.14, _person(GREEN, SKIN)),
    (WorldObject("plant", "potted plant", "green"),
     (-3.4, 3.4), 0, [("cylinder", (.15, .15), (0, 0, .15), TERRACOTTA), ("sphere", (.28,), (0, 0, .5), GREEN)]),
    (WorldObject("couch", "couch", "grey"),
     (-0.8, 3.45), 0, [("box", (.9, .4, .2), (0, 0, .2), GREY), ("box", (.9, .1, .25), (0, .3, .55), GREY)]),
]

GSO_DIR = DEPS / "gso"
# Google Scanned Objects on the floor (short name, COCO label YOLO uses, x, y, yaw). CC-BY 4.0, Google LLC.
GSO_PLACEMENT = [("teddy_bear", "teddy bear", .5, -2.8, .8), ("backpack", "backpack", 2.0, 3.25, 3.4),
                 ("dog_bowl", "bowl", -1.6, 1.2, 0.), ("mug", "cup", -.2, 2.3, 1.1), ("toy_dog", "dog", 1.0, .6, 2.6),
                 ("toy_elephant", "elephant", -3.3, -3.2, .4), ("toy_horse", "horse", 3.4, 3.3, -2.), ("alarm_clock", "clock", -2.0, -1.6, 1.9),
                 ("fire_truck", "truck", 1.3, -.35, 1.2), ("laptop", "laptop", -3.2, 1.6, .3), ("keyboard", "keyboard", -.5, -3.35, 0.),
                 ("book", "book", 2.6, -3.0, .7)]
# Scans that don't come the right way up: the turn (quaternion) that stands them up. The teddy bear was scanned
# lying on its back (0.37 m long, 0.12 m tall) and looked like a flat golden shape from Go2's camera; a quarter
# turn about x sits it up facing -y, so yaw turns it as before.
GSO_UPRIGHT = {"teddy_bear": (np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0)}


def _obj_zmin(path, quat=(1, 0, 0, 0)):
    """The lowest point of an OBJ mesh once turned by `quat`."""
    up = np.zeros(9)
    mujoco.mju_quat2Mat(up, np.asarray(quat, dtype=float))
    up = up[6:]                                        # the rotation's z row: height of a turned vertex
    zmin = None
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                z = float(np.dot(up, [float(v) for v in line.split()[1:4]]))
                zmin = z if zmin is None or z < zmin else zmin
    return zmin or 0.


def _add_scanned(spec, short, name, x, y, yaw):
    """Add one Google Scanned Object as body `name`; False if its files aren't downloaded."""
    folder = GSO_DIR / short
    if not (folder / "model.obj").exists():
        return False
    texture = spec.add_texture()
    texture.name, texture.file, texture.type = f"{short}_tex", str(folder / "texture.png"), mujoco.mjtTexture.mjTEXTURE_2D
    material = spec.add_material()
    material.name = f"{short}_mat"
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texture.name
    material.specular, material.shininess = .3, .3
    mesh = spec.add_mesh()
    mesh.name, mesh.file = f"{short}_mesh", str(folder / "model.obj")
    body = spec.worldbody.add_body()
    body.name = name
    upright = GSO_UPRIGHT.get(short, (1, 0, 0, 0))
    body.pos = (x, y, -_obj_zmin(folder / "model.obj", upright) + .002)
    body.quat = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
    geom = body.add_geom()
    geom.quat = upright
    geom.type, geom.meshname, geom.material = mujoco.mjtGeom.mjGEOM_MESH, mesh.name, material.name
    geom.group = 0   # world group: rendered, seen by the LiDAR, solid (convex hull) to the robot
    return True


def _add_scanned_objects(spec, objects):
    if not GSO_DIR.is_dir():
        return   # run simulation/fetch_gso.py to add them
    for short, label, x, y, yaw in GSO_PLACEMENT:
        if _add_scanned(spec, short, short, x, y, yaw):
            objects.append((short, WorldObject(short, label, "")))


GEOM_TYPES = {"sphere": mujoco.mjtGeom.mjGEOM_SPHERE, "box": mujoco.mjtGeom.mjGEOM_BOX,
              "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE, "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER}


def _size3(size):
    return list(size) + [0] * (3 - len(size))


def build(scene: str = "room", world=None):
    """Return (model, objects) where objects maps body id -> WorldObject.

    scene "procedural" builds the room and objects described by `world` (see random_world); scene "house"
    builds the multi-room house described by `world` (see house.py), or the demo house without one."""
    if scene not in ("room", "empty", "procedural", "house"):
        raise ValueError("scene must be 'room', 'empty', 'procedural' or 'house'")
    if scene == "procedural" and world is None:
        raise ValueError("procedural scenes need a world description")
    spec = mujoco.MjSpec.from_file(str(SCENE))
    for geom in list(spec.geoms):  # Unitree's default hurdles are not part of our worlds.
        if geom.parent.name == "world" and geom.name != "floor":
            spec.delete(geom)
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, CAMERA_WIDTH)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, CAMERA_HEIGHT)
    base = spec.body("base_link")
    camera = base.add_camera()
    camera.name = "front_camera"
    camera.pos = CAMERA_POS
    camera.fovy = camera_fovy()
    # MuJoCo cameras look along -z with +y up; this faces body +x with world-up.
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=float).reshape(-1))
    camera.quat = quat
    lidar = base.add_site()
    lidar.name = "lidar"
    lidar.pos = LIDAR_POS
    lidar.quat = [np.cos(LIDAR_PITCH / 2), 0, np.sin(LIDAR_PITCH / 2), 0]
    lidar.size = [.01, 0, 0]
    lidar.group = 4
    objects = []
    if scene == "room":
        # Indoor look for the camera (and YOLO): matte wood floor, 2 m walls, soft indoor light.
        floor_tex = spec.add_texture()
        floor_tex.name, floor_tex.type, floor_tex.builtin = "wood_floor", mujoco.mjtTexture.mjTEXTURE_2D, mujoco.mjtBuiltin.mjBUILTIN_CHECKER
        floor_tex.rgb1, floor_tex.rgb2, floor_tex.width, floor_tex.height = (.62, .5, .38), (.58, .46, .35), 512, 512
        floor_tex.mark, floor_tex.markrgb = mujoco.mjtMark.mjMARK_EDGE, (.5, .4, .3)
        floor_mat = spec.add_material()
        floor_mat.name, floor_mat.reflectance, floor_mat.texrepeat = "wood_floor", 0., (6, 6)
        floor_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = floor_tex.name
        for geom in spec.geoms:
            if geom.name == "floor":
                geom.material = floor_mat.name
        spec.visual.headlight.ambient = (.45, .45, .45)
        spec.visual.headlight.diffuse = (.5, .5, .5)
        ceiling = spec.worldbody.add_body()
        ceiling.name, ceiling.pos = "ceiling", (0, 0, 2.4)
        ceiling_geom = ceiling.add_geom()
        ceiling_geom.type, ceiling_geom.size, ceiling_geom.rgba = mujoco.mjtGeom.mjGEOM_BOX, (4.1, 4.1, .02), (.93, .93, .9, 1)
        ceiling_geom.group = 5           # only the robot's camera renders it; the viewer and LiDAR ignore it
        ceiling_geom.contype = ceiling_geom.conaffinity = 0
        half, height, thick = 4.0, 2.0, .05
        for name, pos, size in [("wall_north", (0, half, height / 2), (half + thick, thick, height / 2)),
                                ("wall_south", (0, -half, height / 2), (half + thick, thick, height / 2)),
                                ("wall_east", (half, 0, height / 2), (thick, half, height / 2)),
                                ("wall_west", (-half, 0, height / 2), (thick, half, height / 2))]:
            body = spec.worldbody.add_body()
            body.name = name
            body.pos = pos
            geom = body.add_geom()
            geom.type = mujoco.mjtGeom.mjGEOM_BOX
            geom.size = size
            geom.rgba = (.86, .84, .8, 1)
            objects.append((name, WorldObject(name, "wall", "white")))
        for item, (x, y), yaw, geoms in ROOM_OBJECTS:
            body = spec.worldbody.add_body()
            body.name = item.name
            body.pos = (x, y, 0)
            body.quat = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
            for kind, size, pos, rgba in geoms:
                geom = body.add_geom()
                geom.type = GEOM_TYPES[kind]
                geom.size = _size3(size)
                geom.pos = pos
                geom.rgba = rgba
                geom.friction = (.8, .02, .01)
            objects.append((item.name, item))
        _add_scanned_objects(spec, objects)
    if scene == "procedural":
        _add_procedural(spec, world, objects)
    if scene == "house":
        import house     # simulation/house.py, kept apart so the room scenes never change
        house.build_house(spec, world if world is not None else house.house_demo(), objects)
    model = spec.compile()
    by_body = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name): item for name, item in objects}
    return model, by_body


# ---- procedural worlds (training data): random rooms, shapes, sizes, colours and labels ----------------
VOCABULARY = [
    # COCO classes that make sense on a floor or at robot height
    "person", "bicycle", "dog", "cat", "backpack", "umbrella", "handbag", "suitcase", "sports ball", "skateboard",
    "bottle", "cup", "bowl", "chair", "couch", "potted plant", "bed", "dining table", "laptop", "keyboard", "book",
    "clock", "vase", "teddy bear", "toaster", "microwave", "refrigerator", "oven", "sink", "tv", "bench", "fire hydrant",
    "stop sign", "frisbee", "kite", "baseball bat", "tennis racket", "wine glass", "banana", "apple", "orange",
    "broccoli", "remote", "cell phone", "scissors", "hair drier", "toothbrush", "traffic cone", "parking meter",
    # everyday things
    "shoe", "boot", "sock", "slipper", "laundry basket", "water bottle", "cardboard box", "storage box", "crate",
    "trash can", "recycling bin", "bucket", "watering can", "lamp", "floor lamp", "cushion", "pillow", "blanket",
    "dog bed", "dog bowl", "dog toy", "chew toy", "tennis ball", "football", "basketball", "beach ball", "toy car",
    "toy truck", "toy robot", "building blocks", "stool", "footstool", "ottoman", "side table", "coffee table",
    "bookshelf", "cabinet", "drawer unit", "speaker", "guitar", "keyboard stand", "yoga mat", "dumbbell",
    "kettlebell", "mop", "broom", "vacuum cleaner", "fan", "heater", "plant pot", "flower pot", "watering can",
    "paint can", "toolbox", "step ladder", "suitcase", "shopping bag", "paper bag", "gift box", "pizza box",
    "delivery box", "rubber duck", "stuffed animal", "plush toy", "pet carrier", "litter box", "scratching post",
    "coat rack", "umbrella stand", "shoe rack", "radiator", "router", "printer", "monitor", "office chair",
    "gaming chair", "bean bag", "rocking chair", "high chair", "baby stroller", "scooter", "helmet", "cooler",
    "picnic basket", "lunch box", "thermos", "jar", "candle", "photo frame", "globe", "trophy", "piggy bank",
    "traffic barrel", "pylon", "safety cone", "sandbag", "tire", "wheelbarrow", "hose reel", "garden gnome",
]
VOCABULARY = list(dict.fromkeys(VOCABULARY))
SHAPES = ("box", "cylinder", "sphere", "capsule", "table", "chair", "stack")
DESCRIPTORS = ("small", "large", "round", "tall", "wooden", "plastic", "metal", "striped", "folded", "old", "empty")


def colour_word(hue, sat, val):
    """Roughly the colour name the perception code reports for an object of this HSV colour (0-1)."""
    if sat < .25 or val < .2:
        return "black" if val < .2 else "white" if val > .75 and sat < .1 else "grey"
    degrees = hue * 360
    if 15 <= degrees <= 45 and val < .65:
        return "brown"
    names = [(0, "red"), (25, "orange"), (55, "yellow"), (110, "green"), (180, "cyan"), (225, "blue"),
             (275, "purple"), (320, "pink"), (360, "red")]
    return min(names, key=lambda n: abs(degrees - n[0]))[1]


def random_world(rng, n_objects=None):
    """A random room (or open floor) with labelled primitive objects and a free robot start pose."""
    walls = rng.random() > .12
    half_x, half_y = rng.uniform(2.5, 6.), rng.uniform(2.5, 6.)
    if n_objects is None:
        n_objects = 0 if rng.random() < .06 else rng.randint(3, 14)
    taken = []
    def free_spot(clearance, wall_margin=.5):
        for _ in range(200):
            x, y = rng.uniform(-half_x + wall_margin, half_x - wall_margin), rng.uniform(-half_y + wall_margin, half_y - wall_margin)
            if all((x - tx) ** 2 + (y - ty) ** 2 > (clearance + tr) ** 2 for tx, ty, tr in taken):
                return x, y
        return None
    start = free_spot(.9, wall_margin=1.1)
    taken.append((start[0], start[1], .9))
    objects = []
    for i in range(n_objects):
        shape = rng.choice(SHAPES)
        scale = rng.choice([.12, .2, .3, .45, .6, .8, 1.0])
        if shape == "box":
            size = [scale * rng.uniform(.4, 1.), scale * rng.uniform(.4, 1.), scale * rng.uniform(.3, 1.)]
        elif shape == "cylinder":
            size = [scale * rng.uniform(.3, .6), scale * rng.uniform(.3, 1.)]
        elif shape in ("sphere",):
            size = [scale * rng.uniform(.3, .6)]
        elif shape == "capsule":
            size = [scale * rng.uniform(.15, .35), scale * rng.uniform(.3, .9)]
        else:
            size = [max(.25, scale)]
        footprint = max(size) * (1.5 if shape in ("table", "chair", "stack") else 1.)
        spot = free_spot(footprint + .25)
        if spot is None:
            continue
        taken.append((spot[0], spot[1], footprint))
        hue, sat, val = rng.random(), rng.choice([0., .15, .5, .8, .95]), rng.uniform(.25, .95)
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        label, extra = rng.choice(VOCABULARY), rng.random()
        if extra < .12:      # open-vocabulary detectors often name the colour: "blue water bottle"
            label = f"{colour_word(hue, sat, val)} {label}"
        elif extra < .22:
            label = f"{rng.choice(DESCRIPTORS)} {label}"
        objects.append({"name": f"obj_{i}", "label": label, "shape": shape, "size": size,
                        "rgba": [round(r, 3), round(g, 3), round(b, 3), 1], "pos": [round(spot[0], 2), round(spot[1], 2)],
                        "yaw": round(rng.uniform(-3.14, 3.14), 2)})
    return {"room": {"half_x": round(half_x, 2), "half_y": round(half_y, 2), "walls": walls},
            "objects": objects, "start": [round(start[0], 2), round(start[1], 2), round(rng.uniform(-3.14, 3.14), 2)]}


def _box(body, size, pos, rgba):
    g = body.add_geom()
    g.type, g.size, g.pos, g.rgba = mujoco.mjtGeom.mjGEOM_BOX, size, pos, rgba
    g.friction = (.8, .02, .01)


def _add_procedural(spec, world, objects):
    room = world["room"]
    if room.get("walls", True):
        hx, hy, height, thick = room["half_x"], room["half_y"], 2.0, .05
        for name, pos, size in [("wall_north", (0, hy, height / 2), (hx + thick, thick, height / 2)),
                                ("wall_south", (0, -hy, height / 2), (hx + thick, thick, height / 2)),
                                ("wall_east", (hx, 0, height / 2), (thick, hy, height / 2)),
                                ("wall_west", (-hx, 0, height / 2), (thick, hy, height / 2))]:
            body = spec.worldbody.add_body()
            body.name, body.pos = name, pos
            _box(body, size, (0, 0, 0), (.86, .84, .8, 1))
            objects.append((name, WorldObject(name, "wall", "white")))
    for i, part in enumerate(room.get("partitions", [])):    # interior walls (held-out test rooms)
        body = spec.worldbody.add_body()
        body.name = f"partition_{i}"
        body.pos = (part["x"], part["y"], 1.0)
        body.quat = (np.cos(part["yaw"] / 2), 0, 0, np.sin(part["yaw"] / 2))
        _box(body, (part["length"] / 2, .05, 1.0), (0, 0, 0), (.78, .8, .84, 1))
        objects.append((body.name, WorldObject(body.name, "wall", "white")))
    for item in world["objects"]:
        body = spec.worldbody.add_body()
        body.name = item["name"]
        body.pos = (item["pos"][0], item["pos"][1], 0)
        body.quat = (np.cos(item["yaw"] / 2), 0, 0, np.sin(item["yaw"] / 2))
        size, rgba, shape = item["size"], item["rgba"], item["shape"]
        if shape == "box":
            _box(body, size, (0, 0, size[2]), rgba)
        elif shape in ("cylinder", "capsule"):
            g = body.add_geom()
            g.type = mujoco.mjtGeom.mjGEOM_CYLINDER if shape == "cylinder" else mujoco.mjtGeom.mjGEOM_CAPSULE
            g.size, g.rgba = [size[0], size[1], 0], rgba
            g.pos = (0, 0, size[1] + (size[0] if shape == "capsule" else 0))
        elif shape == "sphere":
            g = body.add_geom()
            g.type, g.size, g.pos, g.rgba = mujoco.mjtGeom.mjGEOM_SPHERE, [size[0], 0, 0], (0, 0, size[0]), rgba
        elif shape == "table":
            s = size[0]
            for sx in (-1, 1):
                for sy in (-1, 1):
                    _box(body, (.03 * s + .01, .03 * s + .01, .45 * s), (sx * .5 * s, sy * .35 * s, .45 * s), rgba)
            _box(body, (.6 * s, .42 * s, .03), (0, 0, .9 * s), rgba)
        elif shape == "chair":
            s = size[0]
            for sx in (-1, 1):
                for sy in (-1, 1):
                    _box(body, (.02 * s + .01, .02 * s + .01, .22 * s), (sx * .2 * s, sy * .2 * s, .22 * s), rgba)
            _box(body, (.24 * s, .24 * s, .025), (0, 0, .46 * s), rgba)
            _box(body, (.025, .24 * s, .25 * s), (-.22 * s, 0, .72 * s), rgba)
        elif shape == "stack":
            s = size[0]
            _box(body, (.3 * s, .25 * s, .2 * s), (0, 0, .2 * s), rgba)
            _box(body, (.22 * s, .2 * s, .15 * s), (.03 * s, 0, .55 * s), [min(1, c * 1.1) for c in rgba[:3]] + [1])
        objects.append((item["name"], WorldObject(item["name"], item["label"], "")))
