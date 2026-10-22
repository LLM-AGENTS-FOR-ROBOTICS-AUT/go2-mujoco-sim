"""Multi-room houses (#17): descriptions, random houses, reachability, and the house in MuJoCo."""
import copy
import json
import math
import random
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "simulation"))

import house  # noqa: E402  (plain Python: no simulator needed)

try:
    import mujoco  # noqa: F401
    HAVE_MUJOCO = True
except ImportError:
    HAVE_MUJOCO = False
try:
    import torch  # noqa: F401
    HAVE_SIM = HAVE_MUJOCO                   # walking also needs the policy (torch)
except ImportError:
    HAVE_SIM = False


def inside(box, rect):
    return rect[0] <= box[0] and box[2] <= rect[2] and rect[1] <= box[1] and box[3] <= rect[3]


class HouseDescriptionTests(unittest.TestCase):
    def test_demo_house_has_the_teddy_bear_in_bedroom_2(self):
        demo = house.house_demo()
        teddies = [o for o in demo["objects"] if o["label"] == "teddy bear"]
        self.assertEqual(len(teddies), 1)
        self.assertEqual(teddies[0]["room"], "bedroom_2")
        self.assertEqual(house.room_at(demo, *teddies[0]["pos"]), "bedroom_2")
        self.assertIn(teddies[0]["name"], house.objects_in(demo, "bedroom_2"))

    def test_demo_house_is_the_same_every_time_and_passes_its_checks(self):
        demo = house.house_demo()
        self.assertEqual(json.dumps(demo), json.dumps(house.house_demo()))
        self.assertEqual(house.reachability(demo), (True, []))
        self.assertTrue(house.furnished(demo))
        self.assertEqual(house.room_at(demo, *demo["start"][:2]), "hallway")

    def test_100_random_houses_are_valid(self):
        for seed in range(100):
            with self.subTest(seed=seed):
                h = house.random_house(random.Random(seed))
                self.assertEqual(house.reachability(h), (True, []))
                self.assertTrue(house.furnished(h))
                self.assertTrue(2 <= len(h["rooms"]) <= 7)
                self.assertEqual(sum(o["kind"] == "teddy" for o in h["objects"]), 1)
                self.assertTrue(all(d["width"] >= house.DOOR_MIN_WIDTH for d in h["doors"]))
                front = [d for d in h["doors"] if d["name"] == "front_door"]
                self.assertEqual(len(front), 1)
                self.assertFalse(front[0]["open"])
                self.assertEqual(json.loads(json.dumps(h)), h, "JSON-friendly, for the simulator process")

    def test_every_object_lies_inside_the_room_it_names(self):
        for h in [house.house_demo()] + [house.random_house(random.Random(s)) for s in range(20)]:
            rects = {room["name"]: room["rect"] for room in h["rooms"]}
            for item in h["objects"]:
                with self.subTest(object=item["name"], room=item["room"]):
                    self.assertTrue(inside(house.footprint(item), rects[item["room"]]))
                    self.assertEqual(house.room_at(h, *item["pos"]), item["room"])

    def test_same_seed_same_house(self):
        first = house.random_house(random.Random(42))
        self.assertEqual(json.dumps(first), json.dumps(house.random_house(random.Random(42))))
        self.assertNotEqual(json.dumps(first), json.dumps(house.random_house(random.Random(43))))

    def test_room_count_can_be_chosen(self):
        for n in range(2, 8):
            self.assertEqual(len(house.random_house(random.Random(n), n)["rooms"]), n)

    def test_hallways_are_kept_clear(self):
        for seed in range(30):
            h = house.random_house(random.Random(seed))
            halls = {room["name"] for room in h["rooms"] if room["type"] == "hallway"}
            self.assertFalse([o["name"] for o in h["objects"] if o["room"] in halls])

    def test_reachability_finds_what_blocks_go2(self):
        demo = house.house_demo()
        shut = copy.deepcopy(demo)
        next(d for d in shut["doors"] if d["name"] == "door_bathroom")["open"] = False
        self.assertEqual(house.reachability(shut), (False, ["Go2 can't get into bathroom"]))
        narrow = copy.deepcopy(demo)
        next(d for d in narrow["doors"] if d["name"] == "door_living")["width"] = .7
        ok, problems = house.reachability(narrow)
        self.assertFalse(ok)
        self.assertIn("door_living is 0.70 m wide; Go2 needs 0.9 m", problems)
        blocked = copy.deepcopy(demo)              # a couch dumped in the hallway, in front of bedroom 2's door
        blocked["objects"].append({"name": "couch_x", "label": "couch", "kind": "couch", "room": "hallway",
                                   "pos": [3.5, -.3], "yaw": 0., "rgba": [.5, .5, .5, 1]})
        ok, problems = house.reachability(blocked)
        self.assertFalse(ok)
        self.assertIn("Go2 can't get into bedroom_2", problems)

    def test_furniture_with_a_realistic_model_names_it_and_styles_take_turns(self):
        chairs = set()
        for seed in range(30):
            for item in house.random_house(random.Random(seed))["objects"]:
                if item["kind"] in house.MODELS:
                    self.assertIn(item["model"], [m for m, *_ in house.MODELS[item["kind"]]])
                    if item["kind"] == "chair":
                        chairs.add(item["model"])
                else:
                    self.assertNotIn("model", item)
        self.assertEqual(chairs, {m for m, *_ in house.MODELS["chair"]})

    def test_walls_are_merged_and_doorways_cut_out(self):
        demo = house.house_demo()
        pieces = house.wall_pieces(demo)
        lines = {}
        for axis, line, a, b, *_ in pieces:
            for c, d in lines.get((axis, line), []):
                self.assertTrue(b <= c + 1e-6 or d <= a + 1e-6, f"overlapping wall pieces on {axis}={line}")
            lines.setdefault((axis, line), []).append((a, b))
        for door in demo["doors"]:
            along, across = door["pos"] if door["axis"] == "x" else door["pos"][::-1]
            covering = [p for p in pieces if p[0] == door["axis"] and abs(p[1] - across) < 1e-6 and p[2] < along < p[3]]
            self.assertFalse(covering, f"a wall covers {door['name']}")


class FurnitureConverterTests(unittest.TestCase):
    def test_a_gltf_model_becomes_a_z_up_obj_centred_on_the_floor(self):
        import struct
        import tempfile
        import fetch_furniture as F
        # One triangle standing up in glTF's Y-up frame, 2 m tall, and moved away from the origin.
        blob = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 2, 0) + struct.pack("<6f", 0, 0, 1, 0, 0, 1) + \
            struct.pack("<3H", 0, 1, 2)
        gltf = {"scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "translation": [5, 7, 9]}],
                "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "indices": 2, "material": 0}]}],
                "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
                "textures": [{"source": 0}], "images": [{"uri": "textures/t_diff_1k.jpg"}],
                "buffers": [{"uri": "t.bin", "byteLength": len(blob)}],
                "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 36},
                                {"buffer": 0, "byteOffset": 36, "byteLength": 24},
                                {"buffer": 0, "byteOffset": 60, "byteLength": 6}],
                "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
                              {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC2"},
                              {"bufferView": 2, "componentType": 5123, "count": 3, "type": "SCALAR"}]}
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "t.bin").write_bytes(blob)
            (tmp / "textures").mkdir()
            (tmp / "textures/t_diff_1k.png").write_bytes(b"png")
            (tmp / "t.gltf").write_text(json.dumps(gltf))
            saved, F.DEST = F.DEST, tmp / "out"
            (F.DEST / "t").mkdir(parents=True)
            try:
                info = F.convert(tmp, "t")
            finally:
                F.DEST = saved
            obj = (tmp / "out/t/part0.obj").read_text().split("\n")
            copied = (tmp / "out/t/part0.png").read_bytes()
        self.assertEqual(info, {"size": [1.0, 0.0, 2.0], "parts": [{"obj": "part0.obj", "texture": "part0.png"}]})
        vertices = [tuple(float(v) for v in line.split()[1:]) for line in obj if line.startswith("v ")]
        self.assertEqual(vertices, [(-.5, 0, 0), (.5, 0, 0), (-.5, 0, 2)], "up is z, centred, standing on z = 0")
        uvs = [tuple(float(v) for v in line.split()[1:]) for line in obj if line.startswith("vt ")]
        self.assertEqual(uvs, [(0, 1), (1, 1), (0, 0)], "v flipped: glTF's runs top-down, OBJ's bottom-up")
        self.assertIn("f 1/1 2/2 3/3", obj)
        self.assertEqual(copied, b"png")

    def test_a_glb_model_converts_like_its_gltf(self):
        import struct
        import tempfile
        import fetch_furniture as F
        # The same triangle as one .glb file, plain-coloured: a header, then a JSON chunk and a binary chunk.
        blob = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 2, 0) + struct.pack("<3H", 0, 1, 2) + b"\0\0"
        gltf = {"scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
                "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}]}],
                "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [.5, .25, 1, 1]}}],
                "buffers": [{"byteLength": len(blob)}],
                "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 36},
                                {"buffer": 0, "byteOffset": 36, "byteLength": 6}],
                "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
                              {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"}]}
        text = json.dumps(gltf).encode()
        text += b" " * (-len(text) % 4)
        chunks = struct.pack("<I4s", len(text), b"JSON") + text + struct.pack("<I4s", len(blob), b"BIN\0") + blob
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "g.glb").write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(chunks)) + chunks)
            saved, F.DEST = F.DEST, tmp / "out"
            (F.DEST / "g").mkdir(parents=True)
            try:
                info = F.convert(tmp, "g")
            finally:
                F.DEST = saved
            obj = (tmp / "out/g/part0.obj").read_text().split("\n")
        self.assertEqual(info, {"size": [1.0, 0.0, 2.0], "parts": [{"obj": "part0.obj", "rgba": [.5, .25, 1, 1]}]})
        vertices = [tuple(float(v) for v in line.split()[1:]) for line in obj if line.startswith("v ")]
        self.assertEqual(vertices, [(-.5, 0, 0), (.5, 0, 0), (-.5, 0, 2)])


@unittest.skipUnless(HAVE_MUJOCO, "mujoco not installed")
class HouseMujocoTests(unittest.TestCase):
    def test_demo_house_builds_with_the_teddy_bear_in_bedroom_2(self):
        import world
        model, objects = world.build("house")
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        labels = [item.label for item in objects.values()]
        self.assertGreaterEqual(labels.count("wall"), 10)
        self.assertEqual(labels.count("door"), 1, "the closed front door")
        demo = house.house_demo()
        teddy = next(body for body, item in objects.items() if item.label == "teddy bear")
        self.assertEqual(house.room_at(demo, *data.xpos[teddy][:2]), "bedroom_2")

    def test_100_random_houses_build_in_mujoco(self):
        import world
        for seed in range(100):
            with self.subTest(seed=seed):
                h = house.random_house(random.Random(seed))
                model, objects = world.build("house", h)
                self.assertEqual(sum(item.label != "wall" and item.label != "door" for item in objects.values()),
                                 len(h["objects"]))

    def test_without_the_furniture_download_houses_use_plain_shapes(self):
        import tempfile
        import world
        saved = house.FURNITURE_DIR
        with tempfile.TemporaryDirectory() as empty:
            house.FURNITURE_DIR = Path(empty)
            try:
                model, _ = world.build("house")
            finally:
                house.FURNITURE_DIR = saved
        self.assertFalse(any(name.startswith(tuple(m for ms in house.MODELS.values() for m, *_ in ms))
                             for name in (model.mesh(i).name for i in range(model.nmesh))))
        chair = model.body("chair_1").id
        alphas = [model.geom_rgba[g][3] for g in range(model.ngeom) if model.geom_bodyid[g] == chair]
        self.assertTrue(alphas and min(alphas) == 1.0, "the chair's plain shapes are fully visible")

    @unittest.skipUnless((house.FURNITURE_DIR / "dining_chair_02/model.json").exists(),
                         "furniture not downloaded (simulation/fetch_furniture.py)")
    def test_realistic_furniture_looks_real_but_collides_and_is_seen_as_before(self):
        import numpy as np
        import world
        model, _ = world.build("house")
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        for name in ("chair_1", "plant_1", "tv_1", "bed_1", "oven_1", "couch_1"):
            with self.subTest(item=name):
                body = model.body(name).id
                geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body]
                meshes = [g for g in geoms if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
                shapes = [g for g in geoms if g not in meshes]
                self.assertTrue(meshes, "the realistic model is drawn")
                self.assertTrue(all(model.geom_contype[g] == 0 for g in meshes), "the model only looks")
                faded = [g for g in shapes if model.geom_rgba[g][3] == house.FADED]
                self.assertEqual(len(faded), len(shapes) - (1 if name == "tv_1" else 0), "the TV's stand stays")
                self.assertTrue(all(model.geom_contype[g] for g in shapes), "the plain shapes are still the body")
        self.assertGreater(house.FADED, 0, "rays (the LiDAR) skip fully transparent shapes, so never 0")
        chair = next(o for o in house.house_demo()["objects"] if o["name"] == "chair_1")
        hit = np.array([-1], dtype=np.int32)             # a ray in the LiDAR's geom groups, down onto the chair
        mujoco.mj_ray(model, data, np.array([*chair["pos"], 1.5]), np.array([0., 0, -1]),
                      np.array([1, 1, 1, 0, 0, 0], np.uint8), 1, -1, hit)
        self.assertEqual(model.geom_bodyid[hit[0]], model.body("chair_1").id)

    def test_room_scenes_are_unchanged(self):
        import world
        _, objects = world.build("room")
        names = {item.name for item in objects.values()}
        expected = {"wall_north", "wall_south", "wall_east", "wall_west"} | {o.name for o, *_ in world.ROOM_OBJECTS}
        self.assertTrue(expected <= names)
        self.assertFalse(names - expected - {short for short, *_ in world.GSO_PLACEMENT})

    def test_house_physics_is_about_as_fast_as_the_room(self):
        import world

        def realtime(scene, desc=None, steps=1500):
            model, _ = world.build(scene, desc)
            data = mujoco.MjData(model)
            t = time.perf_counter()
            for _ in range(steps):
                mujoco.mj_step(model, data)
            return steps * model.opt.timestep / (time.perf_counter() - t)
        t = time.perf_counter()
        big = house.random_house(random.Random(3), 7)
        world.build("house", big)
        self.assertLess(time.perf_counter() - t, 2.0, "making and building a 7-room house")
        room, home = realtime("room"), realtime("house", big)
        self.assertGreater(home, .6 * room, f"house {home:.1f}x vs room {room:.1f}x real time")


@unittest.skipUnless(HAVE_SIM, "simulation requirements not installed")
class HouseWalkingTests(unittest.TestCase):
    """Acceptance: in the demo house Go2 walks from the hallway into every room through the doorways."""

    @classmethod
    def setUpClass(cls):
        from go2.adapters.sim import SimAdapter
        cls.demo = house.house_demo()
        cls.sim = SimAdapter(python=sys.executable, scene="house", headless=True)
        cls.sim.connect()

    @classmethod
    def tearDownClass(cls):
        cls.sim.close()

    def setUp(self):
        from go2.executor import Executor
        self.sim.reset()
        self.events = []
        self.executor = Executor(on_event=self.events.append)
        self.executor.attach(self.sim)

    def tearDown(self):
        self.executor.close()

    def go(self, x, y):
        """go_to a point in the world, converted to the robot's frame (x forward, y left)."""
        from go2.tools import resolve
        from tests.fakes import wait_for
        s = self.sim.state()
        dx, dy = x - s.x, y - s.y
        args = {"x_m": round(dx * math.cos(s.yaw) + dy * math.sin(s.yaw), 2),
                "y_m": round(-dx * math.sin(s.yaw) + dy * math.cos(s.yaw), 2)}
        before = len(self.events)
        result = self.executor.submit(resolve("go_to", args), "g", 1)
        if result[0]["status"] != "queued":
            return result[0]
        self.assertTrue(wait_for(lambda: len(self.events) > before, 120))
        return self.events[-1]

    def test_walks_from_the_hallway_into_every_room(self):
        for door in self.demo["doors"]:
            if not door["open"] or "hallway" not in door["rooms"]:
                continue
            room = next(r for r in door["rooms"] if r != "hallway")
            with self.subTest(room=room):
                self.sim.reset()
                self.executor.attach(self.sim)
                side = 1 if door["pos"][1] > 0 else -1
                self.assertEqual(self.go(door["pos"][0], 0.).status, "completed")
                task = self.go(door["pos"][0], door["pos"][1] + side * .6)     # all of Go2 is past the wall
                self.assertEqual(task.status, "completed", task.detail)
                s = self.sim.state()
                self.assertEqual(house.room_at(self.demo, s.x, s.y), room)
                self.assertIsNone(s.fault, "no collision or fall")

    def test_yolo_mode_detects_objects_in_the_house(self):
        try:
            import ultralytics  # noqa: F401
        except ImportError:
            self.skipTest("ultralytics not installed")
        if not (ROOT / "models/yolo11n-seg.pt").exists():
            self.skipTest("YOLO weights not downloaded (requirements-vision.txt)")
        # Fixed views in the demo house: in front of the bathroom's toilet, and bedroom 2's corner; with the
        # realistic furniture downloaded, also the kitchen's potted plant and the living room's chair.
        views = [((.47, -3.05, -math.pi / 2), "toilet"), ((4.24, -2.48, .79), "clock")]
        if (house.FURNITURE_DIR / "dining_chair_02/model.json").exists():
            views += [((2.11, 3.3, math.pi / 2), "potted plant"), ((-4.05, 3.9, -math.pi / 2), "chair")]
        self.sim.detector = "yolo"
        try:
            for (x, y, yaw), label in views:
                with self.subTest(label=label):
                    self.sim.reset(x, y, yaw)
                    self.executor.attach(self.sim)
                    labels = {o["label"] for o in self.executor.look().get("objects", [])}
                    self.assertIn(label, labels)
        finally:
            self.sim.detector = "truth"

    def test_look_sees_furniture_through_the_doorways(self):
        report = self.executor.look()
        labels = {o["label"] for o in report["objects"]}
        self.assertTrue(labels & {o["label"] for o in self.demo["objects"]}, report)
        self.assertNotIn("wall", labels)
        self.assertTrue(any(o["distance_m"] for o in report["objects"]))


if __name__ == "__main__":
    unittest.main()
