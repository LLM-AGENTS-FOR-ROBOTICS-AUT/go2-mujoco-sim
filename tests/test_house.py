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
from tests.fakes import needs_camera  # noqa: E402

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
                self.assertTrue(2 <= house.room_count(h) <= 7)
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
            self.assertEqual(house.room_count(house.random_house(random.Random(n), n)), n)

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

    @unittest.skipUnless((ROOT / "_deps/gso/teddy_bear/model.obj").exists(),
                         "scanned objects not downloaded (simulation/fetch_gso.py)")
    def test_the_teddy_bear_stands_upright_on_the_floor(self):
        import world
        for scene in ("room", "house"):
            with self.subTest(scene=scene):
                model, objects = world.build(scene)
                data = mujoco.MjData(model)
                mujoco.mj_forward(model, data)
                teddy = next(body for body, item in objects.items() if item.label == "teddy bear")
                g = next(g for g in range(model.ngeom)
                         if model.geom_bodyid[g] == teddy and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH)
                mesh = model.geom_dataid[g]
                points = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
                z = (data.geom_xpos[g] + points @ data.geom_xmat[g].reshape(3, 3).T)[:, 2]
                self.assertAlmostEqual(z.min(), 0, delta=.01, msg="on the floor")
                self.assertGreater(z.max(), .3, "standing up, not lying on its back (0.12 m tall)")

    def test_room_scenes_are_unchanged(self):
        import world
        _, objects = world.build("room")
        names = {item.name for item in objects.values()}
        expected = {"wall_north", "wall_south", "wall_east", "wall_west"} | {o.name for o, *_ in world.ROOM_OBJECTS}
        self.assertTrue(expected <= names)
        walkers = {name for name, _ in world.WALKERS}       # parked under the floor until asked to walk (#15)
        self.assertFalse(names - expected - walkers - {short for short, *_ in world.GSO_PLACEMENT})

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


def on_furniture(seeds):
    """(house, thing, furniture) for every thing standing on furniture in the random houses of `seeds`."""
    out = []
    for seed in seeds:
        h = house.random_house(random.Random(seed))
        for item in h["objects"]:
            if item.get("on"):
                out.append((h, item, next(o for o in h["objects"] if o["name"] == item["on"])))
    return out


class OnFurnitureTests(unittest.TestCase):
    """Things on beds, desks and tables (#21), and whether Go2's camera, 0.36 m up, can see them."""

    @classmethod
    def setUpClass(cls):
        cls.found = on_furniture(range(120))

    def test_teddy_bears_beds_cups_tables_and_laptops_desks(self):
        pairs = {(item["kind"], furniture["kind"]) for _, item, furniture in self.found}
        self.assertTrue({("teddy", "bed"), ("cup", "dining_table"), ("laptop", "desk"), ("book", "bed")} <= pairs,
                        pairs)
        for _, item, furniture in self.found:
            self.assertIn(item["kind"], house.ON_TOP[furniture["kind"]])

    def test_each_thing_is_on_top_of_its_furniture_in_the_same_room(self):
        for h, item, furniture in self.found:
            with self.subTest(item=item["name"], on=furniture["name"]):
                self.assertEqual(item["room"], furniture["room"])
                self.assertEqual(house.room_at(h, *item["pos"]), item["room"])
                _, (x0, y0, x1, y1) = house.SURFACES[furniture["kind"]]
                u, v = house._local(furniture, *item["pos"])
                self.assertTrue(x0 <= u <= x1 and y0 <= v <= y1, (u, v))
                fx0, fy0, fx1, fy1 = house.footprint(furniture)
                self.assertTrue(fx0 <= item["pos"][0] <= fx1 and fy0 <= item["pos"][1] <= fy1)

    def test_things_on_the_same_furniture_dont_overlap(self):
        groups = {}
        for h, item, furniture in self.found:
            groups.setdefault((id(h), furniture["name"]), []).append(item)
        for items in groups.values():
            for i, a in enumerate(items):
                for b in items[i + 1:]:
                    gap = math.dist(a["pos"], b["pos"])
                    self.assertGreaterEqual(gap, .95 * (math.hypot(*house.KINDS[a["kind"]][1]) +
                                                        math.hypot(*house.KINDS[b["kind"]][1])) - .1)

    def test_every_thing_on_furniture_can_be_seen_from_the_floor(self):
        for h, item, _ in self.found:
            with self.subTest(item=item["name"]):
                share, (x, y, yaw) = house.views(h, item)[0]
                self.assertGreaterEqual(share, house.VISIBLE_SHARE)
                self.assertEqual(house.room_at(h, x, y), item["room"], "Go2 looks from inside the room")

    def test_the_demo_house_keeps_everything_on_the_floor(self):
        self.assertFalse([o["name"] for o in house.house_demo()["objects"] if "on" in o])

    def test_a_house_is_still_quick_to_make(self):
        for seed in range(30):
            t = time.perf_counter()
            house.random_house(random.Random(seed))
            self.assertLess(time.perf_counter() - t, 1.)


class SightLineTests(unittest.TestCase):
    """The sight-line check on one hand-made room: a table in the middle of a 6 x 6 m room."""

    def room(self, *things):
        table = {"name": "table", "label": "dining table", "kind": "dining_table", "room": "r", "pos": [0., 0.],
                 "yaw": 0., "rgba": [.6, .5, .3, 1]}
        objects = [table] + [{"name": f"{kind}_{i}", "label": kind, "kind": kind, "room": "r", "pos": list(pos),
                              "yaw": 0., "rgba": [1, 1, 1, 1], **({"on": "table"} if up else {})}
                             for i, (kind, pos, up) in enumerate(things)]
        return {"scene": "house", "rooms": [{"name": "r", "type": "kitchen", "rect": [-3., -3., 3., 3.]}],
                "doors": [], "objects": objects, "start": [-2., -2., 0.]}

    def share(self, h):
        return house.sight_share(h, h["objects"][1])

    def test_a_cup_at_the_front_shows_one_in_the_middle_does_not(self):
        self.assertGreater(self.share(self.room(("cup", (0., -.3), True))), house.VISIBLE_SHARE)
        self.assertEqual(self.share(self.room(("cup", (0., 0.), True))), 0.)

    def test_a_flat_thing_a_little_way_in_is_hidden_by_the_edge(self):
        self.assertEqual(self.share(self.room(("book", (0., -.2), True))), 0.)     # 4 cm tall, 18 cm in

    def test_furniture_in_the_way_hides_it_from_that_side(self):
        open_ = self.room(("cup", (.53, 0.), True))           # near the east end: seen from the east
        self.assertGreater(self.share(open_), house.VISIBLE_SHARE)
        blocked = self.room(("cup", (.53, 0.), True), ("chair", (1.3, 0.), False), ("chair", (1.3, .6), False),
                            ("chair", (1.3, -.6), False))
        self.assertLess(self.share(blocked), self.share(open_))

    def test_things_on_the_floor_are_not_checked(self):
        h = self.room(("cup", (1.5, 1.5), False))
        self.assertEqual(self.share(h), 1.)


@unittest.skipUnless(HAVE_MUJOCO, "mujoco not installed")
class OnFurnitureMujocoTests(unittest.TestCase):
    def test_things_rest_on_the_furniture_not_in_it_or_above_it(self):
        import numpy as np
        import world
        for h, item, furniture in on_furniture(range(6)):
            model, _ = world.build("house", h)
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            body = model.body(item["name"]).id
            lowest = math.inf
            for g in range(model.ngeom):
                if model.geom_bodyid[g] != body:
                    continue
                if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:          # a scanned object: its vertices
                    mesh = model.geom_dataid[g]
                    points = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
                else:                                                         # a plain shape: its box
                    centre, half = model.geom_aabb[g][:3], model.geom_aabb[g][3:]
                    points = centre + half * np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
                world_z = data.geom_xpos[g][2] + points @ data.geom_xmat[g].reshape(3, 3)[2]
                lowest = min(lowest, world_z.min())
            hit = np.array([-1], dtype=np.int32)
            top = 2. - mujoco.mj_ray(model, data, np.array([*item["pos"], 2.]), np.array([0., 0, -1]),
                                     np.array([1, 1, 1, 0, 0, 0], np.uint8), 1, body, hit)    # under the ceiling
            with self.subTest(item=item["name"], on=furniture["name"]):
                if (house.FURNITURE_DIR / str(furniture.get("model")) / "model.json").exists():
                    continue         # the ray would stop on the faded shapes under the model, not what is drawn
                self.assertEqual(model.geom_bodyid[hit[0]], model.body(furniture["name"]).id)
                self.assertAlmostEqual(lowest, top, delta=.02)

    @needs_camera
    def test_go2s_camera_sees_each_thing_from_where_views_says(self):
        import world
        from sensors import Camera
        for seed in range(8):
            h = house.random_house(random.Random(seed))
            items = [o for o in h["objects"] if o.get("on")]
            if not items:
                continue
            model, _ = world.build("house", h)
            data = mujoco.MjData(model)
            camera = Camera(model)
            try:
                for item in items:
                    _, (x, y, yaw) = house.views(h, item)[0]
                    data.qpos[:3] = (x, y, .32)             # standing height
                    data.qpos[3:7] = (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
                    mujoco.mj_forward(model, data)
                    _, bodies = camera.capture(data)
                    with self.subTest(seed=seed, item=item["name"]):
                        self.assertGreater((bodies == model.body(item["name"]).id).sum(), 40)
            finally:
                camera.close()


class LayoutShapeTests(unittest.TestCase):
    """L, T and Z-shaped houses, two hallways at right angles, and open-plan living rooms with kitchens (#21)."""

    @classmethod
    def setUpClass(cls):
        cls.houses = [house.random_house(random.Random(seed)) for seed in range(150)]

    def test_at_least_a_third_are_not_plain_rectangles(self):
        shaped = sum(not house.rectangular(h) for h in self.houses)
        self.assertGreaterEqual(shaped, len(self.houses) / 3)
        self.assertTrue(any(sum(r["type"] == "hallway" for r in h["rooms"]) == 2 for h in self.houses))
        self.assertTrue(any(d.get("opening") and "kitchen" in d["rooms"] for h in self.houses for d in h["doors"]))

    def test_every_house_is_reachable_with_unique_room_names_and_no_overlaps(self):
        for seed, h in enumerate(self.houses):
            with self.subTest(seed=seed):
                self.assertEqual(house.reachability(h), (True, []))
                names = [r["name"] for r in h["rooms"]]
                self.assertEqual(len(names), len(set(names)))
                rects = [r["rect"] for r in h["rooms"]]
                for i, a in enumerate(rects):
                    for b in rects[i + 1:]:
                        overlap_x = min(a[2], b[2]) - max(a[0], b[0])
                        overlap_y = min(a[3], b[3]) - max(a[1], b[1])
                        self.assertFalse(overlap_x > 1e-6 and overlap_y > 1e-6, (a, b))

    def test_an_opening_is_the_whole_shared_wall_with_no_wall_left_in_it(self):
        found = 0
        for seed, h in enumerate(self.houses):
            rooms = {r["name"]: r for r in h["rooms"]}
            pieces = house.wall_pieces(h)
            for door in (d for d in h["doors"] if d.get("opening")):
                found += 1
                with self.subTest(seed=seed, opening=door["name"]):
                    a, b = (rooms[n] for n in door["rooms"])
                    self.assertIn({a["type"], b["type"]}, ({"hallway"}, {"living_room", "kitchen"}))
                    axis, line, lo, hi = house._shared_wall(a["rect"], b["rect"])
                    self.assertEqual(door["axis"], axis)
                    self.assertAlmostEqual(door["width"], hi - lo, places=2)
                    self.assertTrue(door["open"])
                    for p_axis, p_line, p_a, p_b, *_ in pieces:
                        if p_axis == axis and abs(p_line - line) < 1e-6:
                            self.assertFalse(min(p_b, hi) - max(p_a, lo) > 1e-3, "a wall piece in the opening")
        self.assertGreater(found, 20)

    def test_big_furniture_never_backs_onto_an_opening(self):
        for seed, h in enumerate(self.houses):
            for door in (d for d in h["doors"] if d.get("opening")):
                line = door["pos"][1] if door["axis"] == "x" else door["pos"][0]
                half = door["width"] / 2
                along = door["pos"][0] if door["axis"] == "x" else door["pos"][1]
                for item in h["objects"]:
                    if not house.KINDS[item["kind"]][2] or item["room"] not in door["rooms"]:
                        continue
                    x0, y0, x1, y1 = house.footprint(item)
                    near, lo, hi = ((min(abs(y0 - line), abs(y1 - line)), x0, x1) if door["axis"] == "x"
                                    else (min(abs(x0 - line), abs(x1 - line)), y0, y1))
                    with self.subTest(seed=seed, item=item["name"], opening=door["name"]):
                        self.assertFalse(near < .2 and hi > along - half and lo < along + half,
                                         "backed onto the open side")

    @unittest.skipUnless(HAVE_MUJOCO, "mujoco not installed")
    def test_no_lintel_over_an_opening_in_mujoco(self):
        import world
        h = next(h for h in self.houses if any(d.get("opening") for d in h["doors"]))
        model, _ = world.build("house", h)
        bodies = {model.body(i).name for i in range(model.nbody)}
        for door in h["doors"]:
            self.assertEqual(f"{door['name']}_lintel" in bodies, not door.get("opening"), door["name"])


class ColourTests(unittest.TestCase):
    """Random colours by kind, named in the description (#21)."""

    def test_each_palette_word_is_the_colour_the_simulator_reports(self):
        import colorsys
        import world
        for kind, palette in house.PALETTES.items():
            for word, rgba in palette:
                self.assertEqual(world.colour_word(*colorsys.rgb_to_hsv(*rgba[:3])), word, (kind, rgba))

    def test_random_houses_vary_colours_and_name_them(self):
        seen = {}
        for seed in range(100):
            for item in house.random_house(random.Random(seed))["objects"]:
                if "colour" not in item:
                    continue
                with self.subTest(seed=seed, item=item["name"]):
                    self.assertNotIn("gso", item, "a scanned object shows its own texture")
                    self.assertIn((item["colour"], tuple(item["rgba"])), house.PALETTES[item["kind"]])
                seen.setdefault(item["kind"], []).append(item["colour"])
        for kind, palette in house.PALETTES.items():
            if len(seen.get(kind, [])) >= 15:                   # rare kinds (a second backpack) may show fewer
                self.assertGreaterEqual(len(set(seen[kind])), min(4, len(palette)), (kind, set(seen[kind])))
        for kind in ("ball", "book", "dining_table", "refrigerator", "sink"):    # cups: mostly the one scanned mug
            self.assertIn(kind, seen)

    def test_the_demo_house_keeps_its_colours(self):
        self.assertFalse([o["name"] for o in house.house_demo()["objects"] if "colour" in o])

    @unittest.skipUnless(HAVE_MUJOCO, "mujoco not installed")
    @needs_camera
    def test_go2s_camera_names_each_colour_as_the_description_does(self):
        import world
        from go2.perception import color_name
        from sensors import Camera
        for kind, palette in house.PALETTES.items():
            far = 2.6 if house.KINDS[kind][2] else 1.3          # big furniture further off, to fit in view
            for word, rgba in palette:
                h = {"scene": "house", "rooms": [{"name": "r", "type": "living_room", "rect": [-1., -3., 5., 3.]}],
                     "doors": [], "start": [0., 0., 0.],
                     "objects": [{"name": "thing", "label": kind, "kind": kind, "room": "r", "pos": [far, 0.],
                                  "yaw": 2., "rgba": list(rgba)}]}
                model, _ = world.build("house", h)
                data = mujoco.MjData(model)
                data.qpos[:3], data.qpos[3:7] = (0, 0, .32), (1, 0, 0, 0)
                mujoco.mj_forward(model, data)
                camera = Camera(model)
                try:
                    rgb, bodies = camera.capture(data)
                finally:
                    camera.close()
                with self.subTest(kind=kind, colour=word):
                    self.assertEqual(color_name(rgb[bodies == model.body("thing").id]), word)


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

    @needs_camera
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

    @needs_camera
    def test_look_sees_furniture_through_the_doorways(self):
        report = self.executor.look()
        labels = {o["label"] for o in report["objects"]}
        self.assertTrue(labels & {o["label"] for o in self.demo["objects"]}, report)
        self.assertNotIn("wall", labels)
        self.assertTrue(any(o["distance_m"] for o in report["objects"]))


if __name__ == "__main__":
    unittest.main()
