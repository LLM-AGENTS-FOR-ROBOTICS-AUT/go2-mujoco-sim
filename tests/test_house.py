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

    def test_look_sees_furniture_through_the_doorways(self):
        report = self.executor.look()
        labels = {o["label"] for o in report["objects"]}
        self.assertTrue(labels & {o["label"] for o in self.demo["objects"]}, report)
        self.assertNotIn("wall", labels)
        self.assertTrue(any(o["distance_m"] for o in report["objects"]))


if __name__ == "__main__":
    unittest.main()
