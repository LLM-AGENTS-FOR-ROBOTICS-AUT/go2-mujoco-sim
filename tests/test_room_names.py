"""Rooms (#35): split from the map, named from what Go2 sees in them, and gone to by name, scored against the
houses from #17."""
import collections
import math
import random
import re
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "simulation"))

import house  # noqa: E402  (plain Python: no simulator needed)
from go2.agent import Agent  # noqa: E402
from go2.executor import Executor  # noqa: E402
from go2.map_view import map_view  # noqa: E402
from go2.mock_model import MockModel  # noqa: E402
from go2.perception import Detection  # noqa: E402
from go2.room_map import FREE, OCCUPIED, UNKNOWN, OccupancyGrid  # noqa: E402
from go2.room_names import EVIDENCE, RoomNamer  # noqa: E402
from go2.room_regions import RoomRegions  # noqa: E402
from go2.tools import ToolError, resolve  # noqa: E402
from tests.fakes import FakeRobot, needs_camera, wait_for  # noqa: E402

try:
    import mujoco  # noqa: F401
    import torch  # noqa: F401
    HAVE_SIM = True                          # the simulator, and the walking policy (torch)
except ImportError:
    HAVE_SIM = False


def name_from_house(h, looks=2):
    """Name a house's rooms as if Go2 saw every object in it `looks` times, in the room it really is in."""
    namer = RoomNamer()
    for _ in range(looks):
        for item in h["objects"]:
            namer.saw(item["room"], item["label"])
    return namer


class RoomNamerTests(unittest.TestCase):
    def test_two_sightings_of_a_bed_name_a_bedroom(self):
        namer = RoomNamer()
        namer.saw(1, "bed")
        self.assertIsNone(namer.room_type(1), "one sighting isn't enough")
        namer.saw(1, "bed")
        self.assertEqual(namer.room_type(1), "bedroom")
        self.assertEqual(namer.name(1), "bedroom")

    def test_a_fridge_and_an_oven_name_a_kitchen(self):
        namer = RoomNamer()
        namer.saw(3, "refrigerator")
        namer.saw(3, "oven")
        self.assertEqual(namer.room_type(3), "kitchen")

    def test_one_wrong_detection_does_not_rename_a_room(self):
        namer = RoomNamer()
        for _ in range(3):
            namer.saw(1, "bed")
        namer.saw(1, "refrigerator")
        self.assertEqual(namer.room_type(1), "bedroom")

    def test_unclear_sightings_leave_a_room_unnamed(self):
        namer = RoomNamer()
        for _ in range(4):
            namer.saw(1, "sink")              # kitchen or bathroom
        self.assertIsNone(namer.room_type(1))
        namer.saw(2, "bed")
        namer.saw(2, "bed")
        namer.saw(2, "couch")
        namer.saw(2, "couch")
        self.assertIsNone(namer.room_type(2), "a tie names nothing")

    def test_labels_that_say_nothing_about_the_room_are_ignored(self):
        namer = RoomNamer()
        for label in ("chair", "teddy bear", "potted plant", "sports ball"):
            namer.saw(1, label)
        namer.saw(None, "bed")                # not in any room yet
        self.assertEqual(namer.votes, {})

    def test_labels_are_matched_whatever_their_case_and_spacing(self):
        namer = RoomNamer()
        namer.saw(1, " Bed")
        namer.saw(1, "BED ")
        self.assertEqual(namer.room_type(1), "bedroom")

    def test_the_users_name_wins(self):
        namer = RoomNamer()
        namer.saw(1, "bed")
        namer.saw(1, "bed")
        namer.name_room(1, " office ")
        self.assertEqual(namer.name(1), "office")
        namer.name_room(2, "spare room")      # named before anything is seen in it
        self.assertEqual(namer.name(2), "spare room")
        with self.assertRaises(ValueError):
            namer.name_room(3, "  ")

    def test_several_rooms_of_one_type_are_numbered_in_room_order(self):
        namer = RoomNamer()
        for room in (7, 4):
            namer.saw(room, "bed")
            namer.saw(room, "bed")
        namer.saw(5, "toilet")
        namer.saw(5, "toilet")
        self.assertEqual(namer.names(), {4: "bedroom_1", 7: "bedroom_2", 5: "bathroom"})
        self.assertIsNone(namer.name(9), "a room nothing was seen in has no name")


class HouseScoringTests(unittest.TestCase):
    """The #35 target: at least 80% of rooms named correctly once explored. Hallways hold nothing that tells
    them apart, so they're left out; room splitting (#14) is scored separately."""

    def test_every_room_types_main_piece_votes_for_that_type(self):
        for kind, furnishing in house.FURNISHING.items():
            if furnishing:
                label = house.KINDS[furnishing[0][0]][0]
                with self.subTest(room_type=kind):
                    self.assertIn(kind, EVIDENCE.get(label, {}))

    def test_the_demo_house_rooms_are_all_named_correctly(self):
        h = house.house_demo()
        namer = name_from_house(h)
        for room in h["rooms"]:
            if room["type"] != "hallway":
                with self.subTest(room=room["name"]):
                    self.assertEqual(namer.room_type(room["name"]), room["type"])
        self.assertEqual(namer.name("bedroom_1"), "bedroom_1")
        self.assertEqual(namer.name("kitchen"), "kitchen")

    def test_at_least_80_percent_of_rooms_named_correctly_in_50_random_houses(self):
        right = total = 0
        for seed in range(50):
            h = house.random_house(random.Random(seed))
            namer = name_from_house(h)
            for room in h["rooms"]:
                if room["type"] != "hallway":
                    total += 1
                    right += namer.room_type(room["name"]) == room["type"]
        self.assertGreaterEqual(right / total, .8, f"{right} of {total} rooms named correctly")


def house_map(h, resolution=.1):
    """The map a full exploration of house h would give: rooms free; walls, closed doors and furniture
    occupied; outside unknown."""
    reach = max(abs(v) for room in h["rooms"] for v in room["rect"])
    m = OccupancyGrid(resolution=resolution, size=2 * math.ceil(reach / resolution) + 20)
    m.grid[:] = UNKNOWN

    def mark(x0, y0, x1, y1, value):
        (r0, c0), (r1, c1) = m.world_to_cell(x0, y0), m.world_to_cell(x1, y1)
        m.grid[r0:r1 + 1, c0:c1 + 1] = value

    for room in h["rooms"]:
        mark(*room["rect"], FREE)
    w = house.WALL_THICK
    for axis, line, a, b, *_ in house.wall_pieces(h):
        mark(a, line - w, b, line + w, OCCUPIED) if axis == "x" else mark(line - w, a, line + w, b, OCCUPIED)
    for door in h["doors"]:
        if not door["open"]:
            (dx, dy), half = door["pos"], door["width"] / 2
            mark(dx - half, dy - w, dx + half, dy + w, OCCUPIED) if door["axis"] == "x" else \
                mark(dx - w, dy - half, dx + w, dy + half, OCCUPIED)
    for item in h["objects"]:
        mark(*house.footprint(item), OCCUPIED)
    return m


def name_from_map(h):
    """Split house h's map into rooms (#14), name them from its objects, each seen twice; return
    [(true room type, guessed type)] for its rooms, by the region most of each room's floor fell in."""
    m = house_map(h)
    regions = RoomRegions().update(m)
    namer = RoomNamer()
    for _ in range(2):
        for item in h["objects"]:
            namer.saw(regions.region_near(*item["pos"]), item["label"])
    out = []
    for room in h["rooms"]:
        x0, y0, x1, y1 = room["rect"]
        (r0, c0), (r1, c1) = m.world_to_cell(x0 + .2, y0 + .2), m.world_to_cell(x1 - .2, y1 - .2)
        ids = regions.labels[r0:r1 + 1, c0:c1 + 1]
        top = collections.Counter(ids[ids > 0].tolist()).most_common(1)
        out.append((room["type"], namer.room_type(top[0][0]) if top else None))
    return out


class HouseMapTests(unittest.TestCase):
    """The same scoring, but with rooms split from the map (#14's room regions) rather than told."""

    def test_the_demo_house_rooms_are_named_correctly_from_its_map(self):
        for kind, guess in name_from_map(house.house_demo()):
            if kind != "hallway":
                with self.subTest(room_type=kind):
                    self.assertEqual(guess, kind)

    def test_at_least_80_percent_of_rooms_named_correctly_from_the_map_in_20_random_houses(self):
        named = [pair for seed in range(20) for pair in name_from_map(house.random_house(random.Random(seed)))
                 if pair[0] != "hallway"]
        right = sum(kind == guess for kind, guess in named)
        self.assertGreaterEqual(right / len(named), .8, f"{right} of {len(named)} rooms named correctly")


def split_rooms(h, resolution=.05):
    """{room name: (the region most of its floor fell in, the share of its floor in that region)}, splitting
    house h's map at the executor's resolution."""
    m = house_map(h, resolution)
    regions = RoomRegions().update(m)
    out = {}
    for room in h["rooms"]:
        x0, y0, x1, y1 = room["rect"]
        (r0, c0), (r1, c1) = m.world_to_cell(x0 + .2, y0 + .2), m.world_to_cell(x1 - .2, y1 - .2)
        ids = regions.labels[r0:r1 + 1, c0:c1 + 1]
        ids = ids[ids > 0]
        region, count = collections.Counter(ids.tolist()).most_common(1)[0]
        out[room["name"]] = region, count / len(ids)
    return out


class RoomSplittingTests(unittest.TestCase):
    """#35: the map splits into the house's rooms, with no room cut in two and no two rooms merged."""

    def test_the_demo_house_splits_into_its_six_rooms(self):
        rooms = split_rooms(house.house_demo())
        self.assertEqual(len({region for region, _ in rooms.values()}), 6, rooms)
        for name, (_, share) in rooms.items():
            with self.subTest(room=name):
                self.assertGreater(share, .98, "cut in two")

    def test_no_two_rooms_merge_and_nearly_all_stay_whole_in_20_random_houses(self):
        whole = total = 0
        for seed in range(20):
            rooms = split_rooms(house.random_house(random.Random(seed)))
            with self.subTest(seed=seed):
                self.assertEqual(len({region for region, _ in rooms.values()}), len(rooms), "rooms merged")
            whole += sum(share > .98 for _, share in rooms.values())
            total += len(rooms)
        self.assertGreaterEqual(whole / total, .95, f"{whole} of {total} rooms whole")

    def test_a_gap_beside_furniture_is_not_a_doorway(self):
        m = OccupancyGrid(resolution=.1, size=90)          # one 7 x 7 m room, nearly cut across by furniture
        m.grid[:] = OCCUPIED
        m.grid[10:80, 10:80] = FREE
        m.grid[40:50, 10:70] = OCCUPIED                    # a 1 m deep sideboard, 1 m short of the far wall
        regions = RoomRegions().update(m)
        self.assertEqual(regions.region_near(*m.cell_to_world(25, 40), 0.), regions.region_near(*m.cell_to_world(65, 40), 0.))
        m.grid[40:50, 10:70] = FREE
        m.grid[44:46, 10:80] = OCCUPIED                    # a thin wall across instead, with a 1 m doorway in it
        m.grid[44:46, 40:50] = FREE
        regions = RoomRegions().update(m)
        self.assertNotEqual(regions.region_near(*m.cell_to_world(25, 40), 0.), regions.region_near(*m.cell_to_world(65, 40), 0.))

    def test_a_rooms_spot_is_in_it_and_clear_of_walls(self):
        m = house_map(house.house_demo())
        regions = RoomRegions().update(m)
        for room in house.house_demo()["rooms"]:
            region = regions.region_near(*[(a + b) / 2 for a, b in zip(room["rect"][:2], room["rect"][2:])], 0.)
            x, y = regions.region_point(region, clearance_m=.45)
            with self.subTest(room=room["name"]):
                self.assertEqual(house.room_at(house.house_demo(), x, y), room["name"])
                self.assertEqual(regions.region_near(x, y, 0.), region)


def seen(label, x, y):
    return Detection(id=None, label=label, color=None, distance_m=1., bearing_deg=0., x_m=x, y_m=y,
                     distance_source="lidar", confidence=.9)


class ExecutorRoomNameTests(unittest.TestCase):
    """What Go2 sees during look() names the room it's in; room_at answers with the name."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.robot = FakeRobot()
        self.robot.connect()
        self.addCleanup(self.robot.close)
        self.executor = Executor(memory_path=temp.name + "/memory.json")
        self.executor.attach(self.robot)
        self.addCleanup(self.executor.close)
        m = OccupancyGrid(resolution=.1, size=90)            # two rooms joined by a 1 m doorway
        m.grid[:] = OCCUPIED
        m.grid[10:80, 10:80] = FREE
        m.grid[10:80, 44:46] = OCCUPIED
        m.grid[40:50, 44:46] = FREE
        self.executor.room_map = m
        self.left, self.right = m.cell_to_world(25, 25), m.cell_to_world(25, 65)

    def test_a_bed_seen_twice_names_its_room_and_not_the_other(self):
        for _ in range(2):
            self.executor._note_room_sightings([seen("bed", *self.left)], (0., 0., 0.))
        self.assertEqual(self.executor.room_at(*self.left), "bedroom")
        self.assertRegex(self.executor.room_at(*self.right), r"^room_\d+$")

    def test_the_users_name_wins_over_a_guess(self):
        for _ in range(2):
            self.executor._note_room_sightings([seen("bed", *self.left)], (0., 0., 0.))
        self.assertEqual(self.executor.room_at(*self.left), "bedroom")
        self.executor.room_regions.name_room(*self.left, "office")
        self.assertEqual(self.executor.room_at(*self.left), "office")

    def test_sightings_are_placed_on_the_map_from_the_robots_pose(self):
        x, y = self.left
        self.executor._note_room_sightings([seen("bed", 1., 0.)], (x, y - 1., math.pi / 2))   # 1 m ahead, facing +y
        (sx, sy, label), = self.executor.room_sightings
        self.assertAlmostEqual(sx, x)
        self.assertAlmostEqual(sy, y)
        self.assertEqual(label, "bed")

    def test_noting_sightings_is_quick_and_keeps_only_telling_things(self):
        things = [seen(label, *self.left) for label in ("bed", "chair", "teddy bear", "fridge", "oven")] * 40
        began = time.perf_counter()
        self.executor._note_room_sightings(things, (0., 0., 0.))
        self.assertLess(time.perf_counter() - began, .05, "look() must not wait for the room regions")
        self.assertEqual({label for *_, label in self.executor.room_sightings}, {"bed", "oven"})

    def test_attach_forgets_the_sightings(self):
        self.executor._note_room_sightings([seen("bed", *self.left)], (0., 0., 0.))
        self.executor.attach(self.robot)
        self.assertEqual(list(self.executor.room_sightings), [])

    def test_a_pocket_too_small_to_be_a_room_takes_no_votes(self):
        m = self.executor.room_map                           # a 1.2 x 1.2 m walled-off pocket in the left room's corner
        m.grid[10:23, 10:23] = OCCUPIED
        m.grid[10:22, 10:22] = FREE
        pocket = m.cell_to_world(16, 16)
        self.executor.room_regions.update(m)
        self.assertNotEqual(self.executor.room_regions.region_near(*pocket, 0.),
                            self.executor.room_regions.region_near(*self.left, 0.), "the pocket is a region of its own")
        for _ in range(2):                                   # a fridge in (by) the pocket, an oven in the room
            self.executor._note_room_sightings([seen("refrigerator", *pocket), seen("oven", *self.left)], (0., 0., 0.))
        names = [room["name"] for room in self.executor.rooms(fresh=True)]
        self.assertIn("kitchen", names, "one kitchen, not kitchen_1 with no kitchen_2 listed")
        self.assertFalse(any(name.startswith("kitchen_") for name in names), names)

    def test_names_stay_with_their_rooms_as_the_map_grows(self):
        grown = OccupancyGrid(resolution=.1, size=90, grow=True)
        grown.grid[:] = self.executor.room_map.grid
        self.executor.room_map = grown
        for _ in range(2):
            self.executor._note_room_sightings([seen("bed", *self.left)], (0., 0., 0.))
        self.assertEqual(self.executor.room_at(*self.left), "bedroom")
        grown.ensure_bounds([[-15., -12.], [14., 13.]])
        self.assertEqual(self.executor.room_at(*self.left), "bedroom")
        self.assertRegex(self.executor.room_at(*self.right), r"^room_\d+$")


class GoToRoomTests(unittest.TestCase):
    """go_to_place goes into a room by its name when no saved place has it ("go to the kitchen")."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.events = []
        self.robot = FakeRobot()
        self.robot.connect()
        self.addCleanup(self.robot.close)
        self.executor = Executor(on_event=self.events.append, memory_path=temp.name + "/memory.json")
        self.executor.attach(self.robot)
        self.addCleanup(self.executor.close)
        m = OccupancyGrid(resolution=.1, size=120)           # three rooms off one side of a hallway
        m.grid[:] = OCCUPIED
        m.grid[10:110, 10:30] = FREE                          # the hallway, 2 m wide
        for top, bottom in ((10, 40), (45, 75), (80, 110)):
            m.grid[top:bottom, 31:70] = FREE                  # a room
            m.grid[top + 10:top + 20, 30:31] = FREE           # its 1 m doorway into the hallway
        self.executor.room_map = m
        self.rooms = [m.cell_to_world(25, 50), m.cell_to_world(60, 50), m.cell_to_world(95, 50)]
        for _ in range(2):                                     # a bedroom, a kitchen, and a bedroom
            self.executor._note_room_sightings([seen("bed", *self.rooms[0]), seen("refrigerator", *self.rooms[1]),
                                                seen("oven", *self.rooms[1]), seen("bed", *self.rooms[2])],
                                               (0., 0., 0.))

    def test_rooms_have_names_and_a_spot_inside(self):
        rooms = {r["name"]: r for r in self.executor.rooms()}
        self.assertTrue({"bedroom_1", "kitchen", "bedroom_2"} <= set(rooms), rooms)
        self.assertEqual(self.executor.room_at(rooms["kitchen"]["x"], rooms["kitchen"]["y"]), "kitchen")

    def test_a_room_name_finds_that_room_and_a_room_type_the_nearest_one(self):
        x, y, name = self.executor._room_destination("the Kitchen", 0., 0.)
        self.assertEqual((name, self.executor.room_at(x, y)), ("kitchen", "kitchen"))
        x, y, name = self.executor._room_destination("bedroom", *self.rooms[2])
        self.assertEqual(self.executor.room_at(x, y), name)
        self.assertLess(math.hypot(x - self.rooms[2][0], y - self.rooms[2][1]), 2.)
        self.assertIsNone(self.executor._room_destination("garage", 0., 0.))

    def test_go_to_place_heads_into_the_room(self):
        result = self.executor.submit(resolve("go_to_place", {"name": "kitchen"}), "g", 1)[0]
        self.assertEqual(result["status"], "queued", result)
        self.assertTrue(wait_for(lambda: self.events, 30))
        self.assertEqual(self.events[-1].detail["destination"]["name"], "kitchen")

    def test_a_saved_place_wins_over_a_room_and_an_unknown_name_is_refused(self):
        self.executor.places.remember("kitchen", 0., 0., 0.)
        result = self.executor.submit(resolve("go_to_place", {"name": "kitchen"}), "g", 1)[0]
        self.assertEqual(result["status"], "queued", result)
        self.assertTrue(wait_for(lambda: self.events, 30))
        self.assertEqual(self.events[-1].detail["destination"]["x_m"], 0.)
        result = self.executor.submit(resolve("go_to_place", {"name": "garage"}), "g", 2)[0]
        self.assertEqual(result["status"], "rejected", result)
        self.assertIn("place or room", result["error"])

    def test_query_memory_and_the_live_map_list_the_rooms(self):
        result = self.executor.submit(resolve("query_memory", {"target": "kitchen"}), "g", 1)[0]
        self.assertEqual([r["name"] for r in result["rooms"]], ["kitchen"])
        view = map_view(self.executor)
        self.assertTrue({"bedroom_1", "kitchen", "bedroom_2"} <= {r["name"] for r in view["rooms"]}, view["rooms"])

    def test_the_user_names_the_room_go2_is_in_and_can_go_back_to_it(self):
        self.assertEqual(self.executor.room_at(0., 0.), "kitchen")            # Go2 starts in the kitchen
        result = self.executor.submit(resolve("name_room", {"name": "Pantry"}), "g", 1)[0]
        self.assertEqual(result["status"], "done", result)
        self.assertEqual(self.executor.room_at(0., 0.), "Pantry")
        self.assertIn("Pantry", {r["name"] for r in self.executor.rooms()})
        self.assertNotIn("kitchen", {r["name"] for r in self.executor.rooms()})
        self.assertFalse(any(c[0] == "move" for c in self.robot.calls))
        x, y, name = self.executor._room_destination("pantry", *self.rooms[0])
        self.assertEqual(name, "Pantry")

    def test_naming_a_room_needs_go2_to_be_in_one_on_the_map(self):
        with self.robot.lock:
            self.robot.x, self.robot.y = 4.5, 0.                             # 3.5 m into unmapped wall
        result = self.executor.submit(resolve("name_room", {"name": "office"}), "g", 1)[0]
        self.assertEqual(result["status"], "rejected", result)
        self.assertIn("which room", result["error"])
        for args in ({}, {"name": ""}, {"name": "office", "x_m": 1}):
            with self.subTest(args=args), self.assertRaises(ToolError):
                resolve("name_room", args)

    def test_mock_model_names_the_room(self):
        agent = Agent(MockModel(), self.executor)
        agent.robot_line = lambda: "standing"
        self.addCleanup(agent.close)
        agent.submit("this is the office")
        self.assertTrue(wait_for(lambda: not agent.busy, 5))
        self.assertEqual([m["tool_name"] for m in agent.messages if m["role"] == "tool"], ["name_room"])
        self.assertIn("office", agent.messages[-1]["content"])
        self.assertEqual(self.executor.room_at(0., 0.), "office")

    def test_mock_model_goes_to_a_room_by_name(self):
        agent = Agent(MockModel(), self.executor)
        agent.robot_line = lambda: "standing"
        self.addCleanup(agent.close)
        agent.submit("go to the kitchen")
        self.assertTrue(wait_for(lambda: not agent.busy, 5))
        calls = [(m["tool_name"], m["content"]) for m in agent.messages if m["role"] == "tool"]
        self.assertEqual([name for name, _ in calls[-2:]], ["query_memory", "go_to_place"], calls)
        self.assertIn('"queued"', calls[-1][1])


@unittest.skipUnless(HAVE_SIM, "simulation requirements not installed")
class HouseRoomsInTheSimulatorTests(unittest.TestCase):
    """End to end in the simulator: Go2 looks all round each room of the demo house, then the map it built is
    split into rooms and they're named from what its camera saw."""

    @needs_camera
    def test_the_demo_houses_rooms_are_named_from_what_go2_sees(self):
        from go2.adapters.sim import SimAdapter
        demo = house.house_demo()
        sim = SimAdapter(python=sys.executable, scene="house", headless=True)
        sim.connect()
        self.addCleanup(sim.close)
        executor = Executor(memory_path=tempfile.mkdtemp() + "/memory.json")
        self.addCleanup(executor.close)
        executor.attach(sim)
        middles = {room["name"]: ((room["rect"][0] + room["rect"][2]) / 2, (room["rect"][1] + room["rect"][3]) / 2)
                   for room in demo["rooms"]}
        for x, y in middles.values():
            for k in range(8):                                # look all round, an eighth of a turn at a time
                sim.reset(x, y, k * math.pi / 4)
                time.sleep(.2)
                executor.look()
        rooms = {name: executor.room_at(x, y) for name, (x, y) in middles.items()}
        self.assertEqual(len(set(rooms.values())), len(rooms), f"two rooms share a region: {rooms}")
        for room in demo["rooms"]:
            if room["type"] != "hallway":
                with self.subTest(room=room["name"]):
                    self.assertEqual(re.sub(r"_\d+$", "", rooms[room["name"]] or ""), room["type"], rooms)


if __name__ == "__main__":
    unittest.main()
