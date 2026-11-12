"""Room names from what Go2 sees in them (#35), scored against the houses from #17."""
import collections
import math
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "simulation"))

import house  # noqa: E402  (plain Python: no simulator needed)
from go2.executor import Executor  # noqa: E402
from go2.perception import Detection  # noqa: E402
from go2.room_map import FREE, OCCUPIED, UNKNOWN, OccupancyGrid  # noqa: E402
from go2.room_names import EVIDENCE, RoomNamer  # noqa: E402
from go2.room_regions import RoomRegions  # noqa: E402
from tests.fakes import FakeRobot  # noqa: E402


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
            self.executor._vote_rooms([seen("bed", *self.left)], (0., 0., 0.))
        self.assertEqual(self.executor.room_at(*self.left), "bedroom")
        self.assertRegex(self.executor.room_at(*self.right), r"^room_\d+$")

    def test_the_users_name_wins_over_a_guess(self):
        for _ in range(2):
            self.executor._vote_rooms([seen("bed", *self.left)], (0., 0., 0.))
        self.executor.room_regions.name_room(*self.left, "office")
        self.assertEqual(self.executor.room_at(*self.left), "office")

    def test_attach_forgets_the_votes(self):
        for _ in range(2):
            self.executor._vote_rooms([seen("bed", *self.left)], (0., 0., 0.))
        self.executor.attach(self.robot)
        self.assertEqual(self.executor.room_namer.votes, {})


if __name__ == "__main__":
    unittest.main()
