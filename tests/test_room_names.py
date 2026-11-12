"""Room names from what Go2 sees in them (#35), scored against the houses from #17."""
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "simulation"))

import house  # noqa: E402  (plain Python: no simulator needed)
from go2.room_names import EVIDENCE, RoomNamer  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
