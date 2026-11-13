import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from go2.places import NamedPlaces, place_name


class NamedPlaceTests(unittest.TestCase):
    def test_saved_pose_survives_restart_and_case_insensitive_lookup(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scene.places.json"
            first = NamedPlaces(path)
            first.remember(" Front   Door ", 1.25, -2., 2*math.pi+.2)
            second = NamedPlaces(path)
            pose = second.get("front door")
            self.assertEqual((pose.name, pose.x, pose.y), ("Front Door", 1.25, -2.))
            self.assertAlmostEqual(pose.yaw, .2)

    def test_renaming_same_normalized_place_updates_its_position(self):
        places = NamedPlaces()
        places.remember("Ｄｏｏｒ", 1., 2., 0.)
        places.remember("door", 3., 4., .5)
        self.assertEqual(len(places.all()), 1)
        self.assertEqual(places.get("DOOR").x, 3.)

    def test_invalid_names_and_coordinates_cannot_replace_a_saved_pose(self):
        places = NamedPlaces()
        places.remember("door", 1., 2., 0.)
        for value in ("", " ", "x"*81, "door\n", 12):
            with self.subTest(name=value), self.assertRaises(ValueError):
                place_name(value)
        for value in (math.nan, math.inf, True, None):
            with self.subTest(coordinate=value), self.assertRaises(ValueError):
                places.remember("door", value, 0., 0.)
        self.assertEqual(places.get("door").x, 1.)

    def test_bad_file_is_reported_and_partial_places_are_not_used(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.json"
            path.write_text(json.dumps({"version": 1, "places": [
                {"name": "door", "x": 1., "y": 2., "yaw": 0., "saved_at": 1.},
                {"name": "broken", "x": "oops"}]}))
            places = NamedPlaces(path)
            self.assertTrue(places.load_error)
            self.assertEqual(places.all(), [])

    def test_failed_atomic_save_preserves_previous_file_and_memory(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "places.json"
            places = NamedPlaces(path)
            places.remember("door", 1., 2., 0.)
            original = path.read_bytes()
            with patch("go2.places.os.replace", side_effect=OSError("disk unavailable")):
                with self.assertRaises(OSError):
                    places.remember("door", 9., 9., 0.)
            self.assertEqual(places.get("door").x, 1.)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(folder).glob("*.tmp")), [])

    def test_forget_is_persistent_and_other_scene_is_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            first, second = (NamedPlaces(Path(folder) / (name + ".json")) for name in ("room", "house"))
            first.remember("door", 1., 0., 0.)
            second.remember("door", 5., 0., 0.)
            first.forget("door")
            self.assertEqual(NamedPlaces(first.path).all(), [])
            self.assertEqual(NamedPlaces(second.path).get("door").x, 5.)


if __name__ == "__main__":
    unittest.main()
