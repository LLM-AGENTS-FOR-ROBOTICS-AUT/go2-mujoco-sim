import json
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from go2 import limits as L  # noqa: E402
from go2.tools import TOOLS, TOOL_NAMES, ToolError, describe_calls, resolve, split  # noqa: E402


def one(name, args):
    commands = resolve(name, args)
    assert len(commands) == 1
    return commands[0].params


class MoveTests(unittest.TestCase):
    def test_default_extent_and_speed(self):
        p = one("move", {"direction": "forward"})
        self.assertEqual(p["distance"], 1.0)
        self.assertAlmostEqual(p["speed"], .2)
        self.assertIn("default", p["note"])

    def test_speed_fits_distance_into_five_seconds(self):
        self.assertAlmostEqual(one("move", {"direction": "forward", "distance_m": 3})["speed"], .6)
        self.assertAlmostEqual(one("move", {"direction": "forward", "distance_m": 4})["speed"], .8)
        self.assertAlmostEqual(one("move", {"direction": "forward", "distance_m": .5})["speed"], .2)

    def test_over_distance_is_rejected_with_a_split_plan(self):
        with self.assertRaises(ToolError) as caught:
            resolve("move", {"direction": "forward", "distance_m": 10})
        self.assertIn("per-call limit of 4 m", str(caught.exception))
        self.assertEqual([c["arguments"]["distance_m"] for c in caught.exception.plan], [4, 4, 2])
        self.assertEqual(describe_calls(caught.exception.plan), "move forward 4 m, then forward 4 m, then forward 2 m")

    def test_over_speed_is_rejected_without_a_plan(self):
        with self.assertRaises(ToolError) as caught:
            resolve("move", {"direction": "forward", "speed_mps": 2})
        self.assertIsNone(caught.exception.plan)
        self.assertIn("Do not move yet", str(caught.exception))

    def test_slow_long_move_must_be_split_not_sped_up(self):
        with self.assertRaises(ToolError) as caught:
            resolve("move", {"direction": "forward", "distance_m": 3, "speed_mps": .2})
        self.assertEqual([c["arguments"]["distance_m"] for c in caught.exception.plan], [1, 1, 1])
        self.assertTrue(all(c["arguments"]["speed_mps"] == .2 for c in caught.exception.plan))

    def test_distance_and_duration_imply_speed(self):
        p = one("move", {"direction": "backward", "distance_m": 2, "duration_s": 4})
        self.assertAlmostEqual(p["speed"], .5)
        self.assertAlmostEqual(p["vx"], -.5)
        with self.assertRaises(ToolError):
            resolve("move", {"direction": "forward", "distance_m": 3, "duration_s": 2})

    def test_duration_only_uses_gentle_default(self):
        p = one("move", {"direction": "left", "duration_s": 3})
        self.assertAlmostEqual(p["vy"], .2)
        self.assertEqual(p["duration"], 3)
        self.assertIsNone(p["distance"])

    def test_diagonal_speed_is_vector_magnitude(self):
        p = one("move", {"direction": "forward_left", "distance_m": 2})
        self.assertAlmostEqual(math.hypot(p["vx"], p["vy"]), .4)
        self.assertGreater(p["vx"], 0)
        self.assertGreater(p["vy"], 0)

    def test_lenient_numbers_strict_types(self):
        self.assertEqual(one("move", {"direction": "forward", "distance_m": "2"})["distance"], 2)
        self.assertEqual(one("move", {"direction": "Forward Left", "distance_m": 1})["ux"], one("move", {"direction": "forward_left", "distance_m": 1})["ux"])
        for bad in ({"direction": "up"}, {"direction": "forward", "distance_m": True},
                    {"direction": "forward", "distance_m": -1}, {"direction": "forward", "hops": 2},
                    {"direction": "forward", "distance_m": float("nan")}):
            with self.assertRaises(ToolError, msg=bad):
                resolve("move", bad)

    def test_split_helper(self):
        self.assertEqual(split(10, 4), [4, 4, 2])
        self.assertEqual(split(8, 4), [4, 4])
        self.assertEqual(split(4.5, 4), [4, .5])


class TurnTests(unittest.TestCase):
    def test_defaults(self):
        p = one("turn", {})
        self.assertAlmostEqual(p["angle"], math.pi / 2)
        self.assertGreater(p["wz"], 0)   # left by default
        self.assertAlmostEqual(p["turn_rate"], L.DEFAULT_TURN_RATE_RPS)

    def test_half_turn_fits_five_seconds(self):
        p = one("turn", {"direction": "right", "angle_deg": 180})
        self.assertAlmostEqual(p["turn_rate"], math.pi / 5)
        self.assertLess(p["wz"], 0)

    def test_full_circle_needs_two_calls(self):
        with self.assertRaises(ToolError) as caught:
            resolve("turn", {"direction": "right", "angle_deg": 360})
        self.assertEqual([c["arguments"]["angle_deg"] for c in caught.exception.plan], [180, 180])
        self.assertEqual(caught.exception.plan[0]["arguments"]["direction"], "right")

    def test_turn_rate_limit(self):
        with self.assertRaises(ToolError):
            resolve("turn", {"direction": "left", "speed_dps": 90})
        self.assertAlmostEqual(one("turn", {"direction": "left", "angle_deg": 57, "duration_s": 1})["turn_rate"], math.radians(57))


class OtherToolTests(unittest.TestCase):
    def test_body_pose_signs_follow_sdk_convention(self):
        command = resolve("body", {"pitch_deg": 15, "roll_deg": -10, "yaw_deg": 20})[0]
        self.assertAlmostEqual(command.params["pitch"], -math.radians(15))   # look up = nose up = negative SDK pitch
        self.assertAlmostEqual(command.params["roll"], math.radians(-10))
        self.assertAlmostEqual(command.params["yaw"], math.radians(20))
        with self.assertRaises(ToolError):
            resolve("body", {"pitch_deg": 60})
        with self.assertRaises(ToolError):
            resolve("body", {})

    def test_body_settings_expand_to_separate_commands(self):
        names = [c.params["name"] for c in resolve("body", {"pitch_deg": 5, "speed_level": "slow", "gait": "trot_run"})]
        self.assertEqual(names, ["euler", "speed_level", "gait"])

    def test_acrobatics_always_need_confirmation(self):
        command = resolve("acrobatic", {"name": "front_flip"})[0]
        self.assertTrue(command.needs_confirmation)
        self.assertFalse(command.params["confirmed"])
        self.assertTrue(resolve("acrobatic", {"name": "front_flip", "confirmed": True})[0].params["confirmed"])

    def test_go_to_and_approach_bounds(self):
        self.assertEqual(one("go_to", {"x_m": 3, "y_m": -2})["y"], -2)
        self.assertEqual(one("go_to", {"target": "start"})["target"], "start")
        with self.assertRaises(ToolError):
            resolve("go_to", {"x_m": 25})
        with self.assertRaises(ToolError):
            resolve("go_to", {})
        self.assertEqual(one("approach", {"target": "Red Ball"})["target"], "red ball")
        self.assertEqual(one("approach", {"target": "chair"})["stop_distance"], .5)
        with self.assertRaises(ToolError):
            resolve("approach", {"target": "chair", "stop_distance_m": 5})

    def test_go_to_an_object_means_approach(self):
        command = resolve("go_to", {"target": "chair"})[0]
        self.assertEqual(command.kind, "approach")
        self.assertEqual(command.params["target"], "chair")
        self.assertEqual(resolve("go_to", {"target": "start"})[0].kind, "goto")

    def test_unknown_tool_and_query_tools(self):
        with self.assertRaises(ToolError):
            resolve("fly", {})
        self.assertEqual(resolve("look", {})[0].kind, "look")
        self.assertEqual(resolve("stop", None)[0].kind, "stop")

    def test_schemas_cover_resolvers_and_serialise(self):
        self.assertEqual(len(TOOLS), 12)
        for tool in TOOLS:
            fn = tool["function"]
            self.assertIn(fn["name"], TOOL_NAMES)
            self.assertTrue(fn["description"])
            json.dumps(tool)


if __name__ == "__main__":
    unittest.main()
