"""The command console backend (command_ui/server.py): the page's cards and status, from the LLM agent."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "command_ui")]
try:
    import server
except ImportError:                     # flask not installed: the console's own requirements
    server = None
from tests.fakes import FakeRobot, ScriptedModel, wait_for  # noqa: E402

PAGE = "http://127.0.0.1:5050"


@unittest.skipIf(server is None, "command console requirements not installed")
class ConsoleTests(unittest.TestCase):
    def make(self, responses, **robot):
        self.robot = FakeRobot(**robot)
        self.robot.connect()
        self.console = server.Console(ScriptedModel(responses), self.robot)
        self.client = server.make_app(self.console, 5050).test_client()

    def tearDown(self):
        if hasattr(self, "console"):
            self.console.close()

    def command(self, text, origin=PAGE):
        response = self.client.post("/api/command", json={"text": text}, base_url=PAGE,
                                    headers={"Origin": origin} if origin else {})
        return response.status_code, response.get_json()

    def test_a_move_passes_and_the_robot_walks(self):
        self.make([("Walking forward one metre.", [("move", {"direction": "forward", "distance_m": 1})])])
        code, card = self.command("walk forward 1 metre")
        self.assertEqual(code, 200)
        self.assertEqual(card["command"], "walk forward 1 metre")
        self.assertEqual(card["tool"], "move(direction=forward, distance_m=1)")
        self.assertEqual((card["valid"], card["validation"]), (True, "Passed: within limits"))
        self.assertEqual(card["response"], "Walking forward one metre.")
        self.assertTrue(wait_for(lambda: self.robot.x > .9, 10), "the fake robot walked the metre")
        status = self.client.get("/api/status", base_url=PAGE).get_json()
        self.assertEqual(set(status), {"state", "action", "speed", "note"})

    def test_a_blocked_move_is_shown_as_blocked_and_the_next_step_never_starts(self):
        self.make([("", [("move", {"direction": "forward", "distance_m": 2}), ("turn", {"direction": "left"})]),
                   "I can't go forward, something is right in front of me."], obstacles=[(.5, .9, -.5, .5)])
        _, card = self.command("walk forward 2 m and turn left")
        self.assertFalse(card["valid"])
        self.assertTrue(card["validation"].startswith("Blocked: "), card["validation"])
        self.assertEqual(card["tool"], "move(direction=forward, distance_m=2) then turn(direction=left)")
        self.assertIn("can't go forward", card["response"])

    def test_chat_without_a_tool_needs_no_robot_action(self):
        self.make(["Hi! I'm Go2."])
        _, card = self.command("hello")
        self.assertEqual((card["tool"], card["valid"], card["validation"]), (None, True, "No robot action"))

    def test_stop_halts_at_once_without_the_model(self):
        self.make([("Walking.", [("move", {"direction": "forward", "distance_m": 3})])])
        self.command("walk forward 3 m")
        self.assertTrue(wait_for(lambda: self.robot.x > .2, 10))
        self.client.post("/api/stop", base_url=PAGE, headers={"Origin": PAGE})
        self.assertTrue(wait_for(lambda: abs(self.robot.v[0]) < .05, 5))
        self.assertLess(self.robot.x, 2.5)

    def test_other_websites_cannot_drive_the_robot(self):
        self.make([("Walking.", [("move", {"direction": "forward", "distance_m": 1})])])
        self.assertEqual(self.command("walk forward", origin="https://example.com")[0], 403)
        rebound = self.client.post("/api/command", json={"text": "walk"}, base_url="http://evil.test:5050")
        self.assertEqual(rebound.status_code, 403)
        self.assertEqual(self.robot.x, 0.)

    def test_cards_for_the_other_outcomes(self):
        rows = [{"role": "tool", "name": "trick", "arguments": {"name": "dance1"},
                 "result": {"status": "needs_confirmation", "confirmation_id": "c1"}},
                {"role": "assistant", "content": "Shall I dance?"}]
        self.assertEqual(server.turn_card("dance", rows)["validation"], "Needs your confirmation: say yes to go ahead")
        rows = [{"role": "tool", "name": "move", "arguments": {"distance_m": 9}, "result": {"error": "distance too far"}}]
        self.assertEqual(server.turn_card("go far", rows)["validation"], "Rejected: distance too far")
        self.assertEqual(server.turn_card("hi", [], "model offline")["validation"], "Language model error")
        self.assertEqual(server.latest_update([{"role": "event", "content": "move forward 1 m done"},
                                               {"role": "assistant", "content": "Done!", "proactive": True}]),
                         "Go2: Done!")


if __name__ == "__main__":
    unittest.main()
