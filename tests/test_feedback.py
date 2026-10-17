"""The page's "bad reply" button (command_ui/server.py Console.flag): the turn behind a reply is saved in
feedback/ in the format the training run replays."""
import json
import sys
import tempfile
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
class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.robot = FakeRobot()
        self.robot.connect()
        self.console = server.Console(ScriptedModel([
            ("", [("turn", {"direction": "right"})]), "Turning right!",
        ]), self.robot, {"scene": "room"}, feedback=self.folder)
        self.client = server.make_app(self.console, 5050).test_client()

    def tearDown(self):
        self.console.close()

    def post(self, path, value):
        response = self.client.post(path, json=value, base_url=PAGE, headers={"Origin": PAGE})
        return response.status_code, response.get_json()

    def entries(self):
        return self.client.get("/api/state", base_url=PAGE).get_json()["conversation"]["entries"]

    def reply(self):
        self.assertTrue(wait_for(lambda: any(e["kind"] == "reply" for e in self.entries()), 10))
        return next(e for e in self.entries() if e["kind"] == "reply")

    def test_a_flagged_reply_is_saved_in_the_format_training_replays(self):
        self.post("/api/messages", {"text": "turn left"})
        reply = self.reply()
        self.assertEqual((reply["flaggable"], reply["flag"]), (True, None))
        code, saved = self.post("/api/feedback", {"entry": reply["id"]})
        self.assertEqual(code, 201)
        self.assertTrue(saved["path"].endswith(f"{saved['id']}.json"))
        record = json.loads((self.folder / f"{saved['id']}.json").read_text())
        self.assertEqual(record["kind"], "user")
        self.assertEqual(record["input"]["content"].split("User: ")[-1], "turn left")
        self.assertEqual([m["role"] for m in record["bad_turn"]], ["assistant", "tool", "assistant"])
        self.assertEqual(record["bad_turn"][-1]["content"], "Turning right!")
        self.assertEqual(record["note"], "")
        self.assertEqual(record["robot"]["target"], "fake")
        self.assertEqual(set(record["robot"]), {"target", "scene", "detector", "x", "y", "yaw", "posture", "doing",
                                                "queued", "armed"})
        self.assertEqual(self.reply()["flag"], saved["id"])

    def test_a_note_is_added_to_the_same_file(self):
        self.post("/api/messages", {"text": "turn left"})
        _, saved = self.post("/api/feedback", {"entry": self.reply()["id"]})
        code, _ = self.post("/api/feedback", {"id": saved["id"], "note": "  It should have turned left.  "})
        self.assertEqual(code, 200)
        record = json.loads((self.folder / f"{saved['id']}.json").read_text())
        self.assertEqual(record["note"], "It should have turned left.")
        self.assertEqual(list(self.folder.glob("*")), [self.folder / f"{saved['id']}.json"], "one file, no leftovers")

    def test_only_replies_the_model_wrote_can_be_flagged(self):
        self.post("/api/messages", {"text": "turn left"})
        reply = self.reply()
        self.assertEqual(self.post("/api/feedback", {"entry": 0})[1], {"error": "Pick one of Go2's replies."})
        self.post("/api/stop", {})
        stopped = [e for e in self.entries() if e["kind"] == "reply"][-1]
        self.assertEqual((stopped["text"], stopped["flaggable"]), ("Okay, stopped.", False))
        self.assertEqual(self.post("/api/feedback", {"entry": stopped["id"]})[0], 400)
        for bad in ({"entry": "x"}, {"entry": True}, {}, {"id": "../../etc/passwd", "note": "x"},
                    {"id": "20261018-111503-000000", "note": "x"}):
            self.assertEqual(self.post("/api/feedback", bad)[0], 400, bad)
        self.assertTrue(reply["flaggable"])
        self.assertEqual(list(self.folder.glob("*")), [])


if __name__ == "__main__":
    unittest.main()
