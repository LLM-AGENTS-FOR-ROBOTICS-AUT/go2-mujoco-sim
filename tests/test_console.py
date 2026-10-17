"""The command console backend (command_ui/server.py): the page's cards and status, from the LLM agent."""
import json
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

try:
    import mujoco  # noqa: F401
    from go2.adapters.sim import SimAdapter
    HAVE_SIM = (ROOT / "_deps/unitree_mujoco/unitree_robots/go2/scene.xml").exists()
except ImportError:
    HAVE_SIM = False

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



@unittest.skipIf(server is None, "command console requirements not installed")
class ApiTests(unittest.TestCase):
    """The richer API for a new page (command_ui/API.md)."""

    def make(self, responses=(), **robot):
        self.robot = FakeRobot(**robot)
        self.robot.connect()
        self.console = server.Console(ScriptedModel(list(responses)), self.robot, {"scene": "test"})
        self.client = server.make_app(self.console, 5050).test_client()

    def tearDown(self):
        if hasattr(self, "console"):
            self.console.close()

    def get(self, path):
        return self.client.get(path, base_url=PAGE).get_json()

    def post(self, path, value=None):
        response = self.client.post(path, json=value or {}, base_url=PAGE, headers={"Origin": PAGE})
        return response.status_code, response.get_json()

    def entries(self):
        return self.get("/api/state")["conversation"]["entries"]

    def test_state_describes_the_robot_the_model_and_the_conversation(self):
        self.make()
        state = self.get("/api/state")
        self.assertEqual(set(state), {"revision", "robot", "model", "voice", "agent", "actions", "conversation", "camera"})
        robot = state["robot"]
        self.assertEqual((robot["kind"], robot["connected"], robot["requires_arming"], robot["armed"], robot["scene"]),
                         ("fake", True, False, False, "test"))
        self.assertEqual(set(robot["telemetry"]), {"x_m", "y_m", "heading_deg", "speed_mps", "turn_dps",
                                                   "battery_pct", "fault"})
        self.assertEqual(state["model"]["state"], "scripted")
        self.assertEqual(state["conversation"], {"id": 0, "entries": []})
        self.assertIsNone(state["camera"])
        self.assertEqual(self.get("/api/state")["revision"], state["revision"], "nothing changed, same revision")

    def test_a_message_returns_at_once_and_its_steps_update_live(self):
        self.make([("Walking forward.", [("move", {"direction": "forward", "distance_m": 1})])])
        code, accepted = self.post("/api/messages", {"text": "walk forward 1 metre"})
        self.assertEqual((code, accepted["entry"], accepted["conversation"]), (202, 0, 0))
        self.assertTrue(wait_for(lambda: [e["kind"] for e in self.entries()] == ["user", "call", "reply"], 10))
        user, call, reply = self.entries()
        self.assertEqual((call["tool"], call["outcome"], call["turn"], reply["turn"]), ("move", "queued", 0, 0))
        self.assertEqual(call["arguments"], {"direction": "forward", "distance_m": 1})
        self.assertIn(call["steps"][0]["status"], ("queued", "running"))
        self.assertTrue(wait_for(lambda: self.entries()[1]["steps"][0]["status"] == "completed", 15))
        self.assertEqual(self.get("/api/state")["actions"]["recent"][-1]["status"], "completed")

    def test_a_blocked_call_says_why(self):
        self.make([("", [("move", {"direction": "forward", "distance_m": 2})]), "Something is in the way."],
                  obstacles=[(.5, .9, -.5, .5)])
        self.post("/api/messages", {"text": "walk forward 2 m"})
        self.assertTrue(wait_for(lambda: any(e["kind"] == "reply" for e in self.entries()), 10))
        call = next(e for e in self.entries() if e["kind"] == "call")
        self.assertEqual(call["outcome"], "blocked")
        self.assertTrue(call["message"].startswith("Blocked: "), call["message"])
        self.assertEqual(call["steps"][0]["status"], "blocked")

    def test_empty_messages_are_refused(self):
        self.make()
        code, answer = self.post("/api/messages", {"text": "  "})
        self.assertEqual(code, 400)
        self.assertIn("error", answer)

    def test_clear_starts_a_new_conversation(self):
        self.make(["Hello!"])
        self.post("/api/messages", {"text": "hi"})
        self.assertTrue(wait_for(lambda: len(self.entries()) == 2, 10))
        code, state = self.post("/api/clear")
        self.assertEqual((code, state["conversation"]), (200, {"id": 1, "entries": []}))

    def test_reset_and_arming_only_where_they_apply(self):
        self.make()
        code, answer = self.post("/api/reset")
        self.assertEqual((code, answer["error"]), (400, "Reset is only available in the simulation."))
        self.assertEqual(self.post("/api/arm", {"armed": True})[0], 400)
        self.console.close()
        self.make(requires_arming=True)
        self.assertEqual(self.post("/api/arm", {"armed": "yes"})[0], 400)
        code, state = self.post("/api/arm", {"armed": True})
        self.assertEqual((code, state["robot"]["armed"]), (200, True))
        self.assertFalse(self.post("/api/arm", {"armed": False})[1]["robot"]["armed"])

    def test_events_stream_the_state(self):
        self.make()
        stream = self.console.events(interval=.01)
        first = next(stream)
        self.assertTrue(first.startswith("event: state\ndata: "))
        self.assertEqual(json.loads(first.split("data: ", 1)[1])["robot"]["kind"], "fake")
        self.console.close()
        self.assertEqual(list(stream), [], "the stream ends when the console closes")

    def test_messages_are_refused_at_once_while_the_model_downloads(self):
        import time
        self.make()
        self.console.model_info.update(state="downloading", progress={"done_mb": 1500, "total_mb": 3416})
        started = time.monotonic()
        code, answer = self.post("/api/messages", {"text": "walk forward"})
        self.assertLess(time.monotonic() - started, 1, "refused straight away, not after a wait")
        self.assertEqual(code, 503)
        self.assertIn("Still downloading the language model", answer["error"])
        self.assertIn("44%, 1.5 of 3.4 GB", answer["error"])
        self.assertEqual(answer["model"]["state"], "downloading")
        self.assertEqual(self.entries(), [])
        card = self.client.post("/api/command", json={"text": "walk forward"}, base_url=PAGE,
                                headers={"Origin": PAGE}).get_json()
        self.assertEqual((card["valid"], card["validation"]), (False, "Language model not ready"))

    def test_stop_still_works_while_the_model_downloads(self):
        self.make()
        self.console.model_info.update(state="downloading", progress={"done_mb": 10, "total_mb": 3416})
        code, _ = self.post("/api/messages", {"text": "stop"})
        self.assertEqual(code, 202)
        self.assertEqual(self.entries()[-1]["text"], "Okay, stopped.")

    def test_a_model_that_failed_to_load_says_why(self):
        self.make()
        self.console.model_info.update(state="unavailable", error="Ollama is not installed.")
        code, answer = self.post("/api/messages", {"text": "hello"})
        self.assertEqual((code, answer["error"]), (503, "The language model isn't available: Ollama is not installed."))

    def test_a_message_sent_while_the_model_loads_waits_for_it(self):
        import threading
        self.make(["Hi!"])
        self.console.model_info["state"] = "loading"
        self.console.loaded.clear()

        def finish():
            self.console.model_info["state"] = "ready"
            self.console.loaded.set()
        threading.Timer(.3, finish).start()
        code, _ = self.post("/api/messages", {"text": "hello"})
        self.assertEqual(code, 202)
        self.assertTrue(wait_for(lambda: self.entries()[-1].get("text") == "Hi!", 10))

    def test_entries_mark_go2s_own_turns_and_every_outcome(self):
        rows = [{"role": "user", "content": "dance", "turn": {}},
                {"role": "tool", "name": "acrobatic", "arguments": {"name": "front_flip"},
                 "result": {"status": "needs_confirmation", "confirmation_id": "c1"}},
                {"role": "assistant", "content": "Are you sure?"},
                {"role": "event", "content": "move forward 1 m done", "turn": {}},
                {"role": "assistant", "content": "Done!", "proactive": True},
                {"role": "tool", "name": "move", "arguments": {}, "result": {"error": "distance too far"}},
                {"role": "system", "content": "Reply interrupted."}]
        entries = server.conversation_entries(rows, {})
        self.assertEqual([e["kind"] for e in entries], ["user", "call", "reply", "update", "reply", "call", "note"])
        self.assertEqual((entries[1]["outcome"], entries[1]["turn"]), ("needs_confirmation", 0))
        self.assertEqual((entries[4]["turn"], entries[4]["proactive"]), (None, True))
        self.assertEqual((entries[5]["outcome"], entries[5]["message"]), ("rejected", "Rejected: distance too far"))
        self.assertEqual(entries[3]["text"], "move forward 1 m done")
        self.assertEqual(server.for_people("move forward 4 m stopped early: obstacle ahead after 0.54 of 4.0 m. "
                                           "Tell the user briefly what happened."),
                         "move forward 4 m stopped early: obstacle ahead after 0.54 of 4.0 m.")
        self.assertEqual(server.outcome({"results": [{"status": "queued", "task_id": "t1"},
                                                     {"status": "already_queued"}]}), "already_queued")


@unittest.skipIf(server is None, "command console requirements not installed")
class CleanupTests(unittest.TestCase):
    def test_leftover_model_folders_are_removed_but_other_processes_are_left_alone(self):
        import os
        import shutil
        import tempfile
        root = Path(tempfile.mkdtemp())
        folder = root / "runtime" / "model-test"
        folder.mkdir(parents=True)
        (folder / "server.pid").write_text(str(os.getpid()))     # this test runner: not an Ollama process
        saved = server.ROOT
        server.ROOT = root
        try:
            server.stop_stale_model_servers()
        finally:
            server.ROOT = saved
            shutil.rmtree(root)
        self.assertFalse(folder.exists())


@unittest.skipIf(server is None or not HAVE_SIM, "command console or simulation requirements not installed")
class CameraTests(unittest.TestCase):
    """The camera endpoint, on the real (headless) room simulation."""

    def test_after_a_look_the_frame_and_its_objects_are_served(self):
        adapter = SimAdapter(scene="room", headless=True)
        adapter.connect()
        console = server.Console(ScriptedModel([]), adapter, {"scene": "room"})
        try:
            client = server.make_app(console, 5050).test_client()
            self.assertEqual(client.get("/api/camera.png", base_url=PAGE).status_code, 404)
            console.executor.look()
            image = client.get("/api/camera.png", base_url=PAGE)
            self.assertEqual((image.status_code, image.data[:8]), (200, b"\x89PNG\r\n\x1a\n"))
            camera = client.get("/api/state", base_url=PAGE).get_json()["camera"]
            self.assertTrue(camera["url"].startswith("/api/camera.png?t="))
            self.assertTrue(camera["boxes"], "the room has objects in view")
            box = camera["boxes"][0]["box"]
            self.assertTrue(0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1, box)
        finally:
            console.close()


if __name__ == "__main__":
    unittest.main()
