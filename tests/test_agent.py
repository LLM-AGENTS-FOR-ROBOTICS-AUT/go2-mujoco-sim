import json
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from go2.agent import Agent, clean_reply, describe_outcome, is_stop  # noqa: E402
from go2.executor import Executor  # noqa: E402
from go2.llm import recover_calls  # noqa: E402
from go2.tools import TOOL_NAMES  # noqa: E402
from tests.fakes import FakeRobot, ScriptedModel, wait_for  # noqa: E402


class AgentTests(unittest.TestCase):
    def make(self, responses, robot=True, **robot_kwargs):
        self.robot = FakeRobot(**robot_kwargs) if robot else None
        self.agent = None
        self.executor = Executor(on_event=lambda task: self.agent and self.agent.on_task_event(task))
        if self.robot:
            self.robot.connect()
            self.executor.attach(self.robot)
        self.model = ScriptedModel(responses)
        self.agent = Agent(self.model, self.executor)
        return self.agent

    def tearDown(self):
        self.agent.close()
        self.executor.close()
        if self.robot:
            self.robot.close()

    def say(self, text):
        self.agent.submit(text)
        self.assertTrue(wait_for(lambda: not self.agent.busy, 10))

    def replies(self):
        return [row["content"] for row in self.agent.transcript if row["role"] == "assistant"]

    def tools(self):
        return [(row["name"], row["result"]) for row in self.agent.transcript if row["role"] == "tool"]

    def test_plain_chat(self):
        self.make(["Woof, hi! I'm Go2."])
        self.say("Hi!")
        self.assertEqual(self.replies(), ["Woof, hi! I'm Go2."])
        self.assertEqual(self.tools(), [])

    def test_tool_loop_runs_a_sequence_across_model_steps(self):
        self.make([("", [("move", {"direction": "forward", "distance_m": .5})]),
                   ("", [("posture", {"name": "sit"})]),
                   "Walking half a metre, then sitting!"])
        self.say("walk half a metre then sit")
        self.assertEqual([name for name, _ in self.tools()], ["move", "posture"])
        self.assertTrue(all(result["status"] == "queued" for _, result in self.tools()))
        self.assertEqual(self.replies()[0], "Walking half a metre, then sitting!")
        tool_messages = [m for m in self.agent.messages if m["role"] == "tool"]
        self.assertEqual(len(tool_messages), 2)

    def test_split_plan_go_ahead_is_ephemeral(self):
        def after_nudge(messages):
            assert messages[-1]["role"] == "user" and messages[-1]["content"].startswith("Okay, do it: move forward 4 m")
            return ("", [("move", {"direction": "forward", "distance_m": 4})])
        self.make([("", [("move", {"direction": "forward", "distance_m": 10})]),
                   "I'll do that in three steps!",                       # talks instead of calling
                   after_nudge,
                   ("", [("move", {"direction": "forward", "distance_m": 4})]),
                   ("", [("move", {"direction": "forward", "distance_m": 2})]),
                   "Off I go, ten metres!"])
        self.say("walk forward 10 metres")
        results = self.tools()
        self.assertIn("error", results[0][1])
        self.assertEqual([r["status"] for _, r in results[1:]], ["queued"] * 3)
        self.assertFalse(any(m["content"].startswith("Okay, do it") for m in self.agent.messages if m["role"] == "user"),
                         "the go-ahead must not be stored in history")
        self.assertEqual(self.replies()[-1], "Off I go, ten metres!")
        self.executor.stop()

    def test_empty_reply_after_calls_asks_for_one_sentence(self):
        self.make([("", [("turn", {})]), "", "Turning left!"])
        self.say("turn")
        self.assertEqual(self.replies(), ["Turning left!"])

    def test_repeated_rejected_call_ends_the_loop(self):
        bad = ("", [("move", {"direction": "up"})])
        self.make([bad, bad, bad, bad, "Sorry, I can't do that."])
        self.say("go up")
        self.assertEqual(len(self.tools()), 2)

    def test_stop_word_bypasses_the_model(self):
        self.make([])
        self.say("Stop!")
        self.assertEqual(self.model.requests, [])
        self.assertEqual(self.replies(), ["Okay, stopped."])
        self.assertTrue(is_stop("STOP."))
        self.assertTrue(is_stop("stop moving"))
        self.assertTrue(is_stop("Cancel that, stay where you are"))
        self.assertTrue(is_stop("Don't move!"))
        self.assertFalse(is_stop("don't stop believing"))
        self.assertFalse(is_stop("walk forward then stop"))
        self.assertFalse(is_stop("stop by the kitchen and then walk to the table and sit down"))

    def test_interrupt_discards_the_late_reply(self):
        def slow(messages):
            time.sleep(.5)
            return "late reply"
        self.make([slow])
        self.agent.submit("tell me a story")
        time.sleep(.1)
        self.agent.interrupt()
        time.sleep(.7)
        self.assertNotIn("late reply", self.replies())
        self.assertFalse(self.agent.busy)

    def test_robot_news_reaches_the_model_with_the_next_message(self):
        self.make(["Hello!", "Got it."])
        self.agent.note("MuJoCo simulation connected and ready")
        self.say("hi")
        first_user = [m for m in self.model.requests[0] if m["role"] == "user"][-1]["content"]
        self.assertIn("Robot: MuJoCo simulation connected and ready", first_user)
        self.assertTrue(first_user.endswith("User: hi"))
        self.assertEqual(self.model.requests[0][0]["role"], "system")
        self.say("again")
        self.assertEqual(self.model.requests[0][0], self.model.requests[1][0], "system prompt stays constant for caching")

    def test_compact_prompt_for_fine_tuned_models(self):
        from go2.agent import COMPACT_SYSTEM
        from go2.tools import COMPACT_TOOLS
        self.make(["Hi!"])
        self.model.prompt_style = "compact"
        captured = {}
        original = self.model.chat
        def chat(messages, tools=None, **kw):
            captured["system"], captured["tools"] = messages[0]["content"], tools
            return original(messages, tools, **kw)
        self.model.chat = chat
        self.say("hello")
        self.assertEqual(captured["system"], COMPACT_SYSTEM)
        self.assertEqual(captured["tools"], COMPACT_TOOLS)
        self.assertTrue(all(not t["function"]["parameters"]["properties"] for t in captured["tools"]))

    def test_blocked_motion_is_reported_proactively(self):
        self.make([("", [("move", {"direction": "forward", "distance_m": 2})]), "Heading forward!",
                   "Whoa, I stopped just short of something in front of me."], obstacles=[(1.2, 1.6, -.5, .5)])
        self.say("walk forward 2 m")
        self.assertTrue(wait_for(lambda: len(self.replies()) >= 2, 15))
        events = [row for row in self.agent.transcript if row["role"] == "event"]
        self.assertTrue(events and "stopped early" in events[0]["content"])
        self.assertTrue(self.agent.transcript[-1].get("proactive"))

    def test_report_arriving_mid_reply_is_spoken_when_the_reply_ends(self):
        def slow_reply(messages):
            time.sleep(2.)                  # the block lands while Go2 is still answering
            return "Heading forward!"
        self.make([("", [("move", {"direction": "forward", "distance_m": 2})]), slow_reply,
                   "Whoa, I stopped just short of something in front of me."], obstacles=[(.9, 1.3, -.5, .5)])
        self.say("walk forward 2 m")
        self.assertTrue(wait_for(lambda: len(self.replies()) >= 2 and not self.agent.busy, 10))
        self.assertEqual(self.replies(), ["Heading forward!", "Whoa, I stopped just short of something in front of me."])
        roles = [row["role"] for row in self.agent.transcript]
        self.assertLess(roles.index("assistant"), roles.index("event"), "announced after the reply, not before")
        self.assertIn("stopped early", self.robot_updates()[0])
        self.assertTrue(self.agent.transcript[-1].get("proactive"))

    def test_app_notices_still_wait_for_the_next_message(self):
        self.make(["Hi!"])
        self.agent.note("MuJoCo simulation connected and ready")
        time.sleep(.3)
        self.assertEqual(self.replies(), [], "a notice alone never makes Go2 speak")
        self.say("hello")
        self.assertIn("simulation connected", self.agent.messages[0]["content"])

    def test_motion_refused_at_the_start_is_reported_in_the_tool_result(self):
        self.make([("", [("move", {"direction": "forward", "distance_m": 2}), ("turn", {"direction": "left"})]),
                   "I can't go forward, something is right in front of me."], obstacles=[(.5, .9, -.5, .5)])
        self.say("walk forward 2 m and turn left")
        (_, move), (_, turn) = self.tools()
        self.assertEqual(move["status"], "blocked")
        self.assertIn("path is blocked", move["error"])
        self.assertEqual(turn, {"status": "cancelled", "reason": "previous step blocked"})
        self.assertEqual([t.command.tool for t in self.executor.history], ["move"], "the turn never started")
        time.sleep(.5)
        self.assertFalse([row for row in self.agent.transcript if row["role"] == "event"], "not announced twice")
        self.assertEqual(len(self.replies()), 1)

    def robot_updates(self):
        return [m["content"] for m in self.agent.messages
                if m["role"] == "user" and m["content"].startswith("Robot:") and "User:" not in m["content"]]

    def test_robot_update_says_what_other_requests_still_have_queued(self):
        fwd, back = {"direction": "forward", "distance_m": .2}, {"direction": "backward", "distance_m": .2}
        self.make([("", [("move", fwd), ("move", fwd)]), "Walking forward twice!",
                   ("", [("move", back)]), "I'll back up after that!",
                   "Done going forward, backing up now!",
                   ("", [("move", {"direction": "left", "distance_m": .2}), ("move", {"direction": "right", "distance_m": .2})]),
                   "Shuffling left and right!", "All done!"])
        self.say("move forward twice")
        self.say("move back")                       # queued behind the forward pair
        self.assertTrue(wait_for(lambda: len(self.robot_updates()) >= 1, 15))
        first = self.robot_updates()[0]
        self.assertIn("finished all steps: move forward 0.2 m", first)
        self.assertIn("; move backward 0.2 m at 0.20 m/s (starting now) is already queued. Tell the user briefly.", first)
        self.assertTrue(wait_for(lambda: not self.executor.snapshot()["current"] and not self.executor.queue
                                 and not self.agent.busy, 15))
        self.say("shuffle left then right")         # nothing else queued this time
        self.assertTrue(wait_for(lambda: len(self.robot_updates()) >= 2, 15))
        self.assertNotIn("already queued", self.robot_updates()[-1])
        self.assertTrue(self.robot_updates()[-1].endswith("Tell the user briefly."))

    def test_robot_update_never_repeats_queued_work(self):
        fwd = {"direction": "forward", "distance_m": .3}
        self.make([("", [("move", fwd), ("move", fwd)]), "Walking forward twice!",
                   ("", [("trick", {"name": "dance1"})]), "I'll dance after that!",
                   ("", [("trick", {"name": "dance1"})]), "Done walking, dancing now!"])
        self.say("move forward twice")
        self.say("dance after that")                # queued behind the forward pair
        self.assertTrue(wait_for(lambda: len(self.robot_updates()) >= 1 and not self.agent.busy, 15))
        self.assertIn("; dance1 (starting now) is already queued.", self.robot_updates()[0])
        self.assertEqual(self.tools()[-1], ("trick", {"status": "already_queued", "action": "dance1"}))
        self.assertTrue(wait_for(lambda: not self.executor.snapshot()["current"] and not self.executor.queue, 15))
        self.assertEqual([t.command.tool for t in self.executor.history].count("trick"), 1, "danced once")

    def test_blocked_update_still_announces_other_queued_requests(self):
        self.make([("", [("move", {"direction": "forward", "distance_m": 3, "speed_mps": .6})]), "Walking forward!",
                   ("", [("turn", {"direction": "left"})]), "Then I'll turn left!",
                   "I stopped before something in front of me, turning left now."],
                  obstacles=[(1.4, 1.8, -.5, .5)])
        self.say("walk forward 3 m")
        self.say("then turn left")
        self.assertTrue(wait_for(lambda: len(self.robot_updates()) >= 1, 15))
        update = self.robot_updates()[0]
        self.assertIn("stopped early", update)
        self.assertIn("; turn left 90° at 29°/s (starting now) is already queued. Tell the user briefly what happened.", update)

    def test_no_robot_tool_error_is_returned_to_the_model(self):
        self.make([("", [("move", {"direction": "forward"})]), "I'm not connected to a body yet!"], robot=False)
        self.say("walk")
        self.assertEqual(self.tools()[0][1]["status"], "rejected")


class HelperTests(unittest.TestCase):
    def test_clean_reply(self):
        self.assertEqual(clean_reply("Robot: Walking now!"), "Walking now!")
        self.assertEqual(clean_reply("[On it!]"), "On it!")
        self.assertEqual(clean_reply("<|tool_response> Sure"), "Sure")
        self.assertEqual(clean_reply("Robots are neat"), "Robots are neat")

    def test_recover_gemma_text_calls(self):
        calls, rest = recover_calls('Sure! move{direction:<|"|>backward<|"|>,distance_m:0.5}', TOOL_NAMES)
        self.assertEqual(calls[0]["arguments"], {"direction": "backward", "distance_m": .5})
        self.assertEqual(rest, "Sure!")
        self.assertEqual(recover_calls("dance{fast}", TOOL_NAMES)[0], [])

    def test_outcome_descriptions(self):
        class T:
            def __init__(self, status, summary, detail):
                self.status, self.detail = status, detail
                self.command = type("C", (), {"summary": summary, "tool": "move"})()
        blocked = T("blocked", "move forward 2 m", {"moved_m": .4, "target_m": 2, "blocked": {"reason": "obstacle ahead", "clearance_m": .2}, "cancelled_after": 1})
        text = describe_outcome(blocked, [blocked])
        self.assertIn("without touching anything", text)
        self.assertIn("after 0.4 of 2 m", text)
        self.assertIn("1 later step", text)
        done = [T("completed", "move forward 1 m", {}), T("completed", "sit", {})]
        self.assertIn("move forward 1 m then sit", describe_outcome(done[-1], done))
        json.dumps(text)


if __name__ == "__main__":
    unittest.main()


class SpeechStreamingTests(unittest.TestCase):
    """Go2's words are handed to the voice phrase by phrase while the model is still writing."""

    def test_phrases_are_cut_at_sentence_ends_and_long_clauses_only(self):
        from go2.agent import Phrases
        out = []
        phrases = Phrases(out.append)
        text = "On it, heading over to the blue ball now! I had to stop after about 1.5 metres, because something is in the way."
        for i in range(1, len(text) + 1):
            phrases.feed(text[:i])
        phrases.finish(text)
        self.assertEqual(out, ["On it, heading over to the blue ball now!", "I had to stop after about 1.5 metres,",
                               "because something is in the way."])

    def test_the_first_phrase_is_cut_sooner_so_go2_starts_talking_sooner(self):
        from go2.agent import Phrases
        out = []
        phrases = Phrases(out.append)
        text = "I'm your robot dog companion, ready to explore and follow your lead."
        for i in range(1, len(text) + 1):
            phrases.feed(text[:i])
        phrases.finish(text)
        self.assertEqual(out, ["I'm your robot dog companion,", "ready to explore and follow your lead."])

    def test_tool_calls_written_as_text_are_never_spoken(self):
        from go2.agent import Phrases
        out = []
        phrases = Phrases(out.append)
        for i in range(1, 26):
            phrases.feed('move{direction:"forward"}'[:i])
        self.assertEqual(out, [])

    def test_stream_waits_for_new_words_and_reports_where_it_is(self):
        import threading
        from go2.agent import SpeechStream
        stream = SpeechStream()
        self.assertEqual(stream.after(-1), {"last": 0, "items": []})
        threading.Timer(.1, lambda: stream.say(3, "Hello there!")).start()
        got = stream.after(0, timeout=2)
        self.assertEqual([i["text"] for i in got["items"]], ["Hello there!"])
        stream.end(3)
        self.assertTrue(stream.after(got["last"], timeout=1)["items"][0]["end"])
