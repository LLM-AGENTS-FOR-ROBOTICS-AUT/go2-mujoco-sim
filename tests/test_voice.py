"""Voice: recognition clean-up, Go2's phrases as they are written, the console's voice endpoints, and a spoken
reply's first movement starting with Go2's voice (command_ui/server.py, go2/executor.py before_motion)."""
import struct
import sys
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "command_ui")]
from go2.agent import SpeechStream  # noqa: E402
from go2.executor import Executor  # noqa: E402
from go2.stt import clean  # noqa: E402
from go2.tools import resolve  # noqa: E402
from tests.fakes import FakeRobot, ScriptedModel, wait_for  # noqa: E402

try:
    import server
except ImportError:                     # flask not installed: the console's own requirements
    server = None

PAGE = "http://127.0.0.1:5050"
WALK = [("", [("move", {"direction": "forward", "distance_m": 1})]), "Walking forward one metre now!"]


class RecognitionTests(unittest.TestCase):
    def test_words_are_kept_and_what_whisper_invents_for_silence_is_not(self):
        said = {"segments": [{"text": " Walk forward two metres.", "no_speech_prob": .01, "avg_logprob": -.2}]}
        self.assertEqual(clean(said, rms=.05), "Walk forward two metres.")
        quiet = {"segments": [{"text": " Thank you.", "no_speech_prob": .3, "avg_logprob": -.4}]}
        self.assertEqual(clean(quiet, rms=.005), "")
        self.assertEqual(clean({"segments": [{"text": " [Music] ♪♪"}]}, rms=.05), "")
        noise = {"segments": [{"text": " Hmm.", "no_speech_prob": .9, "avg_logprob": -1.2}]}
        self.assertEqual(clean(noise, rms=.05), "")


class SpeechStreamTests(unittest.TestCase):
    def test_listeners_hear_each_phrase_and_the_end_of_the_turn(self):
        stream, heard = SpeechStream(), []
        stream.listeners.append(heard.append)
        stream.say(3, "On it!")
        stream.end(3)
        self.assertEqual([(i["epoch"], i.get("text"), bool(i.get("end"))) for i in heard],
                         [(3, "On it!", False), (3, None, True)])
        self.assertEqual(stream.after(0, timeout=0)["last"], 2)


class HoldTests(unittest.TestCase):
    """The executor asks before_motion once per task, right before the robot first moves; Stop cuts it short."""

    def setUp(self):
        self.executor = Executor()
        self.robot = FakeRobot()
        self.robot.connect()
        self.executor.attach(self.robot)
        self.gate, self.asked = threading.Event(), []

        def hold(task, cancel):
            self.asked.append(task.command.tool)
            while not self.gate.wait(.01) and not cancel.is_set():
                pass
        self.executor.before_motion = hold

    def tearDown(self):
        self.gate.set()
        self.executor.close()
        self.robot.close()

    def test_the_robot_waits_for_the_hold_then_walks(self):
        results = self.executor.submit(resolve("move", {"direction": "forward", "distance_m": .5}), "g", 1)
        self.assertEqual(results[0]["status"], "queued", "the tool call isn't kept waiting by the hold")
        time.sleep(.4)
        self.assertEqual([c for c in self.robot.calls if c[0] == "move"], [], "nothing moved while held")
        self.gate.set()
        self.assertTrue(wait_for(lambda: self.robot.x > .4, 10))
        self.assertEqual(self.asked, ["move"])

    def test_stop_ends_the_hold_and_nothing_moves(self):
        self.executor.submit(resolve("move", {"direction": "forward", "distance_m": .5}), "g", 1)
        self.assertTrue(wait_for(lambda: self.asked, 5))
        self.executor.stop("user")
        self.assertTrue(wait_for(lambda: self.executor.current is None, 2))
        self.assertEqual([c for c in self.robot.calls if c[0] == "move"], [])

    def test_a_trick_is_held_too(self):
        self.executor.submit(resolve("trick", {"name": "dance1"}), "g", 1)
        self.assertTrue(wait_for(lambda: self.asked, 5))
        time.sleep(.2)
        self.assertEqual([c for c in self.robot.calls if c[0] == "sport"], [])
        self.gate.set()
        self.assertTrue(wait_for(lambda: [c for c in self.robot.calls if c[0] == "sport"], 5))

    def test_standing_up_first_is_held_too(self):
        self.robot.posture = "sitting"
        self.executor.submit(resolve("move", {"direction": "forward", "distance_m": .3}), "g", 1)
        self.assertTrue(wait_for(lambda: self.asked, 5))
        self.assertEqual([c for c in self.robot.calls if c[0] == "sport"], [], "it doesn't stand up while held")
        self.gate.set()
        self.assertTrue(wait_for(lambda: self.robot.x > .25, 10))
        self.assertEqual(self.asked, ["move"], "asked once per task, not again for the walk after standing")


class FakeListener:
    def __init__(self):
        self.ready, self.error, self.heard = threading.Event(), None, []
        self.ready.set()

    def warm_up(self):
        pass

    def transcribe(self, audio, wanted=lambda: True):
        self.heard.append(len(audio))
        return "walk forward one metre"

    def status(self):
        return {"ready": True, "error": None, "model": "fake", "runs_on": "test"}


class FakeSpeaker:
    """Takes `delay` seconds per phrase, like Kokoro."""
    delay = .5

    def __init__(self):
        self.ready, self.error, self.made = threading.Event(), None, []
        self.ready.set()

    def warm_up(self):
        pass

    def wav(self, text, voice="af_heart", wanted=lambda: True):
        time.sleep(self.delay)
        self.made.append(text)
        return b"RIFF" + text.encode()

    def status(self):
        return {"ready": True, "error": None, "voices": {"af_heart": "Heart"}, "default": "af_heart"}


@unittest.skipIf(server is None, "command console requirements not installed")
class ConsoleVoiceTests(unittest.TestCase):
    def make(self, responses=(), voice=True, **robot):
        saved = server.Listener, server.Speaker
        server.Listener, server.Speaker = FakeListener, FakeSpeaker
        try:
            self.robot = FakeRobot(**robot)
            self.robot.connect()
            self.console = server.Console(ScriptedModel(list(responses)), self.robot, voice=voice)
        finally:
            server.Listener, server.Speaker = saved
        self.client = server.make_app(self.console, 5050).test_client()

    def tearDown(self):
        if hasattr(self, "console"):
            self.console.close()

    def post(self, path, value=None, **kw):
        response = self.client.post(path, json=value, base_url=PAGE, headers={"Origin": PAGE}, **kw)
        return response.status_code, response

    def moved_at(self, timeout=5):
        start = time.monotonic()
        wait_for(lambda: self.robot.x > .01, timeout, .005)
        return time.monotonic() - start

    def test_a_spoken_reply_moves_the_robot_as_go2_starts_talking(self):
        self.make(WALK)
        self.post("/api/messages", {"text": "walk forward one metre", "voice": True})
        moved = self.moved_at()
        self.assertGreater(moved, FakeSpeaker.delay * .8, "waited for the audio")
        self.assertLess(moved, server.VOICE_HOLD_S + .5)
        self.assertEqual(self.console.speaker.made, ["Walking forward one metre now!"])

    def test_typed_messages_move_at_once(self):
        self.make(WALK)
        self.post("/api/messages", {"text": "walk forward one metre"})
        self.assertLess(self.moved_at(), FakeSpeaker.delay * .8)
        self.assertEqual(self.console.speaker.made, [], "no audio made for a page that doesn't speak")

    def test_the_wait_is_bounded(self):
        FakeSpeaker.delay = 3
        try:
            self.make(WALK)
            self.post("/api/messages", {"text": "walk forward one metre", "speak": True})
            self.assertLess(self.moved_at(), server.VOICE_HOLD_S + .5)
        finally:
            FakeSpeaker.delay = .5

    def test_stop_never_waits_for_the_voice(self):
        self.make(WALK)
        self.post("/api/messages", {"text": "walk forward one metre", "voice": True})
        time.sleep(.1)
        started = time.monotonic()
        self.post("/api/stop", {})
        self.assertLess(time.monotonic() - started, .3)
        time.sleep(FakeSpeaker.delay + .5)
        self.assertLess(self.robot.x, .01, "it never started")

    def test_the_page_gets_the_audio_made_while_the_model_wrote(self):
        self.make(WALK)
        self.post("/api/messages", {"text": "walk forward one metre", "voice": True})
        code, response = self.post("/api/tts", {"text": "Walking forward one metre now!"})
        self.assertEqual((code, response.mimetype), (200, "audio/wav"))
        self.assertEqual(response.data, b"RIFFWalking forward one metre now!")
        self.assertEqual(self.console.speaker.made, ["Walking forward one metre now!"], "made once, not twice")

    def test_speech_arrives_phrase_by_phrase(self):
        self.make(["Hi there, I'm Go2!"])
        where = self.client.get("/api/speech?after=-1", base_url=PAGE).get_json()
        self.post("/api/messages", {"text": "hello"})
        got = self.client.get(f"/api/speech?after={where['last']}", base_url=PAGE).get_json()
        self.assertEqual(got["items"][0]["text"], "Hi there, I'm Go2!")

    def test_recognition_takes_raw_samples_and_refuses_anything_else(self):
        self.make()
        samples = struct.pack("<1600f", *([0.] * 1600))
        code, response = self.post("/api/stt", data=samples, content_type="application/octet-stream")
        self.assertEqual((code, response.get_json()), (200, {"text": "walk forward one metre"}))
        self.assertEqual(self.console.listener.heard, [1600])
        self.assertEqual(self.post("/api/stt", data=b"abc", content_type="application/octet-stream")[0], 400)
        too_long = b"\0" * (server.STT_MAX_BYTES + 4)
        self.assertEqual(self.post("/api/stt", data=too_long, content_type="application/octet-stream")[0], 413)

    def test_without_voice_the_page_is_told_to_use_its_own(self):
        self.make(voice=False)
        self.assertEqual(self.post("/api/tts", {"text": "Hi"})[0], 503)
        self.assertEqual(self.post("/api/stt", data=b"\0" * 8, content_type="application/octet-stream")[0], 503)
        self.assertEqual(self.client.get("/api/state", base_url=PAGE).get_json()["voice"], {"available": False})

    def test_an_unknown_voice_is_refused(self):
        self.make(WALK)
        code, response = self.post("/api/messages", {"text": "walk", "speak": "nobody"})
        self.assertEqual(code, 400)
        self.assertIn("Unknown voice", response.get_json()["error"])

    def test_interrupting_drops_the_reply_but_not_the_walk(self):
        gate = threading.Event()

        def slow_reply(messages):
            gate.wait(5)
            return "Walking forward one metre now!"
        self.make([WALK[0], slow_reply])
        self.post("/api/messages", {"text": "walk forward one metre"})
        self.assertTrue(wait_for(lambda: self.robot.x > .05, 5))
        self.post("/api/interrupt", {})
        gate.set()
        self.assertTrue(wait_for(lambda: self.robot.x > .9, 10), "the walk carried on")
        entries = self.client.get("/api/state", base_url=PAGE).get_json()["conversation"]["entries"]
        self.assertNotIn("reply", [e["kind"] for e in entries])
        self.assertEqual(entries[-1], {"id": entries[-1]["id"], "kind": "note", "text": "Reply interrupted."})

    def test_a_spoken_stop_halts_at_once_and_shows_what_was_said(self):
        self.make([("Walking.", [("move", {"direction": "forward", "distance_m": 3})])])
        self.post("/api/messages", {"text": "walk forward three metres"})
        self.assertTrue(wait_for(lambda: self.robot.x > .2, 10))
        self.post("/api/stop", {"said": "Stop!"})
        self.assertTrue(wait_for(lambda: abs(self.robot.v[0]) < .05, 3))
        entries = self.client.get("/api/state", base_url=PAGE).get_json()["conversation"]["entries"]
        self.assertEqual([(e["kind"], e["text"]) for e in entries[-2:]], [("user", "Stop!"), ("reply", "Okay, stopped.")])
        self.assertTrue(entries[-2]["voice"])

    def test_a_typed_stop_is_not_marked_as_spoken(self):
        self.make()
        self.post("/api/messages", {"text": "stop"})
        self.post("/api/messages", {"text": "stop", "voice": True})
        users = [e for e in self.client.get("/api/state", base_url=PAGE).get_json()["conversation"]["entries"]
                 if e["kind"] == "user"]
        self.assertEqual([u["voice"] for u in users], [False, True])


if __name__ == "__main__":
    unittest.main()
