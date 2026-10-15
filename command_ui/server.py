# -*- coding: utf-8 -*-
"""Go2 command console - backend.

The web page (static/index.html) sends a typed or spoken command. The LLM agent (go2/agent.py: Gemma 4 E2B
fine-tuned as go2-ft-04, run by a private Ollama) answers it and calls bounded robot tools; the executor
(go2/executor.py) checks every call against the limits and the robot's own sensors before anything moves.
The robot is the MuJoCo room simulation in its own window (simulation/), or the real Go2 EDU with --robot.
Voice uses faster-whisper locally (offline): the mic button records 5 seconds here.

  .venv/bin/python command_ui/server.py              (macOS; Windows: .venv\\Scripts\\python.exe command_ui\\server.py)
  ... --mock                  scripted replies, no language model (checks the page and the robot)
  ... --robot en7 [--arm]     the real Go2 on that network interface; it only moves once armed
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from flask import Flask, abort, jsonify, request, send_from_directory  # noqa: E402

from go2.adapters.base import AdapterError  # noqa: E402
from go2.adapters.sim import SimAdapter  # noqa: E402
from go2.agent import Agent, robot_state_line  # noqa: E402
from go2.executor import Executor  # noqa: E402
from go2.llm import CATALOG, ModelError, OllamaModel  # noqa: E402
from go2.mock_model import MockModel  # noqa: E402

VOICE_SECONDS = 5
SR = 16000
TURN_TIMEOUT_S = 120
PROBLEMS = ("blocked", "failed", "rejected", "cancelled", "needs_confirmation")


# ---- the robot, the executor and the agent ----------------------------------------------------------------
class Console:
    def __init__(self, model, adapter):
        self.model, self.adapter = model, adapter
        self.executor = Executor(on_event=lambda task: self.agent.on_task_event(task))
        self.agent = Agent(model, self.executor)
        self.agent.robot_line = lambda: robot_state_line(self.adapter, self.executor)
        self.executor.attach(adapter)
        self.commands = threading.Lock()        # the page shows one card per command, in order
        self.shown = 0                          # transcript rows already shown on a card
        self.loaded = threading.Event()         # set once the model has loaded (or failed to)
        self.note = ""
        if isinstance(model, OllamaModel):
            self.note = "Loading the language model..."
            threading.Thread(target=self._warm_up, daemon=True).start()
        else:
            self.loaded.set()

    def _warm_up(self):
        """Load the model in the background so the first command doesn't wait for it."""
        try:
            self.model.check()
            self.model.chat([{"role": "user", "content": "Hello"}])
            self.note = ""
        except Exception as error:              # missing weights, no Ollama: shown under the Stop button
            self.note = f"Language model unavailable: {error}"
        finally:
            self.loaded.set()

    def command(self, text, voice=False):
        """One command -> the card the page shows: tool, safety result and Go2's reply."""
        with self.commands:
            self.loaded.wait(TURN_TIMEOUT_S)    # one model request at a time: not during the warm-up
            start = len(self.agent.snapshot()["transcript"])
            try:
                self.agent.submit(text, voice)
            except ValueError as error:
                return card(text, None, False, f"Rejected: {error}", str(error))
            deadline = time.monotonic() + TURN_TIMEOUT_S
            while self.agent.busy and time.monotonic() < deadline:
                time.sleep(.05)
            snapshot = self.agent.snapshot()
            rows = snapshot["transcript"][start:]
            self.shown = len(snapshot["transcript"])
            if snapshot["busy"]:
                self.agent.interrupt()
                return card(text, None, False, "No answer", "The model took too long to answer. Try again.")
            return turn_card(text, rows, snapshot["error"])

    def stop(self):
        return self.agent.stop()

    def status(self):
        snapshot = self.agent.snapshot()
        executor = self.executor.snapshot()
        if not self.adapter.connected:
            return {"state": "disconnected", "action": "—", "speed": 0.0,
                    "note": "The robot is not connected. Restart the console."}
        try:
            s = self.adapter.state()
            posture, speed = s.posture.replace("_", " "), s.speed
        except AdapterError as error:
            posture, speed = "unknown", 0.0
            self.note = str(error)
        current = executor["current"]
        if current:
            action = current["summary"]
        elif executor["recent"]:
            last = executor["recent"][-1]
            action = f"{last['summary']} ({last['status']})"
        else:
            action = "—"
        state = "thinking" if snapshot["busy"] and not current else posture
        return {"state": state, "action": action, "speed": round(float(speed), 2),
                "note": self.note or latest_update(snapshot["transcript"][self.shown:])}

    def close(self):
        self.agent.close()
        self.executor.close()
        for stop in (self.adapter.close, getattr(self.model, "unload", None)):
            try:
                if stop:
                    stop()
            except Exception:
                pass


# ---- cards: the agent's turn in the page's format ----------------------------------------------------------
def card(command, tool, valid, validation, response):
    return {"command": command, "tool": tool, "valid": valid, "validation": validation, "response": response,
            "exec": None}


def describe_call(row):
    args = ", ".join(f"{k}={v}" for k, v in (row.get("arguments") or {}).items())
    return f"{row['name']}({args})"


def problem(result):
    """Why a tool call didn't happen, or None if it went ahead (or was only queued)."""
    for r in [result, *result.get("results", [])]:
        status = r.get("status")
        if status == "needs_confirmation":
            return "Needs your confirmation: say yes to go ahead"
        if "error" in r or status in PROBLEMS:
            reason = r.get("error") or r.get("reason") or status
            verb = {"blocked": "Blocked", "failed": "Failed", "cancelled": "Not started"}.get(status, "Rejected")
            return f"{verb}: {reason}"
    return None


def turn_card(command, rows, error=None):
    calls = [r for r in rows if r["role"] == "tool"]
    replies = [r["content"] for r in rows if r["role"] == "assistant" and r.get("content")]
    problems = [p for p in (problem(r.get("result") or {}) for r in calls) if p]
    tool = " then ".join(describe_call(r) for r in calls) or None
    if error:
        return card(command, tool, False, "Language model error", error)
    if problems:
        validation = problems[0]
    else:
        validation = "Passed: within limits" if calls else "No robot action"
    return card(command, tool, not problems, validation, " ".join(replies) or "Okay.")


def latest_update(rows):
    """What happened since the last card (a move that finished or was blocked, Go2's own report on it)."""
    for row in reversed(rows):
        if row["role"] == "assistant" and row.get("proactive"):
            return "Go2: " + row["content"]
        if row["role"] == "event":
            return "Robot: " + row["content"]
    return ""


# ---- voice (lazy-loaded so the console starts fast) --------------------------------------------------------
_whisper = None
_whisper_lock = threading.Lock()


def get_whisper():
    global _whisper
    with _whisper_lock:
        if _whisper is None:
            from faster_whisper import WhisperModel
            _whisper = WhisperModel("base", device="cpu", compute_type="int8")
    return _whisper


def transcribe_from_mic():
    import sounddevice as sd
    audio = sd.rec(int(VOICE_SECONDS * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    segs, _ = get_whisper().transcribe(audio.flatten(), language="en", beam_size=1)
    return " ".join(s.text.strip() for s in segs).strip()


# ---- web server ------------------------------------------------------------------------------------------
def make_app(console, port):
    app = Flask(__name__, static_folder=str(HERE / "static"))
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    @app.before_request
    def same_origin():
        # Only this page may drive the robot: no other website (cross-site POST) or rebound domain name.
        origin = request.headers.get("Origin")
        if request.host not in hosts or (origin is not None and origin.split("//", 1)[-1] not in hosts):
            abort(403)

    @app.route("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.route("/api/command", methods=["POST"])
    def api_command():
        text = (request.get_json(force=True, silent=True) or {}).get("text", "")
        return jsonify(console.command(str(text)))

    @app.route("/api/voice", methods=["POST"])
    def api_voice():
        try:
            text = transcribe_from_mic()
        except Exception as exc:
            return jsonify(card("(voice)", None, False, "Voice error", str(exc)))
        if not text:
            return jsonify(card("(heard nothing)", None, False, "Rejected: no speech detected",
                                "I didn't catch that. Try again."))
        return jsonify(console.command(text, voice=True))

    @app.route("/api/stop", methods=["POST"])
    def api_stop():
        console.stop()
        return jsonify({"ok": True})

    @app.route("/api/status")
    def api_status():
        return jsonify(console.status())

    return app


def connect_robot(args):
    if args.robot:
        from go2.adapters.go2_sdk import Go2Adapter
        adapter = Go2Adapter(args.robot)
    else:
        adapter = SimAdapter(scene=args.scene, headless=args.headless)
    adapter.connect()
    return adapter


def check(args):
    """Setup check: the simulator starts and reports its state; the model file is there and verified."""
    adapter = SimAdapter(scene="room", headless=True)
    try:
        adapter.connect()
        state = adapter.state()
        print(json.dumps({"simulator": "ok", "posture": state.posture}))
    finally:
        adapter.close()
    try:
        print(json.dumps({"model": OllamaModel(args.model).check()}))
    except ModelError as error:
        print(json.dumps({"model": "not ready", "why": str(error),
                          "fix": "see command_ui/README.md (model file and Ollama)"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=list(CATALOG["models"]), default=CATALOG["default"])
    parser.add_argument("--mock", action="store_true", help="scripted replies instead of the language model")
    parser.add_argument("--scene", choices=["room", "empty"], default="room")
    parser.add_argument("--headless", action="store_true", help="simulate without the MuJoCo window")
    parser.add_argument("--robot", metavar="INTERFACE", help="drive the real Go2 on this network interface")
    parser.add_argument("--arm", action="store_true", help="with --robot: allow the real robot to move")
    # macOS's AirPlay Receiver listens on port 5000, so the Mac default is 5050.
    parser.add_argument("--port", type=int, default=5050 if sys.platform == "darwin" else 5000)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--check", action="store_true", help="check the simulator and model, then exit")
    args = parser.parse_args()
    if args.check:
        return check(args)

    model = MockModel() if args.mock else OllamaModel(args.model)
    try:
        adapter = connect_robot(args)
    except AdapterError as error:
        sys.exit(f"Could not start the robot: {error}")
    console = Console(model, adapter)
    if args.robot and args.arm:
        console.executor.set_armed(True)
        print("The real Go2 is ARMED: it will move when told to. Press Stop on the page to halt it.")
    elif args.robot:
        print("The real Go2 is connected but not armed: it will not move (add --arm).")

    from werkzeug.serving import make_server
    server = make_server("127.0.0.1", args.port, make_app(console, args.port), threaded=True)

    def watch():                                  # closing the MuJoCo window ends the console, as before
        while adapter.connected:
            time.sleep(.5)
        print("Viewer closed." if not args.robot else "The robot disconnected.")
        server.shutdown()
    threading.Thread(target=watch, daemon=True).start()

    url = f"http://127.0.0.1:{args.port}"
    print(f"Go2 command console ready. Open {url} in your browser.", flush=True)
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        console.close()


if __name__ == "__main__":
    main()
