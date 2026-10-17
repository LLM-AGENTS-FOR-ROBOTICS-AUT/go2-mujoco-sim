# -*- coding: utf-8 -*-
"""Go2 command console - backend.

The web page (static/index.html) sends a typed or spoken command. The LLM agent (go2/agent.py: Gemma 4 E2B
fine-tuned as go2-ft-04, run by a private Ollama) answers it and calls bounded robot tools; the executor
(go2/executor.py) checks every call against the limits and the robot's own sensors before anything moves.
The robot is the MuJoCo room simulation in its own window (simulation/), or the real Go2 EDU with --robot.
Voice uses faster-whisper locally (offline): the mic button records 5 seconds here.

API: the page's original endpoints (/api/command, /api/voice, /api/stop, /api/status) still work. The richer
API for a new page (live state over /api/state or /api/events, /api/messages, /api/clear, /api/reset, /api/arm,
/api/camera.png) is documented with real example messages in command_ui/API.md.

  .venv/bin/python command_ui/server.py              (macOS; Windows: .venv\\Scripts\\python.exe command_ui\\server.py)
  ... --mock                  scripted replies, no language model (checks the page and the robot)
  ... --robot en7 [--arm]     the real Go2 on that network interface; it only moves once armed
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import signal
import struct
import sys
import threading
import time
import webbrowser
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from flask import Flask, Response, abort, jsonify, request, send_from_directory  # noqa: E402

from go2 import perception as P  # noqa: E402
from go2 import procs  # noqa: E402
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
    def __init__(self, model, adapter, robot_info=None):
        self.model, self.adapter = model, adapter
        self.robot_info = robot_info or {}       # {"scene": "room"} or {"interface": "en7"}, shown in the state
        self.executor = Executor(on_event=lambda task: self.agent.on_task_event(task))
        self.agent = Agent(model, self.executor)
        self.agent.robot_line = lambda: robot_state_line(self.adapter, self.executor)
        self.executor.attach(adapter)
        self.commands = threading.Lock()        # the page shows one card per command, in order
        self.shown = 0                          # transcript rows already shown on a card
        self.loaded = threading.Event()         # set once the model has loaded (or failed to)
        self.note = ""
        self.model_info = {"id": getattr(model, "choice", model.model_id),
                           "label": getattr(model, "spec", {}).get("label", "Scripted replies (no language model)"),
                           "state": "scripted", "error": None}
        self.conversation_id = 0                # goes up when the conversation is cleared
        self.revision, self.last_state = 0, None
        self.state_lock = threading.Lock()
        self.closed = False
        if isinstance(model, OllamaModel):
            self.note = "Loading the language model..."
            self.model_info["state"] = "loading"
            threading.Thread(target=self._warm_up, daemon=True).start()
        else:
            self.loaded.set()

    def _warm_up(self):
        """Load the model in the background so the first command doesn't wait for it."""
        try:
            self.model.check()
            self.model.chat([{"role": "user", "content": "Hello"}])
            self.note = ""
            self.model_info["state"] = "ready"
        except Exception as error:              # missing weights, no Ollama: shown under the Stop button
            self.note = f"Language model unavailable: {error}"
            self.model_info.update(state="unavailable", error=str(error))
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

    # ---- the richer API (command_ui/API.md) --------------------------------------------------------------
    def post(self, text, voice=False):
        """Accept a message and return at once (the reply and actions arrive in the state). Returns the id of
        the conversation entry it became."""
        with self.commands:
            self.loaded.wait(TURN_TIMEOUT_S)    # only during the first seconds after start-up
            entry = len(self.agent.snapshot()["transcript"])
            self.agent.submit(text, voice)      # ValueError for an empty or overlong message
        return entry

    def clear(self):
        self.agent.clear()
        self.conversation_id += 1
        self.shown = 0

    def reset(self):
        if self.adapter.kind != "sim":
            raise ValueError("Reset is only available in the simulation.")
        self.executor.stop("simulation reset")
        self.adapter.reset()
        self.executor.attach(self.adapter)

    def arm(self, armed):
        if not self.adapter.requires_arming:
            raise ValueError("Only the real robot needs arming; the simulation always moves.")
        self.executor.set_armed(bool(armed))

    def camera_png(self, full=False):
        frame = self.executor.last_frame
        if frame is None:
            return None
        return png_bytes(frame if full else frame[::2, ::2])

    def snapshot(self):
        agent, executor, tasks = self.agent.snapshot(), self.executor.snapshot(), self.executor.tasks()
        robot = {"kind": self.adapter.kind, "label": self.adapter.label, "connected": self.adapter.connected,
                 "requires_arming": executor["requires_arming"], "armed": executor["armed"], **self.robot_info,
                 "posture": None, "telemetry": None, "error": None}
        if robot["connected"]:
            try:
                s = self.adapter.state()
                robot["posture"] = s.posture
                robot["telemetry"] = {"x_m": round(s.x, 2), "y_m": round(s.y, 2),
                                      "heading_deg": round(s.yaw * 57.2958), "speed_mps": round(s.speed, 2),
                                      "turn_dps": round(s.wz * 57.2958), "battery_pct": s.battery_pct,
                                      "fault": s.fault}
            except AdapterError as error:
                robot["error"] = str(error)
        camera = None
        frame = self.executor.last_frame_info()
        if frame:
            camera = {"stamp": frame["stamp"], "url": f"/api/camera.png?t={frame['stamp']}",
                      "objects": [{**d.public(), "direction": P.describe_bearing(d.bearing_deg)}
                                  for d, _ in list(self.executor.last_detections)],
                      "boxes": frame.get("boxes", [])}
        return {"robot": robot, "model": dict(self.model_info),
                "agent": {"thinking": agent["busy"], "error": agent["error"]},
                "actions": {"current": executor["current"], "queued": executor["queue"],
                            "recent": executor["recent"]},
                "conversation": {"id": self.conversation_id, "entries": conversation_entries(agent["transcript"], tasks)},
                "camera": camera}

    def state(self):
        """The snapshot plus a revision number that goes up whenever anything in it changed."""
        snapshot = self.snapshot()
        text = json.dumps(snapshot, sort_keys=True, default=str)
        with self.state_lock:
            if text != self.last_state:
                self.revision, self.last_state = self.revision + 1, text
            return {"revision": self.revision, **snapshot}

    def events(self, interval=.25, heartbeat=15.):
        """Server-sent events: the state each time it changes, and a comment line now and then to keep the
        connection open."""
        sent, quiet = None, 0.
        while not self.closed:
            state = self.state()
            if state["revision"] != sent:
                sent, quiet = state["revision"], 0.
                yield f"event: state\ndata: {json.dumps(state, default=str)}\n\n"
            elif quiet >= heartbeat:
                quiet = 0.
                yield ": still here\n\n"
            time.sleep(interval)
            quiet += interval

    def close(self):
        self.closed = True
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


# ---- conversation entries for the richer API -------------------------------------------------------------
def outcome(result):
    """What happened to one tool call when it was made: queued, done (a query answered), already_queued,
    needs_confirmation, rejected, blocked, failed or cancelled."""
    statuses = [r.get("status") or ("rejected" if "error" in r else "done") for r in [result, *result.get("results", [])]
                if r is not result or "results" not in result]
    for status in ("needs_confirmation", "rejected", "blocked", "failed", "cancelled", "already_queued"):
        if status in statuses:
            return status
    return "queued" if "queued" in statuses else "done"


def steps(result, tasks):
    """The robot actions a tool call became, each with its live status from the executor."""
    found = []
    for r in [result, *result.get("results", [])]:
        if r.get("task_id"):
            found.append(tasks.get(r["task_id"], {"id": r["task_id"], "summary": r.get("action"),
                                                  "status": r.get("status")}))
        elif r.get("action") and r.get("status") in ("blocked", "failed", "cancelled"):
            found.append({"id": None, "summary": r["action"], "status": r["status"],
                          **({"error": r["error"]} if r.get("error") else {})})
    return found


def conversation_entries(transcript, tasks):
    """The agent's transcript as entries for a page: user messages, the tool calls each reply made (with the
    robot actions they became, updated live), Go2's replies, robot updates and notes. `turn` is the id of the
    user message an entry answers, or None when Go2 acts on its own (reporting a move that finished, say)."""
    entries, turn = [], None
    for i, row in enumerate(transcript):
        role, text = row["role"], row.get("content", "")
        if role == "user":
            turn = i
            entries.append({"id": i, "kind": "user", "text": text, "voice": bool(row.get("voice"))})
        elif role == "tool":
            result = row.get("result") or {}
            entries.append({"id": i, "kind": "call", "turn": turn, "tool": row["name"],
                            "arguments": row.get("arguments") or {}, "outcome": outcome(result),
                            "message": problem(result), "steps": steps(result, tasks)})
        elif role == "assistant":
            proactive = bool(row.get("proactive"))
            entries.append({"id": i, "kind": "reply", "turn": None if proactive else turn, "text": text,
                            "proactive": proactive})
        elif role == "event":
            if "turn" in row:                   # a report that starts Go2's own turn
                turn = None
            entries.append({"id": i, "kind": "update", "text": for_people(text)})
        elif role == "error":
            entries.append({"id": i, "kind": "error", "turn": turn, "text": text})
        else:
            entries.append({"id": i, "kind": "note", "text": text})
    return entries


def for_people(report):
    """A robot report without the instruction it carries for the model ("Tell the user briefly.")."""
    return re.sub(r"\.?\s*Tell the user\b[^.]*\.?\s*$", ".", report).strip()


def png_bytes(rgb):
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


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

    # ---- the richer API (command_ui/API.md) ------------------------------------------------------------
    def body():
        value = request.get_json(force=True, silent=True)
        return value if isinstance(value, dict) else {}

    def refused(error, code=400):
        return jsonify({"error": str(error)}), code

    @app.route("/api/state")
    def api_state():
        return jsonify(console.state())

    @app.route("/api/events")
    def api_events():
        return Response(console.events(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.route("/api/messages", methods=["POST"])
    def api_messages():
        value = body()
        try:
            entry = console.post(str(value.get("text", "")), bool(value.get("voice")))
        except ValueError as error:
            return refused(error)
        return jsonify({"accepted": True, "entry": entry, "conversation": console.conversation_id}), 202

    @app.route("/api/clear", methods=["POST"])
    def api_clear():
        console.clear()
        return jsonify(console.state())

    @app.route("/api/reset", methods=["POST"])
    def api_reset():
        try:
            console.reset()
        except (ValueError, AdapterError) as error:
            return refused(error)
        return jsonify(console.state())

    @app.route("/api/arm", methods=["POST"])
    def api_arm():
        value = body()
        if not isinstance(value.get("armed"), bool):
            return refused('Send {"armed": true} or {"armed": false}.')
        try:
            console.arm(value["armed"])
        except ValueError as error:
            return refused(error)
        return jsonify(console.state())

    @app.route("/api/camera.png")
    def api_camera():
        image = console.camera_png(full=request.args.get("size") == "full")
        if image is None:
            return refused("No camera frame yet: ask Go2 what it sees.", 404)
        return Response(image, mimetype="image/png", headers={"Cache-Control": "no-store"})

    return app


def stop_stale_model_servers():
    """Model servers left running by an earlier console that was killed outright (each private server
    records its process id), so they don't keep the model in memory."""
    for pid_file in (ROOT / "runtime").glob("model-*/server.pid"):
        try:
            pid = int(pid_file.read_text())
            if "ollama" in procs.command_of(pid).lower():
                procs.stop_pid(pid)
        except (OSError, ValueError):
            pass
        shutil.rmtree(pid_file.parent, ignore_errors=True)     # only links to the weights and a log


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

    def shut_down(*_):
        raise KeyboardInterrupt                 # runs the clean-up at the end (simulator, model server)
    for name in ("SIGTERM", "SIGHUP", "SIGBREAK"):   # stop signal, terminal window closed, Ctrl+Break
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), shut_down)
    stop_stale_model_servers()

    model = MockModel() if args.mock else OllamaModel(args.model)
    try:
        adapter = connect_robot(args)
    except AdapterError as error:
        sys.exit(f"Could not start the robot: {error}")
    console = Console(model, adapter, {"interface": args.robot} if args.robot else {"scene": args.scene})
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
