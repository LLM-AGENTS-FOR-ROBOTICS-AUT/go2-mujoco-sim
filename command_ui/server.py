# -*- coding: utf-8 -*-
"""Go2 command console - backend.

The web page (static/index.html) sends a typed or spoken command. The LLM agent (go2/agent.py: Gemma 4 E2B
fine-tuned as go2-ft-04, run by a private Ollama) answers it and calls bounded robot tools; the executor
(go2/executor.py) checks every call against the limits and the robot's own sensors before anything moves.
The robot is the MuJoCo room simulation in its own window (simulation/), or the real Go2 EDU with --robot.
Voice runs on this computer too: the page records the microphone and cuts it where you pause, Whisper
large-v3-turbo (go2/stt.py) turns each sentence into text, and Kokoro (go2/tts.py) speaks Go2's replies phrase by
phrase as the model writes them.

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
import random
import re
import select
import shutil
import signal
import socket
import struct
import sys
import threading
import time
import uuid
import webbrowser
import zlib
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FEEDBACK = ROOT / "feedback"    # replies flagged with the page's "bad reply" button (feedback/README.md)
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from flask import Flask, Response, abort, jsonify, request, send_from_directory  # noqa: E402

from go2 import perception as P  # noqa: E402
from go2 import feedback_upload, procs  # noqa: E402
from go2.adapters.base import AdapterError  # noqa: E402
from go2.adapters.sim import SimAdapter  # noqa: E402
from go2.agent import Agent, is_stop, robot_state_line  # noqa: E402
from go2.executor import Executor  # noqa: E402
from go2.llm import CATALOG, ModelError, OllamaModel  # noqa: E402
from go2.mock_model import MockModel  # noqa: E402
from go2.stt import MAX_BYTES as STT_MAX_BYTES  # noqa: E402
from go2.stt import Listener  # noqa: E402
from go2.tts import DEFAULT_VOICE, VOICES, Speaker  # noqa: E402

VOICE_SECONDS = 5
SR = 16000
TURN_TIMEOUT_S = 120
SPEECH_WAIT_S = 20              # how long /api/speech holds a request open waiting for Go2's next phrase
# A spoken reply's words come from a second model step after the tool call, and its audio takes a few tenths of a
# second more, so on its own the robot would start about half a second before Go2 says anything. Its first
# movement waits for the audio instead, but never longer than this (and Stop always cuts through).
VOICE_HOLD_S = 1.2
PREPARED = 32                   # phrases' audio kept ready (replies like "Okay, stopped." repeat)
LOAD_WAIT_S = 30                # a message sent while the model loads (seconds after start-up) waits this long


class NotReady(ValueError):
    """The language model can't take messages yet (still downloading, or failed to load)."""
PROBLEMS = ("blocked", "failed", "rejected", "cancelled", "needs_confirmation")


class SpokenReply:
    """A user turn whose reply the page plays aloud: its first robot movement starts with Go2's voice."""

    def __init__(self, user_turn):
        self.user_turn, self.epoch = user_turn, None
        self.audible = threading.Event()   # the first phrase can be heard now (or the turn ended without one)
        self.preparing = False             # its first phrase's audio is being made
        self.held = False                  # a movement has already waited for it


# ---- the robot, the executor and the agent ----------------------------------------------------------------
class Console:
    def __init__(self, model, adapter, robot_info=None, voice=False, feedback=FEEDBACK, uploader=None):
        self.model, self.adapter = model, adapter
        self.feedback = Path(feedback)
        self.uploader = uploader                # a copy of each flag in Firebase (go2/feedback_upload.py), or None
        self.robot_info = robot_info or {}       # {"scene": "room"} or {"interface": "en7"}, shown in the state
        self.executor = Executor(on_event=lambda task: self.agent.on_task_event(task))
        self.agent = Agent(model, self.executor)
        self.agent.robot_line = lambda: robot_state_line(self.adapter, self.executor)
        self.agent.robot_snapshot = self._robot_snapshot
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
        self.listener = self.speaker = None     # speech recognition and Go2's voice (both load in the background)
        self.voice_name = None                  # the voice the page plays replies in; None while it doesn't
        self.spoken = None                      # the latest user turn the page speaks the reply of (SpokenReply)
        self.cut = 0                            # phrases of turns before this epoch were cut off: not prepared
        self.prepared = OrderedDict()           # (text, voice) -> {"done": Event, "wav": bytes or None}
        self.to_prepare = []
        self.preparing = threading.Condition()
        self.agent.speech.listeners.append(self._on_speech)
        self.executor.before_motion = self._with_voice
        if voice:
            self.listener, self.speaker = Listener(), Speaker()
            self.listener.warm_up()
            self.speaker.warm_up()
            threading.Thread(target=self._prepare_audio, daemon=True, name="voice-prepare").start()
        if isinstance(model, OllamaModel):
            self.note = "Loading the language model..."
            self.model_info["state"] = "loading"
            threading.Thread(target=self._warm_up, daemon=True).start()
        else:
            self.loaded.set()

    def _warm_up(self):
        """Load the model in the background so the first command doesn't wait for it (downloading it first, the
        first time)."""
        try:
            self.model.fetch(progress=self._downloading)
            self.model_info.pop("progress", None)
            self.model_info["state"] = "loading"
            self.note = "Loading the language model..."
            self.model.check()
            self.model.chat([{"role": "user", "content": "Hello"}])
            self.note = ""
            self.model_info["state"] = "ready"
        except Exception as error:              # missing weights, no Ollama: shown under the Stop button
            self.note = f"Language model unavailable: {error}"
            self.model_info.update(state="unavailable", error=str(error))
        finally:
            self.loaded.set()

    def _downloading(self, done, total):
        self.model_info["state"] = "downloading"
        self.model_info["progress"] = {"done_mb": round(done / 1e6), "total_mb": round(total / 1e6)}
        self.note = (f"Downloading the language model (first start only): {done / total:.0%}, "
                     f"{done / 1e9:.1f} of {total / 1e9:.1f} GB")

    def ready_for(self, text):
        """Raise NotReady, with the reason and download progress, unless the model can answer this message
        now. "Stop" always goes through: it halts the robot without the model."""
        if is_stop(text):
            return
        if self.model_info["state"] == "loading":
            self.loaded.wait(LOAD_WAIT_S)       # loading takes seconds: wait rather than refuse
        state = self.model_info["state"]
        if state == "downloading":
            progress = self.model_info.get("progress") or {}
            done, total = progress.get("done_mb", 0), progress.get("total_mb") or 1
            raise NotReady(f"Still downloading the language model (first start only): {done / total:.0%}, "
                           f"{done / 1000:.1f} of {total / 1000:.1f} GB. Try again when it's ready; Stop works now.")
        if state == "loading":
            raise NotReady("The language model is still loading. Try again in a few seconds.")
        if state == "unavailable":
            raise NotReady(f"The language model isn't available: {self.model_info['error']}")

    def command(self, text, voice=False):
        """One command -> the card the page shows: tool, safety result and Go2's reply."""
        try:
            self.ready_for(text)
        except NotReady as error:
            return card(text, None, False, "Language model not ready", str(error))
        with self.commands:
            start = len(self.agent.snapshot()["transcript"])
            self._speaks(False)                 # the original page shows replies; it doesn't speak them
            self._cut_off()
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

    def stop(self, said=None):
        """Halt at once. `said`: the user's words, when they said "stop" (recorded as their spoken message)."""
        self._cut_off()
        return self.agent.stop(spoken=said or None, voice=True)

    # ---- voice --------------------------------------------------------------------------------------------
    def _speaks(self, speak):
        """Remember how the page plays Go2's replies: `speak` is True (the default voice), a voice's name, or
        False (it doesn't). Raises ValueError for a voice that doesn't exist."""
        if speak is True:
            speak = DEFAULT_VOICE
        if speak and speak not in VOICES:
            raise ValueError(f"Unknown voice {speak!r}: use one of {', '.join(VOICES)}.")
        self.voice_name = speak or None

    def _cut_off(self):
        """A new message, Stop, Interrupt or Clear: what Go2 was about to say is no longer wanted."""
        self.cut = self.agent.epoch + 1
        spoken, self.spoken = self.spoken, None
        if spoken:
            spoken.audible.set()

    def _on_speech(self, item):
        """Each phrase as the model writes it: when the page speaks replies with Kokoro, make its audio now (so it
        is ready when the page asks), and let the turn's first movement go once its first phrase can be heard."""
        spoken = self.spoken
        first = (spoken is not None and item["epoch"] == spoken.epoch and not spoken.audible.is_set()
                 and not spoken.preparing)
        if item.get("end") or self.voice_name is None or self.speaker is None or not self.speaker.ready.is_set():
            if first:
                spoken.audible.set()        # the turn said nothing, or the browser's own voice starts at once
            return
        if first:
            spoken.preparing = True
        self._prepare(item["text"], self.voice_name, item["epoch"], spoken if first else None)

    def _prepare(self, text, voice, epoch, spoken=None):
        key = (" ".join(text.split()), voice)
        with self.preparing:
            if key in self.prepared:
                entry = self.prepared[key]
                self.prepared.move_to_end(key)
            else:
                entry = self.prepared[key] = {"done": threading.Event(), "wav": None}
                while len(self.prepared) > PREPARED:
                    self.prepared.popitem(last=False)[1]["done"].set()
                self.to_prepare.append((key, entry, epoch))
                self.preparing.notify()
            if spoken is not None:
                entry.setdefault("then", []).append(spoken)
        if entry["done"].is_set():
            self._prepared(entry)

    @staticmethod
    def _prepared(entry):
        for spoken in entry.pop("then", []):
            spoken.audible.set()

    def _prepare_audio(self):
        while not self.closed:
            with self.preparing:
                self.preparing.wait_for(lambda: self.to_prepare or self.closed)
                if self.closed:
                    return
                key, entry, epoch = self.to_prepare.pop(0)
            if epoch >= self.cut:
                try:
                    entry["wav"] = self.speaker.wav(*key)
                except Exception:              # the page's own request then reports it (or uses the browser voice)
                    pass
            entry["done"].set()
            self._prepared(entry)

    def _with_voice(self, task, cancel):
        """The executor's before_motion: a spoken reply's first movement waits (at most VOICE_HOLD_S) until Go2's
        voice for it is ready, so it moves as it starts talking."""
        spoken = self.spoken
        if spoken is None or spoken.held or task.user_turn != spoken.user_turn:
            return
        spoken.held = True
        deadline = time.monotonic() + VOICE_HOLD_S
        while not spoken.audible.wait(.02) and not cancel.is_set() and time.monotonic() < deadline:
            pass

    def voice_status(self):
        if self.listener is None:
            return {"available": False}
        listen, speak = self.listener.status(), self.speaker.status()
        return {"available": True,
                "recognition": {key: listen[key] for key in ("ready", "error", "model", "runs_on")},
                "speech": {key: speak[key] for key in ("ready", "error", "voices", "default")}}

    def transcribe(self, audio, wanted=lambda: True):
        """Text for one sentence of 16 kHz mono float samples. RuntimeError while recognition loads or is off;
        LookupError when the page stopped waiting."""
        if self.listener is None:
            raise RuntimeError("Voice is off (the console was started with --no-voice).")
        return self.listener.transcribe(audio, wanted)

    def speech_wav(self, text, voice=None, wanted=lambda: True):
        """Go2's voice for one phrase as WAV bytes. RuntimeError while it loads or is off (the page then uses the
        browser's own voice); LookupError when the page dropped the phrase."""
        if self.speaker is None:
            raise RuntimeError("Voice is off (the console was started with --no-voice).")
        voice = voice or DEFAULT_VOICE
        with self.preparing:
            entry = self.prepared.get((" ".join(str(text).split()), voice))
        if entry is not None and entry["done"].wait(30) and entry["wav"]:
            return entry["wav"]             # made while the model was still writing
        return self.speaker.wav(text, voice, wanted)

    def interrupt(self):
        """The user cut in while Go2 was thinking or talking: drop the reply. Robot actions already under way
        carry on (Stop halts them)."""
        self._cut_off()
        self.agent.interrupt()

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
    def post(self, text, voice=False, speak=None):
        """Accept a message and return at once (the reply and actions arrive in the state). Returns the id of
        the conversation entry it became. NotReady while the model is downloading or failed to load.
        `speak` says the page plays the reply aloud (True or a voice's name; by default, when the message was
        spoken): its audio is then made as the model writes it, and the first movement starts with the voice."""
        self.ready_for(text)
        with self.commands:
            entry = len(self.agent.snapshot()["transcript"])
            self._speaks(voice if speak is None else speak)
            self._cut_off()
            spoken = SpokenReply(self.agent.user_turns + 1) if self.voice_name else None
            self.spoken = spoken
            epoch = self.agent.submit(text, voice)      # ValueError for an empty or overlong message
            if spoken is not None:
                if epoch is None:                       # "stop": no movement to wait for
                    self.spoken = None
                spoken.epoch = epoch
        return entry

    # ---- the "bad reply" button --------------------------------------------------------------------------
    def _robot_snapshot(self):
        """Where the robot was and what it was doing when a turn started (saved with flagged replies, so a
        training run can put the simulated robot back there and redo the turn)."""
        if not self.adapter.connected:
            return None
        s, info, executor = self.adapter.state(), self.adapter.status(), self.executor.snapshot()
        return {"target": self.adapter.kind, "scene": info.get("scene"), "detector": info.get("detector"),
                "x": round(s.x, 3), "y": round(s.y, 3), "yaw": round(s.yaw, 4), "posture": s.posture,
                "doing": executor["current"]["summary"] if executor["current"] else None,
                "queued": [t["summary"] for t in executor["queue"]], "armed": executor["armed"]}

    def flag(self, entry):
        """Save the turn behind one of Go2's replies (conversation entry `entry`) as feedback/<id>.json, for the
        next training run to redo properly. ValueError for anything but a reply the model made."""
        if isinstance(entry, bool) or not isinstance(entry, int):
            raise ValueError("Send the id of one of Go2's replies.")
        turn = self.agent.flagged_turn(entry)
        flag_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        record = {"id": flag_id, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "model": self.model_info["id"],
                  "prompt_style": getattr(self.model, "prompt_style", "full"), "note": "", **turn}
        self._save_flag(record)
        self.agent.mark_flag(entry, flag_id)
        return self._flag_answer(flag_id)

    def note_flag(self, flag_id, note):
        """Add what Go2 should have done to a flagged reply."""
        if not isinstance(flag_id, str) or not re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{6}", flag_id):
            raise ValueError("Unknown flagged reply.")
        path = self.feedback / f"{flag_id}.json"
        if not path.is_file():
            raise ValueError("Unknown flagged reply.")
        record = json.loads(path.read_text())
        record["note"] = str(note or "")[:2000].strip()
        self._save_flag(record)
        return self._flag_answer(flag_id)

    def _save_flag(self, record):
        self.feedback.mkdir(parents=True, exist_ok=True)
        path = self.feedback / f"{record['id']}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=1, default=str))
        temporary.replace(path)
        if self.uploader:
            self.uploader.submit(record["id"])

    def _flag_answer(self, flag_id):
        path = self.feedback / f"{flag_id}.json"
        try:
            shown = path.resolve().relative_to(ROOT).as_posix()      # "feedback/<id>.json"
        except ValueError:                      # a folder outside the repository
            shown = str(path)
        return {"id": flag_id, "path": shown}

    def clear(self):
        self._cut_off()
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
        return {"robot": robot, "model": dict(self.model_info), "voice": self.voice_status(),
                "agent": {"thinking": agent["busy"], "error": agent["error"]},
                "actions": {"current": executor["current"], "queued": executor["queue"],
                            "recent": executor["recent"]},
                "conversation": {"id": self.conversation_id, "entries": conversation_entries(agent["transcript"], tasks)},
                "camera": camera,
                # flags leave this computer (Firebase): the page warns before sending one
                "feedback": {"upload": self.uploader is not None}}

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
        with self.preparing:
            self.preparing.notify_all()
        self.agent.close()
        self.executor.close()
        for stop in (self.adapter.close, getattr(self.model, "unload", None), getattr(self.uploader, "close", None)):
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
                            "proactive": proactive, "flaggable": "msg_end" in row, "flag": row.get("flag")})
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


def client_gone(sock):
    """True when the browser has closed this request's connection (the page gave up on it)."""
    if sock is None:
        return False
    try:
        readable, _, _ = select.select([sock], [], [], 0)
        return bool(readable) and sock.recv(1, socket.MSG_PEEK) == b""
    except (OSError, ValueError):
        return True


# ---- the original page's voice button: records 5 seconds here (lazy-loaded so the console starts fast) -----
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
        said = body().get("said")
        console.stop(said[:200] if isinstance(said, str) else None)
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
        speak = value.get("speak")
        try:
            entry = console.post(str(value.get("text", "")), bool(value.get("voice")),
                                 speak if isinstance(speak, (bool, str)) else None)
        except NotReady as error:
            return jsonify({"error": str(error), "model": console.model_info}), 503
        except ValueError as error:
            return refused(error)
        return jsonify({"accepted": True, "entry": entry, "conversation": console.conversation_id}), 202

    @app.route("/api/feedback", methods=["POST"])
    def api_feedback():
        value = body()
        try:
            if "id" in value:
                return jsonify(console.note_flag(value["id"], value.get("note")))
            return jsonify(console.flag(value.get("entry"))), 201
        except ValueError as error:
            return refused(error)

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

    # ---- voice (command_ui/API.md, "Voice") --------------------------------------------------------------
    def gone():
        sock = request.environ.get("werkzeug.socket")
        return lambda: not client_gone(sock)

    @app.route("/api/stt", methods=["POST"])
    def api_stt():
        size = request.content_length or 0
        if not 0 < size <= STT_MAX_BYTES or size % 4:
            why = f"Send one sentence as 16 kHz mono float32 samples (at most {STT_MAX_BYTES} bytes)."
            return refused(why, 413 if size > STT_MAX_BYTES else 400)
        audio = np.frombuffer(request.get_data(cache=False), "<f4")
        try:
            text = console.transcribe(audio, wanted=gone())
        except LookupError:                   # the page stopped waiting for this one
            return "", 204
        except RuntimeError as error:         # still loading, or voice is off
            return refused(error, 503)
        return jsonify({"text": text})

    @app.route("/api/tts", methods=["POST"])
    def api_tts():
        value = body()
        try:
            audio = console.speech_wav(value.get("text", ""), value.get("voice"), wanted=gone())
        except LookupError:                   # the page dropped this phrase while it waited its turn
            return "", 204
        except RuntimeError as error:         # still loading, or voice is off: the page uses the browser's voice
            return refused(error, 503)
        except ValueError as error:
            return refused(error)
        return Response(audio, mimetype="audio/wav", headers={"Cache-Control": "no-store"})

    @app.route("/api/speech")
    def api_speech():
        try:
            last = int(request.args.get("after", "-1"))
        except ValueError:
            last = -1
        return jsonify(console.agent.speech.after(last, timeout=SPEECH_WAIT_S))

    @app.route("/api/interrupt", methods=["POST"])
    def api_interrupt():
        console.interrupt()
        return jsonify({"ok": True})

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
        world = None                                    # the demo house, for --scene house on its own
        if args.scene == "house" and args.seed is not None:
            from simulation.house import random_house
            world = random_house(random.Random(args.seed))
        adapter = SimAdapter(scene=args.scene, headless=args.headless, world=world)
    adapter.connect()
    return adapter


def fetch_model(args):
    """Download the model now (setup scripts), rather than on the console's first start."""
    model = OllamaModel(args.model)
    shown = [-1]

    def progress(done, total):
        percent = int(done * 100 / total)
        if percent != shown[0]:
            shown[0] = percent
            print(f"\rDownloading {model.spec['weights_filename']}: {percent}% of {total / 1e9:.1f} GB", end="",
                  flush=True)
    try:
        path = model.fetch(progress)
    except ModelError as error:
        print()
        sys.exit(f"Model not downloaded: {error}")
    print(f"\nModel ready and verified: {path}")


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
        spec = CATALOG["models"][args.model]
        print(json.dumps({"model": "not ready", "why": str(error),
                          "fix": "it downloads on the console's first start (or run server.py --fetch-model)"
                                 if spec.get("download_url") else "see command_ui/README.md (model file and Ollama)"}))


def start_uploader(args):
    """Copy flagged replies to Firebase (on by default). Not for --mock: scripted replies aren't the model's."""
    if args.mock or args.no_upload or not feedback_upload.enabled_by_env():
        return None
    try:
        return feedback_upload.FeedbackUploader(FEEDBACK).start()
    except (OSError, ValueError, KeyError) as error:
        print(f"Firebase upload is off: {error}")
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=list(CATALOG["models"]), default=CATALOG["default"])
    parser.add_argument("--mock", action="store_true", help="scripted replies instead of the language model")
    parser.add_argument("--scene", choices=["room", "empty", "house", "hazards"], default="room",
                        help="house: the multi-room demo house; hazards: a platform with a ledge, stairs down and glass")
    parser.add_argument("--seed", type=int, help="with --scene house: a random house from this seed instead")
    parser.add_argument("--headless", action="store_true", help="simulate without the MuJoCo window")
    parser.add_argument("--robot", metavar="INTERFACE", help="drive the real Go2 on this network interface")
    parser.add_argument("--arm", action="store_true", help="with --robot: allow the real robot to move")
    # macOS's AirPlay Receiver listens on port 5000, so the Mac default is 5050.
    parser.add_argument("--port", type=int, default=5050 if sys.platform == "darwin" else 5000)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--no-voice", action="store_true", help="don't load speech recognition or Go2's voice")
    parser.add_argument("--no-upload", action="store_true",
                        help="keep flagged replies on this computer only (no Firebase copy; feedback/README.md)")
    parser.add_argument("--check", action="store_true", help="check the simulator and model, then exit")
    parser.add_argument("--fetch-model", action="store_true", help="download the model now (once), then exit")
    args = parser.parse_args()
    if args.seed is not None and args.scene != "house":
        parser.error("--seed needs --scene house")
    if args.check:
        return check(args)
    if args.fetch_model:
        return fetch_model(args)

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
    console = Console(model, adapter, {"interface": args.robot} if args.robot else {"scene": args.scene},
                      voice=not args.no_voice, uploader=start_uploader(args))
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
