# -*- coding: utf-8 -*-
"""Go2 command console - backend.

Runs the MuJoCo Go2 simulation (main thread) and a Flask web server (background
thread). The web UI sends natural-language commands (typed or spoken); each is
interpreted into a bounded tool, validated, and executed by driving the simulated
Go2 via SimSportClient.Move(). Voice uses faster-whisper locally (offline).
"""
from __future__ import annotations
import sys, time, queue, threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from flask import Flask, request, jsonify, send_from_directory
from go2_locomotion import SimSportClient
import agent

GO2_DIR = ROOT / "_deps" / "unitree_mujoco" / "unitree_robots" / "go2"
SCENE = GO2_DIR / "scene.xml"

CMD_Q: "queue.Queue[dict]" = queue.Queue()
STATUS = {"state": "standing", "action": "—", "speed": 0.0, "note": ""}

# ---- voice (lazy-loaded so the UI starts fast) ----
_whisper = None
_whisper_lock = threading.Lock()
VOICE_SECONDS = 5
SR = 16000


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


# ----------------- Flask app -----------------
app = Flask(__name__, static_folder=str(HERE / "static"))


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/command", methods=["POST"])
def api_command():
    text = (request.get_json(force=True) or {}).get("text", "")
    result = agent.process(text)
    if result["exec"]:
        CMD_Q.put(result["exec"])
    return jsonify(result)


@app.route("/api/voice", methods=["POST"])
def api_voice():
    try:
        text = transcribe_from_mic()
    except Exception as exc:
        return jsonify({"command": "(voice)", "tool": None, "valid": False,
                        "validation": "Voice error", "response": str(exc), "exec": None})
    if not text:
        return jsonify({"command": "(heard nothing)", "tool": None, "valid": False,
                        "validation": "Rejected — no speech detected",
                        "response": "I didn't catch that. Try again.", "exec": None})
    result = agent.process(text)
    if result["exec"]:
        CMD_Q.put(result["exec"])
    return jsonify(result)


@app.route("/api/stop", methods=["POST"])
def api_stop():
    CMD_Q.put({"tool": "stop", "vx": 0, "vy": 0, "yaw": 0, "duration": 0})
    return jsonify({"ok": True})


@app.route("/api/status")
def api_status():
    return jsonify(STATUS)


def run_flask():
    app.run(host="127.0.0.1", port=5000, threaded=True, use_reloader=False)


# ----------------- simulation loop (main thread) -----------------
def sim_loop():
    import mujoco, mujoco.viewer
    client = SimSportClient(SCENE)
    active_until = 0.0
    print("Go2 command console ready. Open http://127.0.0.1:5000 in your browser.")
    with mujoco.viewer.launch_passive(client.model, client.data) as viewer:
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.5, 135, -25
        while viewer.is_running():
            tick = time.monotonic()
            # apply any queued commands
            try:
                while True:
                    c = CMD_Q.get_nowait()
                    if c["tool"] == "stop":
                        client.StopMove(); active_until = 0.0
                        STATUS["action"] = "stopped"
                    else:
                        client.Move(c["vx"], c["vy"], c["yaw"])
                        active_until = time.monotonic() + c["duration"]
                        STATUS["action"] = c["tool"].replace("_", " ")
                    STATUS["note"] = ""
            except queue.Empty:
                pass
            # auto-stop when a timed command elapses
            if active_until and time.monotonic() >= active_until:
                client.StopMove(); active_until = 0.0
                STATUS["action"] = "stopped"
            # advance the simulation
            with viewer.lock():
                try:
                    client.step()
                except RuntimeError as exc:
                    client.reset(); active_until = 0.0
                    STATUS["note"] = f"Auto-reset: {exc}"
                viewer.cam.lookat[:] = client.data.qpos[client.base_q:client.base_q + 3]
            viewer.sync()
            m = client.metrics()
            STATUS["speed"] = round(m["speed"], 2)
            STATUS["state"] = "walking" if client.command != (0.0, 0.0, 0.0) else "standing"
            time.sleep(max(0, client.period - (time.monotonic() - tick)))
    print("Viewer closed.")


if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    try:
        import webbrowser
        threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    except Exception:
        pass
    sim_loop()
