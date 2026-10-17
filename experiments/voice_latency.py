"""How long after a command the robot moves, and how long until Go2's spoken reply is ready to play.

Runs the console itself (agent, executor, model go2-ft-04 and Kokoro voice) against the headless room
simulation, sending each command as the page does when it speaks the replies. For each command it records,
from the moment the message is sent: when the model's tool calls come back, when its first phrase appears,
when the robot first moves (speed, turning or a posture change), and when the first phrase's audio is ready.
The gap that matters is speech ready minus motion: positive means the robot moves before it says anything.

  .venv/bin/python experiments/voice_latency.py [--rounds 2] [--no-hold] [--json out.json]

--no-hold lets the robot move as soon as it can (the console otherwise starts a spoken reply's first movement
with the voice; see VOICE_HOLD_S in command_ui/server.py).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "command_ui"))

from go2.adapters.sim import SimAdapter  # noqa: E402
from go2.llm import CATALOG, OllamaModel  # noqa: E402
from server import Console  # noqa: E402

# In this order: the walk after "sit down" has to stand up first.
COMMANDS = ["walk forward one metre", "turn left", "sit down", "walk forward half a metre", "do a little dance",
            "walk forward then turn right"]


class Timeline:
    def __init__(self):
        self.t0, self.marks, self.lock = None, {}, threading.Lock()

    def start(self):
        with self.lock:
            self.t0, self.marks = time.monotonic(), {}

    def mark(self, name, value=None):
        with self.lock:
            if self.t0 is not None and name not in self.marks:
                self.marks[name] = (round(time.monotonic() - self.t0, 3), value)


def instrument(console, timeline):
    agent, executor, model, adapter = console.agent, console.executor, console.model, console.adapter
    say, chat, submit = agent.speech.say, model.chat, executor.submit

    def speech_say(epoch, text, proactive=False):
        timeline.mark("first_phrase", text)
        return say(epoch, text, proactive)

    steps = []

    def model_chat(messages, tools=None, cancelled=lambda: False, on_text=None, timeout=60):
        began = time.monotonic()
        first = []

        def text(content):
            if not first:
                first.append(time.monotonic())
            if on_text:
                on_text(content)
        result = chat(messages, tools, cancelled=cancelled, on_text=text, timeout=timeout)
        if timeline.t0 is not None:
            steps.append({"start": round(began - timeline.t0, 3), "end": round(time.monotonic() - timeline.t0, 3),
                          "first_text": round(first[0] - timeline.t0, 3) if first else None,
                          "calls": [c["name"] for c in result["tool_calls"]], "content": result["content"]})
            if result["tool_calls"]:
                timeline.mark("tool_calls", [c["name"] for c in result["tool_calls"]])
        return result

    def executor_submit(commands, group, user_turn):
        timeline.mark("submitted")
        results = submit(commands, group, user_turn)
        timeline.mark("submit_returned", [r.get("status") for r in results])
        return results

    def watch():
        last = None
        while True:
            try:
                s = adapter.state()
            except Exception:
                time.sleep(.02)
                continue
            if last is not None and (s.speed > .05 or abs(s.wz) > .1 or s.posture != last):
                timeline.mark("motion", s.posture)
            last = s.posture
            time.sleep(.01)

    agent.speech.say, model.chat, executor.submit = speech_say, model_chat, executor_submit
    threading.Thread(target=watch, daemon=True).start()
    return steps


def settle(console, timeout=40):
    """Wait until Go2 has finished talking and moving, robot updates included."""
    deadline = time.monotonic() + timeout
    quiet = 0
    while time.monotonic() < deadline and quiet < 15:
        busy = console.agent.busy or console.executor.current is not None or console.executor.queue
        quiet = 0 if busy else quiet + 1
        time.sleep(.1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--no-hold", action="store_true", help="let the robot move before Go2's voice is ready")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    adapter = SimAdapter(scene="room", headless=True)
    adapter.connect()
    console = Console(OllamaModel(CATALOG["default"]), adapter, {"scene": "room"}, voice=True)
    if args.no_hold:
        console.executor.before_motion = None
    try:
        console.loaded.wait(600)
        console.speaker.ready.wait(600)
        if console.model_info["state"] != "ready" or not console.speaker.ready.is_set():
            sys.exit(f"Not ready: model {console.model_info}, voice {console.speaker.error}")
        timeline = Timeline()
        steps = instrument(console, timeline)
        rows = []
        for round_ in range(args.rounds):
            settle(console)
            console.reset()                     # the same start pose each round
            for command in COMMANDS:
                settle(console)
                steps.clear()
                console.prepared.clear()        # every reply's audio is made afresh, none from earlier rounds
                timeline.start()
                console.post(command, voice=True, speak=True)
                spoken = console.spoken
                threading.Thread(target=lambda: spoken.audible.wait(30) and timeline.mark("first_audio"),
                                 daemon=True).start()
                settle(console)
                marks = dict(timeline.marks)
                timeline.t0 = None
                row = {"round": round_ + 1, "command": command, "marks": marks, "steps": list(steps)}
                motion, audio = marks.get("motion", (None,))[0], marks.get("first_audio", (None,))[0]
                row["speech_minus_motion"] = round(audio - motion, 3) if motion is not None and audio is not None else None
                rows.append(row)
                print(f"\n[{round_ + 1}] {command!r}")
                for name, (at, value) in sorted(marks.items(), key=lambda kv: kv[1][0]):
                    print(f"   {at:6.2f} s  {name:16} {value if value is not None else ''}")
                for i, step in enumerate(row["steps"]):
                    print(f"   model step {i + 1}: {step['start']:.2f}-{step['end']:.2f} s, first text "
                          f"{step['first_text']}, calls {step['calls']}, said {step['content']!r}")
                print(f"   speech ready minus motion: {row['speech_minus_motion']} s")
        gaps = [r["speech_minus_motion"] for r in rows if r["speech_minus_motion"] is not None]
        if gaps:
            print(f"\nspeech ready minus motion over {len(gaps)} commands: median {statistics.median(gaps):.2f} s, "
                  f"range {min(gaps):.2f} to {max(gaps):.2f} s")
        if args.json:
            args.json.write_text(json.dumps(rows, indent=1))
    finally:
        console.close()


if __name__ == "__main__":
    main()
