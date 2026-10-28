"""Scripted stand-in for the LLM (app.py --mock): exercises the UI and tools without a model."""
from __future__ import annotations

import re


class MockModel:
    model_id = "mock"
    options = {}
    loaded = False

    def check(self):
        return {"model": "Mock (scripted, no LLM)", "label": "Mock", "ready": True, "loaded": False}

    def chat(self, messages, tools=None, cancelled=lambda: False, on_text=None, timeout=60):
        last = messages[-1]
        if last["role"] == "tool":
            return {"content": "On it!", "tool_calls": [], "stats": {"mock": True}}
        text = last["content"].lower()
        text = re.sub(r"^\[robot update:.*?\]\s*", "", text, flags=re.S)
        calls = []
        number = re.search(r"(\d+(?:\.\d+)?)", text)
        if "stop following" in text:
            calls = [("stop", {})]
        elif "follow" in text:
            colour = re.search(r"\b(red|orange|yellow|green|blue|purple|pink|white|grey|black|brown)\b", text)
            calls = [("follow", {"target": f"the person in {colour[1]}" if colour else "me",
                                 **({"distance_m": float(number[1])} if number else {})})]
        elif "look" in text or "see" in text:
            calls = [("look", {})]
        elif "turn" in text:
            calls = [("turn", {"direction": "right" if "right" in text else "left",
                               **({"angle_deg": float(number[1])} if number else {})})]
        elif any(w in text for w in ("walk", "move", "go ")):
            direction = next((d for d in ("backward", "left", "right") if d in text), "forward")
            calls = [("move", {"direction": direction, **({"distance_m": float(number[1])} if number else {})})]
        elif "sit" in text:
            calls = [("posture", {"name": "sit"})]
        elif "stand" in text:
            calls = [("posture", {"name": "stand"})]
        elif "status" in text:
            calls = [("get_status", {})]
        if calls:
            return {"content": "", "tool_calls": [{"name": n, "arguments": a} for n, a in calls], "stats": {"mock": True}}
        return {"content": "Mock reply: I'm a scripted stand-in, no language model is running.", "tool_calls": [],
                "stats": {"mock": True}}

    def cancel(self):
        pass

    def unload(self):
        return {"unloaded": True}
