# -*- coding: utf-8 -*-
"""Command interpreter + validation layer for the Go2 command console.

This is a simple, deterministic rule-based interpreter that maps a natural-language
command to one bounded semantic tool. It is a PLACEHOLDER for the team's LLM agent
(Prionti's area): the LLM will later replace `interpret_to_tool()` and return the
same tool structure, so the validation layer, UI and robot execution stay unchanged.

Velocities are body-frame and stay within SimSportClient's demo limits
(|vx| <= 0.6, |vy| <= 0.3, |yaw_rate| <= 0.5).
"""
from __future__ import annotations
import re

# Bounded semantic tools -> velocity command (vx, vy, yaw_rate)
TOOLS = {
    "move_forward":  {"vx": 0.4,  "vy": 0.0,  "yaw": 0.0,  "label": "move(forward)"},
    "move_backward": {"vx": -0.3, "vy": 0.0,  "yaw": 0.0,  "label": "move(backward)"},
    "strafe_left":   {"vx": 0.0,  "vy": 0.3,  "yaw": 0.0,  "label": "move(left)"},
    "strafe_right":  {"vx": 0.0,  "vy": -0.3, "yaw": 0.0,  "label": "move(right)"},
    "turn_left":     {"vx": 0.0,  "vy": 0.0,  "yaw": 0.5,  "label": "turn(left)"},
    "turn_right":    {"vx": 0.0,  "vy": 0.0,  "yaw": -0.5, "label": "turn(right)"},
    "stop":          {"vx": 0.0,  "vy": 0.0,  "yaw": 0.0,  "label": "stop()"},
}
# Actions the velocity policy cannot do in simulation (per the proposal scope).
UNSUPPORTED = {"sit", "stand up", "standup", "jump", "lie down", "roll", "shake", "dance"}

DEFAULT_DURATION = 3.0
MAX_DURATION = 10.0


_WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
            "seven": 7, "eight": 8, "nine": 9, "ten": 10, "a": 1, "an": 1}


def _find_duration(text: str):
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)\b", text)
    if m:
        return float(m.group(1))
    m = re.search(r"\b(" + "|".join(_WORDNUM) + r")\s*(?:seconds?|secs?)\b", text)
    if m:
        return float(_WORDNUM[m.group(1)])
    return None


def interpret_to_tool(text: str):
    """Rule-based NL -> tool. Returns (tool_name, reason) where tool_name may be None."""
    t = text.lower().strip()
    if not t:
        return None, "empty command"
    for word in UNSUPPORTED:
        if word in t:
            return None, f"'{word}' is not supported by the simulated velocity policy"
    if "stop" in t or "halt" in t or "freeze" in t:
        return "stop", None
    if "turn" in t and "left" in t:
        return "turn_left", None
    if "turn" in t and "right" in t:
        return "turn_right", None
    if "back" in t or "reverse" in t:
        return "move_backward", None
    if "forward" in t or "ahead" in t:
        return "move_forward", None
    if "left" in t:
        return "strafe_left", None
    if "right" in t:
        return "strafe_right", None
    if "walk" in t or "go" in t or "move" in t:
        return "move_forward", None
    return None, "no matching robot tool"


def process(text: str):
    """Full pipeline for one command: interpret -> validate -> build execution plan.

    Returns a dict the UI renders directly and the server executes (if valid).
    """
    tool_name, reason = interpret_to_tool(text)

    # --- interpretation failed -> controlled fallback (no robot action) ---
    if tool_name is None:
        return {
            "command": text,
            "tool": None,
            "valid": False,
            "validation": f"Rejected — {reason}",
            "response": "I can't do that. Try: walk, back, left, right, turn, or stop.",
            "exec": None,
        }

    tool = TOOLS[tool_name]

    # --- validation layer (independent of the interpreter/LLM) ---
    duration = _find_duration(text)
    if tool_name == "stop":
        duration = 0.0
    else:
        if duration is None:
            duration = DEFAULT_DURATION
        if not (0.5 <= duration <= MAX_DURATION):
            return {
                "command": text,
                "tool": tool["label"],
                "valid": False,
                "validation": f"Rejected — duration must be 0.5–{int(MAX_DURATION)} s",
                "response": "Duration out of range. Try e.g. 'walk forward for 3 seconds'.",
                "exec": None,
            }

    # bounds check (defensive; tool velocities are already within limits)
    if abs(tool["vx"]) > 0.6 or abs(tool["vy"]) > 0.3 or abs(tool["yaw"]) > 0.5:
        return {
            "command": text, "tool": tool["label"], "valid": False,
            "validation": "Rejected — velocity outside safe limits",
            "response": "Blocked by the safety limits.", "exec": None,
        }

    label = tool["label"]
    if tool_name != "stop":
        label = label[:-1] + f", {duration:g}s)" if label.endswith(")") else label

    return {
        "command": text,
        "tool": label,
        "valid": True,
        "validation": "Passed — within limits",
        "response": ("Stopping." if tool_name == "stop"
                     else f"Executing — {tool['label'][:-1].replace('(', ' ').strip()} for {duration:g}s."),
        "exec": {"vx": tool["vx"], "vy": tool["vy"], "yaw": tool["yaw"],
                 "duration": duration, "tool": tool_name},
    }
