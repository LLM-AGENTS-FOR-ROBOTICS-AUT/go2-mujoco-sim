"""LLM tool schemas and their translation into executor commands.

The model sees TOOLS. resolve() validates one tool call and returns the commands
to run, or raises ToolError whose message is returned to the model verbatim so
it can correct itself or explain the problem to the user.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from . import limits as L

DIRECTIONS = {
    "forward": (1, 0), "backward": (-1, 0), "left": (0, 1), "right": (0, -1),
    "forward_left": (1, 1), "forward_right": (1, -1), "backward_left": (-1, 1), "backward_right": (-1, -1),
}
POSTURES = ("stand", "stand_up", "lie_down", "sit", "recovery_stand", "damp")
TRICKS = ("hello", "stretch", "content", "heart", "scrape", "dance1", "dance2")
ACROBATICS = ("front_flip", "back_flip", "left_flip", "front_jump", "front_pounce",
              "handstand", "walk_upright", "cross_step", "free_bound", "free_jump")
GAITS = ("classic", "trot_run", "static_walk", "economic", "free_walk")
SPEED_LEVELS = {"slow": -1, "normal": 0, "fast": 1}
EXPECTED_S = {"stand": 1.5, "stand_up": 2.0, "lie_down": 2.0, "sit": 2.5, "recovery_stand": 3.0, "damp": 1.0,
              "hello": 7.0, "stretch": 4.5, "content": 4.0, "heart": 7.0, "scrape": 7.0, "dance1": 7.0, "dance2": 7.0,
              "front_flip": 4.0, "back_flip": 4.0, "left_flip": 4.0, "front_jump": 3.0, "front_pounce": 3.0}


class ToolError(ValueError):
    """A rejected tool call; the message is shown to the model. `plan` is a natural-language
    sequence of smaller calls that would do the same thing within the per-call limits."""

    def __init__(self, message, plan=None):
        super().__init__(message)
        self.plan = plan


@dataclass
class Command:
    kind: str                    # velocity | sport | wait | goto | approach | follow | stop | look | status
    tool: str
    args: dict
    params: dict = field(default_factory=dict)
    summary: str = ""
    needs_confirmation: bool = False


def _fn(name, description, properties=None, required=()):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties or {}, "required": list(required)}}}


TOOLS = [
    _fn("move", "Walk in a straight line relative to where you are facing, including backing up. One call "
        "covers at most 4 m, 5 s and 0.8 m/s, so for longer trips call move several times in a row. Give "
        "distance_m or duration_s; with neither you walk 1 m, and 'a little' is about 0.3 m. Leave out "
        "speed_mps unless the user asked for a speed: slowly is 0.2, fast is 0.8.",
        {"direction": {"type": "string", "enum": list(DIRECTIONS)},
         "distance_m": {"type": "number", "description": "metres, up to 4"},
         "duration_s": {"type": "number", "description": "seconds, up to 5"},
         "speed_mps": {"type": "number", "description": "metres per second, up to 0.8 (slow 0.2, fast 0.8)"}},
        ["direction"]),
    _fn("turn", "Turn in place, at most about 280 degrees and 5 s per call (max 57 deg/s); a full circle "
        "is two calls. Leave out angle_deg for a normal quarter turn.",
        {"direction": {"type": "string", "enum": ["left", "right"]},
         "angle_deg": {"type": "number", "description": "only if the user gave an angle or said 'turn around' "
                       "(half turn) or 'a little' (about 30)"},
         "duration_s": {"type": "number", "description": "seconds, up to 5"},
         "speed_dps": {"type": "number", "description": "degrees per second, up to 57"}}),
    _fn("go_to", "Walk to a point relative to where you are now (x_m forward, y_m left; negative for back or "
        "right), or back to where you started. Up to 20 m; you turn toward the point and walk there in "
        "safe legs. For a plain straight line use move instead.",
        {"x_m": {"type": "number"}, "y_m": {"type": "number"},
         "target": {"type": "string", "enum": ["point", "start"], "description": "default point"}}),
    _fn("approach", "Walk up to an object your camera can see, named like 'red ball' or by its id from look. "
        "Stops stop_distance_m in front of it (default 0.5 m; 0.3 m is as close as you can safely get).",
        {"target": {"type": "string"}, "stop_distance_m": {"type": "number", "description": "0.3 to 3 m"}}, ["target"]),
    _fn("follow", "Follow a person you can see, keeping distance_m between you and them (default 1.2 m, 0.8 to "
        "3). target is who: 'me' or 'the person' for the nearest person, or by clothing like 'the person in "
        "red'. It ends by itself after duration_s (default 60, at most 120), when you lose sight of them, or "
        "on stop. You only follow people, never objects, and never switch to someone else.",
        {"target": {"type": "string", "description": "default 'me'"},
         "distance_m": {"type": "number", "description": "0.8 to 3 m"},
         "duration_s": {"type": "number", "description": "seconds, up to 120"}}),
    _fn("look", "See what is around you: objects in front of you with distance and direction, plus the free "
        "space on every side. To check behind or beside you, turn and look again."),
    _fn("posture", "Change posture. stand = balanced and ready to walk; stand_up = stiff locked stand; "
        "lie_down; sit; recovery_stand = get up after a fall; damp = go limp (only if asked). Never needed before "
        "walking, turning, tricks or body poses: you stand up for those automatically.",
        {"name": {"type": "string", "enum": list(POSTURES)}}, ["name"]),
    _fn("trick", "Perform a gesture: hello (wave), stretch, content, heart, scrape (bow greeting), dance1, dance2.",
        {"name": {"type": "string", "enum": list(TRICKS)}}, ["name"]),
    _fn("acrobatic", "Risky acrobatics and special modes (real robot only). They need the user's clear yes: "
        "call with confirmed=false first, ask the user, and call again with confirmed=true only after they "
        "agree. Special modes end when you switch back to the classic gait with body.",
        {"name": {"type": "string", "enum": list(ACROBATICS)}, "confirmed": {"type": "boolean"}}, ["name"]),
    _fn("body", "Adjust body pose or walking style while standing. pitch_deg: + looks up, - looks down; "
        "roll_deg: + leans right; yaw_deg: + twists left (all 0 to reset). speed_level and gait change "
        "how it walks.",
        {"pitch_deg": {"type": "number"}, "roll_deg": {"type": "number"}, "yaw_deg": {"type": "number"},
         "speed_level": {"type": "string", "enum": list(SPEED_LEVELS)},
         "gait": {"type": "string", "enum": list(GAITS)}}),
    _fn("wait", "Pause between steps of a sequence.",
        {"duration_s": {"type": "number", "description": "seconds, up to 30"}}, ["duration_s"]),
    _fn("stop", "Stop moving immediately and cancel everything queued."),
    _fn("get_status", "Your posture, where you are compared with where you started, what you are doing and "
        "what is queued, plus battery and connection, and done_so_far: everything you did since the robot was "
        "connected (distance walked, degrees turned, recent actions and how each ended). Use it to answer what you "
        "have done."),
]
TOOL_NAMES = {t["function"]["name"] for t in TOOLS}
# Tools the fine-tuned models haven't been trained on yet (#16). They are left out of the compact prompt so it
# stays exactly as trained; use --mock or a full-prompt model (e.g. go2-ft-04@full) to try them.
UNTRAINED = {"follow"}
# Fine-tuned models know the tools by heart: declare names only (~270 prompt tokens instead of ~1,450).
COMPACT_TOOLS = [{"type": "function", "function": {"name": t["function"]["name"], "parameters": {"type": "object", "properties": {}}}}
                 for t in TOOLS if t["function"]["name"] not in UNTRAINED]


def _number(args, key, low, high, unit, *, required=False):
    value = args.get(key)
    if value is None:
        if required:
            raise ToolError(f"{key} is required")
        return None
    if isinstance(value, str):
        try:
            value = float(value.strip().split()[0])
        except (ValueError, IndexError):
            raise ToolError(f"{key} must be a number") from None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ToolError(f"{key} must be a number")
    if value <= low:
        raise ToolError(f"{key} must be greater than {low:g}")
    if value > high + 1e-9:
        raise ToolError(f"{key} {value:g} is over the per-call limit of {high:.3g} {unit}")
    return float(value)


def _choice(args, key, options, *, required=False):
    value = args.get(key)
    if value is None:
        if required:
            raise ToolError(f"{key} is required: one of {', '.join(options)}")
        return None
    if isinstance(value, str):
        value = value.strip().lower().replace(" ", "_").replace("-", "_")
    if value not in options:
        raise ToolError(f"{key} must be one of {', '.join(options)}")
    return value


def _exact(name, args, allowed):
    if not isinstance(args, dict):
        raise ToolError("arguments must be an object")
    extra = set(args) - set(allowed)
    if extra:
        raise ToolError(f"{name} does not take {', '.join(sorted(extra))}")


def _hinted(fn, hint, plan=None):
    try:
        return fn()
    except ToolError as error:
        if "per-call limit" in str(error):
            raise ToolError(f"{error}. {hint}", plan) from None
        raise


def describe_calls(calls):
    """Natural phrase for a list of planned calls, e.g. 'move forward 4 m, then 4 m, then 2 m'."""
    parts = []
    for call in calls:
        a = call["arguments"]
        if call["name"] == "move":
            extent = f"{a['distance_m']:g} m" if "distance_m" in a else f"{a['duration_s']:g} seconds"
            parts.append(f"{a['direction'].replace('_', '-')} {extent}" + (f" at {a['speed_mps']:g} m/s" if "speed_mps" in a else ""))
        elif call["name"] == "turn":
            parts.append(f"{a.get('direction', 'left')} {a['angle_deg']:g} degrees")
    verb = calls[0]["name"] if calls else ""
    return f"{verb} " + ", then ".join(parts)


def _plan(name, base, key, pieces):
    return [{"name": name, "arguments": {**base, key: p}} for p in pieces]


def split(total, step):
    """Equal-ish pieces of at most `step`, largest first: 10 by 4 -> [4, 4, 2]."""
    pieces = []
    while total > 1e-9:
        pieces.append(round(min(step, total), 2))
        total -= step
    return pieces


def _move(args):
    _exact("move", args, ("direction", "distance_m", "duration_s", "speed_mps"))
    direction = _choice(args, "direction", list(DIRECTIONS), required=True)
    raw = args.get("distance_m")
    pieces = split(float(raw), L.MAX_DISTANCE_M) if isinstance(raw, (int, float)) and not isinstance(raw, bool) and raw > 0 else []
    distance = _hinted(lambda: _number(args, "distance_m", 0, L.MAX_DISTANCE_M, "m"),
                       f"Do it now as {len(pieces)} move calls in a row with distance_m "
                       f"{', then '.join(f'{p:g}' for p in pieces)} (direction {direction}).",
                       _plan("move", {"direction": direction, **({"speed_mps": args["speed_mps"]} if isinstance(args.get("speed_mps"), (int, float)) else {})}, "distance_m", pieces))
    raw = args.get("duration_s")
    pieces = split(float(raw), L.MAX_DURATION_S) if isinstance(raw, (int, float)) and not isinstance(raw, bool) and raw > 0 else []
    duration = _hinted(lambda: _number(args, "duration_s", 0, L.MAX_DURATION_S, "s"),
                       f"Do it now as {len(pieces)} move calls in a row with duration_s "
                       f"{', then '.join(f'{p:g}' for p in pieces)}.",
                       _plan("move", {"direction": direction}, "duration_s", pieces))
    speed = _hinted(lambda: _number(args, "speed_mps", 0, L.MAX_SPEED_MPS, "m/s"),
                    "Do not move yet. Tell the user your top speed is 0.8 m/s and ask if you should go that fast.")
    note = None
    if distance is None and duration is None:
        distance, note = 1.0, "no distance given, walking the default 1 m"
    if distance is not None and duration is not None:
        implied = distance / duration
        if speed is not None and abs(speed - implied) > .05:
            raise ToolError("distance, duration and speed disagree; give two of them")
        if implied > L.MAX_SPEED_MPS + 1e-9:
            raise ToolError(f"{distance:g} m in {duration:g} s needs {implied:.2f} m/s, over the 0.8 m/s limit")
        speed, duration = implied, None
    if speed is None:
        speed = L.default_speed(distance)
    if distance is not None and distance / speed > L.MAX_DURATION_S + 1e-6:
        chunk = speed * L.MAX_DURATION_S
        pieces = split(distance, chunk)
        raise ToolError(f"{distance:g} m at {speed:g} m/s takes {distance / speed:.1f} s, over 5 s per move. "
                        f"Do it now as {len(pieces)} moves at {speed:g} m/s with distance_m "
                        f"{', then '.join(f'{p:g}' for p in pieces)}.",
                        _plan("move", {"direction": direction, "speed_mps": speed}, "distance_m", pieces))
    ux, uy = DIRECTIONS[direction]
    norm = math.hypot(ux, uy)
    ux, uy = ux / norm, uy / norm
    extent = f"{distance:g} m" if distance is not None else f"{duration:g} s"
    return [Command("velocity", "move", args, {
        "vx": ux * speed, "vy": uy * speed, "wz": 0., "ux": ux, "uy": uy, "speed": speed,
        "distance": distance, "duration": duration, "note": note},
        summary=f"move {direction.replace('_', '-')} {extent} at {speed:.2f} m/s")]


def _turn(args):
    _exact("turn", args, ("direction", "angle_deg", "duration_s", "speed_dps"))
    direction = _choice(args, "direction", ["left", "right"]) or "left"
    raw = args.get("angle_deg")
    halves = f"{float(raw) / 2:g}" if isinstance(raw, (int, float)) and not isinstance(raw, bool) and raw > 0 else "180"
    angle = _hinted(lambda: _number(args, "angle_deg", 0, L.MAX_TURN_DEG, "degrees"),
                    f"Do it now as 2 turn calls in a row, each angle_deg {halves} (direction {direction}).",
                    _plan("turn", {"direction": direction}, "angle_deg", [float(halves)] * 2))
    duration = _hinted(lambda: _number(args, "duration_s", 0, L.MAX_DURATION_S, "s"),
                       "For longer, call turn several times.")
    rate_dps = _hinted(lambda: _number(args, "speed_dps", 0, L.MAX_TURN_RATE_DPS, "deg/s"),
                       "Do not turn yet. Tell the user the fastest turn is 57 degrees per second and ask if that's okay.")
    note = None
    if angle is None and duration is None:
        angle, note = 90.0, "no angle given, turning the default 90 degrees"
    rate = None if rate_dps is None else math.radians(rate_dps)
    if angle is not None and duration is not None:
        implied = math.radians(angle) / duration
        if rate is not None and abs(rate - implied) > math.radians(3):
            raise ToolError("angle, duration and speed disagree; give two of them")
        if implied > L.MAX_TURN_RATE_RPS + 1e-9:
            raise ToolError(f"{angle:g} degrees in {duration:g} s is faster than the 57 deg/s limit")
        rate, duration = implied, None
    if rate is None:
        rate = L.default_turn_rate(None if angle is None else math.radians(angle))
    if angle is not None and math.radians(angle) / rate > L.MAX_DURATION_S + 1e-6:
        raise ToolError(f"{angle:g} degrees at {math.degrees(rate):.0f} deg/s takes over 5 s; split it into "
                        f"turns of at most {math.degrees(rate) * L.MAX_DURATION_S:.0f} degrees, or turn faster.")
    sign = 1 if direction == "left" else -1
    extent = f"{angle:g}°" if angle is not None else f"{duration:g} s"
    return [Command("velocity", "turn", args, {
        "vx": 0., "vy": 0., "wz": sign * rate, "ux": 0., "uy": 0., "speed": 0., "turn_rate": rate,
        "angle": None if angle is None else math.radians(angle), "duration": duration, "note": note},
        summary=f"turn {direction} {extent} at {math.degrees(rate):.0f}°/s")]


def _go_to(args):
    _exact("go_to", args, ("x_m", "y_m", "target"))
    named = args.get("target")
    if isinstance(named, str) and named.strip().lower() not in ("point", "start", "") and "x_m" not in args and "y_m" not in args:
        # "go to the chair": an object, not a point -> walk up to it.
        return _approach({"target": named})
    target = _choice(args, "target", ["point", "start"]) or "point"
    if target == "start":
        return [Command("goto", "go_to", args, {"target": "start"}, summary="return to the start")]
    x = args.get("x_m", 0) or 0
    y = args.get("y_m", 0) or 0
    for key, value in (("x_m", x), ("y_m", y)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ToolError(f"{key} must be a number")
    distance = math.hypot(x, y)
    if distance < .05:
        raise ToolError("the point is where the robot already is; give x_m and/or y_m")
    if distance > 20:
        raise ToolError(f"{distance:.1f} m is over the 20 m go_to limit; use several calls")
    return [Command("goto", "go_to", args, {"target": "point", "x": float(x), "y": float(y)},
                    summary=f"go to {x:g} m forward, {y:g} m left")]


def _approach(args):
    _exact("approach", args, ("target", "stop_distance_m"))
    target = args.get("target")
    if not isinstance(target, str) or not target.strip():
        raise ToolError("target must name an object, e.g. 'red ball' or 'obj2'")
    stop = args.get("stop_distance_m")
    if isinstance(stop, (int, float)) and not isinstance(stop, bool) and 0 <= stop < .3:
        raise ToolError("0.3 m is as close as you can safely get to an object; use stop_distance_m 0.3 or tell the user")
    stop = .5 if stop is None else _number(args, "stop_distance_m", .29, 3.0, "m")
    return [Command("approach", "approach", args, {"target": target.strip().lower(), "stop_distance": stop},
                    summary=f"approach the {target.strip()}")]


PEOPLE = {"me", "person", "people", "man", "woman", "guy", "girl", "boy", "lady", "kid", "child", "him", "her",
          "them", "someone", "somebody", "user", "owner", "human"}
CLOTHES = {"in", "with", "wearing", "shirt", "tshirt", "t", "top", "jumper", "hoodie", "jacket", "sweater", "coat",
           "dress", "clothes", "the", "a", "an", "that", "this", "one", "over", "there", "nearest", "closest"}
COLOURS = {"red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink", "brown", "white", "grey", "gray",
           "black"}


def _follow(args):
    _exact("follow", args, ("target", "distance_m", "duration_s"))
    target = args.get("target")
    if target is None or (isinstance(target, str) and not target.strip()):
        target = "me"
    if not isinstance(target, str):
        raise ToolError("target must say who to follow, e.g. 'me' or 'the person in red'")
    words = re.findall(r"[a-z]+", target.lower())
    colours = [w for w in words if w in COLOURS]
    other = [w for w in words if w not in PEOPLE and w not in CLOTHES and w not in COLOURS]
    if other or len(set(colours)) > 1:
        raise ToolError(f"follow only follows people ('me', 'the person in red'), not {target!r}; "
                        "to walk up to an object use approach")
    lo, hi = L.FOLLOW_DISTANCE_M
    distance = _number(args, "distance_m", lo, hi, "m")
    distance = L.DEFAULT_FOLLOW_DISTANCE_M if distance is None else distance
    duration = _number(args, "duration_s", 1, L.MAX_FOLLOW_S, "s")
    duration = L.DEFAULT_FOLLOW_S if duration is None else duration
    colour = {"gray": "grey"}.get(colours[0], colours[0]) if colours else None
    who = "you" if words in ([], ["me"]) else target.strip()
    return [Command("follow", "follow", args, {"target": target.strip(), "color": colour, "distance": distance,
                                               "duration": duration},
                    summary=f"follow {who} at {distance:g} m for up to {duration:g} s")]


def _posture(args):
    _exact("posture", args, ("name",))
    name = _choice(args, "name", list(POSTURES), required=True)
    return [Command("sport", "posture", args, {"name": name, "expected_s": EXPECTED_S[name]},
                    summary=name.replace("_", " "))]


def _trick(args):
    _exact("trick", args, ("name",))
    name = _choice(args, "name", list(TRICKS), required=True)
    return [Command("sport", "trick", args, {"name": name, "expected_s": EXPECTED_S[name]}, summary=name)]


def _acrobatic(args):
    _exact("acrobatic", args, ("name", "confirmed"))
    name = _choice(args, "name", list(ACROBATICS), required=True)
    confirmed = args.get("confirmed", False)
    if not isinstance(confirmed, bool):
        raise ToolError("confirmed must be true or false")
    return [Command("sport", "acrobatic", args, {"name": name, "confirmed": confirmed,
                                                 "expected_s": EXPECTED_S.get(name, 2.)},
                    summary=name.replace("_", " "), needs_confirmation=True)]


def _body(args):
    _exact("body", args, ("pitch_deg", "roll_deg", "yaw_deg", "speed_level", "gait"))
    commands = []
    angles = {}
    for key, limit in (("pitch_deg", L.MAX_BODY_PITCH_DEG), ("roll_deg", L.MAX_BODY_ROLL_DEG), ("yaw_deg", L.MAX_BODY_YAW_DEG)):
        value = args.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ToolError(f"{key} must be a number")
        if abs(value) > limit + 1e-9:
            raise ToolError(f"{key} {value:g} is beyond the ±{limit:g} degree body limit")
        angles[key] = float(value)
    if angles:
        commands.append(Command("sport", "body", args, {
            "name": "euler", "roll": math.radians(angles.get("roll_deg", 0.)),
            "pitch": -math.radians(angles.get("pitch_deg", 0.)), "yaw": math.radians(angles.get("yaw_deg", 0.)),
            "expected_s": 1.0}, summary="body pose " + ", ".join(f"{k[:-4]} {v:g}°" for k, v in angles.items())))
    level = _choice(args, "speed_level", list(SPEED_LEVELS))
    if level:
        commands.append(Command("sport", "body", args, {"name": "speed_level", "level": SPEED_LEVELS[level], "expected_s": .2},
                                summary=f"speed level {level}"))
    gait = _choice(args, "gait", list(GAITS))
    if gait:
        commands.append(Command("sport", "body", args, {"name": "gait", "gait": gait, "expected_s": .5},
                                summary=f"gait {gait.replace('_', ' ')}"))
    if not commands:
        raise ToolError("give at least one of pitch_deg, roll_deg, yaw_deg, speed_level or gait")
    return commands


def _wait(args):
    _exact("wait", args, ("duration_s",))
    seconds = _number(args, "duration_s", 0, L.MAX_WAIT_S, "s", required=True)
    return [Command("wait", "wait", args, {"duration": seconds}, summary=f"wait {seconds:g} s")]


def _simple(kind):
    def build(args):
        _exact(kind, args or {}, ())
        return [Command({"stop": "stop", "look": "look", "get_status": "status"}[kind], kind, {}, summary=kind)]
    return build


RESOLVERS = {"move": _move, "turn": _turn, "go_to": _go_to, "approach": _approach, "follow": _follow,
             "posture": _posture,
             "trick": _trick, "acrobatic": _acrobatic, "body": _body, "wait": _wait,
             "stop": _simple("stop"), "look": _simple("look"), "get_status": _simple("get_status")}


def resolve(name, args):
    if name not in RESOLVERS:
        raise ToolError(f"unknown tool {name!r}; available: {', '.join(sorted(TOOL_NAMES))}")
    return RESOLVERS[name](args if args is not None else {})
