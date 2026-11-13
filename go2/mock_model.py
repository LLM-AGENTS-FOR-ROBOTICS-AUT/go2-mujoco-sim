"""Scripted stand-in for the LLM (app.py --mock): exercises the UI and tools without a model."""
from __future__ import annotations

import re
import json


class MockModel:
    model_id = "mock"
    options = {}
    loaded = False

    def check(self):
        return {"model": "Mock (scripted, no LLM)", "label": "Mock", "ready": True, "loaded": False}

    def chat(self, messages, tools=None, cancelled=lambda: False, on_text=None, timeout=60):
        last = messages[-1]
        if last["role"] == "tool":
            try:
                result = json.loads(last.get("content", "{}"))
            except (ValueError, TypeError):
                result = {}
            if result.get("error"):
                return self._reply("I couldn't do that: " + str(result["error"]))
            name = last.get("tool_name")
            if name == "remember_place":
                place = result.get("place", {}).get("name")
                return self._reply(f"I've remembered this spot as {place}." if place else "That spot wasn't saved.")
            if name == "name_room":
                room = result.get("room", {}).get("name")
                return self._reply(f"Got it, this room is the {room}." if room else "I couldn't name this room.")
            if name == "list_places":
                names = [p["name"] for p in result.get("places", [])]
                return self._reply("I remember " + ", ".join(names) + "." if names else "I haven't saved any places yet.")
            if name == "query_memory":
                user = next((m["content"].lower() for m in reversed(messages) if m["role"] == "user"), "")
                destination = self._destination(user)
                if destination:
                    key = re.sub(r"[\s_-]+", " ", destination).casefold()
                    match = next((p["name"] for p in result.get("places", []) + result.get("rooms", [])
                                  if re.sub(r"[\s_-]+", " ", p["name"]).casefold() == key), None)
                    if match is None and any(re.sub(r"[\s_-]+\d+$", "", r["name"]).casefold() == key
                                             for r in result.get("rooms", [])):
                        match = destination            # "the bedroom": the executor picks the nearest one
                    return self._calls([("go_to_place", {"name": match})] if match else
                                       [("approach", {"target": destination})])
                remembered = [p["name"] for p in result.get("places", [])]
                remembered += [" ".join(filter(None, (p.get("color"), p["label"])))
                               for p in result.get("objects", [])]
                if not remembered and result.get("unavailable_objects"):
                    return self._reply("The last remembered location was checked and is now clear. "
                                       "The object may have moved; I don't know its new location yet.")
                return self._reply("I remember " + ", ".join(remembered) + "." if remembered else
                                   "I don't remember anything matching that.")
            return {"content": "On it!", "tool_calls": [], "stats": {"mock": True}}
        raw_text = last["content"].split("\n\nUser:", 1)[-1].strip()
        text = raw_text.lower()
        if text.startswith("robot:"):
            # Execution reports are facts to repeat, not new movement requests.
            report = raw_text[6:].split(". Tell the user", 1)[0].strip()
            return self._reply(report + ".")
        text = re.sub(r"^\[robot update:.*?\]\s*", "", text, flags=re.S)
        calls = []
        number = re.search(r"(\d+(?:\.\d+)?)", text)
        save_place = re.search(r"\bremember (?:this (?:spot|place|location)|here)(?: as)?\s+(?:the\s+)?(.+?)\s*[.!]?$", text)
        explicit_place = re.search(r"\b(?:go|return) to (?:the )?(?:named )?place\s+(.+?)\s*[.!]?$", text)
        room_name = re.search(r"^(?:this (?:room )?is (?:the|my)|(?:call|name) this room(?: the)?)\s+(.+?)\s*[.!]?$", text)
        memory = re.search(r"\b(?:where (?:is|was)|what do you remember about|do you remember)\s+(?:the )?(.+?)\s*[?!]?$", text)
        destination = self._destination(text)
        if save_place:
            calls = [("remember_place", {"name": save_place[1].strip()})]
        elif room_name:
            calls = [("name_room", {"name": room_name[1].strip()})]
        elif explicit_place:
            calls = [("go_to_place", {"name": explicit_place[1].strip()})]
        elif destination:
            # Resolve saved names from the executor, including after a restart.
            calls = [("query_memory", {"target": destination, "limit": 10})]
        elif "what places" in text or "list places" in text:
            calls = [("list_places", {})]
        elif memory:
            calls = [("query_memory", {"target": memory[1].strip()})]
        elif "what do you remember" in text:
            calls = [("query_memory", {})]
        elif "stop following" in text:
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
            return self._calls(calls)
        return {"content": "Mock reply: I'm a scripted stand-in, no language model is running.", "tool_calls": [],
                "stats": {"mock": True}}

    @staticmethod
    def _destination(text):
        match = re.search(r"\b(?:go|return|come back|walk back) to (?:the )?(.+?)\s*[.!]?$", text)
        return match[1].strip() if match else None

    @staticmethod
    def _reply(text):
        return {"content": text, "tool_calls": [], "stats": {"mock": True}}

    @staticmethod
    def _calls(calls):
        return {"content": "", "tool_calls": [{"name": n, "arguments": a} for n, a in calls],
                "stats": {"mock": True}}

    def cancel(self):
        pass

    def unload(self):
        return {"unloaded": True}
