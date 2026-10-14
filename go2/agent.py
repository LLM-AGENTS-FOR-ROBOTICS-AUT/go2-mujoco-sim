"""Conversation agent: natural chat with tool use, driving the executor.

One user turn runs a short tool loop: the model's tool calls are validated and
handed to the executor as they arrive (so the robot starts moving at once), the
results go back to the model, and the loop ends when it replies in words.
"""
from __future__ import annotations

import copy
import json
import re
import threading
import uuid

from . import tools as T
from .adapters.base import AdapterError
from .llm import Cancelled, ModelError, recover_calls

SYSTEM_PROMPT = """You are Go2, a cheerful robot dog chatting with your owner by voice. Talk like a friendly companion: short (one sentence, two at most), warm and casual, like "On it!" or "Hmm, I can't see a ball from here." No lists, markdown or emojis.

- Act only through your tools, and never claim you did something without calling one. Actions run in order and take time, so talk about them in the present or future tense.
- Questions, what-ifs, things someone else said and things the user tells you not to do are conversation, not requests to act.
- One call moves at most 4 m, for at most 5 s; for more, call the tool several times in a row. Your top speed is 0.8 m/s: if asked to go faster, say so and offer it instead of calling a tool.
- Use only the numbers the user gave, converted to metres and degrees, and leave out anything they didn't say. Any number you mention must come from the user or a tool.
- If a tool reports a limit or a problem, follow its advice or ask the user. Never quietly change their request.
- Tool results are the truth: if one says blocked, failed or rejected, tell the user it did not happen and why.
- Walking, turning, tricks and body poses stand you up automatically, so never call posture stand before them.
- For acrobatics, call the acrobatic tool and follow what it says about confirming with the user.
- If the user says stop, stop at once.
- Messages that start with "Robot:" come from your own body, not from the user.
- To tell the user what you have done or how far you went, call get_status and answer from its done_so_far.
- If the user doesn't say how far or how much, the default is fine; if they don't say which way or which thing, ask one short question."""

SPEAKER_PREFIX = re.compile(r"^\s*(?:\[\s*)?(?:robot|go2|assistant|user)\s*:\s*", re.I)
DERAILED = re.compile(r"^(?:" + "|".join(sorted(T.TOOL_NAMES)) + r")\b(?!\{)")
# Spoken stop phrases handled instantly, without the model. Stopping is always safe, so short
# utterances that start with one of these count even if more words follow ("stop, wait").
# Short prompt for models fine-tuned on Go2 behaviour (see training/); used with COMPACT_TOOLS.
COMPACT_SYSTEM = "You are Go2, a cheerful robot dog talking with your owner. Reply briefly and act through your tools."

STOP_PHRASE = re.compile(r"^(?:please\s+)?(?:stop|halt|freeze|whoa|emergency stop|stay(?: there| put| still| where you are)?|"
                         r"don'?t move|cancel(?: that| it| everything)?|abort)\b", re.I)
MAX_STEPS = 8


def is_stop(text):
    clean = " ".join("".join(ch if ch.isalnum() or ch in " '" else " " for ch in text.lower()).split())
    return len(clean.split()) <= 6 and bool(STOP_PHRASE.match(clean))


PHRASE_BREAK = re.compile(r"[.!?…](?=\s)|[,;:](?=\s)")


class SpeechStream:
    """What Go2 says, phrase by phrase as the model writes it, for the page's voice. Every turn ends with an
    end marker; a newer turn's phrases replace an older turn's (the page cuts the old speech off)."""

    def __init__(self):
        self.cond = threading.Condition()
        self.items, self.next_id = [], 1

    def _put(self, item):
        with self.cond:
            item["id"] = self.next_id
            self.next_id += 1
            self.items.append(item)
            del self.items[:-300]
            self.cond.notify_all()

    def say(self, epoch, text, proactive=False):
        self._put({"epoch": epoch, "text": text, "proactive": proactive})

    def end(self, epoch):
        self._put({"epoch": epoch, "end": True})

    def after(self, last, timeout=20.):
        """Items newer than `last` (waits up to `timeout` for one); `last` < 0 only asks where the stream is."""
        with self.cond:
            if last >= 0 and last < self.next_id:
                self.cond.wait_for(lambda: self.next_id - 1 > last, timeout)
            newest = self.next_id - 1
            return {"last": newest, "items": [i for i in self.items if i["id"] > last] if 0 <= last <= newest else []}


class Phrases:
    """Cuts one model step's text into phrases as it streams in: at sentence ends, and at commas once a
    phrase is long enough. Anything that could be a tool call written as text is never spoken early; the
    step's final words are completed by finish()."""

    def __init__(self, emit, enabled=True):
        self.emit, self.enabled, self.sent = emit, enabled, ""

    def feed(self, raw):
        if not self.enabled:
            return
        text = clean_reply(raw)
        if "{" in text or "<|" in (raw or "") or DERAILED.match(text) or not text.startswith(self.sent):
            self.enabled = False
            return
        pending = text[len(self.sent):]
        while True:
            cut = None
            for match in PHRASE_BREAK.finditer(pending):
                if len(pending[:match.end()].strip()) >= (12 if pending[match.start()] in ".!?…" else 30):
                    cut = match.end()
                    break
            if cut is None:
                return
            self.emit(pending[:cut].strip())
            self.sent += pending[:cut]
            pending = pending[cut:]

    def finish(self, content):
        """The step's final words: whatever hasn't been spoken yet."""
        rest = content[len(self.sent):].strip() if content.startswith(self.sent) else ("" if self.sent.strip() else content)
        if rest:
            self.emit(rest)
        self.sent = content


class Agent:
    def __init__(self, model, executor, on_change=lambda: None, speak=lambda text: None):
        self.model = model
        self.executor = executor
        self.on_change = on_change
        self.speak = speak
        self.lock = threading.RLock()
        self.messages = []          # model-visible history (user / assistant / tool)
        self.transcript = []        # UI-visible history
        self.user_turns = 0
        self.epoch = 0
        self.busy = False
        self.error = None
        self.last_request = None
        self.pending_events = []    # app notices: the model hears them with the next user message
        self.pending_reports = []   # robot reports that came in while Go2 was talking: spoken right after
        self.closed = False
        self.robot_line = lambda: ""
        self.robot_snapshot = lambda: None   # robot scene/pose/posture, saved with each turn for replays
        self.last_robot_line = None
        self.speech = SpeechStream()

    # ---- public --------------------------------------------------------------
    def submit(self, text, voice=False):
        text = (text or "").strip()
        if not text:
            raise ValueError("Say or type something first.")
        if len(text) > 2000:
            raise ValueError("That message is too long.")
        if is_stop(text):
            self.stop(spoken=text)
            return
        with self.lock:
            if self.busy:
                self._cancel_turn()     # a new utterance interrupts the reply in progress
            self.user_turns += 1
            self.epoch += 1
            epoch = self.epoch
            self.busy, self.error = True, None
            self.messages.append({"role": "user", "content": self._with_events(text)})
            self.transcript.append({"role": "user", "content": text, "voice": voice, "turn": self._turn_start()})
        self.on_change()
        threading.Thread(target=self._turn, args=(epoch,), daemon=True).start()

    def stop(self, spoken=None):
        result = self.executor.stop("user")
        with self.lock:
            self._cancel_turn()
            if spoken:
                self.transcript.append({"role": "user", "content": spoken, "voice": True})
            reply = "Okay, stopped."
            self.messages.append({"role": "user", "content": spoken or "Stop!"})
            self.messages.append({"role": "assistant", "content": reply})
            self.transcript.append({"role": "assistant", "content": reply, "speak": True,
                                    "detail": {"stop": result}})
            self.speech.say(self.epoch, reply)
            self.speech.end(self.epoch)
        self.on_change()
        return result

    def interrupt(self):
        with self.lock:
            self._cancel_turn()
            self.transcript.append({"role": "system", "content": "Reply interrupted."})
        self.on_change()

    def clear(self):
        self.executor.stop("context cleared")
        with self.lock:
            self._cancel_turn()
            self.messages, self.transcript, self.pending_events, self.pending_reports = [], [], [], []
            self.last_request = None
            self.last_robot_line = None
            self.error = None
        self.on_change()

    def note(self, text):
        """Record a robot/app event; the model hears about it with the next user message."""
        with self.lock:
            self.pending_events.append(text)
            self.transcript.append({"role": "event", "content": text})
        self.on_change()

    def close(self):
        with self.lock:
            self.closed = True
            self._cancel_turn()

    def snapshot(self):
        with self.lock:
            return {"transcript": list(self.transcript), "busy": self.busy, "error": self.error,
                    "epoch": self.epoch, "last_request": self.last_request}

    # ---- internals -------------------------------------------------------------------
    def _cancel_turn(self):
        self.epoch += 1
        self.busy = False
        try:
            self.model.cancel()
        except Exception:
            pass

    def _with_events(self, text):
        """Prefix robot news and a changed robot state. Kept out of the system prompt so the
        model's prompt cache (system + tools + history) stays valid between calls."""
        notes = self.pending_reports + self.pending_events
        self.pending_events, self.pending_reports = [], []
        try:
            line = self.robot_line()
        except Exception:
            line = ""
        if line and line != self.last_robot_line:
            notes.append("now: " + line)
            self.last_robot_line = line
        return f"Robot: {'; '.join(n.rstrip('.') for n in notes)}.\n\nUser: {text}" if notes else text

    @property
    def compact(self):
        return getattr(self.model, "prompt_style", "full") == "compact"

    def _system(self):
        return {"role": "system", "content": COMPACT_SYSTEM if self.compact else SYSTEM_PROMPT}

    def _tools(self):
        return T.COMPACT_TOOLS if self.compact else T.TOOLS

    def _turn(self, epoch, proactive=False):
        """Run one model turn. Model calls and robot commands happen outside self.lock so that
        stop(), interrupt() and clear() are never blocked by inference."""
        cancelled = lambda: epoch != self.epoch or self.closed
        group = "g" + uuid.uuid4().hex[:6]
        spoken, failed_calls = [], set()
        nudges, extra, asked_reply, calls_made = 0, [], False, 0
        self._plan = None
        result = {}
        try:
            for _ in range(MAX_STEPS):
                with self.lock:
                    if cancelled():
                        return
                    messages = [self._system()] + list(self.messages) + extra
                    self.last_request = {"messages": messages, "tools": self._tools(), "options": self.model.options}
                say = lambda text: self._say(epoch, text, proactive)
                # A step after a rejected over-limit call may be replaced by a go-ahead: don't speak it early.
                phrases = Phrases(say, enabled=not self._plan)
                result = self._chat(messages, cancelled, phrases.feed)
                content, calls = clean_reply(result["content"]), result["tool_calls"]
                if not calls and self._plan and nudges < 3:
                    # A tool asked for a split but the model stopped short. Give a go-ahead naming the
                    # remaining calls; it is sent with the next call only, never stored in history.
                    nudges += 1
                    extra = [{"role": "assistant", "content": content},
                             {"role": "user", "content": f"Okay, do it: {T.describe_calls(self._plan)}."}]
                    continue
                extra = []
                if not calls and not content and calls_made and not asked_reply:
                    # Calls were made but nothing was said: ask for one sentence, without tools.
                    asked_reply = True
                    # Same tools as every other call so the model's prompt cache stays valid.
                    phrases = Phrases(say)
                    reply = self.model.chat(messages + [{"role": "user", "content": REPLY_NUDGE}], self._tools(),
                                            cancelled=cancelled, on_text=phrases.feed)
                    content = clean_reply(reply["content"])
                if cancelled():
                    return
                assistant = {"role": "assistant", "content": content}
                if calls:
                    assistant["tool_calls"] = [{"function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]
                with self.lock:
                    if cancelled():
                        return
                    self.messages.append(assistant)
                if content:
                    spoken.append(content)
                    phrases.finish(content)
                if not calls:
                    break
                calls_made += len(calls)
                stuck, halted = False, None
                for call in calls:
                    key = json.dumps([call["name"], call["arguments"]], sort_keys=True, default=str)
                    stuck |= key in failed_calls    # the same rejected call again: stop looping
                    if halted and call["name"] != "stop":
                        # Written before the model knew an earlier step was refused; it may depend on it.
                        outcome = {"status": "cancelled", "reason": f"previous step {halted}"}
                    else:
                        outcome = self._execute(call, group, proactive)
                        halted = halted or next((r["status"] for r in [outcome, *outcome.get("results", [])]
                                                 if r.get("status") in ("blocked", "failed")), None)
                    if "error" in outcome:
                        failed_calls.add(key)
                    with self.lock:
                        if cancelled():
                            return
                        self.messages.append({"role": "tool", "tool_name": call["name"] or "unknown",
                                              "content": json.dumps(outcome, separators=(",", ":"))})
                        self.transcript.append({"role": "tool", "name": call["name"], "arguments": call["arguments"],
                                                "result": outcome})
                    self.on_change()
                if stuck:
                    break
            reply = " ".join(spoken).strip() or "Okay!"
            if not spoken:
                self._say(epoch, reply, proactive)
            with self.lock:
                if cancelled():
                    return
                self.transcript.append({"role": "assistant", "content": reply, "speak": True, "proactive": proactive,
                                        "stats": result.get("stats"), "msg_end": len(self.messages)})
        except Cancelled:
            return
        except ModelError as error:
            with self.lock:
                if not cancelled():
                    self.error = str(error)
                    self.transcript.append({"role": "error", "content": str(error)})
        except Exception as error:   # keep the app alive and the failure visible
            with self.lock:
                if not cancelled():
                    self.error = f"Unexpected error: {error}"
                    self.transcript.append({"role": "error", "content": self.error})
        finally:
            follow_up = None
            with self.lock:
                if epoch == self.epoch:
                    self.busy = False
                    self.speech.end(epoch)
                    follow_up = self._report_turn()
            self.on_change()
            if follow_up:
                threading.Thread(target=self._turn, args=(follow_up, True), daemon=True).start()

    def _say(self, epoch, text, proactive=False):
        """Hand a phrase to the voice, unless this turn has been cancelled meanwhile."""
        if text and epoch == self.epoch and not self.closed:
            self.speech.say(epoch, text, proactive)

    def _chat(self, messages, cancelled, on_text=None):
        """One model call, recovering tool calls Gemma wrote as text and retrying a derailed reply once."""
        for attempt in range(2):
            result = self.model.chat(messages, self._tools(), cancelled=cancelled,
                                     on_text=on_text if attempt == 0 else None)
            if not result["tool_calls"] and result["content"]:
                calls, rest = recover_calls(result["content"], T.TOOL_NAMES)
                if calls:
                    result["tool_calls"], result["content"] = calls, rest
            derailed = (not result["tool_calls"] and DERAILED.match(result["content"] or "")) or \
                (not result["tool_calls"] and not result["content"])
            if not derailed or attempt:
                return result
        return result

    def _execute(self, call, group, proactive=False):
        name, args = call["name"], call["arguments"]
        try:
            commands = T.resolve(name, args)
        except T.ToolError as error:
            if error.plan:
                self._plan = list(error.plan)
            return {"error": str(error)}
        if proactive and commands and all(c.kind not in ("stop", "look", "status") for c in commands):
            # Nobody asked for anything new in a robot update; a call for work that is already running or
            # queued only repeats it (the model sometimes re-issues what the update lists as queued).
            with self.executor.lock:
                pending = [t.command.summary for t in [self.executor.current, *self.executor.queue]
                           if t is not None and t.generation == self.executor.generation]
            if all(c.summary in pending for c in commands):
                return {"status": "already_queued", "action": " then ".join(c.summary for c in commands)}
        if self._plan and self._plan[0]["name"] == name and all(
                isinstance(args.get(k), (int, float)) and abs(args[k] - v) < .05 if isinstance(v, (int, float)) else args.get(k, v) == v
                for k, v in self._plan[0]["arguments"].items()):
            self._plan.pop(0)
        elif self._plan and name in ("stop", "look", "get_status"):
            self._plan = None
        results = self.executor.submit(commands, group, self.user_turns)
        if len(results) == 1:
            return results[0]
        return {"results": results}

    # ---- executor events -> proactive speech ------------------------------------------
    def on_task_event(self, task):
        if task.reported:      # the tool result already told the model it was blocked or failed
            return
        status = task.status
        tool = task.command.tool
        group_done = task.detail.get("group_remaining", 0) == 0
        group = [t for t in self.executor.history if t.group == task.group]
        noteworthy = status in ("blocked", "failed") or (
            status == "completed" and group_done and (len(group) > 1 or tool in ("go_to", "approach")))
        if status in ("stopped", "cancelled") or not noteworthy:
            return
        # What other requests still have queued. Read now, under the executor lock and before the agent lock
        # (the order elsewhere is agent -> executor). Not executor.current or the status line: this event fires
        # after the finished task is cleared and before the next one starts, so the robot looks idle here.
        with self.executor.lock:
            still_to_do = [t.command.summary for t in self.executor.queue if t.generation == self.executor.generation]
        summary = describe_outcome(task, group, still_to_do)
        with self.lock:
            if self.closed:
                return
            self.pending_reports.append(summary)
            if self.busy:
                return                  # spoken as soon as the current reply ends (see _turn)
            epoch = self._report_turn()
        self.on_change()
        threading.Thread(target=self._turn, args=(epoch, True), daemon=True).start()

    def _report_turn(self):
        """Under self.lock: start a robot-update turn for the waiting reports; its epoch, or None."""
        if not self.pending_reports or self.closed:
            return None
        reports, self.pending_reports = self.pending_reports, []
        self.epoch += 1
        self.busy = True
        text = reports[0] if len(reports) == 1 else "; ".join(r.rstrip(".") for r in reports) + "."
        self.messages.append({"role": "user", "content": f"Robot: {text}"})
        for i, report in enumerate(reports):
            self.transcript.append({"role": "event", "content": report, **({"turn": self._turn_start()} if i == 0 else {})})
        return self.epoch

    def _turn_start(self):
        """Under self.lock, right after a turn's message is added: where it starts, and the robot's state."""
        try:
            robot = self.robot_snapshot()
        except Exception:
            robot = None
        return {"msg": len(self.messages) - 1, "robot": robot}

    # ---- flagged replies (the page's "bad reply" button) -------------------------------
    def flagged_turn(self, index):
        """The turn that produced transcript row `index` (one of Go2's replies), for replaying it: the model
        history before it, the message that started it, what Go2 did, and the robot's state at the start."""
        with self.lock:
            rows = self.transcript
            if not (0 <= index < len(rows)) or rows[index]["role"] != "assistant":
                raise ValueError("Pick one of Go2's replies.")
            start = next((i for i in range(index, -1, -1) if "turn" in rows[i]), None)
            if start is None or "msg_end" not in rows[index]:
                raise ValueError("That reply can't be replayed (it wasn't made by the model).")
            first, end = rows[start]["turn"]["msg"], rows[index]["msg_end"]
            return {"kind": "user" if rows[start]["role"] == "user" else "robot_update",
                    "robot": rows[start]["turn"]["robot"], "history": copy.deepcopy(self.messages[:first]),
                    "input": copy.deepcopy(self.messages[first]), "bad_turn": copy.deepcopy(self.messages[first + 1:end]),
                    "transcript": [{k: v for k, v in row.items() if k not in ("turn", "stats")} for row in rows[start:index + 1]]}

    def mark_flag(self, index, flag_id):
        with self.lock:
            if 0 <= index < len(self.transcript):
                self.transcript[index]["flag"] = flag_id
        self.on_change()

    def replay(self, history, message, proactive=False):
        """Teacher replays: continue from a saved history with the message that started a flagged turn."""
        with self.lock:
            self._cancel_turn()
            self.messages = copy.deepcopy(history) + [dict(message)]
            self.user_turns = sum(1 for m in self.messages if m["role"] == "user"
                                  and not (m["content"].startswith("Robot:") and "User:" not in m["content"]))
            self.epoch += 1
            epoch = self.epoch
            self.busy = True
            self.transcript.append({"role": "event" if proactive else "user", "content": message["content"]})
        threading.Thread(target=self._turn, args=(epoch, proactive), daemon=True).start()


def robot_state_line(adapter, executor):
    """The one-line robot state the agent puts in front of a user message when it has changed."""
    if adapter is None or not adapter.connected:
        return "no robot is connected, so movement tools will fail until the user opens the simulator or connects the Go2"
    try:
        s = adapter.state()
    except AdapterError:
        return f"{adapter.label}, telemetry unavailable"
    parts = [adapter.label, s.posture.replace("_", " ")]
    snap = executor.snapshot()
    parts.append(f"doing: {snap['current']['summary']}" if snap["current"] else "idle")
    if adapter.requires_arming and not snap["armed"]:
        parts.append("motion DISARMED (moving will fail until the user arms it)")
    if s.battery_pct is not None:
        parts.append(f"battery {s.battery_pct:.0f}%")
    return ", ".join(parts)


REPLY_NUDGE = "(Now say one short, friendly sentence to me about what you're doing.)"


CONTROL_TOKEN = re.compile(r"<\|[a-z_\"]*>|<[a-z_]*\|>|<\|\"\|>")


def clean_reply(text):
    """Drop control tokens, speaker labels and wrapping brackets the model sometimes imitates."""
    text = CONTROL_TOKEN.sub("", text or "").strip()
    while True:
        stripped = SPEAKER_PREFIX.sub("", text, count=1)
        if stripped == text:
            break
        text = stripped
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    return text.replace("] [", " ").strip()


def describe_outcome(task, group, still_to_do=()):
    """Plain-language report of a finished or failed task for the model. `still_to_do`: summaries of tasks
    other requests still have queued (the first starts right away); when there are none the report is the
    same as it always was."""
    d = task.detail
    ahead = ""
    if still_to_do:
        items = " then ".join([f"{still_to_do[0]} (starting now)", *still_to_do[1:]])
        ahead = f"; {items} {'is' if len(still_to_do) == 1 else 'are'} already queued"
    if task.status == "completed":
        if len(group) > 1:
            steps = " then ".join(t.command.summary for t in group if t.status == "completed")
            text = f"finished all steps: {steps}"
        else:
            text = f"finished: {task.command.summary}"
        if "remaining_m" in d:
            text += f" (ended {d['remaining_m']} m from the target)"
        if "final_gap_m" in d:
            text += f" (now {d['final_gap_m']} m from it)"
        return text + ahead + ". Tell the user briefly."
    progress = ""
    if "moved_m" in d and "target_m" in d:
        progress = f" after {d['moved_m']} of {d['target_m']} m"
    blocked = d.get("blocked")
    if blocked:
        detail = blocked["reason"] + ","
        if blocked.get("clearance_m") is not None:
            detail = (f"{blocked['reason']} (it could only check {blocked['clearance_m']} m {blocked.get('toward', 'ahead')})"
                      if "can't see" in blocked["reason"] else f"{blocked['reason']}, {blocked['clearance_m']} m short of it")
        reason = f"stopped early, without touching anything: {detail}{progress}"
    else:
        reason = f"could not finish: {d.get('error', 'unknown problem')}{progress}"
    text = f"{task.command.summary} {reason}"
    if d.get("cancelled_after"):
        text += f"; the {d['cancelled_after']} later step(s) were cancelled"
    return text + ahead + ". Tell the user briefly what happened."
