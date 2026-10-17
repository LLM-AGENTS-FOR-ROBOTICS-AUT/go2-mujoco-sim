# Command console API

What the console backend (`command_ui/server.py`) offers a web page: the live state of the robot, the
model and the conversation, plus the actions a page can take. Every example below is a real message from
the console running go2-ft-04 with the room simulation, shortened only where marked `...`.

The original four endpoints (`/api/command`, `/api/voice`, `/api/stop`, `/api/status`) still work exactly
as before, so the current page keeps working while a new one is built. They're described at the end.

## The short version

```js
// Everything the page shows comes from one object, pushed whenever it changes.
const events = new EventSource('/api/events');
events.addEventListener('state', e => render(JSON.parse(e.data)));

// Send what the user typed (returns at once; the reply and actions arrive in the state).
fetch('/api/messages', {method: 'POST', headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({text: 'walk forward one metre then turn left'})});

// Stop the robot at once (doesn't wait for the model).
fetch('/api/stop', {method: 'POST'});
```

If server-sent events don't suit, poll `GET /api/state` (every 300 to 500 ms): same object.

## Rules for every request

- The console only listens on `127.0.0.1` (port 5050 on macOS, 5000 on Windows; `--port` changes it).
- Requests must come from the page the console itself serves. A request with an `Origin` header from any
  other site, or a `Host` other than `127.0.0.1:<port>` / `localhost:<port>`, gets **403**. A page served
  by the console needs to do nothing special. (This stops other websites driving the robot.)
- Request and response bodies are JSON. Errors are `{"error": "message for the user"}` with **400**
  (refused) or **404** (nothing there yet).

## Endpoints

| Method and path | Body | Answer |
|---|---|---|
| `GET /api/state` | | The state object (below) |
| `GET /api/events` | | Server-sent events: `event: state` with the state object each time it changes, and a `: still here` comment every 15 s |
| `POST /api/messages` | `{"text": "...", "voice": false, "speak": false}` | **202** `{"accepted": true, "entry": 0, "conversation": 0}`: `entry` is the id the message got in the conversation |
| `POST /api/stop` | Optional `{"said": "Stop!"}` | `{"ok": true}`: the robot stops at once and queued steps are cancelled. `said` (when the user said it) is added as their spoken message |
| `POST /api/interrupt` | | `{"ok": true}`: Go2's reply in progress is dropped (the user cut in); robot steps carry on |
| `GET /api/speech?after=N` | | Go2's words phrase by phrase, held open until there's something new (see "Voice") |
| `POST /api/tts` | `{"text": "...", "voice": "af_heart"}` | The phrase spoken by Go2, as a WAV file. **503** while the voice loads (use the browser's own voice) |
| `POST /api/stt` | One sentence: 16 kHz mono float32 samples, raw (`application/octet-stream`) | `{"text": "walk forward one metre"}` (`""` for silence or noise). **503** while recognition loads |
| `POST /api/clear` | | The state, with an empty conversation and a new `conversation.id` |
| `POST /api/reset` | | Simulation only: the robot back at its start. The state |
| `POST /api/arm` | `{"armed": true}` or `{"armed": false}` | Real robot only: allow or forbid motion. The state |
| `GET /api/camera.png` | `?size=full` for full size | The last camera frame Go2 looked at (640 x 360, or 1280 x 720 full). **404** before the first look |

`voice: true` marks a message as spoken (shown in its entry). `speak` says the page plays Go2's reply aloud:
`true` (the default voice) or a voice's name from `voice.speech.voices`, and by default the same as `voice`.
The console then makes the reply's audio while the model is still writing it, and starts the robot's first
movement together with Go2's voice (see "Voice"). A short message starting with a stop
phrase ("stop", "halt", "freeze", "whoa", "don't move", "stay there", "cancel that", "abort") stops the robot
at once without the model, exactly like `/api/stop`. Stopping also cancels every queued step (their
status becomes `cancelled`).

Refusals, as they really come back:

```json
POST /api/messages {"text": ""}        -> 400 {"error": "Say or type something first."}
POST /api/arm {"armed": true} (sim)    -> 400 {"error": "Only the real robot needs arming; the simulation always moves."}
POST /api/reset (real robot)           -> 400 {"error": "Reset is only available in the simulation."}
POST /api/messages {"speak": "nobody"} -> 400 {"error": "Unknown voice 'nobody': use one of af_heart, af_bella, ..."}
GET  /api/camera.png (no look yet)     -> 404 {"error": "No camera frame yet: ask Go2 what it sees."}
POST /api/messages while downloading   -> 503 {"error": "Still downloading the language model (first start only): 3%, 0.1 of 3.4 GB. Try again when it's ready; Stop works now.",
                                               "model": {"state": "downloading", "progress": {"done_mb": 118, "total_mb": 3416}, "...": "..."}}
```

While the model is downloading (first start only), or if it failed to load (`model.state: "unavailable"`),
messages are refused at once with **503** and the reason, so the page can show it; nothing is queued.
A message sent while the model is loading (the first seconds after start-up) waits for it instead.
A stop phrase ("stop", "halt", ...) always goes through, as does `/api/stop`.

## The state object

```json
{
 "revision": 3,
 "robot": {
  "kind": "sim", "label": "MuJoCo simulation", "connected": true,
  "requires_arming": false, "armed": false, "scene": "room",
  "posture": "walking",
  "telemetry": {"x_m": -2.52, "y_m": 0.0, "heading_deg": 0, "speed_mps": 0.03, "turn_dps": 3,
                "battery_pct": null, "fault": null},
  "error": null
 },
 "model": {"id": "go2-ft-04", "label": "Go2 ft-04: newest fine-tune (teacher data, 4-bit)",
           "state": "ready", "error": null},
 "voice": {"available": true,
           "recognition": {"ready": true, "error": null, "model": "Whisper large-v3-turbo", "runs_on": "mlx"},
           "speech": {"ready": true, "error": null, "default": "af_heart",
                      "voices": {"af_heart": "Heart · warm (US)", "...": "..."}}},
 "agent": {"thinking": true, "error": null},
 "actions": {
  "current": {"id": "tcd0905", "tool": "move", "summary": "move forward 1 m at 0.20 m/s", "status": "running"},
  "queued": [{"id": "t10f101", "tool": "turn", "summary": "turn left 90° at 29°/s", "status": "queued"}],
  "recent": []
 },
 "conversation": {"id": 0, "entries": ["... see below"]},
 "camera": null
}
```

| Field | Meaning |
|---|---|
| `revision` | Goes up by one whenever anything in the state changed. Skip re-rendering when it's the same. |
| `robot.kind` | `"sim"` (room simulation) or `"go2"` (real Go2 EDU, then `robot.interface` is the network interface instead of `robot.scene`) |
| `robot.requires_arming`, `robot.armed` | The real robot moves only while `armed` is true. **Show this prominently when `requires_arming` is true.** The simulation never needs arming. |
| `robot.posture` | `standing`, `walking`, `sitting`, `lying`, `busy` (a trick or posture change in progress), `holding` (holding a body pose), `locked_stand`, `damp` (motors relaxed), `fallen`, `unknown`; `null` while disconnected |
| `robot.telemetry` | Position from the start (m), heading (degrees, positive = left), speed (m/s) and turning speed (degrees/s), battery % and fault text (real robot only) |
| `robot.error` | Why telemetry couldn't be read, if it couldn't |
| `robot.connected` | False once the simulation window is closed or the robot disconnects; the console then shuts down |
| `model.state` | `downloading` (first start only: `model.progress` is `{"done_mb", "total_mb"}`), `loading` (the first seconds after start-up; messages wait for it), `ready`, `unavailable` (then `model.error` says why, such as no internet for the first download), `scripted` (started with `--mock`: no language model) |
| `voice` | `available: false` when the console was started with `--no-voice`. Otherwise speech recognition (`runs_on`: `mlx` on Apple Silicon, `faster-whisper (GPU)` or `faster-whisper (CPU)` elsewhere) and Go2's voice, each `ready` once loaded (a few seconds; the first start downloads them: 1.6 GB and 330 MB) or with an `error` |
| `agent.thinking` | The model is working on a reply. A message sent now interrupts that reply and is answered instead. |
| `agent.error` | The last model error, if the last reply failed |
| `actions` | What the robot is doing (`current`), what's waiting (`queued`, in order) and the last 8 finished (`recent`): steps, described below |
| `conversation` | `id` changes when the conversation is cleared (start a fresh log); `entries` are below |
| `camera` | Set after Go2 has looked: see "Camera" |

## Conversation entries

Entries come in order, each with an `id` (its position, stable until the conversation is cleared) and a
`kind`. To build one card per user message, group the entries by `turn`: the id of the user message they
answer. Entries with `turn: null` are Go2 acting on its own, such as reporting that a move finished;
show those as their own entries in the log.

| `kind` | Fields | Meaning |
|---|---|---|
| `user` | `text`, `voice` | What the user typed or said |
| `call` | `turn`, `tool`, `arguments`, `outcome`, `message`, `steps` | A tool the model called (one reply can make several) |
| `reply` | `turn`, `text`, `proactive` | What Go2 said. `proactive: true` (with `turn: null`) when Go2 spoke on its own |
| `update` | `text` | A robot report: a request finished, was blocked, or failed |
| `error` | `turn`, `text` | The model failed (for example it isn't loaded) |
| `note` | `text` | Anything else, such as "Reply interrupted." |

### `call.outcome` and `call.message`

`outcome` is the robot's answer **when the call was made**; it doesn't change afterwards.

| `outcome` | Meaning |
|---|---|
| `queued` | Accepted. It will run (or is running): follow `steps[].status` for what happens next |
| `done` | A question answered at once (`look`, `get_status`), nothing to run |
| `already_queued` | The same request is already waiting, so it wasn't added twice |
| `needs_confirmation` | Go2 asks first (see "Confirmations") |
| `rejected` | Refused before starting: outside the limits, not possible here, wrong arguments |
| `blocked` | Refused at the start by the safety check (an obstacle) |
| `failed` | Couldn't start (for example the robot faulted) |
| `cancelled` | Not started because an earlier step in the same request was refused |

`message` is a short line for the user when the outcome isn't `queued` or `done`, such as
`"Rejected: front flip only runs on the real Go2; the simulator can't do acrobatics"`, otherwise `null`.

### Steps: what a call became, live

`call.steps` and `actions.current/queued/recent` hold the same step objects, so a card can update as its
steps run. A step's `status` is `queued`, `running`, then one of `completed`, `stopped` (by Stop or a newer
request), `blocked` (the safety check stopped it on the way), `failed` or `cancelled`.

Always present: `id`, `tool`, `summary` (plain words), `status`. Present when they apply:

| Field | Meaning |
|---|---|
| `target_m`, `moved_m` | Distance asked for and walked so far (move, go to, approach) |
| `target_deg`, `turned_deg` | Angle asked for and turned so far |
| `error` | Why it stopped early (blocked or failed), in words |
| `blocked` | `{"reason", "clearance_m", "toward"}` when the safety check stopped it |
| `note` | Something worth showing, such as a default the robot used |
| `group_remaining` | Steps of the same request still to come after this one |

## Example: "walk forward one metre then turn left"

Right after `POST /api/messages` (202, `"entry": 0`), the model is thinking (`agent.thinking: true`). Then:

```json
{"id": 0, "kind": "user", "text": "walk forward one metre then turn left", "voice": false},
{"id": 1, "kind": "call", "turn": 0, "tool": "move", "arguments": {"direction": "forward", "distance_m": 1},
 "outcome": "queued", "message": null,
 "steps": [{"id": "tcd0905", "tool": "move", "summary": "move forward 1 m at 0.20 m/s", "status": "running"}]},
{"id": 2, "kind": "call", "turn": 0, "tool": "turn", "arguments": {"direction": "left"},
 "outcome": "queued", "message": null,
 "steps": [{"id": "t10f101", "tool": "turn", "summary": "turn left 90° at 29°/s", "status": "queued"}]}
```

When both steps have run:

```json
{"id": 1, "kind": "call", "...": "...",
 "steps": [{"id": "tcd0905", "tool": "move", "summary": "move forward 1 m at 0.20 m/s", "status": "completed",
            "target_m": 1.0, "moved_m": 1.03, "group_remaining": 1}]},
{"id": 2, "kind": "call", "...": "...",
 "steps": [{"id": "t10f101", "tool": "turn", "summary": "turn left 90° at 29°/s", "status": "completed",
            "target_deg": 90, "turned_deg": 91, "note": "no angle given, turning the default 90 degrees",
            "group_remaining": 0}]},
{"id": 3, "kind": "reply", "turn": 0, "proactive": false, "text": "Walking forward one metre and turning left now!"},
{"id": 4, "kind": "update", "text": "finished all steps: move forward 1 m at 0.20 m/s then turn left 90° at 29°/s."},
{"id": 5, "kind": "reply", "turn": null, "proactive": true, "text": "All done, I'm right where you wanted me!"}
```

## Example: a move blocked on the way ("walk forward 4 metres")

The call was accepted (`outcome: "queued"`), then the safety check stopped the robot after 0.54 m:

```json
{"id": 10, "kind": "call", "turn": 9, "tool": "move", "arguments": {"direction": "forward", "distance_m": 4},
 "outcome": "queued", "message": null,
 "steps": [{"id": "tc57e1f", "tool": "move", "summary": "move forward 4 m at 0.80 m/s", "status": "blocked",
            "target_m": 4.0, "moved_m": 0.54, "error": "obstacle ahead: 0.20 m of free space",
            "blocked": {"reason": "obstacle ahead", "clearance_m": 0.2, "toward": "ahead"}, "group_remaining": 0}]},
{"id": 11, "kind": "reply", "turn": 9, "proactive": false, "text": "On it, walking forward four metres!"},
{"id": 12, "kind": "update", "text": "move forward 4 m at 0.80 m/s stopped early, without touching anything: obstacle ahead, 0.2 m short of it after 0.54 of 4.0 m."},
{"id": 13, "kind": "reply", "turn": null, "proactive": true, "text": "I had to stop early because there was an obstacle right in front of me."}
```

A move refused at the very start comes back with `outcome: "blocked"` and a `message`, and any later
steps of the same request with `outcome: "cancelled"`.

## Camera ("what do you see?")

The `look` call answers at once (`outcome: "done"`, no steps) and the state gains `camera`:

```json
"camera": {
 "stamp": 1792212267.4898748,
 "url": "/api/camera.png?t=1792212267.4898748",
 "boxes": [
  {"label": "potted plant", "score": 1.0, "box": [0.2844, 0.4722, 0.3484, 0.6083]},
  {"label": "couch", "score": 1.0, "box": [0.4781, 0.475, 0.6703, 0.6111]},
  {"label": "bowl", "score": 1.0, "box": [0.4531, 0.7194, 0.5062, 0.7972]},
  "..."
 ],
 "objects": [
  {"id": "obj1", "label": "bowl", "color": "blue", "distance_m": 1.12, "bearing_deg": 4,
   "direction": "straight ahead", "x_m": 1.12, "y_m": 0.07, "distance_source": "lidar", "confidence": 1.0},
  "..."
 ]
}
```

- `url` changes with every new look, so it can go straight into an `<img>`.
- `boxes` are where each object is in the image: `[left, top, right, bottom]` as fractions of the image
  size, so they fit either image size.
- `objects` are what Go2 placed in the room: distance from the body (m), bearing (degrees, positive =
  left) and a direction in words. They are listed separately from `boxes` and aren't matched one to one.

## Confirmations

Acrobatics (flips, jumps, handstand and the like) only run on the real robot, and only after the user
agrees. The model's first call comes back with `outcome: "needs_confirmation"` and Go2 asks in its reply.
To answer from buttons, send the answer as the next message:

```js
yes.onclick = () => send('yes');   // the model then calls the trick again with confirmed=true
no.onclick  = () => send('no');
```

The yes has to be the very next message, within 60 seconds. In the simulation acrobatics are refused
outright (`outcome: "rejected"`), so there's nothing to confirm.

## Voice

Everything runs on this computer and no audio is stored. `static/live-voice.js` does all of the page's part
and can be reused as it is: `Go2LiveVoice` takes spoken commands (one at a time, or a hands-free conversation)
and `Go2Speaker` plays Go2's replies. `static/index.html` shows how they're wired in.

**Listening.** The page records the microphone itself (with the browser's echo cancellation) and cuts it
into sentences: one starts when the level rises above the room's noise and ends after a 600 ms pause. Each
sentence goes to `POST /api/stt` as raw 16 kHz mono float32 samples (at most 30 s), and the text it answers
with goes to `/api/messages` with `voice: true`. If the user carries on talking before a sentence's text is
back, it is joined to what they say next rather than sent (or lost) on its own. While the user is still
talking, the sentence so far is also sent about once a second, for a live caption and so that "stop" halts
the robot (`/api/stop {"said": "Stop!"}`) without waiting for the end of the sentence. Each transcription
takes about 0.65 s on an M1 Pro, whatever its length.

```json
POST /api/stt  (117,760 bytes: 1.8 s of audio)  -> {"text": "Walk backward 2 meters."}
POST /api/stt  (64,000 bytes of silence)        -> {"text": ""}
POST /api/stt  (while it loads)                 -> 503 {"error": "speech recognition is still loading"}
```

**Go2's voice.** `GET /api/speech?after=N` answers with every phrase newer than item `N`, waiting up to 20 s
for one (start with `after=-1`, which only says where the stream is). The model's reply arrives phrase by
phrase as it is written; each turn ends with an `end` item, and `epoch` tells turns apart: a newer turn
replaces what is still playing, except a `proactive` one (Go2 reporting on its own, such as a walk that
finished), which waits its turn.

```json
GET /api/speech?after=-1  -> {"last": 8, "items": []}
GET /api/speech?after=8   -> {"last": 10, "items": [
                               {"epoch": 6, "id": 9, "proactive": false, "text": "Turning to the left now!"},
                               {"epoch": 6, "id": 10, "end": true}]}
```

`POST /api/tts {"text", "voice"}` gives each phrase's audio (Kokoro, 24 kHz WAV). When the message was sent
with `speak`, the console has already started making it while the model wrote the phrase, so it is usually
ready by the time the page asks. **503** means the voice is still loading or off: speak the phrase with the
browser's own voice instead (`speechSynthesis`), as `Go2Speaker` does.

**Moving with the voice.** The model writes its spoken reply after the tool call, so on its own the robot
would start moving about half a second before Go2 says anything. For a message sent with `speak`, the
robot's first movement waits until the first phrase's audio is ready, at most 1.2 s. The safety checks run
as usual, and Stop still halts at once.

**Cutting in.** When the user talks over Go2 (or while it is still thinking), the page stops playing and calls
`POST /api/interrupt`: the reply is dropped (a "Reply interrupted." note is added), while robot steps already
started carry on. The new sentence then goes to `/api/messages` as usual. A robot update that arrives while
the user is talking is held until they pause, and dropped if what they said is a new request.

## The original endpoints

Kept for the current page:

- `POST /api/command` `{"text"}`: waits for the whole reply, then answers with one card:
  `{"command", "tool", "valid", "validation", "response", "exec": null}`. `tool` lists the calls as text
  (`"move(direction=forward, distance_m=1) then turn(direction=left)"`); `validation` is
  `"Passed: within limits"`, `"No robot action"` or the first problem (`"Blocked: ..."`,
  `"Rejected: ..."`, `"Needs your confirmation: say yes to go ahead"`).
- `POST /api/voice`: records 5 seconds from the computer's microphone, then the same as `/api/command`.
- `POST /api/stop`: `{"ok": true}`.
- `GET /api/status`: `{"state", "action", "speed", "note"}` for the side panel; `note` carries the latest
  robot update or Go2's own report.

## Developing against it

- `command_ui/server.py --mock` runs everything with scripted replies instead of the language model (no
  model file or Ollama needed): "walk forward 2 m", "turn left", "sit", "look" and "status" work.
- `--headless` runs the simulation without its window.
- `--no-voice` starts without speech recognition and Go2's voice (saves about 2.5 GB of memory); the
  voice endpoints then answer **503**.
- `experiments/voice_latency.py` measures, for a set of commands, when the robot starts moving and when
  Go2's spoken reply is ready.
