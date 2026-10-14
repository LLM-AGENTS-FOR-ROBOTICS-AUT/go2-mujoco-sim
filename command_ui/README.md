# Go2 Command Console (UI + voice)

A simple natural-language command interface for the Go2 MuJoCo simulation.
Type or speak a command; it is interpreted into a bounded tool, validated, and
executed by driving the simulated Go2.

```
you type / speak  ->  interpret (tool)  ->  validation layer  ->  Move() in MuJoCo
```

## Run

Double-click **`Launch Command UI.cmd`**, or:

```powershell
..\.venv\Scripts\python.exe server.py
```

Two things open: the **MuJoCo window** (watch the robot) and the **web UI** at
<http://127.0.0.1:5000> (your browser opens automatically).

## Use

- **Type** a command and press Enter / the send button.
- **Speak**: click the mic, speak within 5 seconds (offline voice-to-text via faster-whisper).
- **Stop**: the red Stop button halts the robot immediately (independent software stop).

Examples: `walk forward for 3 seconds`, `turn left`, `back for 2 seconds`, `stop`.
`sit` / `jump` show the controlled fallback for unsupported commands.

## Pieces

| File | Role |
|------|------|
| `server.py` | Flask backend + MuJoCo sim loop; serves the UI and the `/api/*` endpoints |
| `agent.py` | Rule-based command interpreter + validation layer (**placeholder for the LLM agent**) |
| `static/index.html` | The web UI |

### Note for the team
`agent.interpret_to_tool()` is a deterministic placeholder. The LLM agent can
replace it and return the same tool structure, so the validation layer, UI and
robot execution stay unchanged. Voice uses the `base` faster-whisper model
(downloaded on first use; not committed to the repo).

## Requirements (first time)
Already have the simulator set up (`windows/setup.ps1 -Locomotion`, so `mujoco`
and `torch` are installed in `.venv`). Then install the UI's extra deps:

```powershell
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

That adds `flask`, `faster-whisper`, `sounddevice`, `soundfile`, `av==13.1.0`.
The faster-whisper `base` model downloads automatically the first time you use voice.
