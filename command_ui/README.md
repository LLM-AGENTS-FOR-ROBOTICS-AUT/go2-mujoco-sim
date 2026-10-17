# Go2 Command Console (LLM agent, UI + voice)

Type or speak a command to the Go2. A language model understands it and answers, calls bounded robot
tools, and every call is checked against the per-call limits and the robot's own sensors before anything
moves. The page is the same as before; what's behind it changed:

```
you type / speak  ->  LLM agent (go2-ft-04)  ->  tool calls  ->  safety executor  ->  robot (simulated room or real Go2)
```

| | Before (`main`) | This branch (`llm-agent`) |
|---|---|---|
| Understanding | Keyword rules (`agent.py`, the placeholder) | Gemma 4 E2B fine-tuned for Go2 (`go2-ft-04`), via a private Ollama |
| Actions | Walk, strafe, turn, stop | 12 tools: move, turn, go_to, approach, posture (sit, stand, lie), trick, acrobatic, body pose, look, wait, get_status, stop |
| Safety | Speed and duration bounds | Per-call limits (0.8 m/s, 5 s, 4 m, 57 deg/s) plus LiDAR clearance checks before and during each move |
| Robot | Flat floor, walking only | Indoor room with household objects, simulated L1 LiDAR and front camera, or the real Go2 EDU |

## Set up (once)

1. Install **Ollama** (free): <https://ollama.com/download>. The console starts its own private Ollama
   process; it never touches your own Ollama models.
2. Run the setup with the console option. It installs everything else, downloads the room's objects and
   the language model (**go2-ft-04**, 3.4 GB, from the project's Hugging Face page; once):

   - **macOS (Apple Silicon, macOS 14+):** `bash macos/setup.sh --agent`
   - **Windows 10/11:** `powershell -ExecutionPolicy Bypass -File .\windows\setup.ps1 -Agent`

   The setup ends with a check of the room simulation and the model. If the model download didn't finish,
   the console downloads it on its first start instead (resuming where it stopped), showing progress
   under the Stop button. It is checked against the SHA-256 in `models/catalog.json`, and a different
   file is refused. To use a copy you already have, put it in `models/` or point `GO2_FT_WEIGHTS` at it.

## Run

- **macOS:** double-click **`Launch Command UI.command`**, or `.venv/bin/python command_ui/server.py`
- **Windows:** double-click **`Launch Command UI.cmd`**, or `.venv\Scripts\python.exe command_ui\server.py`

The MuJoCo window opens with the Go2 in the room, then the page: <http://127.0.0.1:5050> on macOS
(AirPlay Receiver holds port 5000 there) and <http://127.0.0.1:5000> on Windows. Closing the MuJoCo
window ends the console. The first command waits a few seconds while the model loads.

Options: `--mock` (scripted replies, no language model: checks the page and robot without the model
file), `--scene empty`, `--headless` (no MuJoCo window), `--port N`, `--no-browser`, `--check`.

## Use

- **Type** a command and press Enter, e.g. `walk forward 2 metres then turn left`, `sit down`,
  `what do you see?`, `go to the teddy bear`, `do a little dance`.
- **Speak**: click the mic and speak within 5 seconds (offline speech recognition, faster-whisper).
- **Stop**: the red Stop button halts the robot at once, without waiting for the model.

Each command gets a card: **Tool** is what the model called, **Safety** says whether it went ahead
(or why not: blocked by an obstacle, outside the limits, waiting for your confirmation), **Robot** is
Go2's reply. Updates that arrive later, such as a move finishing or being blocked halfway, appear under
the Stop button.

## Real Go2 EDU

Install CycloneDDS 0.10.x and `pip install -r requirements-robot.txt`, connect the computer to the Go2's
network (192.168.123.x), then:

```
.venv/bin/python command_ui/server.py --robot en7          # connected, but it will not move
.venv/bin/python command_ui/server.py --robot en7 --arm    # armed: it moves when told to
```

Use the name of the network interface cabled to the robot. Follow the safety rules in the main README.

## Pieces

| File | Role |
|------|------|
| `server.py` | Flask backend: the page's endpoints, the agent, the robot connection |
| `API.md` | The backend API for building a page: live state, messages, steps, camera, arming (with real examples) |
| `static/index.html` | The web page (unchanged) |
| `../go2/agent.py` | The LLM conversation and tool loop, with proactive robot updates |
| `../go2/tools.py`, `../go2/limits.py` | Tool definitions, validation and per-call limits |
| `../go2/executor.py` | Runs tool calls safely: clearance checks, queueing, stopping |
| `../go2/llm.py` | The private Ollama process and tool-calling chat |
| `../go2/adapters/` | The robot: `sim.py` (room simulation process) and `go2_sdk.py` (real Go2) |
| `../simulation/` | The room: world, simulated robot (sport mode over the walking policy), LiDAR and camera |
| `../models/` | The model catalog and checksums (the weights themselves are shared separately) |

## Troubleshooting

- **"Language model unavailable"** under the Stop button: install Ollama (step 1), or the model download
  was interrupted (no internet): start the console again and it resumes. `server.py --check` says what is
  missing; `server.py --fetch-model` downloads the model on its own.
- **Windows, slow start every time:** without Developer Mode Windows can't make symbolic links, so the
  console hard-links the model (instant on the same drive) or copies it (3.4 GB, every start). Keep the
  model on the same drive as the repo, or turn on Developer Mode.
- **The page doesn't load on a Mac at port 5000:** that's AirPlay Receiver; use the printed address (5050).
