# Missions (#18)

Long tasks Go2 carries out on its own in a house: `search`, `guard`, `patrol` and `explore`. The code is in
`go2/missions.py`; the tools in `go2/tools.py`. This page fixes what the model sees, for training go2-ft-06 (#19).

## How they run

- A mission is one executor task. **Stop** (the button, Esc, saying "stop", the `stop` tool) cancels it like any move.
- Every movement is an ordinary executor step, with the same limits and LiDAR checks; missions only choose where to go.
- Each has a time limit, and ends with one `outcome` and a one-sentence `result`, reported as a robot update.
- While it runs it reports progress ("checked bedroom 1: no teddy bear"), as robot updates too.
- A stopped or timed-out `search`, `patrol` or `explore`, asked for again within 10 minutes, carries on where it
  left off ("carry on" → the same call again). `guard` always starts afresh.
- Simulator only for now: on the real Go2 they're refused until its safety work is done (#34).
- They're full-prompt tools: left out of the compact prompt until a model is trained on them (`UNTRAINED`).

## The tools (names and arguments are fixed)

| Tool | Arguments | What it does |
|---|---|---|
| `search` | `target` (required), `where`, `time_limit_s` (default 300, up to 900) | What's remembered first; else looks all round where it is; then room by room, likeliest rooms first (`missions.LIKELY`), exploring where the map has no rooms yet; walks up to the target and says the room. `where`: only that room (a room name or type) or saved place. |
| `guard` | `place`, `duration_s` (default 600, up to 3600) | Goes to the room or saved place (default: stays), faces the way in (a saved place's heading, or the nearest doorway), sits, and looks about once a second. When someone comes in after nobody was in view for 10 s, it alerts with a camera snapshot. It never moves once there. |
| `patrol` | `rooms` (list; default all known), `every_s` (60 to 3600, default 600), `duration_s` (default 1800, up to 7200) | Explores first if no rooms are known. Each round: every room nearest-next, looking all round; then reports things moved (same label and colour more than 0.8 m away), new, gone, and people seen. |
| `explore` | `time_limit_s` (default 600, up to 1800) | Visits exploration spots (#14) until none are left, looking all round at each; names rooms as it finds them (#35); then lists them. |

## Results (fixed)

The tool call itself answers `{"status": "queued", ...}` at once. When the mission ends, `actions.current.mission`
(and the finished task's `detail.mission`) holds:

| Field | Values |
|---|---|
| `kind` | `search`, `guard`, `patrol`, `explore` |
| `outcome` | `found`, `not_found`, `blocked` (seen but no way to it), `explored`, `done` (guard, patrol), `timed_out` |
| `result` | One sentence for the user, e.g. "I found the teddy bear in bedroom 2 and I'm next to it" |
| `room` | `search` found: the room it was in |
| `checked` | `search`: the rooms looked round in |
| `alerts` | `guard`: how many |
| `rounds` | `patrol`: how many |
| `rooms` | `explore`: the room names on the map |
| `log` | The last 30 progress lines |

The robot update the model gets at the end is the `result`, e.g.
`Robot: I found the teddy bear in bedroom 2 and I'm next to it. Tell the user briefly.`
Progress updates read `Robot: search for the teddy bear: checked bedroom 1: no teddy bear`, and alerts
`Robot: alert from guard the hallway: someone came in: a person in red, 1.0 m away, straight ahead`.

## Evaluation

`python simulation/evaluate_missions.py search --houses 50` and `... explore --houses 10` run the missions end to end in
random houses on simulated time (the offline simulator from #14, with the camera added) and print a line per house and
a summary. The #18 targets: search finds the teddy bear in at least 90% of 50 houses with no collisions; explore names
at least 80% of rooms correctly.
