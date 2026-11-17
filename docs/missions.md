# Missions (#18)

Long tasks Go2 carries out on its own in a house: `search`, `guard`, `patrol` and `explore`. The code is in
`go2/missions.py`; the tools in `go2/tools.py`. This page fixes what the model sees, for training go2-ft-06 (#19).

## How they run

- A mission is one executor task. **Stop** (the button, Esc, saying "stop", the `stop` tool) cancels it like any move.
- Every movement is an ordinary executor step, with the same limits and LiDAR checks; missions only choose where to go.
- Each has a time limit, and ends with one `outcome` and a one-sentence `result`, reported as a robot update.
- While it runs it reports progress ("checked bedroom 1: no teddy bear"), as robot updates too.
- A stopped or timed-out `search`, `patrol` or `explore`, asked for again within 10 minutes of when it stopped,
  carries on where it left off ("carry on" → the same call again). `guard` always starts afresh.
- Room names in `where` and `rooms` are read as `go_to_place` reads them: "the kitchen" and "bedroom 2" are
  those rooms; a room type ("bedroom") is every room of that type.
- Simulator only for now: on the real Go2 they're refused until its safety work is done (#34).
- The simulator's walking people (`walker_red`, `walker_blue`) are in a house only when its description has
  `"people": true` (the guard test uses it); other house scenes are unchanged.
- They're full-prompt tools: left out of the compact prompt until a model is trained on them (`UNTRAINED`).

## The tools (names and arguments are fixed)

| Tool | Arguments | What it does |
|---|---|---|
| `search` | `target` (required), `where`, `time_limit_s` (default 600, 10 to 1800) | What's remembered first; else looks all round where it is; then room by room, likeliest rooms first (`missions.LIKELY`), exploring where the map has no rooms yet; walks up to the target and says the room. A room counts as checked only once it got in and looked round; a room it can't get into is tried once more after the others, and if it still can't, the search ends `blocked` ("I couldn't get into bedroom 2…") rather than `not_found`. `where`: only that room (a room name or type) or saved place, including what's remembered and what's in view; if it can't get to a saved place it ends `blocked` (not "it isn't there"). |
| `guard` | `place`, `duration_s` (default 600, 10 to 3600) | Goes to the room or saved place (default: stays), faces the way in (the nearest doorway; a saved place's own heading only when the room has no known way in), sits, and looks about once a second. When someone comes in after nobody was in view for 10 s, it alerts with a camera snapshot. It never moves once there. |
| `patrol` | `rooms` (list; default all known), `every_s` (60 to 3600, default 600), `duration_s` (default 1800, 10 to 7200) | Explores first if no rooms are known. Each round: every room nearest-next, looking all round; then reports things moved (same label and colour more than 0.8 m away), new, gone, and people seen. Each thing's room is worked out from where it was seen (the room most of the floor within 1 m of it belongs to), with the rooms' current names, so a room named between rounds isn't a change. |
| `explore` | `time_limit_s` (default 1200, 10 to 3600) | Visits exploration spots (#14) until none are left, looking all round at each; a spot it can't reach gets one more try once nowhere new is left, then is given up; names rooms as it finds them (#35); then lists them. |

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

Results (houses 0-49 and 0-9, 14 November 2026):

| Mission | Result | Target |
|---|---|---|
| search, 50 houses | found in 43 of the 45 houses where the teddy bear can be reached (95.6%); 0 collisions; 265 s of simulated time on average | 90%, no collisions |
| search, unreachable teddy bear | 4 of 5 ended `blocked` (seen, no way to it); 1 was reached anyway (the plan check, `reachable()`, is slightly cautious) | says so, doesn't loop |
| explore, 10 houses | 36 of 37 rooms named right (97.3%); 0 collisions | 80% |

A teddy bear counts as reachable when, by the house plan, Go2's centre can get within 1.1 m of it (`reachable()`).

Known limits:
- Houses 32 and 46 (search): small two-room houses with the teddy bear in a tight spot in the bedroom; Go2 sees it
  but can't find a clear spot near enough, so the search ends `blocked`.
- House 9 (explore): the kitchen stayed `room_N`; not enough kitchen things were seen from where Go2 looked.
- House 4 (explore): a large kitchen with a dining table was split into two regions, both named kitchen
  (`kitchen_1`, `kitchen_2`); the name is right, but it's listed twice.
- Rooms of one type are numbered in the order they were found (`bedroom_1`, `bedroom_2`), not the house plan's order.
- Once, the offline evaluation stopped part-way through a house with no processor use (houses 6-9 were re-run and
  finished normally, with the same results as the run before). The cause wasn't found; it didn't happen again.

The explore results are from the code as committed (re-run after the last change to room naming).
