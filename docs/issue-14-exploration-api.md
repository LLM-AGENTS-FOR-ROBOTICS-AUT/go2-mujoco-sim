# Exploration implementation and handoff

Repository: https://github.com/LLM-AGENTS-FOR-ROBOTICS-AUT/go2-mujoco-sim

Issue: https://github.com/LLM-AGENTS-FOR-ROBOTICS-AUT/go2-mujoco-sim/issues/14

Branch: `prionti/issue-14-exploration`, based on main at `4873df2`.

## API for Ali's mission runner

```python
viewpoint = executor.exploration_viewpoint(visited)
if viewpoint is not None:
    result = executor.submit_viewpoint(
        viewpoint, group=mission_id, user_turn=user_turn,
        timeout_s=min(90, remaining_seconds),
    )
    # Wait for the returned task_id through the normal executor event/history API.
    # Check completed/blocked/failed/stopped before scanning and updating visited.

room = executor.room_at(object_x, object_y)
```

The independent `next_viewpoint(ExplorationMap(...), visited)` API remains
available. ExplorationMap contains the occupancy grid and current robot pose;
the raw grid alone has no pose. All x/y positions are odometry-frame metres;
yaw is absolute radians. These world-frame positions must not be passed as
body-relative `go_to` offsets.

`exploration_viewpoint` refreshes observations and starts no motion. Selection
and execution use the same heading-aware navigation graph. It models a planned
body rectangle calibrated from empty-world walking, diagonal motion, turning and
braking. Measured collision geometry stayed within half-length 0.396 m and
half-width 0.202 m; live checks round these up to 0.42 m and 0.23 m. Planning
adds one cell for tracking and position quantisation: 0.52/0.33 m on the 10 cm
planning map, or 0.47/0.28 m on the 5 cm fallback. A coarse cell is free only
when all its sensing cells are free. Refinement preserves obstacles and unknown
cells and is retried when coarse planning cannot find a route.

The graph retains the exact current body heading and can plan horizontal or
vertical body headings. Body rotations are explicit actions, allowed only in
observed space with 0.60 m circular clearance, or 0.55 m on the fine fallback.
Sideways translation remains
available, but forward/backward travel is preferred so the narrower side of the
body crosses doorways. A change in travel direction does not require a body turn.

Every translation reserves the executor's body-wide stopping corridor, with
0.30 m lookahead beyond the body. Planner and live checks share an exact swept
rectangle calculation, including diagonal motion. Legs follow the planned world
direction from the measured position, preserving the tracking allowance.
Execution also checks fresh LiDAR, ledges,
bumps, telemetry, stall, cancellation and deadlines. Legs are at most 0.5 m at
0.15 m/s, refreshing observations before the next leg. New obstacles stop or
invalidate a route. Repeated steps with no measured progress stop the task.
Viewpoint translation also corrects lateral drift from its planned world line
using fresh odometry. Corrections retain the 0.15 m/s velocity limit, and the
corrected body direction passes the live clearance check before it is sent.
Other command types retain their existing velocity controller.

Viewpoint positions prefer 0.65 m circular clearance; a 0.55 m fine-map fallback handles
smaller observation pockets. Frontier ranking combines graph travel cost with
connected frontier size, reducing repeated stops for tiny nearby holes. Visited
positions are suppressed within 0.5 m, or 0.25 m near the end of a sweep. Further
scans fill non-repetitive LiDAR holes before returning no viewpoint. If no nearby
viewpoint has a clear sightline to a frontier, selection retries positions up to
1.5 m behind it, allowing observation through narrow openings without entering
them. Route and footprint requirements are unchanged.

`submit_viewpoint` queues movement through the existing safety worker. Arrival
tolerance is 0.12 m; the final requested yaw is attempted only in observed
turning space. Stop cancels current and queued work without automatic resumption.
These viewpoint tasks are limited to simulation. No model tool or prompt has
been added or retrained.

Near furniture, bounded recovery may translate at a fixed heading to observed
turning space, checking an oriented rectangular footprint and live directional
clearance. Its footprint retains the measured position within the map cell and
includes half-cell area projected onto the calibrated body axes. Recovery steps
keep that same position offset instead of snapping toward a cell centre beside
a wall. Recovery can account for an already-overlapping confirmed low surface
in its private planning snapshot only when overlap does not increase. It never
clears high obstacles or unknown cells; live clearance still applies. A recovered
pose is reported as a partial outcome so the mission requests a fresh viewpoint.
Recovery is also considered when the standing body fits but the normal route
graph cannot reach turning space. A failed normal route must not bypass this
check and be mistaken for completed exploration. Recovery continues toward
observed turning space before reporting the partial outcome; every leg retains
the same live safety checks and task deadline.
The executor tries all normal coarse/fine planning contexts before enabling
recovery. Independent callers can use `ExplorationMap(..., allow_recovery=False)`
for normal selection and enable the bounded recovery only after those options
are exhausted. A coarse recovery must not bypass a valid normal fine-map route.

The standalone selector preserves its circular-clearance default for existing
callers. Use `ExplorationMap(..., heading_aware=True)` for the Go2 body model, or
use the executor wrapper above, which selects that mode automatically.

After a completed attempt or useful partial travel, scan and append the actual
observed stop pose to `visited`. Do not permanently mark an unreached requested
goal as visited after a timeout. Keep no-progress goals in a short temporary
exclusion list instead, clearing it after progress or before declaring exhaustion.
The evaluator's `remember_attempt` demonstrates this policy; its five consecutive
no-progress limit and overall budget belong to the mission runner. Cancellation
ends the mission. `None` means no currently available safe
viewpoint, which can include a blocked or insufficiently observed current pose.
It does not prove the house was fully searched or a laptop is absent. Ali's
mission still decides target detection, search progress, failure and time limits.

## Growing maps and room regions

The executor starts with a 20 m square at 5 cm resolution, growing up to 2048
cells per side while keeping world positions fixed. Snapshots include the map's
origin and are independent of ongoing LiDAR writes. Changing the attached robot
clears the map and room identities so scenes do not contaminate one another.

Simulation maps also accumulate repeated low-height evidence. A raised surface
whose height exceeds floor noise with a conservative confidence bound is an
obstacle, including a book below the existing 6 cm body-obstacle threshold.
Uncertain raised cells stay unknown. Confirmed low surfaces participate in live
clearance checks without altering raw camera depth or detector evidence. This
noise calibration is enabled only for simulation; real-LiDAR calibration remains
a separate hardware task.
Flat simulation scenes use the base's world height to map returns above the
fixed floor at z=0. A local ground ray hitting a low object under the robot must
not shift the map's floor reference.
In this simulation mapping mode, a projected ray crossing an obstacle cell does
not erase the obstacle: the beam may have passed above it. Repeated direct floor
returns in that cell can clear it after an object moves. Current obstacle hits
take priority when one scan contains both obstacle and floor returns in a cell.

`RoomRegions` erodes free space by 0.65 m to disconnect narrow doors, identifies
room cores, then expands region labels back over observed free floor. IDs are
matched by world-space overlap; the strongest overlap inherits an old ID on
splits/merges. Unknown, occupied or outside positions return `None`.

Unnamed regions are `room_1`, `room_2`, etc. Explicit semantic labels can be set:

```python
executor.room_at(x, y)  # update regions from the latest observed map
executor.room_regions.name_room(x, y, "kitchen")
```

Geometry cannot identify an office or kitchen by itself. Wide arches/open-plan
spaces may stay one region; furniture may split a room. This is a geometric
estimate, not perfect semantic room recognition. Simulator room names never
feed the exploration planner.

## Approach

Approach plans to a free stand-off point near an object and follows route
waypoints around furniture. It keeps a locked world position when a detour turns
the camera away, then reacquires and confirms visually. A missing target at the
old location or no stand-off route produces a clear failure. Approach has a
180-second motion budget and a finite movement limit.

## Validation

```bash
python -m unittest tests.test_exploration tests.test_exploration_map \
  tests.test_exploration_evaluation tests.test_exploration_executor \
  tests.test_exploration_navigation tests.test_exploration_perception \
  tests.test_path_planner tests.test_room_map

python -m simulation.evaluate_exploration --backend grid \
  --houses 20 --starts-per-house 3 --output runtime/exploration-grid.json

python -m simulation.evaluate_exploration --backend mujoco-step \
  --houses 20 --timeout-s 1200 --output runtime/exploration-mujoco-step.json

python -m unittest tests.test_exploration_sim

python -m simulation.calibrate_exploration_footprint runtime/robot-envelope.json
```

The grid backend uses ideal planar rays and kinematic movement for fast
regression checks. It is not physics acceptance. New scans revalidate the next
footprint before the simulated robot follows a previously computed route.

`mujoco-step` runs the actual walking policy, physics, L1 sensor and executor
in-process. In its standalone process only, an executor simulation clock
advances physics during control waits and pauses physics while planning. Its
budget is simulated time; it does not establish real-time/RPC performance.
`mujoco` uses the ordinary headless process/RPC adapter and wall-time budget.
Keep the normal-adapter smoke check above alongside offline physics evaluation.
The footprint calibration is a finite empty-world motion sample, not a formal
bound for every gait or hardware configuration. Its saved measurements are in
`docs/results/issue-14-robot-footprint-2026-11-12.json`.

Both physics backends count robot/world contact onsets at every physics substep,
excluding normal floor/self contact. Low-object contacts also count. The count
persists until reset and is exposed in simulator state.
Evaluation also checks contacts during braking and waits for a stopped standing
state before closing physics. An unsettled result cannot pass.

Coverage measures exposed floor samples currently marked free by the observed
LiDAR map. Scoring uses 5 cm cell centres across all rooms, excluding walls,
closed doors and furniture footprints. It does not measure camera coverage,
semantic room accuracy or laptop-search success. The team must confirm this
interpretation of "floor seen" before accepting the workstream.

Truth supplies only scene construction, clear random starts and scoring. The
policy starts with an unknown map and uses observations only. Reports retain
seeds, starts, outcomes, failures, coverage, collision counts and elapsed time.
`all_passed` only becomes true after all requested cases finish and pass.
The current acceptance protocol allows 1200 simulated seconds per house and
90 seconds per viewpoint, up to 120 attempts. The earlier baseline used 600
seconds, so its pass count cannot isolate the effect of planner changes.

Twenty houses with varied starts sample possible starts; they do not prove
every continuous starting pose. Ali's search loop and Arman's placed-laptop
scenario/50-house success test remain separate integration dependencies.

## Acceptance status

The local implementation passed all 20 seeded MuJoCo house cases under the
1200-simulated-second protocol: at least 90% LiDAR-mapped exposed-floor coverage,
zero contacts, no faults and a stopped standing robot in every case. The final
source hashes match all cases. See the [validation report](issue-14-validation.md)
for the metric, timing limits, regression results and the load-sensitive RPC check.

These are exploration simulation results. Camera target-search success, semantic
room labels and the full team's mission remain separate integration requirements.
The code is local and has not been pushed; the broader issue has not been closed.
