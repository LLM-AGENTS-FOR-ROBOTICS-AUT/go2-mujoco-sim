# Issue #14: Prionti's exploration work

Repository: https://github.com/LLM-AGENTS-FOR-ROBOTICS-AUT/go2-mujoco-sim

Issue: https://github.com/LLM-AGENTS-FOR-ROBOTICS-AUT/go2-mujoco-sim/issues/14

This implements the exploration workstream in the November 11 "Find the laptop"
plan, building on merged object memory (#28) and occupancy mapping/A* (#29).
The broader issue includes work outside this assignment.

## Implemented locally

- **Frontiers and viewpoints:** `go2/exploration.py` chooses known-free positions
  behind the boundary between observed and unknown cells. It uses the current
  robot pose, footprint clearance, remembered obstacles and actual observed stop positions.
- **Map growth:** `go2/room_map.py` grows a bounded map while preserving world
  coordinates, provides consistent snapshots and accumulates simulation-only
  evidence for low obstacles such as books.
- **Room regions:** `go2/room_regions.py` separates geometric room cores at
  narrow doorways, expands them across observed floor and associates IDs across
  updates. `executor.room_at(x, y)` returns a region or an explicit human label.
- **Viewpoint execution:** `executor.submit_viewpoint(...)` follows short,
  slow known-free route legs through the existing worker and safety checks.
  Cancellation and deadlines remain active. These tasks are limited to simulation.
- **Route-aware approach:** the existing approach action plans to a stand-off
  point, follows detours and visually confirms the target again.
- **Evaluation:** `simulation/evaluate_exploration.py` reports coverage, contact
  counts, failures, seeds, starts and timing. Ideal-grid and real-MuJoCo backends
  are reported separately.

See [the integration guide](issue-14-exploration-api.md) for exact calls,
coordinate frames, clearances, evaluation commands and limitations.

## Acceptance and current status

The exploration simulation target is met under the stated metric: all 20 random
houses with varied clear starts reached at least 90% LiDAR-mapped exposed-floor
coverage with zero robot/world contacts, no fault and a stopped standing robot.
The protocol allows 1200 simulated seconds per house. The implementation and
results are local on `prionti/issue-14-exploration` and have not been pushed.
See the [validation report](issue-14-validation.md) for all cases and checks.

"Floor seen" is interpreted as LiDAR occupancy coverage of exposed floor. Camera
coverage and laptop-search success need separate evaluation, and the team must
confirm that metric interpretation. Twenty sampled starts do not prove every
continuous starting pose.

`None` from the selector means no currently available safe viewpoint. It must
not be interpreted as proof that a target is absent. The executor tries normal
coarse/fine routes before bounded recovery and keeps live safety checks active.

## Team handoff

| Owner | Responsibility |
| --- | --- |
| Prionti, #14 | Frontiers, viewpoints, geometric room regions, route-aware approach and exploration coverage |
| Ali, #18 | Search runner, memory-first order, scans, progress, cancellation and mission time limits |
| Luka, #16/#19 | Search model tool, mock and detection checks |
| Arman | Target placement, combined scenario, 50-house evaluation and demonstration |

Ali can call `executor.exploration_viewpoint(visited)` and execute the returned
world pose using `executor.submit_viewpoint(...)`. The independent contract is
`next_viewpoint(ExplorationMap(...), visited) -> (x, y, yaw) | None`.
`ExplorationMap` contains the occupancy grid and fresh robot pose because a raw
grid has no current pose. Coordinates are odometry-frame metres and radians.

Semantic names such as "kitchen" require explicit labels; geometry alone cannot
identify them. Simulator room names are used only for evaluation diagnostics.
Ali's target-search loop and Arman's placed-laptop scenario remain separate
integration dependencies. Do not close the broader issue on the strength of
this exploration implementation alone.
