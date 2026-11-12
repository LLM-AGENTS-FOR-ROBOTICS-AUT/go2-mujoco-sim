# Issue #14 validation

**The exploration simulation coverage target is met: 20/20 houses passed.**
All cases reached at least 90% LiDAR-mapped exposed-floor coverage with zero
recorded robot/world contacts, no fault and a stopped standing robot.
Coverage ranged from 90.03% to 93.67%, averaging 91.45%.

This is Prionti's exploration workstream on `prionti/issue-14-exploration`, based
on `4873df2`, submitted in PR #36. It does not
complete the broader laptop-search mission or all work in issue #14.

## Automated checks

| Check | Result |
| --- | --- |
| Exploration, mapping, navigation, executor and safety regression suite | Passed: 174 tests run, including 8 dependency-related skips |
| Normal headless MuJoCo/RPC viewpoint smoke test | Passed in isolation in 28.1 seconds; requested position/yaw reached without contact |
| Camera-and-approach integration | Passed on the final build |
| Seeded MuJoCo coverage cases | 20/20 passed; zero contacts |

The normal-RPC viewpoint test first reached its 40-second task deadline while
four offline physics jobs were running. It passed with those workers briefly
paused; they were then resumed. The camera-and-approach check passed in the
concurrent run. These results do not establish control latency under heavy load.

This is a focused regression result, not a claim that every repository test
passed. Earlier full lightweight discovery encountered missing simulation/voice
dependencies, localhost restrictions and a load-sensitive timing failure;
focused environment-specific rechecks passed. Unavailable camera/voice checks
were not all rerun. No large language or speech model was downloaded.

```bash
python -m unittest tests.test_exploration tests.test_exploration_map \
  tests.test_exploration_evaluation tests.test_exploration_executor \
  tests.test_exploration_navigation tests.test_exploration_perception \
  tests.test_exploration_sim tests.test_path_planner tests.test_room_map \
  tests.test_executor tests.test_navigation tests.test_safety tests.test_hazards

# Use the existing simulation environment for these integration checks:
python -m unittest tests.test_exploration_sim \
  tests.test_sim.SimulationTests.test_approach_stops_at_the_requested_gap
```

## Coverage protocol

The final build uses seeds 0–19, one reproducible clear random start per house,
a 1200 simulated-second house budget, a 90-second viewpoint limit and at most
120 attempts. The longest completed case took 809.0 simulated seconds.
18/20 final runs finished within 600 simulated seconds; this is an observation
of these 1200-second-budget runs, not a separate 600-second-budget experiment.

`mujoco-step` drives the actual Go2 walking policy, MuJoCo physics, L1 sensor
and Executor. It pauses physics during planning and advances physics during
control waits. Contact onsets are counted at every physics substep, including
braking; normal floor/self contacts are excluded. An unsettled result cannot
pass. This backend does not establish real-time/RPC performance.

Coverage samples exposed floor at 5 cm across all rooms, excluding walls,
closed doors and furniture footprints. It measures the current observed LiDAR
occupancy map, not camera coverage or laptop-search success. Ground truth is
used only for scene construction, clear starts and scoring, never policy input.

The local machine-readable report `docs/results/issue-14-exploration-final-2026-11-12.json`
retains starts, room coverage, timings, task failures, collision data, matching
source hashes and dependency fingerprints. All 20 cases use the same source
hashes, identifying the original validated build `5c77d17`, before subsequent
CI corrections. Generated reports are ignored by Git; regenerate them with
the commands above rather than committing new result dumps. The pinned policy asset checksum
was also verified. Simulator versions: MuJoCo 3.10.0, Torch
2.2.2 and NumPy 1.26.4.

The 10–15 evaluation chunk was intentionally stopped after seed 13 when an idle
worker took seeds 14–15. All 20 unique cases completed; no completed result was
discarded. Workers were briefly paused for the isolated RPC recheck; simulated
time did not advance during the pause and recorded wall elapsed includes it.
Blocked or timed-out viewpoint attempts remain in the report even when the
house ultimately passes. Such an outcome records a stopped/retried task and
is not itself a physical collision.

| Seed | Coverage | Contacts | Simulated seconds |
| --- | ---: | ---: | ---: |
| 0 | 90.76% | 0 | 268.7 |
| 1 | 90.19% | 0 | 116.0 |
| 2 | 90.39% | 0 | 117.1 |
| 3 | 90.06% | 0 | 120.1 |
| 4 | 90.48% | 0 | 308.0 |
| 5 | 90.62% | 0 | 480.4 |
| 6 | 90.03% | 0 | 405.2 |
| 7 | 92.51% | 0 | 286.9 |
| 8 | 92.00% | 0 | 104.8 |
| 9 | 93.07% | 0 | 538.7 |
| 10 | 93.67% | 0 | 809.0 |
| 11 | 92.24% | 0 | 543.5 |
| 12 | 90.45% | 0 | 453.5 |
| 13 | 91.87% | 0 | 377.1 |
| 14 | 91.67% | 0 | 77.8 |
| 15 | 92.14% | 0 | 165.0 |
| 16 | 93.38% | 0 | 523.4 |
| 17 | 91.35% | 0 | 550.7 |
| 18 | 91.57% | 0 | 279.0 |
| 19 | 90.54% | 0 | 664.9 |

## Calibration and comparison

The finite empty-world calibration measured collision half-length below 0.396 m
and half-width below 0.202 m. Live checks use a 0.42/0.23 m rectangle; planning
adds one cell. Local footprint measurements in `docs/results/issue-14-robot-footprint-2026-11-12.json`
are evidence for this simulation configuration, not a formal hardware bound.

| Build | House budget, simulated seconds | Passed | Contacts |
| --- | ---: | ---: | ---: |
| Earlier circular planner | 600 | 10/20 | 0 |
| Preceding recovery build | 1200 | 19/20 | 0 |
| Final selection-order fix | 1200 | 20/20 | 0 |

The earlier circular baseline (`docs/results/issue-14-exploration-2026-11-12.json`)
and preceding recovery build (`docs/results/issue-14-exploration-progress-2026-11-12.json`)
are retained locally as ignored reports. The changed budget prevents attributing the first comparison
solely to planner changes. The final fix exhausts normal coarse/fine plans
before enabling fine-map recovery; the live collision envelope is unchanged.

## Integration scope

CI corrections on 13 November keep LiDAR buffer timestamps in the executor's
clock domain, reproject safety scans at the current pose after map processing,
and select navigation bends with turning clearance. Stopping thresholds and
test tolerances are unchanged. Local checks passed: 29 clock/follow checks
(3 dependency skips), the existing safety/navigation cases with targeted
rechecks of the new clearance regression, 17 planner/physics checks, and all
four real-simulator following/navigation checks, including one minute following
a moving person. Hosted-platform CI still determines cross-platform acceptance.

The team still needs to agree that LiDAR-mapped exposed floor is the intended
meaning of "floor seen". Twenty sampled starts do not prove every continuous
starting pose. Geometric room IDs need explicit semantic labels. Ali's search
runner, camera target detection and Arman's placed-laptop/50-house acceptance
remain separate team integration work. The broader issue should be reviewed
against those requirements before closure.
