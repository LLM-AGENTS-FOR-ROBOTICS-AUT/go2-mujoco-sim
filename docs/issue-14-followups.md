# Issue #14 follow-up implementation and validation

Branch: `prionti/issue-14-followups`, based on exploration commit `5c77d17`.
PR #36 and its recorded coverage results are unchanged by this worktree.

## Implemented

- [x] Named places: scene-specific atomic persistence, current-spot saving and routed return.
- [x] Missing object state, with conservative negative-observation checks and recovery on rediscovery.
- [x] Memory/place query tools for mock and full-prompt models; compact tools unchanged.
- [x] Optional configurable odometry drift and reusable scan-matching correction.
- [x] Live console map: robot, route, named places and active/missing object memory.
- [x] Regression tests, physics integration, two five-minute drift runs and browser verification.

## Trying the features

Use the Python environment created by the repository setup scripts, with the
console requirements installed. From this checkout:

```sh
python command_ui/server.py --mock --no-voice --no-upload --headless --odometry-drift
```

In the page, try `remember this spot as the door`, `what do you see?`,
`walk forward 1 m`, `go to the door`, `what places do you remember?`, and
`where is the red ball?`. The mock resolves named destinations through saved
memory; an unknown object goes through the existing checked `approach` tool.
Execution reports describe actual outcomes and do not repeat movement commands.

Drift is off unless requested. `--odometry-drift` uses 2% distance scale error,
0.01 rad/m heading bias, and seeded distance-dependent noise. To configure it:

```sh
python command_ui/server.py --mock --no-voice --headless \
  --odometry-drift '{"translation_scale":0.02,"yaw_bias_per_m":0.01,"translation_noise":0.003,"yaw_noise":0.002,"seed":1}'
```

`--odometry-seed N` overrides the JSON seed. `--no-scan-correction` with drift
provides an uncorrected baseline. These switches are refused with `--robot`.
The four new tools are `remember_place`, `go_to_place`, `list_places` and
`query_memory`. Their schemas are available to full-prompt models, including
`go2-ft-04@full`. The 12 compact tools and the public `look` format are preserved;
new model training remains #16/#19. The model verification here uses `--mock`,
not a downloaded or retrained language model.

## Interfaces and safety

`NamedPlaces` stores map-frame x/y/yaw under a normalised name. The executor
derives `<scene>.places.json` beside the existing scene-specific object-memory
file. Saving starts no motion. A return freezes the saved destination before
queueing, plans through the normal occupancy grid, and obeys existing arming,
clearance, drop, cancellation and per-leg checks, plus a 180-second overall
budget. Unknown names and destinations over 20 m away are refused.

Object records now have stable persistent IDs and active/missing states. Three
separate clear views, at least 0.15 seconds apart, can retire an old location.
Each requires a fresh stationary camera view with a stable standing pose and
accepted localisation (when enabled), the saved location inside the
image, direct floor returns there, and no raised obstacle or occluder. Missing
detections alone do not erase memory. Pending negative evidence resets on a
new connection, and a new sighting reactivates a nearby matching record.
Queries separate unavailable locations from active sightings. Arrival reports
say whether the old location was verified clear or could not be checked.

This negative-evidence check is intentionally limited to observable floor
locations in the simulator. A table, occlusion or sparse scan leaves the
record uncertain. Far-apart lookalikes remain separate; #32 supplies appearance
identity rather than this code guessing which physical object moved.

`ScanLocalizer.update(raw_odom_pose, floor_relative_cloud, sensor_time)` returns
a corrected map-frame pose. `pose`, `status` and `reset` complete the interface.
It uses only NumPy and knows no simulator geometry, truth pose or robot SDK.
It aligns a short motion-compensated L1 window to an incrementally built map
using bounded ICP and point-to-plane refinement. Rejected scans do not enter
the occupancy map; after three consecutive rejected scans, navigation stops.
New clear scans can recover tracking. The simulator error model alone uses
truth to generate imperfect odometry, once per physics step rather than once
per client request. Camera, LiDAR and telemetry use the same corrected frame.

The simulator begins in its scene's world frame, so saved places can be reused
after restarting the same scene. The localiser map is rebuilt from sensors;
this is not global relocalisation into a saved map. Real Go2 places remain
connection-local, and drift correction is not enabled on hardware.

`GET /api/map` is a bounded, read-only presentation endpoint described in
`command_ui/API.md`. A separate idle simulator sensor loop keeps the map fresh.
The page draws it once per second. Fetching the map never starts sensing,
camera inference or movement. Occupied cells survive display downsampling.

## Acceptance evidence (12 November 2026)

The two original runs at commit `3fbd1f0` use an unfurnished 8 m × 8 m test room, actual MuJoCo physics,
the walking policy, L1 rays, and production executor motion/safety checks. The
stepped clock advances every physics step; it does not teleport the robot or
replace physics with a grid model. Truth supplies scene construction, the
odometry error sensor and scoring only.

| Noise seed | Walking exercise | Saved-place return error | Raw odometry error before return | Occupied-wall distance, 95th percentile | Contacts / faults |
| --- | --- | --- | --- | --- | --- |
| 0 | 310.74 sim s, 31 patrol corners | 0.08475 m | 1.14613 m | 0.075 m | 0 / none |
| 1 | 301.80 sim s, 30 patrol corners | 0.11305 m | 1.05685 m | 0.075 m | 0 / none |

Both returned within the issue's 0.3 m bound and settled standing. The largest
pose errors sampled at patrol corners were 0.00673 m and 0.00852 m respectively;
these are sampled errors, not a claim about every physics step. The wall check
requires the 95th percentile to be below 0.15 m, a separate numerical check of
map sharpness. This establishes the criterion in this test room, not every
possible house, noise level, or moving-obstacle layout.

Full configurations, runtime versions, samples and source hashes are retained
locally in the Git-ignored reports `docs/results/issue-14-drift-seed0-2026-11-12.json`
and `docs/results/issue-14-drift-seed1-2026-11-12.json`. Their hashes identify
the original validated build, not later CI corrections. Regenerate reports with
the command below; generated JSON dumps are not committed.

Reproduce each five-minute acceptance run with:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python simulation/evaluate_drift.py --seed 0 --output runtime/drift-seed0.json
```

Final local checks:

- 205 focused regression tests passed (places, memory, localisation, map API,
  tools, agent, existing console, grid/planner and exploration).
- 56 safety/executor/memory checks passed.
- 8 navigation/physics integration checks passed, including the furniture
  detour, saved-place return with real LiDAR and drift, and normal RPC viewpoint.
- 15 additional API/cleanup and remembered-object revisit checks passed.
- A final 46-test feature check passed after adding the stable-pose/localisation
  guard to negative memory evidence, before the CI corrections described below.
- Browser: saved `door`, looked at room objects, walked away, and returned via
  `query_memory` → `go_to_place`; live map and remembered locations rendered.

These are named test sets, with some overlap, rather than a claim that the
entire repository suite ran. Local physics runtime: Python 3.12.5, NumPy 1.26.4,
MuJoCo 3.10.0, Torch 2.2.2, macOS x86_64. The exploration-only 20-house coverage
results in PR #36 remain evidence for commit `5c77d17`, not a new twenty-house
validation of every follow-up feature.

## CI corrections (13 November 2026)

This branch incorporates PR #36's three-platform passing head `46e02e2` and
the team's merged semantic-memory implementation. The follow-up fixes retain
live LiDAR checks and the existing safety and accuracy bounds:

- Scan timestamps use the executor's monotonic clock, including the explicit
  simulated clock used by offline evaluation. Clearance uses the current pose
  after scan processing; stall telemetry is checked before and after mapping.
- Every corrected movement direction is checked in the same send cycle.
  Waypoint legs use the measured post-turn heading and lateral path feedback.
- Distance moves slow near their target with a conservative braking profile,
  while respecting the existing default minimum walking speed. Turn commands
  also slow near their measured target. Regression tests model weaker brakes
  without changing the original accuracy limits.
- Navigation retains the rest of a valid route between completed legs and
  replans when live sensing blocks a leg. Live start probes also apply to
  ordinary routes, and planning uses the map evidence collected by those probes.
  Simplification preserves the initial bend so it cannot restore a refused
  direction through the cleared start area.
- Intermediate bends use a 2 cm pre-movement tolerance to avoid repeated no-op
  replans. Completion uses the established navigation arrival tolerance and
  live safety checks.
- The missing-object test waits for the actual executor completion report,
  including when submission first returns an asynchronous acknowledgement.

New regressions cover alternating slow/fast scans, live-blocked starts, fresh
scan evidence, nearby bends, post-turn heading error, lateral walking bias,
retaining a valid detour, and slow braking at full speed. Deterministic replay
also exercises the waypoint sequence that failed on CI. Five repeated real-time
room detours passed, together with the final real-physics distance/turn,
full-speed distance, and wall-gap checks. Generated raw reports
remain local and ignored. Hosted checks must confirm the final reviewed head
on Apple Silicon, Intel and Windows before merging.

The delayed-mapping deadline regression uses a controlled clock across 100
contact phases, with 200 ms scan work and 80 ms ticks, retaining the 600 ms
stop-command bound. The earlier controller fails 20 phases; the corrected
controller passes all 100. The existing real-time FakeRobot glass-stop test
retains its 600 ms assertion. Controlled timing keeps host descheduling from
being counted as controller latency in the added stress case.

## What still needs team action

- Review the follow-up PRs #39 and #40, which share the same head branch. PR #39
  is stacked on #36; retarget it to main after #36 is merged. No PR is merged
  automatically by this work.
- Confirm all three hosted platform checks on the final reviewed follow-up head
  and review the navigation adjustments.
- Review the conservative missing-object behaviour and the drift test's
  environment limits before closing #14. Broader furnished-house drift tests
  would extend the evidence beyond the issue's five-minute test case.

Semantic image embeddings (#32) have landed on main and are preserved here.
Room naming (#35), model fine-tuning (#16/#19), missions (#18) and physical Go2
tuning (#34) remain their respective workstreams. This branch provides their
map/place interfaces and the simulator-side features requested by #14.
