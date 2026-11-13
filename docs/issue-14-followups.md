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

PR #36 now passes all three hosted platform checks at commit `46e02e2`. This
branch incorporates those fixes and the team's merged semantic-memory work.
Its full local suite passed 409 tests with 11 optional-dependency/model skips.

The follow-up corrections reproduce and fix each previously reported failure:

- The missing-object test forces asynchronous acknowledgement, then waits for
  the executor's completion report and final agent turn. It no longer reads the
  initial "On it" reply as the result of the inspection.
- Measured turns slow near their target. A slow-braking test that previously
  overshot by about 11 degrees now passes the unchanged 8-degree bound. The
  unchanged real-physics move/turn accuracy check passed three consecutive runs.
- Stall decisions refresh telemetry and time after scan processing, and use
  current velocity alongside progress integrated over the existing confirmation
  window. Three delayed-mapping stress runs stopped in 0.36–0.37 seconds, inside
  the unchanged 0.6-second limit. Slow starts, slow motion and following checks
  still pass.

All 96 executor/safety/following/place/memory/map/drift regression checks passed
after these corrections. The semantic-memory integration checks also passed
(55 checks, 7 optional-model skips). Hosted follow-up CI must confirm the final
reviewed head on Apple Silicon, Intel and Windows before merging.

The first follow-up CI run exposed additional timing cases despite a complete
443-test local pass (11 optional-dependency/model skips):

- Tracking now updates a direction and checks its clearance in the same send
  cycle. Alternating slow/fast scans reproduced unchecked commands on the old
  code; the regression passes with the fix.
- Intermediate navigation bends use a 2 cm arrival tolerance rather than the
  ordinary 8 cm destination tolerance. A 6 cm bend previously became a no-op
  on every replan; it now produces measured movement before replanning.
- Stall telemetry is checked both before and after scan processing, so a
  confirmed contact does not wait for another map update. The existing delayed
  mapping test passed five consecutive runs with its 0.6-second limit intact.

The existing furniture-detour and direction-tracking checks also passed with
these corrections. Final hosted results still need to confirm this revised head.

A subsequent run passed both Apple Silicon jobs and one Windows job, while the
other Windows job exposed post-turn heading error in the real-room detour.
Navigation now projects the next waypoint into the measured post-turn body
frame. A regression injecting a 7-degree turn error previously drifted 14 cm
off its planned line; it now passes a 4 cm lateral bound. This retains live
clearance checks for the exact commanded translation and the existing turn
accuracy limits. Three consecutive real-simulator room detours passed after
this correction, as did the remembered-object furniture-detour check.

## Earlier PR check

An earlier PR #36 Apple Silicon log contained a furniture-detour navigation
failure: a route waypoint lacked enough room to turn. The latest GitHub status
read on 12 November lists that job as cancelled; Intel and Windows succeeded.
This follow-up chooses bends with turning clearance while retaining live
safety checks, and the local real-physics furniture test passes. Apple CI uses
MuJoCo 3.13.0 and a newer walking runtime; this Intel machine cannot establish
that Apple run is green. Rerun CI on the published changes before merging.

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
