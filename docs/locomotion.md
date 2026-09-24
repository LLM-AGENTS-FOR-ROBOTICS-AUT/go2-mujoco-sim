# Go2 policy comparison and working local demo

The selected local configuration is **Walk These Ways from go2_deploy**, a
neutral trot, and feedback from the simulator's actual planar position and yaw.
The pretrained network drives the twelve joints; the feedback controller adjusts
its velocity commands to follow the requested path. No network weights were
retrained. This is not a claim that the raw policy tracks perfectly on its own.

## Results on Apple Silicon

Tested on 24 September 2026 with macOS 27 / arm64, Python 3.12.14, MuJoCo 3.13.0
and CPU PyTorch 2.8.0. The optional installation requires macOS 14+, matching
the PyTorch wheel. Intel Mac locomotion is outside this change.

| Check | Result |
| --- | --- |
| Forward 0.4 m/s, 10 seconds | 3.912 m, maximum sideways deviation 4.3 cm, maximum yaw error 0.73 degrees |
| Forward 0.4 m/s, 60 seconds | 23.912 m, maximum sideways deviation 4.3 cm, maximum yaw error 0.73 degrees |
| Forward 0.2 and 0.6 m/s | Both passed distance, sideways deviation and heading checks |
| Backward 0.3 m/s | Passed |
| Left/right at both 0.2 and 0.3 m/s | Passed |
| Turn left/right at 0.5 rad/s | Passed |
| Straight walking from two rotated starts | Passed |
| Stop after each movement | Passed: final-second mean speed below 5 mm/s and rotation below 0.01 rad/s |
| Continuous 37-second sequence, with stop and restart | Passed |
| Stand still for 20 seconds | Passed |
| Native Mac viewer | Completed the full demo; keyboard walk, stop, reset and turn controls checked |

The 15 automated cases passed. The headless control-step 95th percentile was
about 0.7 ms, inside the 20 ms control period. This is not a graphics benchmark.
[Full local result](results/wtw-macos-arm64.json).

**Windows execution is pending.** Its separate setup installs the same CPU
runtime and weights, and the shared checks are wired into the Windows workflow.
They have not run for this change because it remains local. Neither Windows
graphics nor the physical Go2 EDU is claimed as tested.

## Other policies tested

All six alternatives below used the same official Go2 dynamics, a flat plane,
a ten-second forward command of 0.4 m/s, and no heading/position feedback.
Each then received the policy's own zero command for three seconds. Sideways
deviation and heading error are maxima over the moving portion.

| Policy | Forward distance | Sideways deviation | Heading error | Assessment |
| --- | ---: | ---: | ---: | --- |
| HIMLoco from rl_sar | 3.438 m | 17.9 cm | 3.87 degrees | Best speed tracking, but lateral drift; further trials exposed poor in-place turning |
| Simple RL from go2_deploy | 2.574 m | 10.7 cm | 0.82 degrees | Holds heading well at this speed, but slow and less straight at higher speeds |
| Teacher–Student from go2_deploy | 2.647 m | 40.8 cm | 15.58 degrees | Rejected for heading drift |
| Walk These Ways from go2_deploy | 2.833 m | 5.6 cm | 1.94 degrees | Best raw straight-line result here; still too slow and dependent on gait/speed |
| Explicit Estimator from go2_deploy_python | 2.753 m | 37.9 cm | 7.06 degrees | Rejected for drift |
| DreamWaQ from go2_deploy_python | 3.174 m | 21.8 cm | 3.12 degrees | Better speed tracking, but too much lateral drift |

[Raw comparison data](results/raw-policy-comparison.json). The benchmark
requires 3.2–4.8 m forward, at most 10 cm sideways deviation and at most 5 degrees
heading error. None passed all three unassisted. The earlier robot_lab/Efferent
baseline had about 25 degrees of yaw and 36.5 cm sideways displacement in just
four seconds at the same forward command, so it was rejected.

Additional trials covered speeds, reversed motion, rotated starts, gait settings,
and contact sensitivity. Those experiments did not justify claiming that a raw
policy met the full requirement. The final controller retains the original
robot mass, joint dynamics, and foot contact parameters. Its default walking
scene removes the default scene's hurdles to provide an unobstructed plane.

## What makes the final demo different

- **Neutral WTW gait:** period 0.5 seconds, body-height command 0.30 m,
  foot-clearance command 0.08 m, pitch command 0. These are supported gait inputs.
  The raw comparison used the upstream reset defaults, including pitch 0.3 rad.
- **Position and heading feedback:** the requested velocity integrates into a
  reference trajectory. Measured MuJoCo base pose provides the error used to
  adjust policy commands. This requires the simulator's pose; it is not an
  unassisted-policy or physical-robot result.
- **Stop and hold:** zero velocity blends to a standing pose over 0.5 seconds
  using joint PD control. Physics keeps running. Starting again resets gait
  history and the reference from the current pose.
- **Bounded control:** joint targets respect the model's limits and full PD
  torque is clipped to 23.5 Nm per joint. A fall or a reference error over
  0.5 m stops the run with an error.

The local API implements only `Move(vx, vy, yaw_rate)`, `StopMove()`, reset and
simulation stepping. Move uses body-frame m/s and rad/s. It is not the Unitree
SDK or a complete SportClient. Call `step()` every 20 ms to advance physics.
Allowed commands are up to 0.6 m/s forward/back, 0.3 m/s sideways and 0.5 rad/s
turning. Not every simultaneous combination has been tested.

These are level-ground results. Terrain traversal, stairs, obstacle avoidance,
recovery, sit/lie-down and factory gait modes are not validated or implemented.
The original go2_deploy C++/DDS application was not installed: its policy weights
were exercised through our local MuJoCo adapter.

## Run the local demo

From the repository root on an M-series Mac:

```bash
bash macos/setup.sh --locomotion
bash macos/run.sh --demo
```

For Windows, the prepared PowerShell path is:

```powershell
powershell -ExecutionPolicy Bypass -File .\windows\setup.ps1 -Locomotion
& '.\windows\Launch Go2 Viewer.cmd' --demo
```

Use `--walk` for keyboard control: W/S forward/back, A/D sideways, Q/E turn,
Space stop, R reset. Tap a key to set a persistent command; releasing it does
not stop movement. Use `--walk --validate` to run the quantitative suite.
Reports go to `_deps/locomotion-results.json`; `--report <path>` selects a path.

For the separate candidate comparison after installing walking dependencies:

```bash
.venv/bin/python experiments/fetch_candidates.py
.venv/bin/python experiments/compare_policies.py
.venv/bin/python experiments/compare_policies.py --policies wtw --feedback --gait 0.5 0.30 0.08 0 --extended
```

On Windows use `.venv\Scripts\python.exe` for those three commands.
Candidate weights/configs are checksum-verified against
`experiments/sources.json`. No candidate's physical deployment program is run.
