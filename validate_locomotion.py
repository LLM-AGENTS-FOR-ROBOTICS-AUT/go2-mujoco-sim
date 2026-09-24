"""Quantitative tracking, stopping and continuous-motion tests in MuJoCo."""

from __future__ import annotations

import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

from go2_locomotion import DEMO, GAIT, SimSportClient, demo_command
from policy_assets import FILES, REVISION


def validate(client: SimSportClient, report_path: Path) -> int:
    started = time.perf_counter()
    steps_ms, inference_ms, results = [], [], []

    def advance(command, duration):
        client.Move(*command)
        history = []
        for _ in range(round(duration/client.period)):
            tick = time.perf_counter()
            client.step()
            steps_ms.append((time.perf_counter()-tick)*1000)
            if client.inference_ms:
                inference_ms.append(client.inference_ms)
            history.append(client.metrics())
        return history

    cases = [
        ("forward", (.4,0,0), 10, 0), ("slow_forward", (.2,0,0), 10, 0),
        ("fast_forward", (.6,0,0), 10, 0), ("backward", (-.3,0,0), 10, 0),
        ("left", (0,.3,0), 10, 0), ("right", (0,-.3,0), 10, 0),
        ("slow_left", (0,.2,0), 10, 0), ("slow_right", (0,-.2,0), 10, 0),
        ("turn_left", (0,0,.5), 10, 0), ("turn_right", (0,0,-.5), 10, 0),
        ("rotated_start_left", (.4,0,0), 10, .7),
        ("rotated_start_right", (.4,0,0), 10, -1.2),
        ("long_straight", (.4,0,0), 60, 0),
    ]
    for name, command, duration, yaw0 in cases:
        try:
            client.reset(yaw0)
            initial = client.metrics()
            moving = advance(command, duration)
            delta = np.array([[m["x"]-initial["x"], m["y"]-initial["y"]] for m in moving])
            rotation = np.array([[np.cos(yaw0), -np.sin(yaw0)], [np.sin(yaw0), np.cos(yaw0)]])
            body_delta = delta @ rotation
            yaw = np.unwrap([initial["yaw"]]+[m["yaw"] for m in moving])[1:]-initial["yaw"]
            if command[2]:
                progress = yaw[-1]*np.sign(command[2])
                expected = abs(command[2])*duration
                cross = float(np.max(np.linalg.norm(delta, axis=1)))
                heading_error = abs(float(yaw[-1]-command[2]*duration))
            else:
                axis = 0 if command[0] else 1
                progress = body_delta[-1,axis]*np.sign(command[axis])
                expected = abs(command[axis])*duration
                cross = float(np.max(np.abs(body_delta[:,1-axis])))
                heading_error = float(np.max(np.abs(yaw)))
            end = moving[-1]
            stopped = advance((0,0,0), 3)
            stop_speed = float(np.mean([m["speed"] for m in stopped[-50:]]))
            stop_yaw = float(np.mean([abs(m["yaw_rate"]) for m in stopped[-50:]]))
            stop_drift = float(np.hypot(stopped[-1]["x"]-end["x"], stopped[-1]["y"]-end["y"]))
            checks = {
                "distance_within_20_percent": bool(.8*expected <= progress <= 1.2*expected),
                "cross_track_under_10cm": cross <= .10,
                "heading_error_under_5deg": np.degrees(heading_error) <= 5,
                "stopped_speed_under_5mm_s": stop_speed <= .005,
                "stopped_rotation_under_0_01rad_s": stop_yaw <= .01,
                "stop_drift_under_25cm": stop_drift <= .25,
            }
            result = dict(name=name, command=command, seconds=duration, start_yaw=yaw0,
                          distance=float(progress), expected_distance=expected,
                          cross_track_m=cross, heading_error_deg=float(np.degrees(heading_error)),
                          final_yaw_deg=float(np.degrees(yaw[-1])),
                          max_tilt_deg=max(m["tilt_deg"] for m in moving+stopped),
                          stop_speed_m_s=stop_speed, stop_yaw_rate_rad_s=stop_yaw,
                          stop_drift_m=stop_drift,
                          checks={k: bool(v) for k,v in checks.items()}, passed=bool(all(checks.values())))
        except Exception as exc:
            result = dict(name=name, passed=False, error=str(exc))
        results.append(result)
        print(f"{name}: {'PASS' if result['passed'] else 'FAIL'} | " +
              (f"distance={result['distance']:.3f}, cross-track={result['cross_track_m']:.3f} m, "
               f"heading error={result['heading_error_deg']:.2f} deg" if "distance" in result else result["error"]),
              flush=True)

    try:
        client.reset()
        for step in range(round(sum(d[0] for d in DEMO)/client.period)):
            _, command = demo_command(step*client.period)
            client.Move(*command)
            client.step()
        stopped = advance((0,0,0), 3)
        results.append(dict(name="continuous_demo", passed=max(m["speed"] for m in stopped) < .005))
        client.reset()
        initial = client.metrics()
        idle = advance((0,0,0), 20)
        drift = float(np.hypot(idle[-1]["x"]-initial["x"], idle[-1]["y"]-initial["y"]))
        idle_yaw = float(idle[-1]["yaw"]-initial["yaw"])
        results.append(dict(name="idle_20s", drift_m=drift, yaw_rad=idle_yaw,
                            passed=drift < .01 and abs(idle_yaw) < .01))
    except Exception as exc:
        results.append(dict(name="continuous_demo_or_idle", passed=False, error=str(exc)))

    report = dict(
        passed=all(r["passed"] for r in results), platform=platform.platform(),
        architecture=platform.machine(), python=platform.python_version(),
        packages={p: version(p) for p in ("mujoco", "torch", "numpy")},
        policy="go2_deploy Walk These Ways", policy_revision=REVISION,
        policy_sha256=FILES["wtw_model.pt"], gait=GAIT,
        feedback="Simulated planar position and yaw; not an unassisted policy result",
        scope="Level ground only, original Go2 dynamics; no physical robot validation",
        cases=results, step_p95_ms=float(np.percentile(steps_ms,95)),
        inference_p95_ms=float(np.percentile(inference_ms,95)) if inference_ms else None,
        wall_seconds=time.perf_counter()-started,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(f"Tracking validation: {'PASS' if report['passed'] else 'FAIL'}; "
          f"control-step p95={report['step_p95_ms']:.2f} ms (20 ms budget). Report: {report_path}")
    return 0 if report["passed"] else 1
