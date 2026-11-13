"""Five-minute real-physics acceptance: saved place, drift and wall sharpness.

An in-process stepped clock accelerates wall time, not physics. All travel uses
the production Executor's route and live LiDAR checks. Truth is used only by
the odometry sensor model and this scorer, never by ScanLocalizer.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"simulation")]
import numpy as np
import mujoco
import torch
from go2 import executor as X
from go2.tools import resolve
from go2.room_map import OCCUPIED
from exploration_clock import SteppedSimAdapter
from evaluate_exploration import settle_robot
from odometry import DriftConfig


def run(seed=0, seconds=300.):
    world = dict(scene="house", rooms=[dict(name="test_room", type="living_room", rect=[-4., -4., 4., 4.])],
                 doors=[], objects=[], start=[-1.1, -1.1, 0.])
    sim = SteppedSimAdapter(world, odometry_drift={"seed": seed})
    wall_start = time.monotonic()
    original = X.time
    X.time = sim.clock
    errors, raw_errors, failures, samples = [], [], [], []
    with tempfile.TemporaryDirectory() as folder:
        executor = X.Executor(memory_path=str(Path(folder)/"test.json"))
        try:
            sim.connect()
            executor.attach(sim)
            for _ in range(15):
                executor._refresh_cloud()
                sim.clock.sleep(.1)
            saved = executor.remember_place("door")
            truth = sim.sim.state()
            saved_truth = np.array([truth["x"], truth["y"]])
            began = sim.clock.monotonic()
            corners = [(1.1, -1.1), (1.1, 1.1), (-1.1, 1.1), (-1.1, -1.1)]
            visited = 0
            def place_task(name):
                command = resolve("go_to_place", {"name": name})[0]
                place = executor.places.get(name)
                command = replace(command, params={**command.params, "x": place.x, "y": place.y})
                return X.Task(command, "drift-evaluation", executor.generation)
            while sim.clock.monotonic()-began < seconds:
                x, y = corners[visited % len(corners)]
                executor.places.remember("patrol corner", x, y, 0.)
                task = place_task("patrol corner")
                try:
                    executor._run(task, threading.Event())
                except Exception as error:
                    failures.append(str(error)+"; "+json.dumps(sim.localizer.status()))
                    break
                executor._safe_stop()
                state = sim.state()
                truth = sim.sim.state()
                actual = np.array([truth["x"], truth["y"]])
                error = float(np.linalg.norm(np.array([state.x, state.y])-actual))
                raw_error = float(np.linalg.norm(sim.odometry.pose[:2]-actual))
                errors.append(error)
                raw_errors.append(raw_error)
                samples.append(dict(sim_time_s=sim.clock.monotonic(), pose_error_m=error,
                                    raw_error_m=raw_error, localization=sim.localizer.status()))
                visited += 1
                print(json.dumps(dict(corner=visited, elapsed_s=sim.clock.monotonic()-began,
                                      pose_error_m=error, raw_error_m=raw_error)), flush=True)
                if truth.get("collision_count") or truth.get("fault"):
                    failures.append("collision or robot fault")
                    break
            walk_elapsed = sim.clock.monotonic()-began
            returned = False
            if not failures:
                task = place_task("door")
                try:
                    executor._run(task, threading.Event())
                    returned = True
                except Exception as error:
                    failures.append("return: "+str(error))
            state, settled, _ = settle_robot(sim, sim.clock)
            actual = sim.sim.state()
            return_error = math.hypot(actual["x"]-saved_truth[0], actual["y"]-saved_truth[1])
            # Evaluate occupied cells near the four known test-room walls.
            grid = executor.room_map.snapshot()
            rows, cols = np.where(grid.grid == OCCUPIED)
            x = (cols+grid._min_col+.5)*grid.resolution
            y = (rows+grid._min_row+.5)*grid.resolution
            distance = np.minimum(abs(abs(x)-4.), abs(abs(y)-4.))
            wall95 = float(np.quantile(distance, .95)) if len(distance) else None
            sharp = wall95 is not None and wall95 < .15
            result = dict(seed=seed, requested_walk_s=seconds, elapsed_walk_s=walk_elapsed,
                          completed_corners=visited, saved_place=saved, returned=returned,
                          return_error_m=return_error, max_pose_error_m=max(errors, default=None),
                          raw_final_error_m=raw_errors[-1] if raw_errors else None,
                          wall_distance_p95_m=wall95, wall_cells=len(distance), map_sharp=sharp,
                          collision_count=actual.get("collision_count"), fault=actual.get("fault"),
                          settled=settled, failures=failures, samples=samples,
                          drift=asdict(DriftConfig(seed=seed)), wall_time_s=time.monotonic()-wall_start)
            result["passed"] = (not failures and returned and walk_elapsed >= seconds and return_error < .3
                                and sharp and actual.get("collision_count") == 0 and not actual.get("fault") and settled)
            return result
        finally:
            executor.close()
            sim.close()
            X.time = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=300.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("seconds must be positive and finite")
    result = run(args.seed, args.seconds)
    result["runtime"] = dict(python=sys.version, platform=platform.platform(), mujoco=mujoco.__version__, torch=torch.__version__)
    result["sources"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [ROOT/"go2/localization.py", ROOT/"simulation/odometry.py", ROOT/"go2/executor.py",
                                   ROOT/"simulation/exploration_clock.py", Path(__file__).resolve()]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: v for k,v in result.items() if k not in ("samples", "sources")}), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
