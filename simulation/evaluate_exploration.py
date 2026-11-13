"""Reproducible issue #14 coverage evaluation.

Default backend drives real headless MuJoCo through Executor. The fast `grid`
backend is an idealised ray-sensor regression check, NOT physics acceptance.
Ground truth supplies only scene creation, start selection and scoring. The
exploration policy receives an initially unknown map and sensor observations.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "simulation")]

import numpy as np

import house
from go2.exploration import ExplorationMap, _safe_cells, _oriented_safe, recovery_route, next_viewpoint
from go2.room_map import FREE, OCCUPIED, OccupancyGrid


class FloorTruth:
    """Exposed floor samples at cell centres; furniture footprints are excluded.

    This metric is LiDAR-observed exposed floor, not camera-visible floor or
    object-search success. All rooms count, including parts never reached.
    """
    def __init__(self, world, resolution=.05):
        self.rooms = [{"name": room["name"], "rect": list(room["rect"])} for room in world["rooms"]]
        extent = max(abs(v) for room in world["rooms"] for v in room["rect"])
        self.map = OccupancyGrid(resolution, 2 * math.ceil((extent + 1) / resolution))
        m = self.map
        r, c = np.indices(m.grid.shape)
        self.x = (c + m._min_col + .5) * resolution
        self.y = (r + m._min_row + .5) * resolution
        floor = np.zeros_like(m.grid, dtype=bool)
        for room in world["rooms"]:
            x0, y0, x1, y1 = room["rect"]
            floor |= (self.x > x0) & (self.x < x1) & (self.y > y0) & (self.y < y1)
        m.grid[:] = OCCUPIED
        m.grid[floor] = FREE

        def box(x0, y0, x1, y1):
            hit = (self.x >= x0) & (self.x <= x1) & (self.y >= y0) & (self.y <= y1)
            m.grid[hit] = OCCUPIED

        for axis, line, a, b, before, after in house.wall_pieces(world):
            lo = a - (0 if before or axis == "y" else house.WALL_THICK)
            hi = b + (0 if after or axis == "y" else house.WALL_THICK)
            if axis == "x":
                box(lo, line - house.WALL_THICK, hi, line + house.WALL_THICK)
            else:
                box(line - house.WALL_THICK, lo, line + house.WALL_THICK, hi)
        for door in world["doors"]:
            if not door["open"]:
                x, y = door["pos"]
                hx, hy = (door["width"] / 2, .03) if door["axis"] == "x" else (.03, door["width"] / 2)
                box(x - hx, y - hy, x + hx, y + hy)
        for item in world["objects"]:
            dx, dy = self.x - item["pos"][0], self.y - item["pos"][1]
            co, sn = math.cos(item["yaw"]), math.sin(item["yaw"])
            hx, hy = house.KINDS[item["kind"]][1]
            m.grid[(abs(co * dx + sn * dy) <= hx) & (abs(-sn * dx + co * dy) <= hy)] = OCCUPIED
        self.floor = m.grid == FREE
        self.samples = np.stack([self.x[self.floor], self.y[self.floor]], axis=1)

    def coverage(self, observed):
        return float(self.observed_samples(observed).mean())

    def observed_samples(self, observed):
        m = observed.snapshot()
        r, c, inside = m._cells(self.samples)
        seen = np.zeros(len(self.samples), dtype=bool)
        seen[inside] = m.grid[r[inside], c[inside]] == FREE
        return seen

    def coverage_by_room(self, observed):
        seen = self.observed_samples(observed)
        x, y = self.samples[:, 0], self.samples[:, 1]
        result = {}
        for room in self.rooms:
            x0, y0, x1, y1 = room["rect"]
            mask = (x > x0) & (x < x1) & (y > y0) & (y < y1)
            result[room["name"]] = float(seen[mask].mean()) if mask.any() else None
        return result

    def start(self, seed):
        cells = np.argwhere(_safe_cells(self.map, .5))
        if not len(cells):
            raise ValueError("no clear evaluation start")
        rng = random.Random(seed + 8000)
        r, c = cells[rng.randrange(len(cells))]
        return [*self.map.cell_to_world(r, c), rng.uniform(-math.pi, math.pi)]


def grid_scan(truth, observed, pose, radius=4.):
    """Ideal 360-degree planar rays, separate from MuJoCo's real L1 pattern."""
    angles = np.arange(720) * (2 * math.pi / 720)
    distance = np.arange(.025, radius + .025, .05)
    xy = np.stack([pose[0] + np.cos(angles[:, None]) * distance,
                   pose[1] + np.sin(angles[:, None]) * distance], axis=-1)
    rows, cols, inside = truth.map._cells(xy.reshape(-1, 2))
    hits = np.ones(rows.shape, dtype=bool)
    hits[inside] = truth.map.grid[rows[inside], cols[inside]] == OCCUPIED
    hits = hits.reshape(len(angles), -1)
    any_hit = hits.any(axis=1)
    end = np.where(any_hit, hits.argmax(axis=1), len(distance) - 1)
    points = np.column_stack([xy[np.arange(len(angles)), end], np.where(any_hit, .3, 0.)])
    observed.mark_scan(pose[:2], points)


def run_grid(world, truth, timeout_s, max_viewpoints):
    """Ideal sensor/motion diagnostic using the production heading-aware graph."""
    from go2.exploration_navigation import ExplorationNavigation, _offsets
    observed = OccupancyGrid(resolution=.05, size=400, grow=True)
    pose, visited, collisions = tuple(world["start"]), [], 0
    began, status, travelled = time.monotonic(), "viewpoint_limit", 0.
    for _ in range(max_viewpoints):
        for _ in range(6):
            grid_scan(truth, observed, pose)
        if truth.coverage(observed) >= .9:
            status = "coverage_reached"
            break
        if time.monotonic()-began >= timeout_s:
            status = "timed_out"
            break
        goal = None
        for visited_radius,turning,resolution in ((.5,.65,.1),(.25,.65,.1),(.25,.6,.1),(.25,.65,.05),(.25,.55,.05)):
            goal = next_viewpoint(ExplorationMap(observed,pose,clearance_m=.42,
                visited_radius_m=visited_radius,turn_clearance_m=turning,heading_aware=True,
                planning_resolution_m=resolution),visited)
            if goal is not None:
                break
            for _ in range(5):
                grid_scan(truth,observed,pose)
        if goal is None:
            status = "no_viewpoint"
            break
        navigation=ExplorationNavigation(observed,pose)
        if not navigation.start_fits and navigation.can_refine:
            refined=navigation.refined()
            if refined.start_fits:
                navigation=refined
        recovering=not navigation.start_fits
        if recovering:
            path=recovery_route(observed,pose,clearance=navigation.turning_radius,visited=visited)
            route=None if path is None else [(*cell,0) for cell in path]
            route_map=observed.snapshot()
        else:
            route=navigation.route(*goal[:2])
            if route is None and navigation.can_refine:
                navigation=navigation.refined()
                route=navigation.route(*goal[:2])
            route_map=navigation.map
        if route is None:
            raise RuntimeError("selector returned an unreachable viewpoint")
        truth_masks={}
        offsets={}
        for h,yaw in enumerate(navigation.headings):
            truth_masks[h]=_oriented_safe(truth.map,yaw)
            co,sn=math.cos(yaw),math.sin(yaw)
            length,width=(.45,.25) if recovering else (navigation.half_length,navigation.half_width)
            offsets[h]=_offsets(observed.resolution,math.hypot(length,width),
                lambda x,y: abs(co*x+sn*y)<=length and abs(-sn*x+co*y)<=width)
        for a,b in zip(route,route[1:]):
            x,y=route_map.cell_to_world(*b[:2])
            live=observed.world_to_cell(x,y)
            if a[:2]==b[:2] and not _safe_cells(observed,.5)[live]:
                break
            if live is None or any(not (0<=live[0]+dr<observed.size and 0<=live[1]+dc<observed.size)
                    or observed.grid[live[0]+dr,live[1]+dc]!=FREE for dr,dc in offsets[b[2]]):
                break
            tc=truth.map.world_to_cell(x,y)
            if tc is None or not truth_masks[b[2]][tc] or (a[:2]==b[:2] and not _safe_cells(truth.map,.4)[tc]):
                collisions+=1
                status="collision"
                break
            travelled+=math.dist(pose[:2],(x,y))
            pose=(x,y,navigation.headings[b[2]])
            grid_scan(truth,observed,pose)
        if collisions:
            break
        if math.dist(pose[:2],goal[:2])<.12 and _safe_cells(observed,.5)[observed.world_to_cell(*pose[:2])]:
            pose=(*pose[:2],goal[2])
        visited.append(pose)
    coverage=truth.coverage(observed)
    if coverage>=.9 and not collisions:
        status="coverage_reached"
    return dict(status=status,coverage=coverage,collisions=collisions,
                coverage_by_room=truth.coverage_by_room(observed),viewpoints=len(visited),
                distance_m=travelled,elapsed_s=time.monotonic()-began)


def settle_robot(sim, clock, timeout_s=3.):
    """Include braking/stance settling in collision scoring before closing physics."""
    sim.stop_move()
    began = clock.monotonic()
    while True:
        state = sim.state()
        settled = state.speed < .05 and abs(state.wz) < .1 and state.posture == "standing"
        if settled or clock.monotonic() - began >= timeout_s:
            return state, settled, clock.monotonic() - began
        clock.sleep(.05)


def remember_attempt(visited, blocked, goal, before, after, status):
    """Record observed positions; an unreached destination was not searched.

    A timeout after useful travel can be retried from the new pose. Only recent
    failures without translation are temporarily excluded, rather than marking
    the unseen destination as permanently visited.
    """
    moved=math.hypot(after.x-before.x,after.y-before.y)
    if status=='completed' or moved>=.05:
        visited.append((after.x,after.y,after.yaw))
        blocked.clear()
    else:
        blocked.append(tuple(goal))
        del blocked[:-3]
    return moved


def run_mujoco(world, truth, timeout_s, max_viewpoints, *, stepped=False, progress=False):
    from go2.adapters.sim import SimAdapter
    from go2 import executor as X
    if stepped:
        from exploration_clock import SteppedSimAdapter
        sim = SteppedSimAdapter(world)
    else:
        sim = SimAdapter(scene="house", headless=True, world=world)
    clock = sim.clock if stepped else time
    wall_began, began = time.monotonic(), clock.monotonic()
    visited, failures, blocked, stalled, attempted = [], [], [], 0, 0
    status, collisions, fault = "viewpoint_limit", None, None
    with tempfile.TemporaryDirectory() as temp:
        original_clock = X.time
        if stepped:
            X.time = clock
        executor = X.Executor(memory_path=str(Path(temp) / "memory.json"))
        try:
            sim.connect()
            executor.attach(sim)
            for _ in range(max_viewpoints):
                left = timeout_s - (clock.monotonic() - began)
                if left <= 0:
                    status = "timed_out"
                    break
                goal = executor.exploration_viewpoint(visited+blocked)
                if goal is None and blocked:
                    blocked.clear()
                    goal=executor.exploration_viewpoint(visited)
                if progress:
                    print(json.dumps({"progress": attempted, "coverage": truth.coverage(executor.room_map),
                                      "next": None if goal is None else [float(v) for v in goal]}), flush=True)
                if truth.coverage(executor.room_map) >= .9:
                    status = "coverage_reached"
                    break
                if goal is None:
                    status = "no_viewpoint"
                    break
                before = sim.state()
                attempted+=1
                result = executor.submit_viewpoint(goal, timeout_s=min(left, 90))
                if result["status"] in ("blocked", "failed", "rejected"):
                    failures.append(result)
                    after = sim.state()
                    moved=remember_attempt(visited,blocked,goal,before,after,result['status'])
                    if after.extra.get("collision_count") or after.fault:
                        status, fault = "collision_or_fault", after.fault
                        break
                    stalled = stalled+1 if moved<.05 else 0
                    if stalled >= 5:
                        status = "stalled"
                        break
                    continue
                while clock.monotonic() - began < timeout_s:
                    with executor.lock:
                        task = next((t for t in executor.history if t.id == result.get("task_id")), None)
                    if task is not None:
                        break
                    time.sleep(.05)
                else:
                    executor.stop("evaluation timed out")
                    status = "timed_out"
                    break
                if task.status != "completed":
                    failures.append(task.public())
                state = sim.state()
                moved=remember_attempt(visited,blocked,goal,before,state,task.status)
                stalled = (stalled + 1 if task.status != "completed"
                           and moved<.05 else 0)
                collisions = state.extra.get("collision_count")
                if collisions is None:
                    raise RuntimeError("simulator lacks collision instrumentation")
                if collisions or state.fault:
                    status, fault = "collision_or_fault", state.fault
                    break
                if stalled >= 5:
                    status = "stalled"
                    break
            executor.stop("evaluation finished")
            until = time.monotonic() + 3.
            while executor.current is not None and time.monotonic() < until:
                time.sleep(.01)
            elapsed = clock.monotonic() - began
            state, settled, settle_s = settle_robot(sim, clock)
            collisions = state.extra.get("collision_count")
            result = dict(status=status, coverage=truth.coverage(executor.room_map),
                          coverage_by_room=truth.coverage_by_room(executor.room_map),
                          collisions=collisions, fault=fault or state.fault,
                          collision_events=state.extra.get("collision_events", []),
                          viewpoints=attempted, observed_stops=len(visited), failures=failures,
                          sim_time_s=state.extra.get("sim_time"), elapsed_s=elapsed,
                          settled=settled, settle_s=settle_s,
                          wall_elapsed_s=time.monotonic() - wall_began)
            if result["coverage"] >= .9 and not result["collisions"] and not result["fault"]:
                result["stop_reason"], result["status"] = status, "coverage_reached"
        finally:
            executor.stop("evaluation finished")
            executor.close()
            # Let the cancelled worker finish before restoring its time source.
            until = time.monotonic() + 3.
            while executor.current is not None and time.monotonic() < until:
                time.sleep(.01)
            try:
                sim.close()
            finally:
                X.time = original_clock
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mujoco", "mujoco-step", "grid"), default="mujoco")
    parser.add_argument("--houses", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=300.)
    parser.add_argument("--max-viewpoints", type=int, default=120)
    parser.add_argument("--starts-per-house", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "runtime/exploration-results.json")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    if min(args.houses, args.starts_per_house, args.max_viewpoints, args.timeout_s) <= 0:
        parser.error("counts and timeout must be positive")
    report = dict(backend=args.backend, metric="lidar_observed_exposed_floor_fraction",
                  physics_acceptance=args.backend != "grid", target=.9,
                  map_resolution_m=.05, scoring_resolution_m=.05,
                  navigation_model="heading_and_stopping_corridor",
                  collision_envelope_half_length_m=.42, collision_envelope_half_width_m=.23,
                  planning_resolution_m=.1,
                  body_half_length_m=.52, body_half_width_m=.33,
                  refined_planning_resolution_m=.05, refined_body_half_length_m=.47,
                  refined_body_half_width_m=.28, refined_turning_radius_m=.55,
                  turning_radius_m=.6, stopping_lookahead_m=.30, motion_speed_mps=.15,
                  collision_measurement=("MuJoCo robot/world contact onsets at each physics substep"
                                         if args.backend != "grid" else "idealised grid footprint overlap"),
                  seed_start=args.seed, houses=args.houses, starts_per_house=args.starts_per_house,
                  timeout_s=args.timeout_s, max_viewpoints=args.max_viewpoints,
                  viewpoint_timeout_s=90,
                  budget_units="simulated_seconds" if args.backend == "mujoco-step" else "wall_seconds",
                  python=sys.version, numpy=np.__version__,
                  thread_environment={k: os.environ.get(k) for k in
                                      ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
                  source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                 for name in ("go2/exploration.py", "go2/adapters/sim.py", "go2/limits.py", "go2/perception.py", "go2/footprint.py", "go2/exploration_navigation.py", "go2/room_map.py", "go2/path_planner.py",
                                              "go2/room_regions.py", "go2/executor.py", "simulation/robot.py",
                                              "simulation/house.py", "simulation/sensors.py", "simulation/world.py", "simulation/evaluate_exploration.py",
                                              "simulation/exploration_clock.py", "simulation/calibrate_exploration_footprint.py")},
                  runs=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for seed in range(args.seed, args.seed + args.houses):
        for start_index in range(args.starts_per_house):
            world = house.random_house(random.Random(seed))
            truth = FloorTruth(world)
            world["start"] = [float(v) for v in truth.start(seed * 100 + start_index)]
            try:
                if args.backend == "grid":
                    run = run_grid(world, truth, args.timeout_s, args.max_viewpoints)
                else:
                    run = run_mujoco(world, truth, args.timeout_s, args.max_viewpoints,
                                     stepped=args.backend == "mujoco-step", progress=args.progress)
            except Exception as exc:
                run = dict(status="error", error=str(exc), coverage=None, collisions=None)
            run.update(seed=seed, start_index=start_index, start=world["start"],
                       rooms=len(world["rooms"]), floor_samples=len(truth.samples))
            run["passed"] = (run.get("coverage") is not None and run["coverage"] >= .9
                             and run.get("collisions") == 0 and not run.get("fault")
                             and run.get("settled", True)
                             and run.get("elapsed_s", float("inf")) <= args.timeout_s + .1)
            report["runs"].append(run)
            report["passed_runs"] = sum(r["passed"] for r in report["runs"])
            report["complete"] = len(report["runs"]) == args.houses * args.starts_per_house
            report["all_passed"] = report["complete"] and all(r["passed"] for r in report["runs"])
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            summary = {k: run[k] for k in ("seed", "start_index", "status", "coverage", "collisions",
                                           "fault", "viewpoints", "elapsed_s", "passed", "error") if k in run}
            summary["failed_viewpoints"] = len(run.get("failures", []))
            print(json.dumps(summary), flush=True)
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
