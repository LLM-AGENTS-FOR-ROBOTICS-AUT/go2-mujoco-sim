"""Mission evaluation (#18): search and explore in random houses, end to end, in simulated time.

  python simulation/evaluate_missions.py search --houses 50     # find the teddy bear: the share found
  python simulation/evaluate_missions.py explore --houses 10    # explore an unknown house: the share of rooms named

Each run builds random house N (its seed), starts Go2 where the house says with nothing remembered and nothing mapped,
and runs the mission through the normal Executor and its safety checks. Physics, the walking policy, the LiDAR and the
camera (the simulator's own labels) run in this process on simulated time (exploration_clock.py), so it runs faster
than real time; it's the same mission as on the console. The house description is used only to score: a search
succeeds when the mission says it found the target and Go2 ended within REACHED_M of it, with no collision or fall;
explore is scored by the type each room's middle is named as. A line of JSON per run, then a summary.
"""
import argparse
import json
import math
import queue
import random
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "simulation")]

import numpy as np  # noqa: E402

import house  # noqa: E402
from exploration_clock import SteppedSimAdapter  # noqa: E402
from go2 import executor as X  # noqa: E402
from go2.adapters.base import AdapterError, CameraFrame  # noqa: E402
from go2.adapters.sim import SimAdapter  # noqa: E402
from go2.perception import Mask  # noqa: E402
from go2.tools import MISSION_LIMITS, resolve  # noqa: E402
from sensors import Camera, camera_pose_in_base  # noqa: E402

REACHED_M = 1.5            # a search succeeded if Go2 ended this close to the target
REACHABLE_M = 1.1          # a target counts as reachable if, by the house plan, Go2's centre can get this close


def reachable(world, x, y, within=REACHABLE_M):
    """From the house plan: can Go2's centre get within `within` m of (x, y)? Walls, closed doors and furniture
    (other than a thing at (x, y) itself) grown by Go2's radius, then flood-filled from the start, as
    house.reachability does. #18's target counts only reachable targets."""
    from collections import deque
    grid = house.GRID
    xs = [v for room in world["rooms"] for v in room["rect"][0::2]]
    ys = [v for room in world["rooms"] for v in room["rect"][1::2]]
    x0, y0 = min(xs), min(ys)
    nx, ny = int(round((max(xs) - x0) / grid)) + 1, int(round((max(ys) - y0) / grid)) + 1

    def cell(px, py):
        return int(round((px - x0) / grid)), int(round((py - y0) / grid))
    blocked = np.zeros((nx, ny), bool)

    def mark(bx0, by0, bx1, by1):
        (i0, j0), (i1, j1) = cell(bx0, by0), cell(bx1, by1)
        blocked[max(i0, 0):max(i1 + 1, 0), max(j0, 0):max(j1 + 1, 0)] = True
    w = house.WALL_THICK
    for axis, line, a, b, *_ in house.wall_pieces(world):
        mark(a, line - w, b, line + w) if axis == "x" else mark(line - w, a, line + w, b)
    for door in world["doors"]:
        if not door["open"]:
            (dx, dy), half = door["pos"], door["width"] / 2
            mark(dx - half, dy - w, dx + half, dy + w) if door["axis"] == "x" else mark(dx - w, dy - half, dx + w, dy + half)
    for item in world["objects"]:
        if math.hypot(item["pos"][0] - x, item["pos"][1] - y) > .05:
            mark(*house.footprint(item))
    r = int(round(house.ROBOT_RADIUS / grid))
    padded, grown = np.pad(blocked, r), np.zeros_like(blocked)
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            if di * di + dj * dj <= r * r:
                grown |= padded[r + di:r + di + nx, r + dj:r + dj + ny]
    start = cell(*world["start"][:2])
    seen = np.zeros_like(grown)
    queue = deque([start]) if not grown[start] else deque()
    seen[start] = True
    goal = cell(x, y)
    reach = within / grid
    while queue:
        i, j = queue.popleft()
        if (i - goal[0]) ** 2 + (j - goal[1]) ** 2 <= reach * reach:
            return True
        for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if 0 <= a < nx and 0 <= b < ny and not grown[a, b] and not seen[a, b]:
                seen[a, b] = True
                queue.append((a, b))
    return False


class CameraSteppedSim(SteppedSimAdapter):
    """The offline simulator, with the front camera: images and the simulator's own labels, as sim_server gives.
    Rendering has a thread of its own, as in sim_server: a GL context must be made, used and freed by one thread
    (freeing it from another crashes on Windows)."""

    def __init__(self, world):
        super().__init__(world)
        self._requests = queue.Queue()
        self._renderer = threading.Thread(target=self._render_loop, daemon=True, name="camera")
        self._renderer.start()

    def _render_loop(self):
        camera = None
        try:
            while True:
                reply = self._requests.get()
                if reply is None:
                    return
                try:
                    if camera is None:
                        camera = Camera(self.sim.model)
                    with self.sim.lock:
                        rgb, bodies = camera.capture(self.sim.data)
                        raw = self._raw_state()
                    reply["value"] = rgb, bodies, raw, camera.intrinsics()
                except Exception as error:
                    reply["error"] = error
                reply["done"].set()
        finally:
            if camera is not None:
                camera.close()

    def camera(self):
        reply = {"done": threading.Event()}
        self._requests.put(reply)
        if not reply["done"].wait(30):
            raise AdapterError("camera render timed out")
        if "error" in reply:
            raise AdapterError(f"camera render failed: {reply['error']}")
        rgb, bodies, raw, intrinsics = reply["value"]
        masks = [Mask(item.label, bodies == body, 1.)
                 for body, item in self.sim.objects.items()
                 if item.label != "wall" and (bodies == body).sum() >= 12]
        pos, rot = camera_pose_in_base(self.sim.model)
        return CameraFrame(rgb=rgb, masks=masks, intrinsics=intrinsics, camera_pos=pos, camera_rot=rot,
                           state=SimAdapter._local_state(self, raw))

    def close(self):
        super().close()
        self._requests.put(None)
        self._renderer.join(10)

    def sport(self, name, **args):
        if name in ("stand", "stand_up", "balance_stand"):
            return super().sport("stand" if name == "balance_stand" else name)
        return None                             # postures beyond standing don't matter here


def run(world, mission, args, time_limit):
    """Run one mission in `world`; (task, final state, rooms on the map, simulated seconds, wall seconds)."""
    sim = CameraSteppedSim(world)
    clock = sim.clock
    original = X.time
    X.time = clock
    wall, began = time.monotonic(), clock.monotonic()
    simulated = 0.
    events = []
    with tempfile.TemporaryDirectory() as temp:
        executor = X.Executor(on_event=events.append, memory_path=str(Path(temp) / "memory.json"))
        try:
            sim.connect()
            executor.attach(sim)
            result = executor.submit(resolve(mission, {**args, "time_limit_s": time_limit}), "g", 1)[0]
            if result["status"] != "queued":
                raise RuntimeError(f"the mission wasn't started: {result}")
            while not events:
                time.sleep(.05)
            task, state = events[-1], sim.state()
            simulated = clock.monotonic() - began
            rooms = executor.rooms(fresh=True)
        finally:
            executor.stop("evaluation finished")
            executor.close()
            until = time.monotonic() + 3.
            while executor.current is not None and time.monotonic() < until:
                time.sleep(.01)
            sim.close()
            X.time = original
    return task, state, rooms, simulated, time.monotonic() - wall


def evaluate_search(seed, time_limit):
    world = house.random_house(random.Random(seed))
    teddy = next(o for o in world["objects"] if o["kind"] == "teddy")
    task, state, _, sim_s, wall_s = run(world, "search", {"target": "teddy bear"}, time_limit)
    mission = task.detail.get("mission", {})
    gap = math.hypot(state.x - teddy["pos"][0], state.y - teddy["pos"][1])
    collisions = state.extra.get("collision_count")
    ok = mission.get("outcome") == "found" and gap <= REACHED_M and not collisions and not state.fault
    return {"seed": seed, "ok": ok, "reachable": reachable(world, *teddy["pos"]), "outcome": mission.get("outcome"),
            "status": task.status,
            "error": task.detail.get("error"), "teddy_room": teddy["room"], "said_room": mission.get("room"),
            "gap_m": round(gap, 2), "collisions": collisions, "fault": state.fault,
            "sim_s": round(sim_s or 0.), "wall_s": round(wall_s), "rooms": len(world["rooms"])}


def evaluate_explore(seed, time_limit):
    world = house.random_house(random.Random(seed))
    task, state, rooms, sim_s, wall_s = run(world, "explore", {}, time_limit)
    named = {}
    regions = {r["region"]: r["name"] for r in rooms}
    for room in world["rooms"]:
        if room["type"] == "hallway":
            continue
        x0, y0, x1, y1 = room["rect"]
        middle = min(rooms, key=lambda r: (not (x0 <= r["x"] <= x1 and y0 <= r["y"] <= y1),
                                           math.hypot(r["x"] - (x0 + x1) / 2, r["y"] - (y0 + y1) / 2)), default=None)
        guess = middle["name"] if middle and x0 <= middle["x"] <= x1 and y0 <= middle["y"] <= y1 else None
        named[room["name"]] = (room["type"], guess)
    right = sum(1 for kind, guess in named.values()
                if guess is not None and (guess == kind or (guess[-1].isdigit() and guess.rsplit("_", 1)[0] == kind)))
    return {"seed": seed, "named_right": right, "rooms": len(named), "outcome": task.detail.get("mission", {})
            .get("outcome"), "status": task.status, "error": task.detail.get("error"), "named": named,
            "regions": len(regions), "collisions": state.extra.get("collision_count"), "fault": state.fault,
            "sim_s": round(sim_s or 0.), "wall_s": round(wall_s)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mission", choices=("search", "explore"))
    parser.add_argument("--houses", type=int, default=50)
    parser.add_argument("--first", type=int, default=0, help="seed of the first house")
    parser.add_argument("--time-limit", type=float, default=None, help="seconds per mission (simulated)")
    args = parser.parse_args()
    results = []
    for seed in range(args.first, args.first + args.houses):
        if args.mission == "search":
            result = evaluate_search(seed, args.time_limit or MISSION_LIMITS["search"][1])
        else:
            result = evaluate_explore(seed, args.time_limit or MISSION_LIMITS["explore"][1])
        results.append(result)
        print(json.dumps(result, default=str), flush=True)
    if args.mission == "search":
        reach = [r for r in results if r["reachable"]]
        ok = sum(r["ok"] for r in reach)
        summary = {"houses": len(results), "reachable": len(reach), "found": ok,
                   "rate": round(ok / max(len(reach), 1), 3),
                   "unreachable_outcomes": [r["outcome"] for r in results if not r["reachable"]],
                   "collisions": sum(bool(r["collisions"]) for r in results),
                   "mean_sim_s": round(float(np.mean([r["sim_s"] for r in results])))}
    else:
        right, total = sum(r["named_right"] for r in results), sum(r["rooms"] for r in results)
        summary = {"houses": len(results), "rooms_named_right": right, "rooms": total,
                   "rate": round(right / max(total, 1), 3),
                   "collisions": sum(bool(r["collisions"]) for r in results)}
    print(json.dumps({"summary": summary}), flush=True)


if __name__ == "__main__":
    main()
