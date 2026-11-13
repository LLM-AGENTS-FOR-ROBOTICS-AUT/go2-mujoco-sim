"""Choose safe observation positions on the LiDAR map (#14).

This module chooses positions; the mission runner must execute them through the
existing executor and scan at each position. It never commands a robot directly.
"""

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from .room_map import FREE, UNKNOWN, OccupancyGrid


@dataclass(frozen=True)
class ExplorationMap:
    """Inputs for one planning decision, all in odometry coordinates.

    pose is (x metres, y metres, yaw radians). clearance_m is a conservative
    circular body clearance. extra_obstacles carries the executor's remembered
    bump points, which LiDAR alone can miss. The mission runner supplies fresh
    inputs after every move/scan.

    heading_aware uses the calibrated moving-body model. Planning can aggregate
    to planning_resolution_m, retaining a fine-map fallback for narrow routes.
    """

    occupancy: OccupancyGrid
    pose: tuple
    clearance_m: float = .5
    visited_radius_m: float = .75
    extra_obstacles: object = None
    turn_clearance_m: float | None = None
    heading_aware: bool = False
    forbidden_steps: tuple = ()
    planning_resolution_m: float = .1
    allow_recovery: bool = True


_STEPS = ((-1, 0), (0, -1), (0, 1), (1, 0))


def _pose(value, name):
    try:
        values = tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite (x, y, yaw) pose") from exc
    if len(values) != 3 or not all(math.isfinite(v) for v in values):
        raise ValueError(f"{name} must be a finite (x, y, yaw) pose")
    return values


def _neighbours(cell, shape):
    r, c = cell
    for dr, dc in _STEPS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < shape[0] and 0 <= nc < shape[1]:
            yield nr, nc


def _safe_cells(m, clearance):
    """Keep the complete body footprint in known free space, inside the map."""
    reach = math.ceil(clearance / m.resolution)
    unsafe = m.grid != FREE
    padded = np.pad(unsafe, reach, constant_values=True)
    blocked = np.zeros_like(unsafe)
    for dr in range(-reach, reach + 1):
        for dc in range(-reach, reach + 1):
            if dr * dr + dc * dc <= (clearance / m.resolution) ** 2 + 1e-9:
                blocked |= padded[reach + dr:reach + dr + m.size,
                                  reach + dc:reach + dc + m.size]
    return ~blocked


def _oriented_safe(m, yaw, *, offset=(0., 0.), dimensions=(.45, .25)):
    """Known-free cells fitting the moving body at one fixed heading."""
    length, width = dimensions
    reach = math.ceil((math.hypot(length, width) + math.hypot(*offset)) / m.resolution)
    unsafe = np.pad(m.grid != FREE, reach, constant_values=True)
    blocked = np.zeros_like(m.grid, dtype=bool)
    co, sn = math.cos(yaw), math.sin(yaw)
    for dr in range(-reach, reach + 1):
        for dc in range(-reach, reach + 1):
            x, y = dc * m.resolution - offset[0], dr * m.resolution - offset[1]
            if abs(co*x+sn*y)<=length+1e-9 and abs(-sn*x+co*y)<=width+1e-9:
                blocked |= unsafe[reach + dr:reach + dr + m.size,
                                  reach + dc:reach + dc + m.size]
    return ~blocked


def recovery_route(m, pose, clearance=.42, max_distance_m=1.5, *, visited=(), forbidden_steps=()):
    """Translate without rotating to nearby turning space, on observed floor.

    A robot beside furniture may fit at its current heading but lack room to
    turn. This bounded escape path uses its oriented rectangular footprint.
    None means do not attempt recovery; every executed step still needs LiDAR.
    """
    x, y, yaw = _pose(pose, "pose")
    start = m.world_to_cell(x, y)
    if start is None:
        return None
    sx, sy = m.cell_to_world(*start)
    offset = x - sx, y - sy
    # Keep the measured subcell position during fixed-heading recovery. A
    # cell-centre snap can move the footprint into the very wall we must leave.
    # Expand the calibrated body by the projection of half a grid cell so an
    # occupied cell's area, not just its centre, is kept outside the body.
    from .exploration_navigation import BODY_HALF_LENGTH_M, BODY_HALF_WIDTH_M
    allowance = .5 * m.resolution * (abs(math.cos(yaw)) + abs(math.sin(yaw)))
    dimensions = BODY_HALF_LENGTH_M + allowance, BODY_HALF_WIDTH_M + allowance
    def fixed_mask(occupancy):
        return _oriented_safe(occupancy, yaw, offset=offset, dimensions=dimensions)
    def position(cell):
        wx, wy = m.cell_to_world(*cell)
        return wx + offset[0], wy + offset[1]
    fixed = fixed_mask(m)
    low_overlap = np.empty((0, 2))
    if not fixed[start] and m.detect_low:
        # A newly detected low surface may be under the belly without a body
        # collision. Permit only a fixed-heading escape with non-increasing
        # overlap. Walls/high obstacles and unknown cells are never cleared.
        low = m.low_obstacles()[:, :2]
        co, sn = math.cos(yaw), math.sin(yaw)
        if len(low):
            dx, dy = low[:, 0] - x, low[:, 1] - y
            low_overlap = low[(abs(co*dx+sn*dy)<=dimensions[0]) &
                              (abs(-sn*dx+co*dy)<=dimensions[1])]
        if len(low_overlap):
            planning = m.snapshot()
            for point in low_overlap:
                planning.grid[planning.world_to_cell(*point)] = FREE
            fixed = fixed_mask(planning)
    if not fixed[start]:
        return None

    def overlaps(cell):
        if not len(low_overlap):
            return 0
        wx, wy = position(cell)
        dx, dy = low_overlap[:, 0] - wx, low_overlap[:, 1] - wy
        co, sn = math.cos(yaw), math.sin(yaw)
        return np.count_nonzero((abs(co*dx+sn*dy)<=dimensions[0]) &
                               (abs(-sn*dx+co*dy)<=dimensions[1]))
    turning = _safe_cells(m, clearance)
    queue, parent = deque([start]), {start: None}
    while queue:
        cell = queue.popleft()
        wx, wy = position(cell)
        if turning[cell] and not any(math.hypot(wx - vx, wy - vy) <= .2 for vx, vy, _ in visited):
            path = [cell]
            while parent[path[-1]] is not None:
                path.append(parent[path[-1]])
            return list(reversed(path))
        for n in _neighbours(cell, m.grid.shape):
            if cell == start and (n[0] - cell[0], n[1] - cell[1]) in forbidden_steps:
                continue
            wx, wy = position(n)
            if (n not in parent and fixed[n] and math.hypot(wx - x, wy - y) <= max_distance_m
                    and overlaps(n) <= overlaps(cell)):
                parent[n] = cell
                queue.append(n)
    return None


def _frontiers(grid):
    """Connected groups of free cells adjoining unknown cells on four sides."""
    unknown = grid == UNKNOWN
    adjacent = np.zeros_like(unknown)
    adjacent[1:] |= unknown[:-1]
    adjacent[:-1] |= unknown[1:]
    adjacent[:, 1:] |= unknown[:, :-1]
    adjacent[:, :-1] |= unknown[:, 1:]
    remaining = set(zip(*np.nonzero((grid == FREE) & adjacent)))
    groups = []
    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        group, queue = [], deque([seed])
        while queue:
            cell = queue.popleft()
            group.append(cell)
            for neighbour in _neighbours(cell, grid.shape):
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    queue.append(neighbour)
        groups.append(group)
    return groups


def next_viewpoint(map, visited):
    """Return an unvisited (x, y, yaw), or None if no safe frontier is reachable.

    map is an ExplorationMap. visited contains observation poses in metres and
    radians; nearby positions are suppressed regardless of yaw
    because the search mission scans all round at each stop. Unknown cells are
    never traversed, and clearance applies along the complete known-free route.

    Prefer the shortest safe route, breaking ties by larger frontier group then
    cell order. The viewpoint stands behind the frontier, facing unseen space.
    None means no currently available viewpoint, not proof a target is absent.
    Temporary blocked-goal exclusions may be combined with visited. A timeout
    partway through a route must not permanently mark its unreached goal seen.
    """
    x, y, _ = _pose(map.pose, "pose")
    for name in ("clearance_m", "visited_radius_m"):
        value = getattr(map, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    visited = [_pose(v, "visited entry") for v in visited]
    if map.turn_clearance_m is not None and (not math.isfinite(map.turn_clearance_m)
                                            or map.turn_clearance_m < map.clearance_m):
        raise ValueError("turn clearance must be finite and at least the transit clearance")
    original = map.occupancy
    # Work on a private snapshot: planning must not edit the live LiDAR map.
    m = original.snapshot()
    m.grow = False
    if map.extra_obstacles is not None:
        m.mark_obstacles(map.extra_obstacles)
    start = m.world_to_cell(x, y)
    safe = _safe_cells(m, map.clearance_m)
    navigation = None
    if map.heading_aware:
        from .exploration_navigation import ExplorationNavigation
        navigation = ExplorationNavigation(m,map.pose,forbidden_steps=map.forbidden_steps,
                                            planning_resolution_m=map.planning_resolution_m)
        if not navigation.start_fits and navigation.can_refine:
            refined=navigation.refined()
            if refined.start_fits:
                navigation=refined
    if start is None:
        return None
    def recover():
        recovery = recovery_route(m,map.pose,navigation.turning_radius if navigation is not None else map.clearance_m,
                                  visited=visited, forbidden_steps=map.forbidden_steps)
        if recovery and len(recovery) > 1:
            rx, ry = m.cell_to_world(*recovery[-1])
            sx, sy = m.cell_to_world(*start)
            return float(rx + x - sx), float(ry + y - sy), float(map.pose[2])
        return None
    if not (navigation.start_fits if navigation is not None else safe[start]):
        return recover() if map.allow_recovery else None
    turn_radius=map.turn_clearance_m
    if navigation is not None:
        turn_radius=max(turn_radius or 0.,navigation.turning_radius)
    turn_safe = safe if turn_radius is None else _safe_cells(m,turn_radius)

    if navigation is not None:
        distances = navigation.distances()
    else:
        distances = {start: 0}
        queue = deque([start])
        while queue:
            cell = queue.popleft()
            for neighbour in _neighbours(cell, safe.shape):
                if safe[neighbour] and neighbour not in distances:
                    distances[neighbour] = distances[cell] + 1
                    queue.append(neighbour)

    # Look from just behind the observed edge so the body stays in known space.
    inset=max(map.clearance_m,turn_radius or 0.,.75 if navigation is not None else 0.)
    reach = math.ceil(inset/m.resolution)+2
    # A doorway or furniture can hide the edge from every nearby turning
    # position. Retry from farther back before declaring exploration exhausted.
    # The same observed-floor route, turning footprint and clear sightline are
    # required; this changes where we look from, not where we may walk.
    ranges = [reach]
    if navigation is not None:
        ranges.append(max(reach, math.ceil(1.5 / m.resolution)))
    groups = _frontiers(m.grid)
    unknown_or_obstacle = m.grid != FREE
    for reach in dict.fromkeys(ranges):
        offsets = [(dr, dc) for dr in range(-reach, reach + 1)
                   for dc in range(-reach, reach + 1)
                   if 0 < dr * dr + dc * dc <= reach * reach]
        best = None
        for group in groups:
            for edge in group:
                for dr, dc in offsets:
                    candidate = edge[0] + dr, edge[1] + dc
                    if not (0<=candidate[0]<m.size and 0<=candidate[1]<m.size) or not turn_safe[candidate]:
                        continue
                    wx,wy=m.cell_to_world(*candidate)
                    route_cell=navigation.map.world_to_cell(wx,wy) if navigation is not None else candidate
                    if route_cell not in distances:
                        continue
                    # Rank before the more expensive visibility and visited checks.
                    travel = distances[route_cell]
                    if map.heading_aware:
                        travel /= math.sqrt(min(len(group), 100))
                    score = (travel, -len(group), candidate,
                             dr * dr + dc * dc, edge)
                    if best is not None and score >= best[0]:
                        continue
                    wx, wy = m.cell_to_world(*candidate)
                    if any(math.hypot(wx - vx, wy - vy) <= map.visited_radius_m
                           for vx, vy, _ in visited):
                        continue
                    if not m.line_clear(unknown_or_obstacle, candidate, edge):
                        continue
                    if best is None or score < best[0]:
                        ex, ey = m.cell_to_world(*edge)
                        best = score, (wx, wy, math.atan2(ey - wy, ex - wx))
        if best is not None:
            return best[1]
    # The padded stopping corridor can trap the normal graph even when the
    # standing footprint fits. Do not declare exhaustion if a fixed-heading,
    # observed-floor retreat can reach turning space. Execution still applies
    # the calibrated body and live stopping/ledge checks to every recovery leg.
    if (map.allow_recovery and groups and navigation is not None
            and not any(navigation.turning[cell] for cell in distances)):
        return recover()
    return None
