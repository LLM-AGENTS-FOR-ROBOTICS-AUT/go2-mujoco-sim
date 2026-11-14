"""Room regions inferred from observed free space, not the simulator floor plan.

Eroding free space removes narrow doorways. Connected room cores then grow back
through free cells with a multi-source flood fill. This is a geometric heuristic:
an open-plan room or a wide arch may stay one region. Furniture can leave a gap
as narrow as a doorway; regions that meet at a gap that isn't one are merged
back (see JAMB_M below, #35). Region IDs are associated by observed world-space
overlap across map updates. Names here are the user's; go2/room_names.py guesses
the others from what is seen in each region (#35).
"""

from collections import deque
import threading
from types import SimpleNamespace

import numpy as np

from .exploration import _neighbours, _safe_cells
from .room_map import FREE, OCCUPIED

# Furniture can leave a gap as narrow as a doorway, so eroding cuts a room in two there (#35). A doorway is a gap
# in a thin wall; a gap beside furniture has something thick on at least one side (the furniture, or a wall
# running along the way through), or is wider than any doorway once the regions' border bends round it. Regions
# that meet at such a gap are one room.
JAMB_M = .45               # a 10 cm wall scans up to 0.3-0.4 m thick; furniture is mostly 0.5 m or deeper
WIDE_GAP_M = 1.6           # wider than any doorway
PROBE_M = (.1, .35)        # how far past each end of a gap to look for its side
# Rooms are split on a map with cells about this size, and finer maps take their labels from it: at 5 cm the
# regions' borders bend round door frames, and over 101 random houses 12 had two rooms merged (none at 10 cm).
SPLIT_AT_M = .1


def _flood(labels, grid):
    """labels grown through free cells from the labelled ones (breadth first: nearest label wins)."""
    labels = labels.copy()
    queue = deque(zip(*np.nonzero(labels)))
    while queue:
        cell = queue.popleft()
        for n in _neighbours(cell, labels.shape):
            if grid[n] == FREE and labels[n] == 0:
                labels[n] = labels[cell]
                queue.append(n)
    return labels


def _coarse(m, k):
    """Map m with k x k cells merged into one: occupied if any of them is, else free if any is, else unknown."""
    rows = m.grid.shape[0] - m.grid.shape[0] % k
    blocks = m.grid[:rows, :rows].reshape(rows // k, k, rows // k, k)
    grid = np.where((blocks == OCCUPIED).any(axis=(1, 3)), OCCUPIED, blocks.max(axis=(1, 3))).astype(m.grid.dtype)
    return SimpleNamespace(grid=grid, resolution=m.resolution * k, size=grid.shape[0])


def _thickness(grid, point, step, reach):
    """Occupied cells in a row through `point` (row, col) along `step` both ways, up to `reach` cells each way;
    None if the cell at `point` isn't occupied."""
    r, c = int(round(point[0])), int(round(point[1]))
    if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]) or grid[r, c] != OCCUPIED:
        return None
    count = 1
    for sign in (1, -1):
        for k in range(1, reach + 1):
            rr, cc = int(round(point[0] + sign * k * step[0])), int(round(point[1] + sign * k * step[1]))
            if not (0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]) or grid[rr, cc] != OCCUPIED:
                break
            count += 1
    return count


def _way_through(labels, a, b, centre, reach):
    """Unit (row, col) direction from region a's floor to region b's, right by the gap at `centre`."""
    r0, c0 = (max(int(v) - reach, 0) for v in centre)
    window = labels[r0:int(centre[0]) + reach + 1, c0:int(centre[1]) + reach + 1]
    sides = []
    for label in (a, b):
        cells = np.argwhere(window == label) + (r0, c0)
        if not len(cells):         # a long or bent border: its middle can be far from one region's floor
            cells = np.argwhere(labels == label)
            cells = cells[[np.argmin(((cells - centre) ** 2).sum(axis=1))]]
        sides.append(cells.mean(axis=0))
    step = sides[1] - sides[0]
    return step / (np.linalg.norm(step) or 1.)


def _is_doorway(border, grid, resolution, across):
    """Is the gap where two regions meet (border: cell positions along it; across: the way through it) a
    doorway, with thin sides?"""
    centre = border.mean(axis=0)
    along = np.array([-across[1], across[0]])
    t = (border - centre) @ along
    if (t.max() - t.min()) * resolution > WIDE_GAP_M:
        return False
    reach = int(np.ceil(1. / resolution))
    for end, sign in ((t.min(), -1), (t.max(), 1)):
        side = None
        for k in range(max(1, int(PROBE_M[0] / resolution)), int(np.ceil(PROBE_M[1] / resolution)) + 1):
            side = _thickness(grid, centre + (end + sign * k) * along, across, reach)
            if side is not None:
                break
        if side is not None and side * resolution > JAMB_M:
            return False               # a thick side: furniture, or a wall running along the way through
    return True                        # thin sides (or unseen ones: don't merge on a guess)


def _gaps(cells):
    """Border cell positions grouped into separate gaps: runs of positions next to each other."""
    left, out = set(cells), []
    while left:
        queue = [left.pop()]
        for r, c in queue:                     # grows as neighbours are found
            near = [p for p in left if abs(p[0] - r) <= 1 and abs(p[1] - c) <= 1]
            left.difference_update(near)
            queue.extend(near)
        out.append(np.array(sorted(queue)))
    return sorted(out, key=lambda gap: tuple(gap[0]))


def _merge_furniture_splits(labels, grid, resolution):
    """Merge neighbouring regions that meet at a gap that isn't a doorway; labels renumbered 1..n."""
    borders = {}
    for dr, dc in ((0, 1), (1, 0)):
        a, b = labels[:labels.shape[0] - dr, :labels.shape[1] - dc], labels[dr:, dc:]
        touch = (a > 0) & (b > 0) & (a != b)
        for r, c, la, lb in zip(*np.nonzero(touch), a[touch], b[touch]):
            borders.setdefault((min(la, lb), max(la, lb)), []).append((r + dr / 2, c + dc / 2))
    parent = list(range(int(labels.max()) + 1))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    reach = int(np.ceil(.6 / resolution))
    for (la, lb), cells in sorted(borders.items()):
        # Two regions can meet at more than one gap (two doors into one room): judge each gap on its own, or
        # together they look like one gap wider than any doorway.
        for gap in _gaps(cells):
            across = _way_through(labels, la, lb, gap.mean(axis=0), reach)
            if root(la) != root(lb) and not _is_doorway(gap, grid, resolution, across):
                parent[max(root(la), root(lb))] = min(root(la), root(lb))
    roots = sorted({root(i) for i in range(1, len(parent))})
    lookup = np.zeros(len(parent), dtype=np.int32)
    for i in range(1, len(parent)):
        lookup[i] = roots.index(root(i)) + 1
    return lookup[labels], len(roots)


def _components(mask):
    remaining = mask.copy()
    for r, c in zip(*np.nonzero(mask)):
        if not remaining[r, c]:
            continue
        component, queue = [], deque([(int(r), int(c))])
        remaining[r, c] = False
        while queue:
            cell = queue.popleft()
            component.append(cell)
            for n in _neighbours(cell, mask.shape):
                if remaining[n]:
                    remaining[n] = False
                    queue.append(n)
        yield component


class RoomRegions:
    def __init__(self, core_radius_m=.65):
        if not np.isfinite(core_radius_m) or core_radius_m <= 0:
            raise ValueError("room core radius must be finite and positive")
        self.core_radius_m = core_radius_m
        self.map = self.labels = None
        self.names, self._next_id = {}, 1
        self._lock = threading.RLock()

    def _split(self, m):
        """Region labels (0 for none) on map m, and how many regions."""
        cores = list(_components(_safe_cells(m, self.core_radius_m)))
        seeds = np.zeros_like(m.grid, dtype=np.int32)
        for i, core in enumerate(cores, 1):
            r, c = zip(*core)
            seeds[r, c] = i
        # A small/narrow connected area with no eroded core still gets a region.
        for component in _components(m.grid == FREE):
            if not any(seeds[cell] for cell in component):
                cores.append([component[0]])
                seeds[component[0]] = len(cores)
        return _merge_furniture_splits(_flood(seeds, m.grid), m.grid, m.resolution)

    def update(self, occupancy):
        m = occupancy.snapshot()
        k = max(1, int(round(SPLIT_AT_M / m.resolution)))
        if k == 1:
            labels, count = self._split(m)
        else:
            # Split on a coarser view (see SPLIT_AT_M), then label the map's own free cells from it.
            coarse, count = self._split(_coarse(m, k))
            labels = np.zeros_like(m.grid, dtype=np.int32)
            fine = np.repeat(np.repeat(coarse, k, axis=0), k, axis=1)
            labels[:fine.shape[0], :fine.shape[1]] = fine
            labels[m.grid != FREE] = 0
            labels = _flood(labels, m.grid)
            for component in _components((m.grid == FREE) & (labels == 0)):   # pockets the coarse view missed
                count += 1
                labels[tuple(zip(*component))] = count
        with self._lock:
            # One old ID can be inherited by at most one new region. Strongest
            # overlaps win; new regions receive monotonic IDs.
            overlaps = []
            if self.map is not None:
                rows, cols = np.nonzero(labels)
                xy = np.array([m.cell_to_world(r, c) for r, c in zip(rows, cols)])
                if len(xy):
                    pr, pc, inside = self.map._cells(xy)
                    old = self.labels[pr[inside], pc[inside]]
                    pairs, counts = np.unique(np.stack([labels[rows[inside], cols[inside]], old], 1),
                                              axis=0, return_counts=True)
                    overlaps = sorted(((int(n), int(o), int(count)) for (n, o), count in zip(pairs, counts)
                                       if o > 0), key=lambda item: (-item[2], item[0], item[1]))
            assigned, used = {}, set()
            for new, old, _ in overlaps:
                if new not in assigned and old not in used:
                    assigned[new] = old
                    used.add(old)
            lookup = np.zeros(count + 1, dtype=np.int32)
            for i in range(1, count + 1):
                if i not in assigned:
                    assigned[i] = self._next_id
                    self._next_id += 1
                lookup[i] = assigned[i]
            self.map, self.labels = m, lookup[labels]
        return self

    def room_at(self, x, y):
        """A named region (or room_N), None for unknown, occupied or outside."""
        with self._lock:
            cell = None if self.map is None else self.map.world_to_cell(x, y)
            if cell is None or self.labels[cell] == 0:
                return None
            identity = int(self.labels[cell])
            return self.names.get(identity, f"room_{identity}")

    def region_near(self, x, y, radius_m=1.):
        """ID of the region most of the free floor within radius_m of (x, y) belongs to, or None. A seen object's
        position is on the object itself, which the map marks occupied; a bed against a wall has much more of its
        own room around it than of the room behind the wall. radius_m=0 is just the region at (x, y)."""
        with self._lock:
            cell = None if self.map is None else self.map.world_to_cell(x, y)
            if cell is None:
                return None
            reach = int(np.ceil(radius_m / self.map.resolution))
            r0, c0 = max(cell[0] - reach, 0), max(cell[1] - reach, 0)
            window = self.labels[r0:cell[0] + reach + 1, c0:cell[1] + reach + 1]
            rows, cols = np.nonzero(window)
            inside = (rows + r0 - cell[0]) ** 2 + (cols + c0 - cell[1]) ** 2 <= reach * reach
            ids = window[rows[inside], cols[inside]]
            return int(np.bincount(ids).argmax()) if len(ids) else None

    def areas(self):
        """{region ID: floor area in m²} of the observed regions."""
        with self._lock:
            if self.map is None:
                return {}
            ids, counts = np.unique(self.labels[self.labels > 0], return_counts=True)
            return {int(i): float(n) * self.map.resolution ** 2 for i, n in zip(ids, counts)}

    def region_point(self, region, clearance_m=None):
        """A spot (x, y) in the region near its middle, or None: the cell nearest its centre, among those at least
        clearance_m from anything in the way when there are any (somewhere the robot can stand)."""
        with self._lock:
            if self.map is None:
                return None
            inside = self.labels == region
            if clearance_m:
                roomy = inside & _safe_cells(self.map, clearance_m)
                inside = roomy if roomy.any() else inside
            cells = np.argwhere(inside)
            if not len(cells):
                return None
            centre = np.argwhere(self.labels == region).mean(axis=0)
            row, col = cells[np.argmin(((cells - centre) ** 2).sum(axis=1))]
            return self.map.cell_to_world(int(row), int(col))

    def name_room(self, x, y, name):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("room name must be nonempty")
        with self._lock:
            cell = None if self.map is None else self.map.world_to_cell(x, y)
            if cell is None or self.labels[cell] == 0:
                raise ValueError("no observed room at this position")
            self.names[int(self.labels[cell])] = name.strip()
