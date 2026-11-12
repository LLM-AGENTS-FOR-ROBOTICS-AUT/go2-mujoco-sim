"""Room regions inferred from observed free space, not the simulator floor plan.

Eroding free space removes narrow doorways. Connected room cores then grow back
through free cells with a multi-source flood fill. This is a geometric heuristic:
an open-plan room or a wide arch may stay one region; furniture may split one.
Region IDs are associated by observed world-space overlap across map updates.
Semantic names are optional explicit labels, never inferred from geometry alone.
"""

from collections import deque
import threading

import numpy as np

from .exploration import _neighbours, _safe_cells
from .room_map import FREE


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

    def update(self, occupancy):
        m = occupancy.snapshot()
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
        labels = seeds.copy()
        queue = deque(zip(*np.nonzero(labels)))
        while queue:
            cell = queue.popleft()
            for n in _neighbours(cell, labels.shape):
                if m.grid[n] == FREE and labels[n] == 0:
                    labels[n] = labels[cell]
                    queue.append(n)
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
            lookup = np.zeros(len(cores) + 1, dtype=np.int32)
            for i in range(1, len(cores) + 1):
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

    def name_room(self, x, y, name):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("room name must be nonempty")
        with self._lock:
            cell = None if self.map is None else self.map.world_to_cell(x, y)
            if cell is None or self.labels[cell] == 0:
                raise ValueError("no observed room at this position")
            self.names[int(self.labels[cell])] = name.strip()
