"""A 2D occupancy grid of the room, built from the LiDAR (#14).

Cells are -1 unknown, 0 free, 1 occupied, in odometry coordinates (the frame the robot's x/y are reported in).
Each scan marks the cells between the robot and every return as free, then the returns that could block the
body as occupied. Marking free space every scan also clears things that have moved, such as a person walking
past.

Floor seen lower than the robot's (beyond a ledge, down the stairs) is marked occupied, so routes keep away from
drops; the executor's drop check (#22) still stops the robot at an edge either way. Spots the stall check bumped
into (glass, thin legs) are added when planning (blocked(extra=...)), since the LiDAR's rays see through them.

Fixtures can keep the default fixed grid; the executor opts into bounded growth,
preserving odometry coordinates when the array expands. Simulation also uses
repeated height evidence for low objects that would otherwise look like floor."""

import math
import threading

import numpy as np

UNKNOWN, FREE, OCCUPIED = -1, 0, 1
FLOOR_Z = .06              # a return this close to floor height is floor, so its cell is free
OBSTACLE_Z = (.06, .55)    # heights that can block the body (perception.OBSTACLE_MIN_Z/MAX_Z)
RAY_RANGE_M = 4.           # free space is only marked this far out; further returns are too sparse to trust
MAX_RAYS = 600             # rays used per scan (a random subset; the scans in between fill in the rest)


class OccupancyGrid:
    """Store what the LiDAR has seen in a 2D grid."""

    def __init__(self, resolution=0.1, size=200, *, grow=False, max_size=2048, detect_low=False):
        if resolution <= 0:
            raise ValueError("resolution must be positive")
        if size <= 0:
            raise ValueError("size must be positive")
        self.resolution = float(resolution)
        self.size = int(size)
        self.grow, self.max_size = grow, max(int(max_size), self.size)
        self._min_row = self._min_col = -(self.size // 2)
        self._lock = threading.RLock()
        self.grid = np.full((self.size, self.size), UNKNOWN, dtype=np.int8)
        self.detect_low = detect_low
        self._height_count = np.zeros_like(self.grid, dtype=np.uint16) if detect_low else None
        self._height_mean = np.zeros_like(self.grid, dtype=np.float32) if detect_low else None
        self._high_mask = np.zeros_like(self.grid, dtype=bool) if detect_low else None
        self._low_cells, self._uncertain_cells = set(), set()
        self._rng = np.random.default_rng(0)

    def snapshot(self):
        """A consistent private map, including its world origin after growth."""
        with self._lock:
            out = OccupancyGrid(self.resolution, self.size, grow=self.grow, max_size=self.max_size,
                                detect_low=self.detect_low)
            out._min_row, out._min_col = self._min_row, self._min_col
            out.grid = self.grid.copy()
            if self.detect_low:
                out._height_count = self._height_count.copy()
                out._height_mean = self._height_mean.copy()
                out._high_mask = self._high_mask.copy()
                out._low_cells, out._uncertain_cells = self._low_cells.copy(), self._uncertain_cells.copy()
            return out

    def ensure_bounds(self, xy):
        """Grow an opt-in map around points without moving their world coordinates.

        Bounded to max_size cells per side; an outlier cannot allocate an
        unbounded grid. Requests beyond that bound raise before changing it.
        """
        if not self.grow:
            return
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        xy = xy[np.isfinite(xy).all(axis=1)]
        if not len(xy):
            return
        with self._lock:
            cells = np.floor(xy / self.resolution).astype(np.int64)
            lo_c = min(self._min_col, int(cells[:, 0].min()) - 2)
            lo_r = min(self._min_row, int(cells[:, 1].min()) - 2)
            hi_c = max(self._min_col + self.size, int(cells[:, 0].max()) + 3)
            hi_r = max(self._min_row + self.size, int(cells[:, 1].max()) + 3)
            needed = max(hi_c - lo_c, hi_r - lo_r)
            if needed == self.size:
                return
            new_size = min(self.max_size, max(needed, self.size + 64))
            if needed > self.max_size:
                raise ValueError("room map extent exceeds its configured maximum")
            new = np.full((new_size, new_size), UNKNOWN, dtype=np.int8)
            row, col = self._min_row - lo_r, self._min_col - lo_c
            new[row:row + self.size, col:col + self.size] = self.grid
            if self.detect_low:
                count = np.zeros_like(new, dtype=np.uint16)
                mean = np.zeros_like(new, dtype=np.float32)
                high = np.zeros_like(new, dtype=bool)
                count[row:row + self.size, col:col + self.size] = self._height_count
                mean[row:row + self.size, col:col + self.size] = self._height_mean
                high[row:row + self.size, col:col + self.size] = self._high_mask
                self._height_count, self._height_mean = count, mean
                self._high_mask = high
            self.grid, self.size = new, new_size
            self._min_row, self._min_col = lo_r, lo_c

    # ---- coordinates ---------------------------------------------------------------------------
    def world_to_cell(self, x, y):
        """Grid (row, col) of a point in metres, or None outside the grid."""
        col = math.floor(x / self.resolution) - self._min_col
        row = math.floor(y / self.resolution) - self._min_row
        if not (0 <= row < self.size and 0 <= col < self.size):
            return None
        return row, col

    def cell_to_world(self, row, col):
        """Centre of a cell in metres."""
        x = (col + self._min_col + 0.5) * self.resolution
        y = (row + self._min_row + 0.5) * self.resolution
        return x, y

    def _cells(self, xy):
        """(rows, cols, inside) for an Nx2 array of points, all at once."""
        cols = np.floor(xy[:, 0] / self.resolution).astype(int) - self._min_col
        rows = np.floor(xy[:, 1] / self.resolution).astype(int) - self._min_row
        inside = (rows >= 0) & (rows < self.size) & (cols >= 0) & (cols < self.size)
        return rows, cols, inside

    # ---- marking -------------------------------------------------------------------------------
    def mark_obstacles(self, points, *, high=True):
        """Mark points (Nx2, metres, odometry frame) as occupied."""
        points = np.asarray(points, dtype=float)
        if points.size == 0:
            return
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points must have shape (N, 2)")
        points = points[np.isfinite(points).all(axis=1)]
        with self._lock:
            self.ensure_bounds(points)
            rows, cols, inside = self._cells(points)
            self.grid[rows[inside], cols[inside]] = OCCUPIED
            if self.detect_low and high:
                self._high_mask[rows[inside], cols[inside]] = True

    def mark_ray(self, start, end, endpoint_occupied=True):
        """Mark the cells from start to end as free, and the end cell as occupied (a single return) unless
        endpoint_occupied is False (then the whole ray is free)."""
        with self._lock:
            self.ensure_bounds([start, end])
            self._mark_ray(start, end, endpoint_occupied)

    def _mark_ray(self, start, end, endpoint_occupied):
        start_cell, end_cell = self.world_to_cell(*start), self.world_to_cell(*end)
        if start_cell is None or end_cell is None:
            return
        (row0, col0), (row1, col1) = start_cell, end_cell
        steps = max(abs(row1 - row0), abs(col1 - col0))
        if steps:
            rows = np.rint(np.linspace(row0, row1, steps + 1)).astype(int)
            cols = np.rint(np.linspace(col0, col1, steps + 1)).astype(int)
            self.grid[rows[:-1], cols[:-1]] = FREE
            if self.detect_low:
                self._high_mask[rows[:-1], cols[:-1]] = False
        self.grid[end_cell] = OCCUPIED if endpoint_occupied else FREE
        if self.detect_low:
            self._high_mask[end_cell] = endpoint_occupied

    def mark_scan(self, origin, points):
        """Add one LiDAR scan: origin is the robot's (x, y); points are Nx3 in the odometry frame with z the
        height above the floor. Cells along each ray are free; floor returns are free; returns at heights that
        can block the body are occupied."""
        with self._lock:
            self._mark_scan(origin, points)

    def _mark_scan(self, origin, points):
        points = np.asarray(points, dtype=float)
        if points.size == 0:
            return
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        points = points[np.isfinite(points).all(axis=1)]
        if not len(points):
            return
        self.ensure_bounds(np.vstack([np.asarray(origin).reshape(1, 2), points[:, :2]]))
        retained = np.nonzero(self.grid == OCCUPIED) if self.detect_low else None
        retained_high = self._high_mask[retained].copy() if self.detect_low else None
        z = points[:, 2]
        ox, oy = origin
        d = np.hypot(points[:, 0] - ox, points[:, 1] - oy)
        floor = (np.abs(z) < FLOOR_Z) & (d <= RAY_RANGE_M)
        blocking = (z > OBSTACLE_Z[0]) & (z < OBSTACLE_Z[1])
        below = z < -FLOOR_Z               # floor lower than the robot's: beyond a ledge or down the stairs
        rays = points[(floor | blocking) & (d <= RAY_RANGE_M) & (d > 1e-6)]
        if len(rays) > MAX_RAYS:
            rays = rays[self._rng.choice(len(rays), MAX_RAYS, replace=False)]
        if len(rays):
            length = np.hypot(rays[:, 0] - ox, rays[:, 1] - oy)
            steps = np.maximum(1, np.ceil(length / self.resolution).astype(int))
            n = int(steps.max())
            t = np.arange(n) / n                                             # 0 .. just short of the return
            frac = np.minimum(t[None, :] * n / steps[:, None], 1 - 1e-9)     # each ray stops before its end
            xs = ox + (rays[:, 0:1] - ox) * frac
            ys = oy + (rays[:, 1:2] - oy) * frac
            rows, cols, inside = self._cells(np.stack([xs.ravel(), ys.ravel()], 1))
            self.grid[rows[inside], cols[inside]] = FREE
            if self.detect_low:
                self._high_mask[rows[inside], cols[inside]] = False
        if retained is not None:
            # A 3D ray can pass above a low obstacle. Its 2D projection is not
            # evidence that the obstacle disappeared. Keep obstacle hits
            # until repeated direct floor returns clear that particular cell.
            self.grid[retained] = OCCUPIED
            self._high_mask[retained] = retained_high
        floor_pts = points[floor]
        if len(floor_pts):
            rows, cols, inside = self._cells(floor_pts[:, :2])
            keep = self.grid[rows[inside], cols[inside]] != OCCUPIED
            self.grid[rows[inside][keep], cols[inside][keep]] = FREE
        if self.detect_low:
            # A noisy book-top return can cross 6 cm; that alone is not proof
            # of a wall/high obstacle. Preserve reliable structural returns for
            # recovery so it cannot clear walls while leaving low surfaces in
            # the ordinary obstacle map and live clearance checks.
            self._mark_low_returns(points[floor])
            structural = points[((z > .12) & (z < OBSTACLE_Z[1])) | below]
            if len(structural):
                rr, cc, keep = self._cells(structural[:, :2])
                self._high_mask[rr[keep], cc[keep]] = True
        # Current obstacle evidence wins over floor returns in the same cell,
        # including objects between the low-height and structural thresholds.
        self.mark_obstacles(points[blocking | below][:, :2], high=False)

    def _mark_low_returns(self, points):
        """Repeated low-height evidence, independent of free-space ray carving.

        The L1 simulation has 1 cm range noise. A conservative 1.2 cm standard
        deviation and a three-sigma bound distinguish a persistent raised
        surface (books/laptops) from isolated noisy floor hits. Uncertain raised
        cells stay unknown; confirmed cells are obstacles to both route planning
        and the executor's live clearance checks. This is sim-first calibration.
        """
        if not len(points):
            return
        rows, cols, inside = self._cells(points[:, :2])
        indices, inverse, counts = np.unique(rows[inside] * self.size + cols[inside],
                                             return_inverse=True, return_counts=True)
        heights = np.bincount(inverse, weights=points[inside, 2])
        old_n = self._height_count.flat[indices].astype(float)
        means = (self._height_mean.flat[indices] * old_n + heights) / (old_n + counts)
        self._height_mean.flat[indices] = means
        self._height_count.flat[indices] = np.minimum(64, old_n + counts).astype(np.uint16)
        n = self._height_count.flat[indices].astype(float)
        confidence = self._height_mean.flat[indices] - .036 / np.sqrt(np.maximum(n, 1))
        floor_confident = ((n >= 3) & (self._height_mean.flat[indices] +
                           .036 / np.sqrt(np.maximum(n, 1)) < .015))
        cleared = indices[floor_confident]
        self._high_mask.flat[cleared] = False
        self.grid.flat[cleared] = FREE
        confirmed = (n >= 3) & (confidence > .015)
        uncertain = (n >= 2) & (self._height_mean.flat[indices] > .02) & ~confirmed
        for index, low, pending in zip(indices, confirmed, uncertain):
            row, col = divmod(int(index), self.size)
            cell = row + self._min_row, col + self._min_col
            self._low_cells.discard(cell)
            self._uncertain_cells.discard(cell)
            if low:
                self._low_cells.add(cell)
            elif pending:
                self._uncertain_cells.add(cell)
        if self._uncertain_cells:
            rc = np.asarray(sorted(self._uncertain_cells)) - (self._min_row, self._min_col)
            rows, cols = rc[:, 0], rc[:, 1]
            keep = self.grid[rows, cols] != OCCUPIED
            self.grid[rows[keep], cols[keep]] = UNKNOWN
        if self._low_cells:
            rc = np.asarray(sorted(self._low_cells)) - (self._min_row, self._min_col)
            self.grid[rc[:, 0], rc[:, 1]] = OCCUPIED

    def low_obstacles(self):
        """Derived low obstacles as odometry x/y and a safety-obstacle height.

        The 0.1 m z marks them for clearance checks; it is not a raw sensor hit
        and must not be used as camera depth or object-detection evidence.
        """
        if not self.detect_low:
            return np.empty((0, 3))
        with self._lock:
            if not self._low_cells:
                return np.empty((0, 3))
            rc = np.asarray(sorted(self._low_cells))
            local = rc - (self._min_row, self._min_col)
            rc = rc[~self._high_mask[local[:, 0], local[:, 1]]]
            return np.column_stack([(rc[:, 1] + .5) * self.resolution,
                                    (rc[:, 0] + .5) * self.resolution,
                                    np.full(len(rc), .1)])

    # ---- planning helpers ----------------------------------------------------------------------
    def blocked(self, radius_m, extra=None):
        """Boolean grid of cells the body's centre can't be in: occupied cells, plus any extra points (Nx2,
        metres; for things the LiDAR can't see, which its free-space rays would otherwise clear), grown by
        radius_m."""
        occupied = self.grid == OCCUPIED
        if extra is not None and len(extra):
            occupied = occupied.copy()
            rows, cols, inside = self._cells(np.asarray(extra, dtype=float)[:, :2])
            occupied[rows[inside], cols[inside]] = True
        reach = int(math.ceil(radius_m / self.resolution))
        if reach <= 0 or not occupied.any():
            return occupied
        out = occupied.copy()
        rows, cols = np.nonzero(occupied)
        for dr in range(-reach, reach + 1):
            for dc in range(-reach, reach + 1):
                if dr * dr + dc * dc > (radius_m / self.resolution) ** 2 + 1e-9:
                    continue
                r, c = rows + dr, cols + dc
                ok = (r >= 0) & (r < self.size) & (c >= 0) & (c < self.size)
                out[r[ok], c[ok]] = True
        return out

    def line_clear(self, blocked, a, b):
        """True if the straight line between cells a and b crosses no blocked cell."""
        steps = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
        if steps == 0:
            return not blocked[a]
        rows = np.rint(np.linspace(a[0], b[0], steps * 2 + 1)).astype(int)
        cols = np.rint(np.linspace(a[1], b[1], steps * 2 + 1)).astype(int)
        return not blocked[rows, cols].any()
