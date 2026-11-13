"""Known-floor planning with the same body poses for reachability and execution.

The Go2 can translate sideways. Translation bends do not rotate its body;
heading changes are explicit actions allowed only in observed turning space.
A direction-dependent lookahead models the executor's stopping corridor.
"""
import heapq
import math

import numpy as np

from .room_map import FREE, UNKNOWN, OCCUPIED, OccupancyGrid
from .footprint import swept_distances

MOVES = ((-1, 0), (0, -1), (0, 1), (1, 0),
         (-1, -1), (-1, 1), (1, -1), (1, 1))

# Empty-world collision-geometry calibration includes walking, turning and
# braking: measured maxima 0.396 x 0.202 m. Round up and retain a safety buffer.
BODY_HALF_LENGTH_M, BODY_HALF_WIDTH_M = .42, .23


def planning_map(occupancy, resolution=.1):
    """Conservative planning cells (10 cm by default) over the sensing map.

    A coarse cell is free only if every contributing sensing cell is free.
    Align to world coordinates even after map growth changes the array origin.
    """
    source = occupancy.snapshot()
    if not math.isfinite(resolution) or resolution<=0:
        raise ValueError('planning resolution must be finite and positive')
    factor = max(1, round(resolution/source.resolution))
    if factor == 1 or not math.isclose(factor*source.resolution,resolution):
        return source
    row, col = source._min_row//factor, source._min_col//factor
    dr, dc = source._min_row-factor*row, source._min_col-factor*col
    size = math.ceil(max(source.size+dr,source.size+dc)/factor)
    fine = np.full((size*factor,size*factor),UNKNOWN,dtype=np.int8)
    fine[dr:dr+source.size,dc:dc+source.size]=source.grid
    tiles=fine.reshape(size,factor,size,factor)
    out=OccupancyGrid(source.resolution*factor,size)
    out._min_row,out._min_col=row,col
    out.grid[:]=np.where((tiles==OCCUPIED).any(axis=(1,3)),OCCUPIED,
                        np.where((tiles==FREE).all(axis=(1,3)),FREE,UNKNOWN))
    return out


def _clear_stencil(grid, resolution, offsets):
    reach = max((max(abs(r), abs(c)) for r, c in offsets), default=0)
    padded = np.pad(grid != FREE, reach, constant_values=True)
    blocked = np.zeros_like(grid, dtype=bool)
    rows, cols = grid.shape
    for dr, dc in offsets:
        blocked |= padded[reach + dr:reach + dr + rows, reach + dc:reach + dc + cols]
    return ~blocked


def _offsets(resolution, reach, predicate):
    cells = math.ceil(reach / resolution)
    return [(r, c) for r in range(-cells, cells + 1) for c in range(-cells, cells + 1)
            if predicate(c * resolution, r * resolution)]


class ExplorationNavigation:
    """A private snapshot and a graph shared by frontier selection and movement.

    Two axis-aligned body headings plus the exact current heading cover ordinary
    house doorways without snapping a robot's pose inside a narrow passage.
    Unknown cells, map boundaries and occupied cells are all impassable.
    """
    def __init__(self, occupancy, pose, *, forbidden_steps=(), cancelled=None, planning_resolution_m=.1):
        self.source_map = occupancy.snapshot()
        self.map = planning_map(self.source_map,planning_resolution_m)
        self.map.grow = False
        self.pose = tuple(pose)
        self.cancelled = cancelled
        headings = [float(pose[2])]
        for yaw in (0., math.pi / 2):
            if all(abs(math.sin(yaw-h)) > math.sin(math.radians(12)) for h in headings):
                headings.append(yaw)
        self.headings = tuple(headings)
        self.start_cell = self.map.world_to_cell(*pose[:2])
        self.start = None if self.start_cell is None else (*self.start_cell, 0)
        self.forbidden = set(forbidden_steps)
        self._motion = {}
        self.fitting = []
        resolution = self.map.resolution
        # One cell covers quantisation, position offset and tracking allowance.
        self.half_length=BODY_HALF_LENGTH_M+resolution
        self.half_width=BODY_HALF_WIDTH_M+resolution
        self.turning_radius=.5+resolution
        for yaw in self.headings:
            co, sn = math.cos(yaw), math.sin(yaw)
            offsets = _offsets(resolution, math.hypot(self.half_length,self.half_width),
                lambda x,y: abs(co*x+sn*y)<=self.half_length+1e-9
                and abs(-sn*x+co*y)<=self.half_width+1e-9)
            self.fitting.append(_clear_stencil(self.map.grid, resolution, offsets))
        self.turning = _clear_stencil(self.map.grid, resolution,
            _offsets(resolution,self.turning_radius,
                     lambda x,y: x*x+y*y<=self.turning_radius**2+1e-9))

    @property
    def can_refine(self):
        return self.map.resolution>self.source_map.resolution+1e-9

    def refined(self):
        return ExplorationNavigation(self.source_map,self.pose,forbidden_steps=self.forbidden,
            cancelled=self.cancelled,planning_resolution_m=self.source_map.resolution)

    @property
    def start_fits(self):
        return self.start is not None and bool(self.fitting[0][self.start_cell])

    def _motion_safe(self, heading, direction):
        key = heading, direction
        if key not in self._motion:
            dr, dc = MOVES[direction]
            length = math.hypot(dr, dc)
            ux, uy = dc / length, dr / length
            yaw = self.headings[heading]
            bx = math.cos(yaw) * ux + math.sin(yaw) * uy
            by = -math.sin(yaw) * ux + math.cos(yaw) * uy
            # At 0.15 m/s the executor requires 0.267 m beyond the body.
            # Sweep the SAME rectangle as live sensing, with grid allowance.
            # A travel-aligned bounding box falsely includes diagonal corners.
            reach=math.ceil((math.hypot(self.half_length,self.half_width)+.30)/self.map.resolution)
            rr,cc=np.mgrid[-reach:reach+1,-reach:reach+1]
            x,y=cc.ravel()*self.map.resolution,rr.ravel()*self.map.resolution
            co,sn=math.cos(yaw),math.sin(yaw)
            body=np.stack([co*x+sn*y,-sn*x+co*y],axis=1)
            blocked=swept_distances(body,bx,by,self.half_length,self.half_width)<=.30+1e-9
            offsets=list(zip(rr.ravel()[blocked],cc.ravel()[blocked]))
            self._motion[key] = _clear_stencil(self.map.grid, self.map.resolution, offsets)
        return self._motion[key]

    def _successors(self, state):
        r, c, h = state
        if self.turning[r, c]:
            for other in range(len(self.headings)):
                if other != h and self.fitting[other][r, c]:
                    angle = abs(math.atan2(math.sin(self.headings[h] - self.headings[other]),
                                           math.cos(self.headings[h] - self.headings[other])))
                    yield (r, c, other), max(1., angle * .2 / self.map.resolution)
        rows, cols = self.map.grid.shape
        for d, (dr, dc) in enumerate(MOVES):
            if state == self.start and (dr, dc) in self.forbidden:
                continue
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols and self.fitting[h][nr, nc]):
                continue
            if dr and dc and (not self.fitting[h][r + dr, c] or not self.fitting[h][r, c + dc]):
                continue
            motion=self._motion_safe(h,d)
            if motion[r,c] and motion[nr,nc]:
                # Prefer the narrow side of the body across a passage. A
                # sideways route can fit geometrically but leave almost no
                # tracking tolerance at a doorway. It remains available for
                # bounded manoeuvres where rotating is impossible.
                yaw = self.headings[h]
                lateral = abs(-math.sin(yaw)*dc + math.cos(yaw)*dr) / math.hypot(dr, dc)
                yield (nr, nc, h), math.hypot(dr, dc) * (1. + 2. * lateral)

    def _search(self, goal=None):
        if not self.start_fits:
            return {}, None
        cost, parent = {self.start: 0.}, {}
        queue, counter, examined = [(0., 0, self.start)], 0, 0
        done, distances = set(), {}
        while queue:
            if self.cancelled is not None and examined % 256 == 0 and self.cancelled():
                return {}, None
            examined += 1
            _, _, state = heapq.heappop(queue)
            if state in done:
                continue
            done.add(state)
            cell = state[:2]
            distances[cell] = min(distances.get(cell, math.inf), cost[state])
            if cell == goal:
                path = [state]
                while path[-1] in parent:
                    path.append(parent[path[-1]])
                return distances, list(reversed(path))
            for nxt, step in self._successors(state):
                distance = cost[state] + step
                if distance < cost.get(nxt, math.inf):
                    cost[nxt], parent[nxt] = distance, state
                    counter += 1
                    heuristic = 0. if goal is None else math.dist(nxt[:2], goal)
                    heapq.heappush(queue, (distance + heuristic, counter, nxt))
        return distances, None

    def distances(self):
        return self._search()[0]

    def route(self, x, y):
        goal = self.map.world_to_cell(x, y)
        if goal is None:
            return None
        return self._search(goal)[1]

    def first_leg(self, path, max_distance_m=.5):
        """World (x,y,yaw) of the first rotation or straight translation run."""
        if len(path)<2:
            return (*self.map.cell_to_world(*path[0][:2]),self.headings[path[0][2]])
        first,nxt=path[:2]
        if first[:2]==nxt[:2]:
            return (*self.map.cell_to_world(*first[:2]),self.headings[nxt[2]])
        direction=nxt[0]-first[0],nxt[1]-first[1]
        end=nxt
        for a,b in zip(path[1:],path[2:]):
            if b[2]!=first[2] or (b[0]-a[0],b[1]-a[1])!=direction:
                break
            end=b
        wx,wy=self.map.cell_to_world(*end[:2])
        x,y=self.pose[:2]
        length=math.hypot(*direction)
        ux,uy=direction[1]/length,direction[0]/length
        distance=min(max_distance_m,(wx-x)*ux+(wy-y)*uy)
        # Follow the graph direction from the measured pose. Steering toward
        # a nearby cell centre can turn a forward step into a large diagonal;
        # the footprint's tracking allowance covers the cell-centre offset.
        return x+ux*distance,y+uy*distance,self.headings[first[2]]
