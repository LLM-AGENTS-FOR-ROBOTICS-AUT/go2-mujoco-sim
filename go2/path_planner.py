"""A* path planning on the room's occupancy grid (#14)."""

import heapq
import math

SQRT2 = math.sqrt(2)


def turning_astar(grid, start, goal, blocked, turning, cancelled=None, forbidden_first_steps=(), unknown_cost=None):
    """Route whose direction changes have room to brake and turn.

    Straight sections can pass through a narrow doorway. Bends use the wider
    turning mask, avoiding waypoints that live clearance checks cannot reach.
    The returned cell path retains direction changes; simplify only collinear
    runs so an unsafe new corner is not introduced afterward.
    """
    if unknown_cost is not None and (not math.isfinite(unknown_cost) or unknown_cost < 1.):
        raise ValueError("unknown cost must be finite and at least one")
    rows, cols = grid.shape
    def free(cell):
        r, c = cell
        return (0 <= r < rows and 0 <= c < cols and not blocked[cell]
                and (grid[cell] == 0 or (grid[cell] == -1 and unknown_cost is not None)))
    if not free(start) or not free(goal):
        return None
    moves = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    initial = (*start, -1)
    queue, cost, parent = [(math.dist(start, goal), 0, initial)], {initial: 0.}, {}
    counter, examined, done = 0, 0, set()
    while queue:
        if cancelled is not None and examined % 256 == 0 and cancelled():
            return None
        examined += 1
        _, _, state = heapq.heappop(queue)
        if state in done:
            continue
        done.add(state)
        r, c, direction = state
        if (r, c) == goal:
            path = [(r, c)]
            while state in parent:
                state = parent[state]
                path.append(state[:2])
            return list(reversed(path))
        for d, (dr, dc) in enumerate(moves):
            if (r, c) == start and (dr, dc) in forbidden_first_steps:
                continue
            if direction != -1 and d != direction and not turning[r, c]:
                continue
            cell = r + dr, c + dc
            if not free(cell) or (dr and dc and (not free((r + dr, c)) or not free((r, c + dc)))):
                continue
            # In open turning space the arrival direction is irrelevant. Merge
            # those states instead of expanding the same room eight times.
            nxt = (*cell, -1 if turning[cell] else d)
            distance = cost[state] + (SQRT2 if dr and dc else 1.) * (unknown_cost if grid[cell] == -1 else 1.)
            if distance < cost.get(nxt, math.inf):
                cost[nxt], parent[nxt] = distance, state
                counter += 1
                heapq.heappush(queue, (distance + math.dist(cell, goal), counter, nxt))
    return None


def keep_turns(path):
    """Collapse collinear cell runs while retaining every planned bend."""
    if not path or len(path) < 3:
        return list(path or [])
    out = [path[0]]
    previous = (path[1][0] - path[0][0], path[1][1] - path[0][1])
    for a, b in zip(path[1:-1], path[2:]):
        direction = (b[0] - a[0], b[1] - a[1])
        if direction != previous:
            out.append(a)
        previous = direction
    return out + [path[-1]]


def astar(grid, start, goal, blocked=None, unknown_cost=None, diagonal=False):
    """Shortest path from start to goal as a list of (row, col) cells, or None if there is none.

    Grid values: -1 unknown, 0 free, 1 occupied. Only free cells are walkable, unless unknown_cost is given:
    then unknown cells are walkable at that cost per step (higher than 1, so known free space is preferred).
    blocked: an optional boolean grid of extra cells to avoid (for example obstacles grown by the body's size).
    diagonal: also step diagonally (cost sqrt 2), which gives shorter, smoother routes.
    """
    rows, cols = grid.shape

    def step_cost(cell):
        r, c = cell
        if not (0 <= r < rows and 0 <= c < cols):
            return None
        if blocked is not None and blocked[r, c]:
            return None
        value = grid[r, c]
        if value == 0:
            return 1.
        if value == -1 and unknown_cost is not None:
            return float(unknown_cost)
        return None

    if step_cost(start) is None or step_cost(goal) is None:
        return None

    if diagonal:
        moves = [(-1, 0, 1.), (1, 0, 1.), (0, -1, 1.), (0, 1, 1.),
                 (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2)]

        def heuristic(cell):          # octile distance: never more than the true cost
            dr, dc = abs(cell[0] - goal[0]), abs(cell[1] - goal[1])
            return max(dr, dc) + (SQRT2 - 1) * min(dr, dc)
    else:
        moves = [(-1, 0, 1.), (1, 0, 1.), (0, -1, 1.), (0, 1, 1.)]

        def heuristic(cell):          # Manhattan distance
            return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

    counter = 0
    frontier = [(heuristic(start), counter, start)]
    came_from = {}
    cost_so_far = {start: 0.}
    done = set()
    while frontier:
        _, _, current = heapq.heappop(frontier)
        if current in done:
            continue                   # an older, costlier entry for a cell already expanded
        done.add(current)
        if current == goal:
            path = [goal]
            while path[-1] != start:
                path.append(came_from[path[-1]])
            return list(reversed(path))
        cost = cost_so_far[current]   # the real cost so far (the counter only breaks ties in the heap)
        for dr, dc, length in moves:
            neighbour = (current[0] + dr, current[1] + dc)
            each = step_cost(neighbour)
            if each is None:
                continue
            if dr and dc and (step_cost((current[0] + dr, current[1])) is None
                              or step_cost((current[0], current[1] + dc)) is None):
                continue               # don't cut a corner between two blocked cells
            new_cost = cost + each * length
            if new_cost < cost_so_far.get(neighbour, math.inf):
                cost_so_far[neighbour] = new_cost
                came_from[neighbour] = current
                counter += 1
                heapq.heappush(frontier, (new_cost + heuristic(neighbour), counter, neighbour))
    return None


def simplify(path, line_clear):
    """Fewer waypoints: from each kept cell, jump to the furthest later cell still in a straight, clear line.
    line_clear(a, b) says whether the straight line between two cells is clear."""
    if not path:
        return []
    out, i = [path[0]], 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not line_clear(path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out
