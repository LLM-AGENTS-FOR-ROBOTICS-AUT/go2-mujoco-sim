"""A* path planning on the room's occupancy grid (#14)."""

import heapq
import math

SQRT2 = math.sqrt(2)


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
