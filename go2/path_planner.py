
"""A simple A* path planner for a 2D occupancy grid."""

import heapq


def astar(grid, start, goal):
    """Return a path from start to goal, or None if no path exists.

    Grid values:
      -1 = unknown
       0 = free
       1 = occupied

    Coordinates are (row, column).
    """
    rows, cols = grid.shape

    def walkable(cell):
        row, col = cell
        return (
            0 <= row < rows
            and 0 <= col < cols
            and grid[row, col] == 0
        )

    if not walkable(start) or not walkable(goal):
        return None

    def heuristic(cell):
        return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])

    frontier = [(heuristic(start), 0, start)]
    came_from = {}
    cost_so_far = {start: 0}
    counter = 0

    while frontier:
        priority, cost, current = heapq.heappop(frontier)

        if priority != cost_so_far[current] + heuristic(current):
            continue

        if current == goal:
            path = [goal]
            while path[-1] != start:
                path.append(came_from[path[-1]])
            return list(reversed(path))

        row, col = current
        neighbours = [
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ]

        for neighbour in neighbours:
            if not walkable(neighbour):
                continue

            new_cost = cost + 1
            if new_cost < cost_so_far.get(neighbour, float("inf")):
                cost_so_far[neighbour] = new_cost
                came_from[neighbour] = current
                counter += 1
                priority = new_cost + heuristic(neighbour)
                heapq.heappush(
                    frontier, (priority, counter, neighbour)
                )

    return None
