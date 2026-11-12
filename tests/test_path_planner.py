
import unittest
import numpy as np

from go2.path_planner import astar, simplify, turning_astar, keep_turns


class TestAStar(unittest.TestCase):

    def test_turning_route_can_cross_unknown_but_prefers_observed_corridor(self):
        grid = np.zeros((7, 9), dtype=int)
        grid[3, 2:7] = -1
        blocked = grid == 1
        turning = np.ones_like(grid, dtype=bool)
        path = turning_astar(grid, (3, 1), (3, 7), blocked, turning, unknown_cost=5.)
        self.assertTrue(all(grid[cell] == 0 for cell in path))
        grid[:] = -1
        self.assertIsNone(turning_astar(grid, (3, 1), (3, 7), blocked, turning))
        self.assertIsNotNone(turning_astar(grid, (3, 1), (3, 7), blocked, turning, unknown_cost=5.))

    def test_turning_route_uses_wide_corners_and_passes_straight_doorways(self):
        grid = np.ones((9, 9), dtype=int)
        grid[1:8, 1:4] = 0
        grid[1:8, 5:8] = 0
        grid[4, 4] = 0
        blocked, turning = grid != 0, np.zeros_like(grid, dtype=bool)
        turning[2, 2] = turning[4, 2] = turning[4, 6] = turning[6, 6] = True
        path = turning_astar(grid, (2, 2), (6, 6), blocked, turning)
        self.assertIsNotNone(path)
        corners = keep_turns(path)
        self.assertTrue(all(turning[cell] for cell in corners[1:-1]))
        self.assertIn((4, 4), path)

    def test_turning_route_refuses_a_bend_without_turning_clearance(self):
        grid = np.ones((5, 5), dtype=int)
        grid[1, 1:4] = grid[1:4, 3] = 0
        self.assertIsNone(turning_astar(grid, (1, 1), (3, 3), grid != 0, np.zeros_like(grid, dtype=bool)))

    def test_turning_route_can_be_cancelled_during_planning(self):
        grid = np.zeros((20, 20), dtype=int)
        self.assertIsNone(turning_astar(grid, (0, 0), (19, 19), grid != 0,
                                      np.ones_like(grid, dtype=bool), cancelled=lambda: True))

    def test_turning_route_escapes_a_live_blocked_initial_direction(self):
        grid = np.zeros((5, 6), dtype=int)
        path = turning_astar(grid, (2, 1), (2, 4), grid != 0, np.ones_like(grid, dtype=bool),
                             forbidden_first_steps={(0, 1)})
        self.assertIsNotNone(path)
        self.assertNotEqual(path[1], (2, 2))
        self.assertEqual(path[-1], (2, 4))

    def test_turning_route_rechecks_cancellation_after_search_has_started(self):
        grid = np.zeros((400, 400), dtype=int)
        checks = []
        def cancelled():
            checks.append(1)
            return len(checks) >= 2
        self.assertIsNone(turning_astar(grid, (0, 0), (399, 399), grid != 0,
                                      np.ones_like(grid, dtype=bool), cancelled=cancelled))
        self.assertEqual(len(checks), 2)

    def test_finds_path_on_clear_grid(self):
        grid = np.zeros((5, 5), dtype=int)

        path = astar(grid, (0, 0), (4, 4))

        self.assertIsNotNone(path)
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (4, 4))

    def test_routes_around_obstacle(self):
        grid = np.zeros((5, 5), dtype=int)
        grid[1, :] = 1
        grid[1, 2] = 0  # Gap in the obstacle wall

        path = astar(grid, (0, 0), (4, 4))

        self.assertIsNotNone(path)
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (4, 4))

        # Every step should be adjacent, and no step should be occupied.
        for row, col in path:
            self.assertEqual(grid[row, col], 0)

        for current, following in zip(path, path[1:]):
            distance = (
                abs(current[0] - following[0])
                + abs(current[1] - following[1])
            )
            self.assertEqual(distance, 1)

    def test_returns_none_when_goal_is_blocked(self):
        grid = np.zeros((5, 5), dtype=int)
        grid[4, 4] = 1

        self.assertIsNone(astar(grid, (0, 0), (4, 4)))

    def test_returns_none_when_no_route_exists(self):
        grid = np.zeros((5, 5), dtype=int)
        grid[2, :] = 1

        self.assertIsNone(astar(grid, (0, 0), (4, 4)))

    def test_does_not_travel_through_unknown_cells(self):      # from Prionti's handover branch
        grid = np.zeros((5, 5), dtype=int)
        grid[2, :] = -1
        self.assertIsNone(astar(grid, (0, 0), (4, 4)))

    def test_unknown_cells_can_be_allowed_at_a_cost(self):
        grid = np.zeros((5, 5), dtype=int)
        grid[2, :] = -1
        self.assertIsNotNone(astar(grid, (0, 0), (4, 4), unknown_cost=3))

    def test_routes_are_the_shortest(self):
        # The cost used to be the heap's tie-break counter, so some routes came out longer than the shortest.
        rng = np.random.default_rng(7)
        for _ in range(150):
            grid = (rng.random((20, 20)) < .3).astype(int)
            grid[0, 0] = grid[19, 19] = 0
            shortest = bfs_length(grid, (0, 0), (19, 19))
            path = astar(grid, (0, 0), (19, 19))
            self.assertEqual(None if path is None else len(path) - 1, shortest)

    def test_blocked_cells_are_avoided(self):
        grid = np.zeros((5, 5), dtype=int)
        blocked = np.zeros((5, 5), dtype=bool)
        blocked[1, :4] = True
        path = astar(grid, (0, 0), (4, 0), blocked=blocked)
        self.assertTrue(all(not blocked[c] for c in path))

    def test_diagonal_routes_dont_cut_corners(self):
        grid = np.zeros((3, 3), dtype=int)
        grid[0, 1] = grid[1, 0] = 1
        self.assertIsNone(astar(grid, (0, 0), (1, 1), diagonal=True))

    def test_simplify_keeps_only_the_corners(self):
        path = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]
        clear = lambda a, b: a[0] == b[0] or a[1] == b[1]      # only straight lines along rows or columns
        self.assertEqual(simplify(path, clear), [(0, 0), (0, 2), (2, 2)])


def bfs_length(grid, start, goal):
    seen, frontier, steps = {start}, [start], 0
    while frontier:
        if goal in frontier:
            return steps
        nxt = []
        for r, c in frontier:
            for n in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if 0 <= n[0] < grid.shape[0] and 0 <= n[1] < grid.shape[1] and grid[n] == 0 and n not in seen:
                    seen.add(n)
                    nxt.append(n)
        frontier, steps = nxt, steps + 1
    return None


if __name__ == "__main__":
    unittest.main()
