"""Frontier selection uses observations only; no simulator ground-truth rooms."""

import math
import unittest
from dataclasses import replace

import numpy as np

from go2.exploration import ExplorationMap, next_viewpoint, recovery_route, _safe_cells, _oriented_safe
from go2.path_planner import astar
from go2.room_map import FREE, OCCUPIED, UNKNOWN, OccupancyGrid


class ExplorationTests(unittest.TestCase):
    def setUp(self):
        self.m = OccupancyGrid(resolution=.1, size=40)
        self.m.grid[:] = OCCUPIED
        self.m.grid[5:35, 5:25] = FREE
        self.m.grid[5:35, 25:35] = UNKNOWN
        self.pose = (*self.m.cell_to_world(20, 15), 0.)

    def context(self, **kwargs):
        values = dict(occupancy=self.m, pose=self.pose,
                      clearance_m=.2, visited_radius_m=.35)
        values.update(kwargs)
        return ExplorationMap(**values)

    def test_viewpoint_faces_unknown_space_from_known_free_space(self):
        viewpoint = next_viewpoint(self.context(), [])
        self.assertIsNotNone(viewpoint)
        x, y, yaw = viewpoint
        self.assertEqual(self.m.grid[self.m.world_to_cell(x, y)], FREE)
        self.assertGreater(x, self.pose[0])
        self.assertAlmostEqual(y, self.pose[1])
        self.assertAlmostEqual(yaw, 0.)

    def test_complete_route_has_body_clearance(self):
        x, y, _ = next_viewpoint(self.context(), [])
        route = astar(self.m.grid, self.m.world_to_cell(*self.pose[:2]),
                      self.m.world_to_cell(x, y), blocked=self.m.blocked(.2))
        self.assertIsNotNone(route)
        for r, c in route:
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    if dr * dr + dc * dc <= 4:
                        self.assertEqual(self.m.grid[r + dr, c + dc], FREE)

    def test_yaw_is_absolute_odometry_heading_not_relative_to_robot(self):
        viewpoint = next_viewpoint(self.context(), [])
        turned = next_viewpoint(self.context(pose=(*self.pose[:2], math.pi / 2)), [])
        self.assertEqual(turned, viewpoint)

    def test_nearer_frontier_is_preferred(self):
        self.m.grid[5:35, :5] = UNKNOWN
        right_start = (*self.m.cell_to_world(20, 18), 0.)
        x, _, yaw = next_viewpoint(self.context(pose=right_start), [])
        self.assertGreater(x, right_start[0])
        self.assertAlmostEqual(yaw, 0.)

    def test_wall_separating_frontier_prevents_selection(self):
        self.m.grid[:, 19] = OCCUPIED
        self.assertIsNone(next_viewpoint(self.context(), []))

    def test_remembered_bumps_are_avoided(self):
        bumps = [self.m.cell_to_world(r, 19) for r in range(5, 35)]
        self.assertIsNone(next_viewpoint(self.context(extra_obstacles=bumps), []))

    def test_visited_position_is_not_selected_again_even_with_other_yaw(self):
        first = next_viewpoint(self.context(), [])
        visited = [(first[0], first[1], math.pi)]
        second = next_viewpoint(self.context(), visited)
        self.assertIsNotNone(second)
        self.assertGreater(math.dist(first[:2], second[:2]), .35)

    def test_exhausted_viewpoints_return_none(self):
        self.assertIsNone(next_viewpoint(self.context(visited_radius_m=10), [self.pose]))

    def test_no_unknown_cells_means_no_frontier(self):
        self.m.grid[self.m.grid == UNKNOWN] = OCCUPIED
        self.assertIsNone(next_viewpoint(self.context(), []))

    def test_observe_through_a_slit_from_farther_back_without_entering_it(self):
        m = OccupancyGrid(resolution=.05, size=100)
        m.grid[:] = OCCUPIED
        m.grid[10:65, 10:60] = FREE
        m.grid[32:38, 60:72] = FREE
        m.grid[32:38, 72:85] = UNKNOWN
        pose = (*m.cell_to_world(35, 35), 0.)
        context = ExplorationMap(m, pose, clearance_m=.42, heading_aware=True,
                                 turn_clearance_m=.55, planning_resolution_m=.05)
        before = m.grid.copy()
        goal = next_viewpoint(context, [])
        self.assertIsNotNone(goal)
        cell = m.world_to_cell(*goal[:2])
        self.assertTrue(_safe_cells(m, .55)[cell])
        self.assertLess(cell[1], 60, 'stay in the room, outside the narrow slit')
        self.assertGreater((71 - cell[1]) * m.resolution, .85)
        self.assertTrue(any(m.line_clear(m.grid != FREE, cell, (r, 71))
                            for r in range(32, 38)))
        np.testing.assert_array_equal(m.grid, before)

    def test_unknown_start_does_not_allow_exploration_through_unknown(self):
        self.m.grid[:] = UNKNOWN
        self.assertIsNone(next_viewpoint(self.context(), []))

    def test_start_outside_map_returns_none(self):
        self.assertIsNone(next_viewpoint(self.context(pose=(100., 100., 0.)), []))

    def test_narrow_free_space_does_not_fit_body(self):
        self.m.grid[5:19, :] = OCCUPIED
        self.m.grid[22:35, :] = OCCUPIED
        self.assertIsNone(next_viewpoint(self.context(), []))

    def test_map_boundary_is_not_a_frontier_or_safe_body_space(self):
        self.m.grid[:] = FREE
        self.assertIsNone(next_viewpoint(self.context(), []))
        self.m.grid[:, 25:] = UNKNOWN
        boundary_pose = (*self.m.cell_to_world(0, 15), 0.)
        self.assertIsNone(next_viewpoint(self.context(pose=boundary_pose), []))

    def test_selection_is_deterministic_and_does_not_mutate_inputs(self):
        before = self.m.grid.copy()
        visited = []
        first = next_viewpoint(self.context(), visited)
        self.assertEqual(first, next_viewpoint(self.context(), visited))
        np.testing.assert_array_equal(self.m.grid, before)
        self.assertEqual(visited, [])

    def test_bad_poses_and_distances_are_rejected(self):
        for pose in ((0., 0.), (math.nan, 0., 0.), (math.inf, 0., 0.)):
            with self.subTest(pose=pose), self.assertRaises(ValueError):
                next_viewpoint(self.context(pose=pose), [])
        for field in ("clearance_m", "visited_radius_m"):
            for value in (-1., math.inf, math.nan):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    next_viewpoint(self.context(**{field: value}), [])
        with self.assertRaises(ValueError):
            next_viewpoint(self.context(), [(0., 0.)])

    def test_fixed_heading_recovery_exits_a_tight_channel_without_turning(self):
        m = OccupancyGrid(resolution=.1, size=60)
        m.grid[:] = OCCUPIED
        m.grid[20:25, 5:35] = FREE
        m.grid[10:40, 30:50] = FREE
        pose = (*m.cell_to_world(22, 27), 0.)
        self.assertFalse(_safe_cells(m, .42)[m.world_to_cell(*pose[:2])])
        path = recovery_route(m, pose)
        self.assertIsNotNone(path)
        self.assertTrue(all(_oriented_safe(m, 0.)[cell] for cell in path))
        self.assertTrue(_safe_cells(m, .42)[path[-1]])
        viewpoint = next_viewpoint(ExplorationMap(m, pose, clearance_m=.42), [])
        self.assertAlmostEqual(viewpoint[2], 0.)

    def test_recovery_does_not_assume_it_can_rotate_in_a_narrow_channel(self):
        m = OccupancyGrid(resolution=.1, size=60)
        m.grid[:] = OCCUPIED
        m.grid[20:25, 5:35] = FREE
        m.grid[10:40, 30:50] = FREE
        pose = (*m.cell_to_world(22, 27), math.pi / 2)
        self.assertIsNone(recovery_route(m, pose))

    def test_recovery_preserves_measured_offset_beside_a_wall(self):
        m = OccupancyGrid(resolution=.05, size=160)
        m.grid[:] = FREE
        for r in range(m.size):
            if m.cell_to_world(r, 0)[1] >= .72:
                m.grid[r] = OCCUPIED
        pose = (.8113, .4057, .1297)
        self.assertFalse(_oriented_safe(m, pose[2])[m.world_to_cell(*pose[:2])])
        path = recovery_route(m, pose, clearance=.55,
                              forbidden_steps=((1, 0), (1, -1), (1, 1)))
        self.assertIsNotNone(path)
        self.assertLess(path[-1][0], path[0][0], 'recover away from the wall')
        context = ExplorationMap(m, pose, clearance_m=.42, heading_aware=True)
        goal = next_viewpoint(context, [])
        self.assertIsNotNone(goal)
        self.assertAlmostEqual((goal[0] - pose[0]) / m.resolution,
                               round((goal[0] - pose[0]) / m.resolution))
        self.assertAlmostEqual((goal[1] - pose[1]) / m.resolution,
                               round((goal[1] - pose[1]) / m.resolution))

    def test_recover_when_body_fits_but_normal_routes_cannot_reach_turning_space(self):
        from go2.exploration_navigation import ExplorationNavigation
        m = OccupancyGrid(resolution=.05, size=160)
        m.grid[:] = OCCUPIED
        r, c = np.indices(m.grid.shape)
        x = (c + m._min_col + .5) * m.resolution
        y = (r + m._min_row + .5) * m.resolution
        m.grid[(x > -.6) & (x < .9) & (abs(y) < .3)] = FREE
        m.grid[(x > -2.8) & (x < -.6) & (abs(y) < 1.2)] = FREE
        m.grid[(x > -.6) & (x < -.3)] = OCCUPIED
        m.grid[(x > -.6) & (x < -.3) & (abs(y) < .25)] = FREE
        m.grid[(x > -3.) & (x < -2.8) & (abs(y) < 1.)] = UNKNOWN
        pose = (.3, 0., 0.)
        nav = ExplorationNavigation(m, pose, planning_resolution_m=.05)
        self.assertTrue(nav.start_fits)
        self.assertFalse(any(nav.turning[cell] for cell in nav.distances()))
        before = m.grid.copy()
        context = ExplorationMap(m, pose, clearance_m=.42,
                                 heading_aware=True, planning_resolution_m=.05)
        self.assertIsNone(next_viewpoint(replace(context, allow_recovery=False), []))
        goal = next_viewpoint(context, [])
        self.assertIsNotNone(goal)
        self.assertLess(goal[0], -.6)
        self.assertEqual(goal[2], pose[2])
        np.testing.assert_array_equal(m.grid, before)

    def test_recovery_does_not_repeat_a_failed_destination(self):
        m = OccupancyGrid(resolution=.1, size=60)
        m.grid[:] = OCCUPIED
        m.grid[20:25, 5:35] = FREE
        m.grid[10:40, 30:50] = FREE
        context = ExplorationMap(m, (*m.cell_to_world(22, 27), 0.), clearance_m=.42)
        first = next_viewpoint(context, [])
        second = next_viewpoint(context, [first])
        self.assertTrue(second is None or math.dist(first[:2], second[:2]) > .2)

    def test_turning_viewpoint_has_extra_clearance(self):
        viewpoint = next_viewpoint(self.context(turn_clearance_m=.5), [])
        self.assertIsNotNone(viewpoint)
        self.assertTrue(_safe_cells(self.m, .5)[self.m.world_to_cell(*viewpoint[:2])])
        with self.assertRaises(ValueError):
            next_viewpoint(self.context(turn_clearance_m=.1), [])

    def test_recovery_from_newly_observed_low_surface_only_reduces_overlap(self):
        m = OccupancyGrid(resolution=.05, size=80, detect_low=True)
        m.grid[:] = FREE
        m.mark_scan((.025, .025), [[.225, .025, .04]] * 8)
        before = m.grid.copy()
        pose = (.025, .025, 0.)
        path = recovery_route(m, pose, forbidden_steps={(0, 1)})
        self.assertIsNotNone(path)
        self.assertLess(m.cell_to_world(*path[-1])[0], pose[0])
        np.testing.assert_array_equal(m.grid, before)

    def test_low_surface_recovery_never_clears_a_high_obstacle(self):
        m = OccupancyGrid(resolution=.05, size=80, detect_low=True)
        m.grid[:] = FREE
        m.mark_scan((.025, .025), [[.225, .025, .04]] * 8)
        m.mark_obstacles([[.225, .025]])
        self.assertIsNone(recovery_route(m, (.025, .025, 0.)))


if __name__ == "__main__":
    unittest.main()
