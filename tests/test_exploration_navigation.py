import math
import unittest

import numpy as np

from go2.exploration_navigation import ExplorationNavigation, planning_map
from go2.room_map import OccupancyGrid, FREE, OCCUPIED


class HeadingNavigationTests(unittest.TestCase):
    def make_map(self, rectangles):
        m = OccupancyGrid(resolution=.1, size=120)
        m.grid[:] = OCCUPIED
        r,c = np.indices(m.grid.shape)
        x=(c+m._min_col+.5)*m.resolution
        y=(r+m._min_row+.5)*m.resolution
        for x0,y0,x1,y1 in rectangles:
            m.grid[(x>x0)&(x<x1)&(y>y0)&(y<y1)] = FREE
        return m

    def test_narrow_doorways_use_explicit_body_rotations_in_open_space(self):
        m=self.make_map([(-5,-1.5,-2,1.5),(-2,-.4,.8,.4),(-.8,-.8,.8,.8),
                         (-.4,.8,.4,2),(-1.5,2,1.5,5)])
        nav=ExplorationNavigation(m,(-3.,0.,0.))
        path=nav.route(0.,3.5)
        self.assertTrue(path)
        rotations=[(a,b) for a,b in zip(path,path[1:]) if a[2]!=b[2]]
        self.assertTrue(rotations, 'vertical doorway requires a different body heading')
        for a,b in rotations:
            self.assertEqual(a[:2],b[:2])
            self.assertTrue(nav.turning[a[:2]])
        for a,b in zip(path,path[1:]):
            self.assertIn(b,dict(nav._successors(a)))
        self.assertIn(m.world_to_cell(0.,3.5),nav.distances())

    def test_a_tight_corner_cannot_be_used_for_a_body_rotation(self):
        m=self.make_map([(-5,-1.5,-2,1.5),(-2,-.4,.4,.4),(-.4,-.4,.4,2),
                         (-1.5,2,1.5,5)])
        nav=ExplorationNavigation(m,(-3.,0.,0.))
        self.assertIsNone(nav.route(0.,3.5))

    def test_prefer_forward_travel_through_a_door_that_also_fits_sideways(self):
        m=self.make_map([(-2,-3,2,-1),(-.7,-1,.7,2),(-2,2,2,5)])
        nav=ExplorationNavigation(m,(0.,-2.,0.))
        path=nav.route(0.,3.5)
        self.assertTrue(path)
        in_door=[node for node in path if -.5 < m.cell_to_world(*node[:2])[1] < 1.5]
        self.assertTrue(in_door)
        self.assertTrue(all(abs(math.sin(nav.headings[node[2]]))>.9 for node in in_door))

    def test_stopping_corridor_is_required_even_when_the_body_fits(self):
        m=self.make_map([(-5,-5,.7,5)])
        nav=ExplorationNavigation(m,(0.,0.,0.))
        self.assertTrue(nav.fitting[1][m.world_to_cell(.3,0.)])
        self.assertIsNone(nav.route(.3,0.))

    def test_unknown_floor_is_not_a_route(self):
        m=OccupancyGrid(resolution=.1,size=100)
        m.grid[45:55,45:55]=FREE
        nav=ExplorationNavigation(m,(0.,0.,0.))
        self.assertIsNone(nav.route(2.,0.))

    def test_forbidden_translations_cannot_be_bypassed_in_a_tight_passage(self):
        m=self.make_map([(-4,-.4,4,.4)])
        nav=ExplorationNavigation(m,(0.,0.,0.),forbidden_steps=((-1,0),(0,-1),(0,1),(1,0),
                                  (-1,-1),(-1,1),(1,-1),(1,1)))
        self.assertIsNone(nav.route(2.,0.))

    def test_current_heading_is_preserved_until_turning_space(self):
        m=self.make_map([(-4,-.4,4,.4)])
        yaw=math.radians(4)
        nav=ExplorationNavigation(m,(0.,0.,yaw))
        path=nav.route(2.,0.)
        self.assertTrue(path)
        self.assertTrue(all(node[2]==0 for node in path))
        self.assertAlmostEqual(nav.first_leg(path)[2],yaw)

    def test_cancellation_and_private_snapshot(self):
        m=self.make_map([(-5,-5,5,5)])
        before=m.grid.copy()
        nav=ExplorationNavigation(m,(0.,0.,0.),cancelled=lambda:True)
        self.assertEqual(nav.distances(),{})
        self.assertIsNone(nav.route(2.,0.))
        nav.map.grid[:]=OCCUPIED
        np.testing.assert_array_equal(m.grid,before)

    def test_fine_obstacles_and_unknown_cells_survive_coarsening(self):
        m=OccupancyGrid(resolution=.05,size=80)
        m.grid[:]=FREE
        m.mark_obstacles([[.12,.22]])
        m.grid[m.world_to_cell(-.12,-.22)]=-1
        coarse=planning_map(m)
        self.assertEqual(coarse.resolution,.1)
        self.assertEqual(coarse.grid[coarse.world_to_cell(.12,.22)],OCCUPIED)
        self.assertEqual(coarse.grid[coarse.world_to_cell(-.12,-.22)],-1)

    def test_planning_coordinates_survive_an_odd_map_origin_after_growth(self):
        m=OccupancyGrid(resolution=.05,size=40,grow=True)
        m.grid[:]=FREE
        m.mark_obstacles([[.12,.22]])
        m.ensure_bounds([[-3.07,-4.03]])
        coarse=planning_map(m)
        self.assertEqual(coarse.grid[coarse.world_to_cell(.12,.22)],OCCUPIED)
        x,y=coarse.cell_to_world(*coarse.world_to_cell(.12,.22))
        self.assertAlmostEqual(x,.15)
        self.assertAlmostEqual(y,.25)

    def test_off_centre_pose_follows_the_planned_axis_instead_of_a_diagonal(self):
        m=self.make_map([(-5,-5,5,5)])
        nav=ExplorationNavigation(m,(.038,.046,math.pi/2))
        path=nav.route(.038,2.046)
        x,y,yaw=nav.first_leg(path)
        self.assertAlmostEqual(x,.038)
        self.assertAlmostEqual(y,.546)
        self.assertAlmostEqual(yaw,math.pi/2)

    def test_collision_envelope_reserves_subcell_tracking_space(self):
        m=OccupancyGrid(resolution=.05,size=160)
        m.grid[:]=FREE
        m.mark_obstacles([[.5,.05]])
        nav=ExplorationNavigation(m,(.099,.05,0.))
        self.assertFalse(nav.start_fits)
        self.assertFalse(nav.refined().start_fits)

    def test_refinement_keeps_a_safe_narrow_passage_without_erasing_obstacles(self):
        m=OccupancyGrid(resolution=.05,size=200)
        m.grid[:]=OCCUPIED
        r,c=np.indices(m.grid.shape)
        y=(r+m._min_row+.5)*m.resolution
        x=(c+m._min_col+.5)*m.resolution
        m.grid[(abs(y)<.35)&(abs(x)<4.)]=FREE
        nav=ExplorationNavigation(m,(0.,0.,0.))
        self.assertFalse(nav.start_fits)
        self.assertTrue(nav.can_refine)
        fine=nav.refined()
        self.assertTrue(fine.start_fits)
        self.assertTrue(fine.route(2.,0.))
        self.assertEqual(fine.map.grid[fine.map.world_to_cell(0.,.5)],OCCUPIED)


if __name__=='__main__':
    unittest.main()
