"""How far away a seen thing is (perception.fuse): by LiDAR on it, by where it meets the floor, or, when it's on
furniture above the LiDAR's view, by the LiDAR range to what it's on."""
import unittest

import numpy as np

from go2 import perception as P

INTRINSICS = {"fx": 300., "fy": 300., "cx": 320., "cy": 180.}
CAMERA_POS = np.array([.3, 0., .05])
CAMERA_ROT = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])    # OpenCV camera axes in the base frame
HEIGHT = .32                                                          # of the base above the floor


def seen(rows, cols):
    mask = np.zeros((360, 640), bool)
    mask[rows[0]:rows[1], cols[0]:cols[1]] = True
    return [P.Mask("teddy bear", mask, 1.)]


def grid(xs, ys, zs):
    return np.array([(x, y, z) for x in xs for y in ys for z in zs], float)


FLOOR = grid(np.arange(.8, 1.95, .1), np.arange(-.3, .31, .1), [-HEIGHT])
BED_SIDE = grid([2.], np.arange(-.3, .31, .05), np.arange(-.25, -.04, .05))      # .07 to .27 m above the floor
ON_A_BED = seen((120, 160), (310, 330))                                         # all of it above the horizon


def fuse(masks, cloud):
    rgb = np.zeros((360, 640, 3), np.uint8)
    return P.fuse(masks, rgb, cloud, INTRINSICS, CAMERA_POS, CAMERA_ROT, 0., 0., HEIGHT)[0]


class RangeTests(unittest.TestCase):
    def test_a_thing_on_furniture_is_ranged_by_the_furniture_below_it(self):
        det = fuse(ON_A_BED, np.vstack([FLOOR, BED_SIDE]))
        self.assertEqual(det.distance_source, "support")
        self.assertAlmostEqual(det.distance_m, 2., delta=.1)
        self.assertAlmostEqual(det.x_m, 2., delta=.1)
        self.assertAlmostEqual(det.y_m, 0., delta=.1)

    def test_the_floor_in_front_of_it_doesnt_count_as_what_its_on(self):
        det = fuse(ON_A_BED, FLOOR)
        self.assertEqual(det.distance_source, "unknown")
        self.assertIsNone(det.distance_m)

    def test_another_thing_in_front_of_it_doesnt_count_as_what_its_on(self):
        # A book on the floor 1 m away, in the same line of sight as a teddy bear on a bed 2 m away (seen through a
        # doorway, say): the LiDAR points on the book are below the teddy bear in the image, but it isn't on it.
        book = grid([1.], np.arange(-.08, .085, .01), np.arange(-.25, -.195, .01))   # 7 to 12 cm above the floor
        book_mask = seen((270, 320), (270, 370))[0].mask
        masks = ON_A_BED + [P.Mask("book", book_mask, 1.)]
        det = next(d for d in P.fuse(masks, np.zeros((360, 640, 3), np.uint8), np.vstack([FLOOR, BED_SIDE, book]),
                                     INTRINSICS, CAMERA_POS, CAMERA_ROT, 0., 0., HEIGHT) if d.label == "teddy bear")
        self.assertEqual(det.distance_source, "support")
        self.assertAlmostEqual(det.distance_m, 2., delta=.1, msg="ranged by the bed, not the book in front")

    def test_furniture_it_could_be_on_still_counts_when_its_seen_too(self):
        bed_mask = seen((200, 360), (250, 390))[0].mask                            # the bed's own outline
        det = next(d for d in P.fuse(ON_A_BED + [P.Mask("bed", bed_mask, 1.)], np.zeros((360, 640, 3), np.uint8),
                                     np.vstack([FLOOR, BED_SIDE]), INTRINSICS, CAMERA_POS, CAMERA_ROT, 0., 0., HEIGHT)
                   if d.label == "teddy bear")
        self.assertEqual(det.distance_source, "support")
        self.assertAlmostEqual(det.distance_m, 2., delta=.1)

    def test_furniture_off_to_one_side_doesnt_count(self):
        det = fuse(ON_A_BED, np.vstack([FLOOR, BED_SIDE + (0., 1.5, 0.)]))
        self.assertEqual(det.distance_source, "unknown")

    def test_a_thing_on_the_floor_is_still_ranged_by_where_it_meets_the_floor(self):
        det = fuse(seen((200, 260), (310, 330)), BED_SIDE + (0., 1.5, 0.))    # no LiDAR points on it
        self.assertEqual(det.distance_source, "ground_plane")

    def test_lidar_points_on_the_thing_itself_still_come_first(self):
        on_it = grid([1.5], [-.02, 0., .02], [-.2, -.18])
        det = fuse(seen((200, 260), (300, 340)), np.vstack([FLOOR, BED_SIDE, on_it]))
        self.assertEqual(det.distance_source, "lidar")
        self.assertAlmostEqual(det.distance_m, 1.5, delta=.05)


if __name__ == "__main__":
    unittest.main()
