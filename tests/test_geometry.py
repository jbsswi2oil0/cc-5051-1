"""Geometry primitives and the exact Euclidean distance transform."""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relic_nav.geometry import (
    Vec, dist2_point_seg, closest_point_on_seg, segments_intersect,
    euclidean_distance_field, edt_1d, path_clears_corner, polyline_length,
)


class TestVec(unittest.TestCase):
    def test_basic_arithmetic(self):
        a = Vec(1, 2)
        b = Vec(3, -1)
        self.assertEqual(a + b, Vec(4, 1))
        self.assertEqual(b - a, Vec(2, -3))
        self.assertEqual(a * 2, Vec(2, 4))
        self.assertAlmostEqual(Vec(3, 4).length(), 5.0)
        self.assertAlmostEqual(a.dot(b), 1.0)
        self.assertEqual(Vec(1, 0).cross(Vec(0, 1)), 1.0)
        self.assertEqual(Vec(1, 0).perp_left(), Vec(0, 1))

    def test_normalised_zero(self):
        self.assertEqual(Vec(0, 0).normalised(), Vec(0, 0))


class TestPointSeg(unittest.TestCase):
    def test_interior_and_endpoint(self):
        a, b = Vec(0, 0), Vec(10, 0)
        self.assertAlmostEqual(dist2_point_seg(5, 3, a.x, a.y, b.x, b.y), 9.0)
        self.assertEqual(closest_point_on_seg(Vec(5, 3), a, b), Vec(5, 0))
        # clamped beyond the endpoint
        self.assertEqual(closest_point_on_seg(Vec(-4, 2), a, b), Vec(0, 0))
        self.assertEqual(closest_point_on_seg(Vec(14, 2), a, b), Vec(10, 0))

    def test_degenerate_segment(self):
        a = Vec(2, 2)
        self.assertAlmostEqual(dist2_point_seg(5, 6, a.x, a.y, a.x, a.y), 25.0)


class TestSegmentsIntersect(unittest.TestCase):
    def test_crossing_and_disjoint(self):
        self.assertTrue(segments_intersect(Vec(0, 0), Vec(4, 4), Vec(0, 4), Vec(4, 0)))
        self.assertFalse(segments_intersect(Vec(0, 0), Vec(1, 1), Vec(2, 2), Vec(3, 3)))

    def test_collinear_overlap(self):
        self.assertTrue(segments_intersect(Vec(0, 0), Vec(4, 0), Vec(2, 0), Vec(6, 0)))
        self.assertFalse(segments_intersect(Vec(0, 0), Vec(1, 0), Vec(2, 0), Vec(3, 0)))


class TestEDT(unittest.TestCase):
    def test_1d_known_values(self):
        # single blocked site at index 0
        f = [0.0, 1e18, 1e18, 1e18, 1e18]
        d = [0.0] * 5
        v = [0] * 5
        z = [0.0] * 6
        edt_1d(f, 5, d, v, z)
        self.assertEqual(d, [0.0, 1.0, 4.0, 9.0, 16.0])

    def test_2d_matches_brute_force(self):
        h = w = 12
        blocked = [((x * 7 + y * 3) % 11 == 0) for y in range(h) for x in range(w)]
        blocked[0] = True  # guarantee at least one site
        field = euclidean_distance_field(blocked, h, w)
        for y in range(h):
            for x in range(w):
                i = y * w + x
                if blocked[i]:
                    self.assertEqual(field[i], 0.0)
                    continue
                brute = min(
                    (x - bx) ** 2 + (y - by) ** 2
                    for by in range(h) for bx in range(w)
                    if blocked[by * w + bx]
                )
                self.assertAlmostEqual(field[i] ** 2, brute, places=6)

    def test_empty_row_band_does_not_skip_sites(self):
        # regression: a long free band must still see blocked cells beyond it
        h, w = 4, 9
        blocked = [False] * (h * w)
        for y in range(h):
            blocked[y * w + 0] = True   # wall column on the left
            blocked[y * w + 8] = True   # wall column on the right
        field = euclidean_distance_field(blocked, h, w)
        # centre cell distance to either wall column is 4 cells
        self.assertAlmostEqual(field[2 * w + 4], 4.0)


class TestCapsuleClearance(unittest.TestCase):
    def setUp(self):
        # vertical wall through x=5 from y=0 to y=10
        self.solids = [(Vec(5, 0), Vec(5, 10))]

    def test_parallel_pass_and_clip(self):
        # centre line 0.4 away with radius 0.3 clears; radius 0.5 does not
        self.assertTrue(path_clears_corner(Vec(4.6, 1), Vec(4.6, 9),
                                           self.solids, 0.3))
        self.assertFalse(path_clears_corner(Vec(4.6, 1), Vec(4.6, 9),
                                            self.solids, 0.5))

    def test_diagonal_corner_cut_is_rejected(self):
        # endpoints individually valid but the segment crosses the wall
        p0, p1 = Vec(4.6, 1), Vec(5.4, 9)
        self.assertFalse(path_clears_corner(p0, p1, self.solids, 0.3))

    def test_thin_wall_between_distant_endpoints(self):
        # long segment with a tiny wall near its middle: dense length-based
        # sampling must not leap over it
        thin = [(Vec(5, 4.95), Vec(5, 5.05))]
        self.assertFalse(path_clears_corner(Vec(0, 5), Vec(10, 5), thin, 0.3))
        self.assertTrue(path_clears_corner(Vec(0, 2), Vec(10, 2), thin, 0.3))


class TestPolylineLength(unittest.TestCase):
    def test_length(self):
        pts = [Vec(0, 0), Vec(3, 0), Vec(3, 4)]
        self.assertAlmostEqual(polyline_length(pts), 7.0)


if __name__ == "__main__":
    unittest.main()
