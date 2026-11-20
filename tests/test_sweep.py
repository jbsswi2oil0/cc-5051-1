"""Independent swept-volume check: exact segment distance and attribution."""

import math
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relic_nav.geometry import Vec, closest_point_on_seg
from relic_nav.sweep import (
    seg_seg_closest, sweep_polyline, disc_clearance, turn_contact,
    polyline_turns, SweepResult, Contact,
)


class TestSegSegClosest(unittest.TestCase):
    def test_intersecting_segments_zero(self):
        d, pa, pb = seg_seg_closest(Vec(0, 0), Vec(2, 2),
                                    Vec(0, 2), Vec(2, 0))
        self.assertAlmostEqual(d, 0.0)
        self.assertAlmostEqual(pa.x, 1.0)
        self.assertAlmostEqual(pa.y, 1.0)
        self.assertAlmostEqual(pb.x, 1.0)

    def test_parallel_offset(self):
        d, _, _ = seg_seg_closest(Vec(0, 0), Vec(4, 0),
                                  Vec(0, 1.5), Vec(4, 1.5))
        self.assertAlmostEqual(d, 1.5)

    def test_endpoint_to_endpoint(self):
        d, pa, pb = seg_seg_closest(Vec(0, 0), Vec(1, 0), Vec(4, 3), Vec(5, 3))
        self.assertAlmostEqual(d, math.hypot(3, 3))
        self.assertAlmostEqual(pa.x, 1.0)
        self.assertAlmostEqual(pb.x, 4.0)

    def test_matches_brute_force_sampling(self):
        rng = random.Random(5)
        for _ in range(50):
            a = Vec(rng.uniform(-2, 2), rng.uniform(-2, 2))
            b = Vec(rng.uniform(-2, 2), rng.uniform(-2, 2))
            c = Vec(rng.uniform(-2, 2), rng.uniform(-2, 2))
            dvec = Vec(rng.uniform(-2, 2), rng.uniform(-2, 2))
            exact, _, _ = seg_seg_closest(a, b, c, dvec)
            brute = 1e9
            for i in range(201):
                p = a + (b - a) * (i / 200)
                q = closest_point_on_seg(p, c, dvec)
                brute = min(brute, (q - p).length())
            self.assertLessEqual(exact, brute + 1e-9)
            self.assertLess(brute - exact, 0.03)


class TestSweepPolyline(unittest.TestCase):
    def setUp(self):
        # a 1.0 m opening between two collinear wall segments at y = 0
        self.solids = [(Vec(-3, 0), Vec(0, 0)), (Vec(1, 0), Vec(4, 0))]

    def test_straight_channel_clearance(self):
        poly = [Vec(0.5, -2), Vec(0.5, 2)]
        res = sweep_polyline(poly, self.solids, radius=0.4)
        # nearest solid is the wall tip at (0,0) or (1,0), 0.5 m away
        self.assertAlmostEqual(res.min_clearance, 0.5)
        self.assertTrue(res.feasible)

    def test_radius_above_clearance_infeasible(self):
        poly = [Vec(0.5, -2), Vec(0.5, 2)]
        res = sweep_polyline(poly, self.solids, radius=0.55)
        self.assertFalse(res.feasible)

    def test_radius_half_width_passes(self):
        poly = [Vec(0.5, -2), Vec(0.5, 2)]
        self.assertTrue(sweep_polyline(poly, self.solids, radius=0.5).feasible)

    def test_turn_contact_localised(self):
        # route bends at (0.5, 0.5); the scrape is the wall tip right there
        poly = [Vec(0.5, -2), Vec(0.5, 0.5), Vec(3.0, 0.5)]
        res = sweep_polyline(poly, self.solids, radius=0.6)
        self.assertFalse(res.feasible)
        tc = turn_contact(res, Vec(0.5, 0.5), 0.6)
        self.assertIsNotNone(tc)
        self.assertLess((tc.point - Vec(0.5, 0.5)).length(), 1.0)

    def test_exact_matches_full_scan(self):
        rng = random.Random(11)
        solids = []
        for _ in range(300):
            a = Vec(rng.uniform(0, 8), rng.uniform(0, 8))
            b = a + Vec(rng.uniform(-1, 1), rng.uniform(-1, 1))
            solids.append((a, b))
        poly = [Vec(2, 2), Vec(5, 3), Vec(6, 6)]
        res = sweep_polyline(poly, solids, radius=0.3)
        brute = float("inf")
        for a, b in zip(poly, poly[1:]):
            for w0, w1 in solids:
                d, _, _ = seg_seg_closest(a, b, w0, w1)
                brute = min(brute, d)
        self.assertAlmostEqual(res.min_clearance, brute, places=9)

    def test_empty_polyline(self):
        res = sweep_polyline([], self.solids, 0.5)
        self.assertIsNone(res.min_contact)
        self.assertTrue(res.feasible)  # inf clearance + eps >= r

    def test_as_dict_serialisable(self):
        poly = [Vec(0.5, -2), Vec(0.5, 2)]
        d = sweep_polyline(poly, self.solids, 0.4).as_dict()
        self.assertEqual(d["radius"], 0.4)
        self.assertIn("min_contact", d)


class TestDiscClearance(unittest.TestCase):
    def test_agrees_with_exact_on_straight(self):
        solids = [(Vec(-3, 0), Vec(0, 0)), (Vec(1, 0), Vec(4, 0))]
        poly = [Vec(0.5, -2), Vec(0.5, 2)]
        exact = sweep_polyline(poly, solids, 0.4).min_clearance
        sampled, _, n = disc_clearance(poly, solids, 0.4, sample_step=0.01)
        self.assertAlmostEqual(sampled, exact, places=2)
        self.assertGreater(n, 0)


class TestTurns(unittest.TestCase):
    def test_detects_sharp_turn(self):
        poly = [Vec(0, 0), Vec(1, 0), Vec(1, 1)]
        turns = polyline_turns(poly, min_angle_deg=20)
        self.assertEqual(len(turns), 1)
        self.assertAlmostEqual(turns[0][2], 90.0)

    def test_straight_has_no_turns(self):
        self.assertEqual(polyline_turns([Vec(0, 0), Vec(1, 0), Vec(2, 0)]), [])


class TestContactRecord(unittest.TestCase):
    def test_contact_as_dict_keys(self):
        c = Contact(0.3, 1, Vec(1, 1), (Vec(0, 0), Vec(2, 0)), 7, 0.5)
        d = c.as_dict()
        for key in ("distance_m", "seg_index", "point", "solid", "solid_index",
                    "along_segment"):
            self.assertIn(key, d)


if __name__ == "__main__":
    unittest.main()
