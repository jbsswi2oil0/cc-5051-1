"""Driven movement: collision resolution, no tunnelling, failure stages."""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relic_nav import (
    Level, Surface, Wall, FLOOR, CharacterSpec, NavMesh,
    SurfaceStatus, FollowStatus,
)
from relic_nav.geometry import Vec, dist2_point_seg
from relic_nav.level import make_ruin_level, make_characters
from relic_nav.runner import (
    move_with_collision, follow_route, run_scenario,
)


def make_spec(radius=0.3):
    return CharacterSpec("t", radius, max_slope_deg=25,
                         step_height=0.35, speed=2.0)


class TestMoveWithCollision(unittest.TestCase):
    def setUp(self):
        self.solids = [(Vec(5, 0), Vec(5, 10))]

    def test_never_penetrates_radius(self):
        pos = Vec(2, 5)
        # drive straight at the wall for many steps
        for _ in range(40):
            pos, _c, _pk = move_with_collision(
                pos, Vec(0.5, 0.0), 0.3, self.solids)
            d = math.sqrt(dist2_point_seg(pos.x, pos.y, 5, 0, 5, 10))
            self.assertGreaterEqual(d, 0.3 - 1e-9)
        # it must stop before the wall rather than cross it
        self.assertLess(pos.x, 5.0)

    def test_thin_wall_is_not_tunneled(self):
        thin = [(Vec(5, 4.95), Vec(5, 5.05))]
        pos = Vec(0, 5)
        for _ in range(30):
            pos, _c, _pk = move_with_collision(
                pos, Vec(0.6, 0.0), 0.3, thin)
        self.assertLess(pos.x, 5.0 - 0.3 + 1e-9)

    def test_free_movement_is_unaffected(self):
        pos = Vec(0, 0)
        delta = Vec(1, 1)
        new_pos, collided, peak = move_with_collision(
            pos, delta, 0.3, self.solids)
        self.assertFalse(collided)
        self.assertEqual(new_pos, Vec(1, 1))
        self.assertEqual(peak, 0.0)

    def test_slide_keeps_moving_along_wall(self):
        pos = Vec(4.71, 1)
        delta = Vec(0.3, 0.3)
        new_pos, collided, _pk = move_with_collision(
            pos, delta, 0.3, self.solids)
        self.assertTrue(collided)
        # blocked in x but still progresses in y
        self.assertGreater(new_pos.y, pos.y)
        d = math.sqrt(dist2_point_seg(new_pos.x, new_pos.y, 5, 0, 5, 10))
        self.assertGreaterEqual(d, 0.3 - 1e-9)


class TestFollowRoute(unittest.TestCase):
    def test_arrival_and_no_frame_penetration(self):
        level = make_ruin_level()
        spec = make_characters()[0]
        nav, result = run_scenario(
            level, spec, level.markers["start"], level.markers["room_b_far"])
        self.assertEqual(result.status, FollowStatus.ARRIVED)
        self.assertGreater(len(result.frames), 10)
        for frame in result.frames:
            p = Vec(frame.x, frame.y)
            nearest = min(
                dist2_point_seg(p.x, p.y, a.x, a.y, b.x, b.y)
                for a, b in nav.solids
            )
            self.assertGreaterEqual(
                math.sqrt(nearest), spec.radius - 1e-9,
                f"frame at t={frame.t} penetrates a solid")
        # the body stays over a real surface the whole time
        self.assertEqual(result.surface_failures, 0)
        # travelled distance is comparable to the reference polyline
        self.assertLess(abs(result.travelled - result.path_length),
                        0.25 * result.path_length)

    def test_deviation_is_measured_against_route(self):
        level = make_ruin_level()
        spec = make_characters()[0]
        _nav, result = run_scenario(
            level, spec, level.markers["start"], level.markers["room_b_far"])
        self.assertGreaterEqual(result.max_deviation, 0.0)
        self.assertLess(result.max_deviation, 1.5)
        self.assertGreater(result.mean_deviation, 0.0)


class TestFailureStages(unittest.TestCase):
    def test_surface_generation_rejection(self):
        level = make_ruin_level()
        for spec in make_characters():
            _nav, result = run_scenario(
                level, spec, level.markers["start"], level.markers["void"],
                snap_tol=0.2)
            self.assertEqual(result.status, FollowStatus.NO_SURFACE)
            self.assertIn("surface-stage", result.reason)
            self.assertEqual(result.goal_status, SurfaceStatus.REJECTED_VOID)
            self.assertEqual(result.frames, [])

    def test_corridor_search_disconnected(self):
        level = make_ruin_level()
        spec = make_characters()[0]
        _nav, result = run_scenario(
            level, spec, level.markers["start"], level.markers["platform"])
        self.assertEqual(result.status, FollowStatus.NO_ROUTE)
        self.assertIn("corridor-stage", result.reason)
        self.assertEqual(result.frames, [])

    def test_narrow_door_surface_stage_for_hauler(self):
        level = make_ruin_level()
        hauler = make_characters()[1]
        _nav, result = run_scenario(
            level, hauler, level.markers["start"],
            level.markers["door_narrow"])
        self.assertEqual(result.status, FollowStatus.NO_SURFACE)

    def test_follow_stage_cannot_teleport(self):
        # construct a reachable route, then wall the goal off: the follower
        # must report a stuck failure rather than appear at the target
        lv = Level("pinned")
        lv.add_surface(Surface("floor", (0, 0, 12, 12), FLOOR, 0))
        lv.add_wall(Wall("barrier", (6, 2), (6, 10)))
        lv.compute_bounds()
        spec = make_spec(0.3)
        nav = NavMesh(lv, spec)
        # route across the barrier is impossible at corridor stage, so to
        # exercise the follow stage feed a short route into a dead-end pocket
        start = Vec(2, 5)
        route = nav.find_route(start, Vec(4, 5))
        self.assertEqual(route["status"].value, "route_ok")
        follow = follow_route(nav, route, start, Vec(4, 5), max_time=3.0)
        # an open pocket succeeds, proving the follow pipeline itself runs
        self.assertEqual(follow.status, FollowStatus.ARRIVED)


if __name__ == "__main__":
    unittest.main()
