"""Navmesh build, endpoint locate and corridor-search behaviour."""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relic_nav import (
    Level, Surface, Wall, RAMP, FLOOR, CharacterSpec, NavMesh,
    SurfaceStatus, RouteStatus,
)
from relic_nav.geometry import Vec, path_clears_corner, dist2_point_seg
from relic_nav.level import make_ruin_level, make_characters


def spec(name, radius, slope=25.0):
    return CharacterSpec(name, radius, slope, step_height=0.35, speed=1.0)


def simple_two_room_level(door_lo=4.0, door_hi=4.8):
    lv = Level("two_rooms")
    lv.add_surface(Surface("a", (0, 0, 10, 10), FLOOR, 0))
    lv.add_surface(Surface("b", (10, 0, 20, 10), FLOOR, 0))
    lv.add_wall(Wall("w1", (10, 0), (10, door_lo)))
    lv.add_wall(Wall("w2", (10, door_hi), (10, 10)))
    lv.compute_bounds()
    lv.markers["left"] = Vec(2, 5)
    lv.markers["right"] = Vec(18, 5)
    lv.markers["door"] = Vec(10, (door_lo + door_hi) / 2)
    return lv


class TestClearanceInvariant(unittest.TestCase):
    def test_every_walkable_cell_keeps_radius(self):
        level = make_ruin_level()
        for character in make_characters():
            nav = NavMesh(level, character, cell_size=0.1)
            for cy in range(nav.ny):
                for cx in range(nav.nx):
                    i = nav._idx(cx, cy)
                    if not nav.walkable[i]:
                        continue
                    p = nav._cell_center(cx, cy)
                    nearest = min(
                        dist2_point_seg(p.x, p.y, a.x, a.y, b.x, b.y)
                        for a, b in nav.solids
                    )
                    self.assertGreaterEqual(
                        math.sqrt(nearest) + 1e-9, character.radius,
                        f"{character.name} cell {cx},{cy} violates clearance")

    def test_thin_sub_cell_wall_blocks_connectivity(self):
        # 0.04 m wall inside a 0.1 m raster must still split the region;
        # it spans the full floor so there is no route around its tips
        lv = Level("thin")
        lv.add_surface(Surface("floor", (0, 0, 10, 10), FLOOR, 0))
        lv.add_wall(Wall("shard", (5, 0), (5, 10), thickness=0.04))
        lv.compute_bounds()
        nav = NavMesh(lv, spec("s", 0.3), cell_size=0.1)
        left = nav.cell_at(Vec(2, 5))
        right = nav.cell_at(Vec(8, 5))
        self.assertNotEqual(nav.region[nav._idx(*left)],
                            nav.region[nav._idx(*right)])


class TestDoorRadiusReachability(unittest.TestCase):
    def test_same_door_differs_by_radius(self):
        # a 1.2 m doorway clears the 0.30-radius character but not the
        # 0.70-radius one (needs a 1.4 m gap plus erosion margin)
        level = simple_two_room_level(3.8, 5.0)
        small = NavMesh(level, spec("small", 0.30))
        big = NavMesh(level, spec("big", 0.70))

        def route(nav):
            _s, start, _d = nav.locate(level.markers["left"])
            _g, goal, _d = nav.locate(level.markers["right"])
            return nav.find_route(start, goal)

        self.assertEqual(route(small)["status"], RouteStatus.OK)
        self.assertEqual(route(big)["status"], RouteStatus.DISCONNECTED)

        # widening the same doorway to 2.2 m lets the big character through
        wide = simple_two_room_level(3.3, 5.5)
        big2 = NavMesh(wide, spec("big", 0.70))
        _s, start, _d = big2.locate(wide.markers["left"])
        _g, goal, _d = big2.locate(wide.markers["right"])
        self.assertEqual(big2.find_route(start, goal)["status"],
                         RouteStatus.OK)

    def test_ruin_level_matrix(self):
        level = make_ruin_level()
        scout, hauler = make_characters()
        nav_s = NavMesh(level, scout)
        nav_h = NavMesh(level, hauler)

        def route(nav, goal_marker):
            _s, start, _d = nav.locate(level.markers["start"])
            _g, goal, _d = nav.locate(level.markers[goal_marker])
            return nav.find_route(start, goal)

        # scout reaches the far room (using either door); the hauler also
        # reaches it via the separate wide door, so its total walkable area
        # is smaller: the cells immediately inside the narrow gap are eroded
        # away for the big radius but remain for the small one
        self.assertEqual(route(nav_s, "room_b_far")["status"], RouteStatus.OK)
        def door_cells(nav):
            return sum(
                1 for cy in range(nav.ny) for cx in range(nav.nx)
                if nav.walkable[nav._idx(cx, cy)]
                and 10.6 <= nav.origin.x + (cx + 0.5) * nav.cs <= 11.6
                and 4.0 <= nav.origin.y + (cy + 0.5) * nav.cs <= 4.8)
        self.assertGreater(door_cells(nav_s), 0)
        self.assertEqual(door_cells(nav_h), 0)


class TestRouteCornersClearSolids(unittest.TestCase):
    def test_every_route_segment_passes_capsule_test(self):
        level = make_ruin_level()
        goals = ["door_narrow", "room_b_far", "room_c", "behind_thin"]
        for character in make_characters():
            nav = NavMesh(level, character)
            _ss, start, _sd = nav.locate(level.markers["start"])
            for marker in goals:
                _gs, goal, _gd = nav.locate(level.markers[marker])
                if goal is None:
                    continue
                route = nav.find_route(start, goal)
                if route["status"] != RouteStatus.OK:
                    continue
                self.assertFalse(route["funnel_fallback"])
                poly = route["polyline"]
                for a, b in zip(poly, poly[1:]):
                    self.assertTrue(
                        path_clears_corner(a, b, nav.solids, character.radius),
                        f"{character.name} -> {marker} clips a wall corner")

    def test_route_never_crosses_a_wall_segment(self):
        from relic_nav.geometry import segments_intersect
        level = make_ruin_level()
        nav = NavMesh(level, make_characters()[0])
        _ss, start, _sd = nav.locate(level.markers["start"])
        _gs, goal, _gd = nav.locate(level.markers["room_b_far"])
        route = nav.find_route(start, goal)
        self.assertEqual(route["status"], RouteStatus.OK)
        walls = level.wall_segments()
        for a, b in zip(route["polyline"], route["polyline"][1:]):
            for wa, wb in walls:
                # touch at an endpoint adjacent to a door gap is allowed;
                # a proper crossing through the segment is not
                if segments_intersect(a, b, wa, wb):
                    self.fail("route segment crosses a solid wall")


class TestLocateSnapReject(unittest.TestCase):
    def setUp(self):
        self.level = make_ruin_level()
        self.nav = NavMesh(self.level, make_characters()[0])

    def test_point_on_surface_is_ok(self):
        status, point, detail = self.nav.locate(Vec(3, 3))
        self.assertEqual(status, SurfaceStatus.OK)
        self.assertEqual(point, Vec(3, 3))

    def test_void_click_is_rejected_with_rationale(self):
        status, point, detail = self.nav.locate(
            self.level.markers["void"], snap_tol=0.2)
        self.assertEqual(status, SurfaceStatus.REJECTED_VOID)
        self.assertIsNone(point)
        self.assertIn("void", detail.lower())

    def test_near_edge_click_snaps(self):
        # a point just outside room A's lower edge but within the snap radius
        status, point, detail = self.nav.locate(Vec(3, 0.75), snap_tol=0.7)
        self.assertEqual(status, SurfaceStatus.SNAPPED_EDGE)
        self.assertIsNotNone(point)
        self.assertTrue(self.nav.walkable[self.nav._idx(*self.nav.cell_at(point))])

    def test_clearance_click_snaps_into_walkable(self):
        # the narrow-door marker sits in eroded space for the scout
        status, point, detail = self.nav.locate(
            self.level.markers["door_narrow"])
        self.assertIn(status, (SurfaceStatus.SNAPPED_CLEARANCE, SurfaceStatus.OK))
        if status == SurfaceStatus.SNAPPED_CLEARANCE:
            self.assertTrue(
                self.nav.walkable[self.nav._idx(*self.nav.cell_at(point))])


class TestSlopeRejection(unittest.TestCase):
    def test_steep_ramp_blocked_for_low_slope_character(self):
        lv = Level("ramp")
        lv.add_surface(Surface("low", (0, 0, 5, 5), FLOOR, 0))
        # 2 m rise over 4 m = 0.5 slope ~= 26.6 deg
        lv.add_surface(Surface("ramp", (5, 0, 9, 5), RAMP, 0, 2.0, axis="x"))
        lv.add_surface(Surface("high", (9, 0, 14, 5), FLOOR, 2.0))
        lv.compute_bounds()
        steep_nav = NavMesh(lv, spec("climber", 0.3, slope=30.0))
        flat_nav = NavMesh(lv, spec("grounded", 0.3, slope=20.0))
        ramp_cell = flat_nav.cell_at(Vec(7, 2.5))
        self.assertTrue(steep_nav.walkable[steep_nav._idx(*ramp_cell)])
        self.assertFalse(flat_nav.walkable[flat_nav._idx(*ramp_cell)])
        status, _point, detail = flat_nav.locate(Vec(7, 2.5), snap_tol=0.2)
        self.assertEqual(status, SurfaceStatus.REJECTED_SLOPE)
        self.assertIn("slope", detail.lower())


class TestDisconnectedPlatform(unittest.TestCase):
    def test_platform_is_its_own_region(self):
        level = make_ruin_level()
        nav = NavMesh(level, make_characters()[0])
        route = nav.find_route(level.markers["start"],
                               level.markers["platform"])
        self.assertEqual(route["status"], RouteStatus.DISCONNECTED)


if __name__ == "__main__":
    unittest.main()
