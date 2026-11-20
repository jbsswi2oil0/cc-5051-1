"""Diagonal-gate scene, injected faults, phase classification and diagnosis."""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relic_nav.geometry import Vec
from relic_nav.scene import (
    GateScenario, ErosionShortfall, RadiusMisread, repaired, build_gate_level,
    evaluate, classify, diagnosis, phase_shift, DEFAULT_PHASES,
)


class TestSceneGeometry(unittest.TestCase):
    def test_jamb_span_equals_door_width(self):
        scn = GateScenario()
        lo, hi = scn.jamb_points()
        self.assertAlmostEqual((hi - lo).length(), scn.door_width, places=6)

    def test_door_centre_sits_between_jambs(self):
        scn = GateScenario()
        lo, hi = scn.jamb_points()
        mid = (lo + hi) * 0.5
        self.assertLess((mid - scn.door_center()).length(), 1e-9)

    def test_phase_shift_moves_door_and_character_together(self):
        scn = GateScenario()
        shifted = phase_shift(scn, 0.05, 0.05)
        # continuous sizes unchanged
        self.assertEqual(shifted.door_width, scn.door_width)
        self.assertEqual(shifted.radius, scn.radius)
        self.assertEqual(shifted.ramp_gap, scn.ramp_gap)
        # door group and body translated by the same vector
        self.assertAlmostEqual(shifted.door_center().x, scn.door_center().x + 0.05)
        self.assertAlmostEqual(shifted.start_point().x, scn.start_point().x + 0.05)
        # slanted phase translation preserves jamb span
        lo0, hi0 = scn.jamb_points()
        lo1, hi1 = shifted.jamb_points()
        self.assertAlmostEqual((hi1 - lo1).length(), (hi0 - lo0).length(), places=6)

    def test_level_builds_two_jambs_and_markers(self):
        lv = build_gate_level(GateScenario())
        names = {w.name for w in lv.walls}
        self.assertEqual(names, {"jamb_low", "jamb_high"})
        for m in ("start", "goal", "door", "jamb_low", "jamb_high", "turn"):
            self.assertIn(m, lv.markers)

    def test_ramp_starts_past_the_door(self):
        scn = GateScenario()
        rx0, _, _, _ = scn.ramp_rect()
        self.assertGreater(rx0, scn.door_x)


class TestFaultModels(unittest.TestCase):
    def test_erosion_shortfall_lowers_erode_radius(self):
        plan_r, margin = ErosionShortfall(1.0).planner_params(0.5, 0.1)
        self.assertEqual(plan_r, 0.5)
        self.assertAlmostEqual(margin, -0.1)

    def test_radius_misread_underreads_planner_radius(self):
        plan_r, margin = RadiusMisread(0.15).planner_params(0.5, 0.1)
        self.assertAlmostEqual(plan_r, 0.35)
        self.assertEqual(margin, 0.0)

    def test_repair_restores_continuous_dimensions(self):
        for fault in (ErosionShortfall(1.0), RadiusMisread(0.15)):
            fix = repaired(fault)
            plan_r, margin = fix.planner_params(0.5, 0.1)
            self.assertEqual(plan_r, 0.5)
            self.assertEqual(margin, 0.0)
            self.assertTrue(fix.name.startswith("repaired_"))


class TestEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scn = GateScenario()
        cls.misread = evaluate(cls.scn, RadiusMisread(0.15), drive=True)
        cls.erosion = evaluate(cls.scn, ErosionShortfall(1.0), drive=True)
        cls.fix = evaluate(cls.scn, repaired(RadiusMisread(0.15)), drive=True)

    def test_baseline_channel_is_passable(self):
        self.assertTrue(self.fix.passed)
        self.assertGreaterEqual(self.fix.sweep.min_clearance, self.scn.radius)

    def test_radius_misread_fails_sweep(self):
        self.assertTrue(self.misread.failed)
        self.assertLess(self.misread.sweep.min_clearance, self.scn.radius)

    def test_erosion_shortfall_fails_sweep(self):
        self.assertTrue(self.erosion.failed)

    def test_failure_localises_to_a_corner_at_the_turn(self):
        for res in (self.misread, self.erosion):
            self.assertTrue(res.turns)
            self.assertIsNotNone(res.turn_anchor)
            self.assertTrue(res.contact_on_jamb or res.turn_anchor is not None)

    def test_driven_body_actually_walks(self):
        self.assertGreater(len(self.misread.driven.frames), 1)
        self.assertGreater(self.misread.driven.travelled, 1.0)

    def test_planner_edges_are_portal_segments(self):
        for edge in self.misread.planner_edges:
            self.assertEqual(len(edge), 2)
            self.assertEqual(len(edge[0]), 2)

    def test_drive_off_matches_drive_on_verdict(self):
        fast = evaluate(self.scn, RadiusMisread(0.15), drive=False)
        self.assertEqual(fast.failed, self.misread.failed)
        self.assertIsNone(fast.driven)


class TestPhaseStudy(unittest.TestCase):
    def test_radius_misread_is_sweep_dependent(self):
        study = classify(GateScenario(), RadiusMisread(0.15))
        self.assertTrue(study.depends_on_sweep)
        self.assertFalse(study.depends_on_phase)
        diag = diagnosis(study)
        self.assertEqual(diag["kind"], "sweep_dependent")
        self.assertIsNotNone(diag["turn_contact"])
        self.assertIn("contact_on_jamb", diag)

    def test_diagnosis_turn_is_nearest_the_contact(self):
        # the scrape sits at the turn the body actually makes it at, not at
        # some earlier bend on the route
        study = classify(GateScenario(), RadiusMisread(0.15))
        diag = diagnosis(study)
        tx, ty = diag["turn_point"]
        cx, cy = diag["turn_contact"]["point"]
        self.assertLess(math.hypot(tx - cx, ty - cy), 0.6)

    def test_small_radius_misread_is_phase_dependent(self):
        study = classify(GateScenario(), RadiusMisread(0.04))
        self.assertTrue(study.depends_on_phase)
        self.assertFalse(study.depends_on_sweep)
        diag = diagnosis(study)
        self.assertEqual(diag["kind"], "phase_dependent")
        self.assertTrue(diag["failing_phases"])

    def test_repaired_control_stable_across_phases(self):
        study = classify(GateScenario(), repaired(RadiusMisread(0.15)))
        self.assertTrue(study.stable_pass)
        self.assertEqual(diagnosis(study)["kind"], "passes")

    def test_verdicts_cover_all_phases(self):
        study = classify(GateScenario(), RadiusMisread(0.15))
        self.assertEqual(len(study.verdicts), len(DEFAULT_PHASES))


if __name__ == "__main__":
    unittest.main()
