"""Automatic reduction of failing gate samples and the level-view loader."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relic_nav.scene import GateScenario, RadiusMisread, ErosionShortfall
from relic_nav.reducer import (
    reduce_failure, default_faults, phase_probe_fault, _in_turn_region,
)
from relic_nav.view import catalog, load_sample, deliver_view, Sample


class TestDefaultFaults(unittest.TestCase):
    def test_two_literal_faults(self):
        names = [type(f).__name__ for f in default_faults()]
        self.assertEqual(names, ["RadiusMisread", "ErosionShortfall"])
        erosion = default_faults()[1]
        self.assertEqual(erosion.cells, 1.0)

    def test_phase_probe_is_marginal(self):
        self.assertEqual(phase_probe_fault().cells, 0.4)


class TestReducer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scn = GateScenario()
        cls.misread = reduce_failure(cls.scn, RadiusMisread(0.15),
                                     steps_per_axis=4)
        cls.erosion = reduce_failure(cls.scn, ErosionShortfall(1.0),
                                     steps_per_axis=4)
        cls.phase = reduce_failure(cls.scn, phase_probe_fault(),
                                   steps_per_axis=4)

    def test_reduced_radius_fault_still_fails(self):
        self.assertTrue(self.misread.reduced_result.failed)

    def test_reduced_erosion_fault_still_fails(self):
        self.assertTrue(self.erosion.reduced_result.failed)

    def test_control_door_stays_passable_and_same_width(self):
        for red in (self.misread, self.erosion):
            self.assertTrue(red.control_result.passed)
            self.assertEqual(red.control_result.scenario.door_width,
                             red.reduced_result.scenario.door_width)

    def test_every_axis_shrinks_or_stays(self):
        for red in (self.misread, self.erosion):
            for axis in ("door_width", "radius", "ramp_gap"):
                self.assertLessEqual(getattr(red.reduced, axis),
                                     getattr(red.original, axis) + 1e-9)

    def test_turn_position_reduces_for_radius_fault(self):
        self.assertLess(self.misread.reduced.ramp_gap, self.misread.original.ramp_gap)

    def test_door_width_reduces_for_both(self):
        for red in (self.misread, self.erosion):
            self.assertLess(red.reduced.door_width, red.original.door_width)

    def test_erosion_backs_off_to_preserve_diagnosis_kind(self):
        # pushing every axis to its floor turns the sweep-dependent scrape into
        # a phase-dependent one; the reducer must keep the most-reduced
        # checkpoint whose kind still matches, not the raw floor
        self.assertTrue(self.erosion.reduced_result.failed)
        self.assertEqual(self.erosion.reduced_diagnosis["kind"],
                         "sweep_dependent")
        self.assertLess(self.erosion.reduced.radius, self.erosion.original.radius)

    def test_scrape_stays_at_the_turn(self):
        for red in (self.misread, self.erosion):
            self.assertTrue(_in_turn_region(red.reduced_result,
                                            red.reduced))

    def test_reduced_erosion_radius_actually_shrinks(self):
        # the turn-region criterion (not a brittle jamb-name match) lets the
        # body shrink; the one-cell erosion must still fire on a smaller body
        self.assertLess(self.erosion.reduced.radius, self.erosion.original.radius)
    def test_diagnosis_kind_is_preserved(self):
        for red in (self.misread, self.erosion, self.phase):
            self.assertEqual(red.base_diagnosis.get("kind"),
                             red.reduced_diagnosis.get("kind"))

    def test_both_diagnosis_kinds_reachable(self):
        self.assertEqual(self.misread.base_diagnosis["kind"], "sweep_dependent")
        self.assertEqual(self.erosion.base_diagnosis["kind"], "sweep_dependent")
        self.assertEqual(self.phase.base_diagnosis["kind"], "phase_dependent")

    def test_phase_dependent_reduction_lists_failing_phases(self):
        self.assertTrue(self.phase.reduced_diagnosis["failing_phases"])

    def test_sweep_dependent_reduction_names_turn_contact(self):
        for red in (self.misread, self.erosion):
            diag = red.reduced_diagnosis
            self.assertIsNotNone(diag["turn_contact"])
            self.assertIn("turn_point", diag)

    def test_report_is_serialisable(self):
        d = self.misread.as_dict()
        self.assertIn("shrink", d)
        self.assertEqual(d["shrink"]["ramp_gap"]["to"],
                         self.misread.reduced.ramp_gap)
        self.assertIn("half_cell_shift", d)
        self.assertTrue(d["kind_preserved"])

    def test_half_cell_shift_is_reported(self):
        d = self.erosion.as_dict()
        self.assertEqual(d["half_cell_shift"]["phase"], [0.05, 0.0])
        self.assertIn("failed", d["half_cell_shift"])

    def test_refuses_non_failing_sample(self):
        from relic_nav.scene import repaired
        with self.assertRaises(ValueError):
            reduce_failure(self.scn, repaired(RadiusMisread(0.15)),
                           steps_per_axis=1)


class TestViewSamples(unittest.TestCase):
    def test_catalog_has_all_samples(self):
        ids = {s.id for s in catalog()}
        self.assertEqual(ids,
                         {"radius_misread", "erosion_shortfall", "erosion_phase"})

    def test_load_sample_by_id(self):
        s, faulted, control = load_sample("radius_misread",
                                          out_dir="/nonexistent")
        self.assertEqual(s.id, "radius_misread")
        self.assertTrue(faulted.failed)
        self.assertTrue(control.passed)

    def test_unknown_sample_raises(self):
        with self.assertRaises(KeyError):
            load_sample("nope", out_dir="/nonexistent")

    def test_sample_dict_has_continuous_dims(self):
        d = catalog()[0].as_dict()
        for key in ("door_width", "radius", "ramp_gap", "slant_deg"):
            self.assertIn(key, d["reduced"])

    def test_deliver_writes_panels_and_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            idx = deliver_view(out_dir=tmp)
            self.assertEqual(len(idx["samples"]), 3)
            for s in idx["samples"]:
                self.assertTrue(s["control_passable"])
                self.assertTrue(os.path.exists(
                    os.path.join(tmp, s["panel"])))
            self.assertTrue(os.path.exists(
                os.path.join(tmp, "reduce_samples.json")))
            # loader reads the delivered index
            s, f, c = load_sample("radius_misread", out_dir=tmp)
            self.assertTrue(f.failed)

    def test_view_draws_the_reduced_geometry(self):
        # the panel must load the scenario the reducer kept, not the
        # unreduced reference — otherwise "one-click load reduced sample"
        # would silently show the full-size door
        reduced = {"radius_misread": GateScenario(door_width=1.404,
                                                  radius=0.436,
                                                  ramp_gap=0.12)}
        sample = {s.id: s for s in catalog(reduced)}["radius_misread"]
        self.assertAlmostEqual(sample.reduced.door_width, 1.404)
        self.assertAlmostEqual(sample.reduced.radius, 0.436)

    def test_delivered_index_reloads_reduced_scenario(self):
        with tempfile.TemporaryDirectory() as tmp:
            reduced = {"radius_misread": GateScenario(door_width=1.404,
                                                      radius=0.436,
                                                      ramp_gap=0.12)}
            deliver_view(out_dir=tmp, reduced_by_id=reduced)
            s, faulted, control = load_sample("radius_misread", out_dir=tmp)
            # restored from the index, not re-tuned back to the reference
            self.assertAlmostEqual(s.reduced.door_width, 1.404, places=6)
            self.assertAlmostEqual(s.reduced.radius, 0.436, places=6)
            self.assertTrue(faulted.failed)
            self.assertTrue(control.passed)


if __name__ == "__main__":
    unittest.main()
