"""Automatic minimisation of a failing gate sample.

A failure is only useful as a test case if it can be shrunk: the same fault
should still break a smaller door, a smaller body and a turn pulled closer to
the corner. This module walks those three axes — geometry (door width), radius
and turn position (ramp-foot gap) — always downhill, and only accepts a step
when *all* of the following survive:

  * the faulted sample still fails (independent sweep, not the driver),
  * the failure still localises to the same wall corner / turn,
  * the repaired control at the same width still passes, so the reduced door
    remains a genuinely passable doorway that only the fault ruins,
  * the final sample still classifies into the same diagnosis kind as the
    original (sampling-layout dependence vs swept-volume dependence).

Nothing continuous is rewritten to make the numbers small: the scenario is a
real, runnable level at every accepted step, and the emitted report carries the
unreduced and reduced continuous dimensions side by side.
"""

from dataclasses import dataclass, field, replace

from .scene import (
    GateScenario, ErosionShortfall, RadiusMisread, repaired,
    evaluate, classify, diagnosis, phase_shift, DEFAULT_PHASES,
)


AXES = ("door_width", "radius", "ramp_gap")

# how far a contact may sit from the ramp-foot turn and still count as "the
# body's radius crossing the turned-in corner". Reducing the radius or pulling
# the turn closer slides the scrape between the jamb tip and the ramp-foot
# ledge; both are the same corner region, and a shrink that throws the scrape
# onto an unrelated wall is beyond this band.
TURN_BAND = 1.5


def _corner_key(res):
    """Human-readable label of the corner a failure localises to."""
    if res.contact_on_jamb:
        return res.contact_on_jamb
    if res.turn_anchor is not None and res.turns:
        return "turn"
    return ""


def _contact_point(res):
    if res.first_contact is not None:
        return res.first_contact.point
    if res.turn_anchor is not None:
        return res.turn_anchor.point
    return None


def _in_turn_region(res, scn, band=TURN_BAND):
    """The failure still fires where the body carries its radius across the
    door/ramp corner, rather than somewhere else on the map."""
    p = _contact_point(res)
    if p is None:
        return False
    return (p - scn.turn_reference()).length() <= band


def _lower_bound(scn, axis):
    """Physical floor a shrink axis may not cross."""
    if axis == "door_width":
        # a door narrower than the body has no passable centre line at all
        return 2.0 * scn.radius + 0.02
    if axis == "radius":
        return 0.18
    if axis == "ramp_gap":
        return 0.12
    raise KeyError(axis)


def _with(scn, axis, value):
    return replace(scn, **{axis: float(value)})


def _accept(candidate, fault, cell_size):
    """Return (ok, fault_result) for one reduced candidate.

    Faulted run first; the control is only evaluated when the fault still
    fires, because a candidate that repairs the bug is rejected regardless."""
    fr = evaluate(candidate, fault, cell_size=cell_size, drive=False)
    if not fr.failed or not _in_turn_region(fr, candidate):
        return False, fr
    cr = evaluate(candidate, repaired(fault), cell_size=cell_size, drive=False)
    if not cr.passed:
        return False, fr
    return True, fr


@dataclass
class ReductionStep:
    axis: str
    before: float
    after: float
    accepted: bool
    reason: str = ""


def _compact(res):
    """Verdict-level slice of an EvalResult for the reduction report.

    ``EvalResult.summary`` carries the full per-solid sweep contact list, which
    makes a multi-sample report megabytes wide. The reduction report only
    needs the verdict, the clearance, the attributed contact and the route, so
    the heavy contact list is dropped here."""
    if res is None:
        return None
    s = res.summary()
    if s.get("sweep"):
        s["sweep"] = {
            "radius": s["sweep"]["radius"],
            "length_m": s["sweep"]["length_m"],
            "min_clearance_m": s["sweep"]["min_clearance_m"],
            "feasible": s["sweep"]["feasible"],
            "min_contact": s["sweep"]["min_contact"],
        }
    return s


@dataclass
class Reduction:
    original: GateScenario
    fault: object
    cell_size: float
    base_result: object = None
    reduced: GateScenario = None
    reduced_result: object = None
    control_result: object = None
    base_diagnosis: dict = field(default_factory=dict)
    reduced_diagnosis: dict = field(default_factory=dict)
    steps: list = field(default_factory=list)
    corner: str = ""

    def as_dict(self):
        o, r = self.original, self.reduced
        return {
            "fault": self.fault.name,
            "fault_cause": self.fault.describe(),
            "cell_size": self.cell_size,
            "corner": self.corner,
            "base_diagnosis": self.base_diagnosis,
            "reduced_diagnosis": self.reduced_diagnosis,
            "kind_preserved": (self.base_diagnosis.get("kind")
                               == self.reduced_diagnosis.get("kind")),
            "base": _compact(self.base_result),
            "reduced": _compact(self.reduced_result),
            "control": _compact(self.control_result),
            "shrink": {
                axis: {
                    "from": getattr(o, axis),
                    "to": getattr(r, axis),
                    "ratio": round(getattr(r, axis) / getattr(o, axis), 4),
                }
                for axis in AXES
            },
            "steps": [vars(s) for s in self.steps],
            "half_cell_shift": {
                "phase": [0.05, 0.0],
                "failed": self._half_cell_failed(),
            },
        }

    def _half_cell_failed(self):
        """Whether the reduced sample still fails when the slanted door and the
        character are translated together by half a navmesh cell (0.05 m of a
        0.1 m cell), continuous sizes untouched."""
        res = evaluate(phase_shift(self.reduced, 0.05, 0.0), self.fault,
                       cell_size=self.cell_size, drive=False)
        return res.failed


def _shrink_axis(scn, axis, fault, cell_size, steps_per_axis, log,
                 trajectory):
    """Walk ``axis`` toward its floor, keeping the smallest accepted value.

    Linear stepping (not bisection) because a raster-phase flip can make the
    fail predicate non-monotone in a continuous parameter; each candidate is
    still an exact rerun, so a skipped minimum is impossible to report as
    accepted. Every accepted candidate is appended to ``trajectory`` so the
    caller can back off if the fully-reduced sample stops being the same kind
    of bug."""
    before = getattr(scn, axis)
    lo = _lower_bound(scn, axis)
    if before <= lo + 1e-9:
        log.append(ReductionStep(axis, before, before, False, "already minimal"))
        return scn
    best = scn
    for k in range(1, steps_per_axis + 1):
        val = before - (before - lo) * k / steps_per_axis
        cand = _with(best, axis, val)
        ok, fr = _accept(cand, fault, cell_size)
        if ok:
            best = cand
            trajectory.append(cand)
            log.append(ReductionStep(axis, before, round(val, 4), True))
        else:
            why = ("fault no longer fires" if not fr.failed
                   else "scrape left the turn" if not _in_turn_region(fr, cand)
                   else "control door blocked")
            log.append(ReductionStep(axis, before, round(val, 4), False, why))
    return best


def reduce_failure(scn, fault, cell_size=0.1, steps_per_axis=6,
                   phases=None, log=None):
    """Greedily shrink a failing sample along geometry, radius and turn
    position while preserving the fault cause."""
    log = [] if log is None else log
    base = evaluate(scn, fault, cell_size=cell_size)
    if not base.failed:
        raise ValueError("reduce_failure needs a failing sample; this one "
                         "passes or is invalid")
    corner = _corner_key(base)
    phases = DEFAULT_PHASES if phases is None else phases
    base_study = classify(scn, fault, cell_size=cell_size, phases=phases)
    base_diag = diagnosis(base_study)

    work = scn
    # geometry first (changes the jamb span), then radius, then the turn:
    # each later axis is re-searched against the already-reduced layout so the
    # reduction is monotone in every dimension
    trajectory = []
    for axis in AXES:
        work = _shrink_axis(work, axis, fault, cell_size, steps_per_axis,
                            log, trajectory)

    reduced_study = classify(work, fault, cell_size=cell_size, phases=phases)
    reduced_diag = diagnosis(reduced_study)
    if reduced_diag.get("kind") != base_diag.get("kind"):
        # The fully-shrunk sample stopped being the same kind of bug (e.g. a
        # reduction pushed a sweep-dependent scrape into phase dependence).
        # Rather than throw the whole reduction away, keep the most-reduced
        # checkpoint that still classifies like the original.
        work = scn
        reduced_study = base_study
        reduced_diag = base_diag
        for cand in trajectory:
            study = classify(cand, fault, cell_size=cell_size, phases=phases)
            diag = diagnosis(study)
            if diag.get("kind") == base_diag.get("kind"):
                work, reduced_study, reduced_diag = cand, study, diag
        log.append(ReductionStep(
            "diagnosis", 1.0, 0.0, work is not scn,
            "kept the most-reduced checkpoint that preserves the diagnosis "
            "kind" if work is not scn else
            "no reduced checkpoint preserved the diagnosis kind; keeping the "
            "unreduced sample"))
    else:
        log.append(ReductionStep("diagnosis", 1.0, 0.0,
                                 work is not scn, "kind preserved"))

    reduced_result = evaluate(work, fault, cell_size=cell_size)
    control = evaluate(work, repaired(fault), cell_size=cell_size)
    return Reduction(
        original=scn, fault=fault, cell_size=cell_size,
        base_result=base, reduced=work, reduced_result=reduced_result,
        control_result=control, base_diagnosis=base_diag,
        reduced_diagnosis=reduced_diag, steps=log, corner=corner,
    )


def default_faults():
    """The two injected faults at their literal severity.

    ``ErosionShortfall(1.0)`` is a walkable surface eroded exactly one raster
    cell short — the fault as specified. On the reference scene even the best
    cell centre it allows is inside the true radius, so it is *sweep*
    dependent: the reduced report names the turn / wall-corner contact.
    ``RadiusMisread(0.15)`` likewise fails at every raster phase."""
    return [RadiusMisread(0.15), ErosionShortfall(1.0)]


def phase_probe_fault():
    """A marginal erosion (four tenths of a cell) that is *only* fatal for
    some raster phases.

    This is not the specified one-cell fault; it exists to exercise the other
    branch of the reducer's report — a failure that depends on the sampling
    layout rather than the swept volume, whose reduction must say so and list
    the failing phases."""
    return ErosionShortfall(0.4)
