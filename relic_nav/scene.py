"""Diagonal-door test scene, fault models, phase experiments and evaluation.

    room (one floor) --[ slanted doorway ]--> ramp --> upper landing

A single slanted two-jamb wall cuts the room in two. The opening is the only
way through; a ramp sits just past it, dropped below the door centre line so a
body that crosses the opening and then turns toward the ramp carries its
radius across the lower/upper jamb tip. The straight centre line through the
opening is passable on its own — the failure only appears once a radius is
carried through the turn.

Two faults are modelled. Both leave every *continuous* dimension intact and
only corrupt how the planner reasons:

``ErosionShortfall``
    Walkable cells are eroded short of the body radius (by default four
    tenths of a raster cell), so cells the body cannot actually occupy are
    declared walkable. The planner's own corner verification trusts the same
    short radius; a fault that verified corners at the true radius would
    silently repair itself and never reach the emitted route.

``RadiusMisread``
    The planner believes the collider is smaller than physics uses.

``evaluate`` builds the planner mesh under the fault, sweeps the planned
channel with an independent disc of the *true* radius, and drives the body
along the same channel with the true radius against the shared solids. A board
*fails* when the sweep finds the channel too tight; the driven run supplies
first-contact evidence so the failure can be localised to the turn.

``classify`` runs the same fault over a set of rigid raster phases (the slanted
door and the character translated together by a fraction of a cell). If the
verdict changes with phase while the continuous layout is identical, the
failure depends on the sampling layout; if the sweep stays infeasible at every
phase, the failure depends on the swept volume and is reported against the
turn and wall-corner contact.
"""

import math
from dataclasses import dataclass, field, replace

from .geometry import Vec, dist2_point_seg
from .level import Level, Surface, Wall, FLOOR, RAMP, CharacterSpec
from .navmesh import NavMesh, RouteStatus
from .runner import follow_route
from .sweep import sweep_polyline, polyline_turns, turn_contact


# --------------------------------------------------------------- scene model
@dataclass
class GateScenario:
    """Continuous parameters of the diagonal-door scene.

    The raster phase is expressed separately via ``phase_x/phase_y`` so a
    phase experiment never changes the size relationships: the slanted door
    and the character translate together, the room floor stays put."""

    door_width: float = 1.5
    slant_deg: float = 40.0
    door_x: float = 9.0
    door_y: float = 3.4
    radius: float = 0.5
    ramp_gap: float = 0.6          # foot of the ramp past the door centre
    ramp_y0: float = 2.2
    ramp_y1: float = 4.8
    ramp_len: float = 2.8
    ramp_rise: float = 1.1
    room: tuple = (0.0, 0.0, 18.0, 9.0)
    start_x: float = 2.5
    start_y: float = 5.5
    # rigid translation of the door + character relative to the fixed raster
    phase_x: float = 0.0
    phase_y: float = 0.0

    def door_center(self):
        return Vec(self.door_x + self.phase_x, self.door_y + self.phase_y)

    def wall_direction(self):
        """Unit direction of the jamb line (A->B)."""
        a = math.radians(self.slant_deg)
        return Vec(math.sin(a), math.cos(a))

    def jamb_points(self):
        c = self.door_center()
        d = self.wall_direction()
        return c - d * (self.door_width / 2.0), c + d * (self.door_width / 2.0)

    def ramp_rect(self):
        x0 = self.door_x + self.phase_x + self.ramp_gap
        return (x0, self.ramp_y0, x0 + self.ramp_len, self.ramp_y1)

    def start_point(self):
        return Vec(self.start_x + self.phase_x, self.start_y + self.phase_y)

    def goal_point(self):
        x0, y0, x1, y1 = self.ramp_rect()
        return Vec(self.room[2] - 2.0, (y0 + y1) / 2.0)

    def turn_reference(self):
        """Where the body is expected to turn into the ramp."""
        x0, y0, x1, y1 = self.ramp_rect()
        return Vec(x0, (y0 + y1) / 2.0)


def _stretch_to_room(p, direction, room):
    """Extend a jamb from p along `direction` until it leaves the room box."""
    x0, y0, x1, y1 = room
    ts = []
    if abs(direction.y) > 1e-9:
        for edge in (y0, y1):
            t = (edge - p.y) / direction.y
            q = p + direction * t
            if x0 - 1e-6 <= q.x <= x1 + 1e-6 and t > 0:
                ts.append(t)
    if abs(direction.x) > 1e-9:
        for edge in (x0, x1):
            t = (edge - p.x) / direction.x
            q = p + direction * t
            if y0 - 1e-6 <= q.y <= y1 + 1e-6 and t > 0:
                ts.append(t)
    return p + direction * (min(ts) if ts else 14.0)


def build_gate_level(scn):
    """One room split by a slanted two-jamb doorway, plus a ramp just past it."""
    lv = Level("diagonal_gate")
    x0, y0, x1, y1 = scn.room
    lv.add_surface(Surface("floor", (x0, y0, x1, y1), FLOOR, 0.0))

    rx0, ry0, rx1, ry1 = scn.ramp_rect()
    lv.add_surface(Surface("ramp", (rx0, ry0, rx1, ry1), RAMP,
                           0.0, scn.ramp_rise, axis="x"))
    lv.add_surface(Surface("upper", (rx1, ry0, x1, ry1), FLOOR, scn.ramp_rise))

    lo, hi = scn.jamb_points()
    d = scn.wall_direction()
    a = _stretch_to_room(lo, -d, scn.room)
    b = _stretch_to_room(hi, d, scn.room)
    lv.add_wall(Wall("jamb_low", a, lo))
    lv.add_wall(Wall("jamb_high", hi, b))

    lv.compute_bounds()
    lv.markers["start"] = scn.start_point()
    lv.markers["goal"] = scn.goal_point()
    lv.markers["door"] = scn.door_center()
    lv.markers["jamb_low"] = lo
    lv.markers["jamb_high"] = hi
    lv.markers["turn"] = scn.turn_reference()
    return lv


# --------------------------------------------------------------- fault models
class Fault:
    """A corruption of how the planner reasons, with an explicit repair so a
    same-width passable control can be evaluated without it."""

    name = "fault"
    knob = ""

    def planner_params(self, radius, cell_size):
        """Return (planner_radius, erosion_margin)."""
        raise NotImplementedError

    def physics_radius(self, radius):
        return radius

    def describe(self):
        raise NotImplementedError


class ErosionShortfall(Fault):
    name = "erosion_shortfall"
    knob = "erosion_margin"

    def __init__(self, cells=1.0):
        self.cells = float(cells)

    def planner_params(self, radius, cell_size):
        return radius, -self.cells * cell_size

    def describe(self):
        return (f"walkable surface eroded {self.cells:g} cell short of the "
                f"character radius")


class RadiusMisread(Fault):
    name = "radius_misread"
    knob = "planner_radius"

    def __init__(self, underread=0.15):
        self.underread = float(underread)

    def planner_params(self, radius, cell_size):
        return max(1e-3, radius - self.underread), 0.0

    def describe(self):
        return (f"planner reads collider radius {self.underread:.2f} m "
                f"smaller than physics uses")


class FaultCorrection(Fault):
    """Repair of ``fault`` that leaves every continuous dimension untouched."""

    def __init__(self, fault):
        self.fault = fault
        self.name = "repaired_" + fault.name
        self.knob = fault.knob

    def planner_params(self, radius, cell_size):
        return radius, 0.0

    def describe(self):
        return f"repaired control ({self.fault.knob} correct)"


def repaired(fault):
    return FaultCorrection(fault)


# --------------------------------------------------------------- evaluation
@dataclass
class EvalResult:
    scenario: object
    fault: object
    cell_size: float
    ok: bool
    invalid_reason: str = ""
    planned: bool = False
    route_status: str = ""
    polyline: list = field(default_factory=list)
    planner_edges: list = field(default_factory=list)
    sweep: object = None
    driven: object = None
    turns: list = field(default_factory=list)
    first_contact: object = None
    turn_anchor: object = None
    contact_on_jamb: str = ""
    driven_contact_events: int = 0
    ramp_rect: tuple = ()

    @property
    def failed(self):
        """Fails when the independent sweep finds the planner's channel too
        tight for the true physics radius."""
        return (self.ok and self.planned and self.sweep is not None
                and not self.sweep.feasible)

    @property
    def passed(self):
        return (self.ok and self.planned and self.sweep is not None
                and self.sweep.feasible)

    def summary(self):
        return {
            "fault": self.fault.name,
            "failed": self.failed,
            "planned": self.planned,
            "route_status": self.route_status,
            "invalid_reason": self.invalid_reason,
            "min_clearance_m": (round(self.sweep.min_clearance, 5)
                                if self.sweep else None),
            "radius": self.scenario.radius,
            "first_contact": (self.first_contact.as_dict()
                              if self.first_contact else None),
            "turn_anchor": (self.turn_anchor.as_dict()
                            if self.turn_anchor else None),
            "contact_on_jamb": self.contact_on_jamb,
            "driven_status": (self.driven.status.value
                              if self.driven is not None else None),
            "driven_collisions": (self.driven.collision_count
                                  if self.driven is not None else None),
            "driven_contact_events": self.driven_contact_events,
            "polyline": [[round(p.x, 4), round(p.y, 4)]
                         for p in self.polyline],
            "planner_edges": [list(e) for e in self.planner_edges],
            "sweep": self.sweep.as_dict() if self.sweep else None,
        }


def _planner_edges(route):
    """Portal spans the planner accepted along its corridor, as world
    segments; the viewer overlays these to show what the navigator committed
    to."""
    edges = []
    for left, right in route.get("portals", []):
        if (left - right).length() < 1e-9:
            continue
        edges.append(((round(left.x, 4), round(left.y, 4)),
                      (round(right.x, 4), round(right.y, 4))))
    return edges


def _attribute_jamb(contact, lv, tolerance):
    """Name the jamb the contact sits at, if any.

    The navmesh emits boundary ledges for eroded cells as well as the analytic
    jamb segments, so a contact is attributed to a jamb when its centreline
    point is within `tolerance` of either jamb tip (the door-frame corners the
    turn wraps around)."""
    if contact is None:
        return ""
    p = contact.point
    best, best_d = "", float("inf")
    for name in ("jamb_low", "jamb_high"):
        d = (p - lv.markers[name]).length()
        if d < best_d:
            best, best_d = name, d
    return best if best_d <= tolerance else ""


def _driven_contacts_near(driven, anchor, radius, band=0.35):
    if driven is None or anchor is None:
        return 0
    n = 0
    for f in driven.frames:
        if not f.collision:
            continue
        if (Vec(f.x, f.y) - anchor).length() <= radius + band:
            n += 1
    return n


def evaluate(scn, fault, cell_size=0.1, snap_tol=1.6, max_time=30.0,
             drive=True):
    """Build planner mesh under ``fault``, verify the channel independently at
    the true radius, and drive the body along it.

    ``drive=False`` skips the driven pass (which only supplies contact
    evidence); the sweep verdict is unchanged, so bulk reduction searches can
    run several times faster."""
    lv = build_gate_level(scn)
    plan_r, erosion = fault.planner_params(scn.radius, cell_size)
    true_r = fault.physics_radius(scn.radius)

    planner_spec = CharacterSpec("planner", plan_r, 50.0, 0.35, 2.2)
    nav = NavMesh(lv, planner_spec, cell_size=cell_size,
                  erosion_margin=erosion)

    s_status, s_point, _ = nav.locate(lv.markers["start"], snap_tol=snap_tol)
    g_status, g_point, _ = nav.locate(lv.markers["goal"], snap_tol=snap_tol)
    if s_point is None or g_point is None:
        return EvalResult(scn, fault, cell_size, ok=False,
                          invalid_reason=f"locate:{s_status.value}/{g_status.value}")

    route = nav.find_route(s_point, g_point)
    if route["status"] != RouteStatus.OK:
        return EvalResult(scn, fault, cell_size, ok=True, planned=False,
                          route_status=route["status"].value)

    poly = route["polyline"]
    sweep = sweep_polyline(poly, nav.solids, true_r)
    turns = polyline_turns(poly)

    # localise to the turn nearest the ramp foot
    turn_anchor = None
    if turns:
        ref = scn.turn_reference()
        nearest = min(turns, key=lambda t: (t[1] - ref).length())
        turn_anchor = turn_contact(sweep, nearest[1], true_r)
    if turn_anchor is None:
        turn_anchor = sweep.min_contact
    jamb = _attribute_jamb(sweep.min_contact, lv, true_r + 0.3)

    driven = None
    events = 0
    if drive:
        # drive the same channel with the true radius against shared solids
        phys_spec = CharacterSpec("physics", true_r, 50.0, 0.35, 2.2)
        phy_nav = NavMesh(lv, phys_spec, cell_size=cell_size)
        _, ps, _ = phy_nav.locate(s_point, snap_tol=0.5)
        _, pg, _ = phy_nav.locate(g_point, snap_tol=0.5)
        driven = follow_route(phy_nav, route, ps or s_point, pg or g_point,
                              max_time=max_time)
        anchor_pt = turn_anchor.point if turn_anchor is not None else None
        events = _driven_contacts_near(driven, anchor_pt, true_r)

    return EvalResult(
        scenario=scn, fault=fault, cell_size=cell_size, ok=True, planned=True,
        route_status=route["status"].value, polyline=poly,
        planner_edges=_planner_edges(route), sweep=sweep, driven=driven,
        turns=turns, first_contact=sweep.min_contact,
        turn_anchor=turn_anchor, contact_on_jamb=jamb,
        driven_contact_events=events, ramp_rect=scn.ramp_rect(),
    )


# --------------------------------------------------------------- phase study
DEFAULT_PHASES = (
    (0.0, 0.0), (0.05, 0.0), (0.0, 0.05), (0.05, 0.05),
    (-0.05, 0.0), (0.10, 0.10), (-0.05, 0.05),
)


def phase_shift(scn, dx, dy):
    """Rigidly translate the slanted door and the character by (dx, dy).

    Continuous dimensions (door width, slant, ramp relationship, radii) are
    untouched; only the door/character group moves relative to the fixed
    raster, which is exactly what changes mesh phase."""
    return replace(scn, phase_x=scn.phase_x + dx, phase_y=scn.phase_y + dy)


@dataclass
class PhaseStudy:
    scenario: object
    fault: object
    cell_size: float
    results: list = field(default_factory=list)

    @property
    def verdicts(self):
        return [("fail" if r.failed else
                 ("pass" if r.passed else "invalid")) for r in self.results]

    @property
    def all_invalid(self):
        return all(v == "invalid" for v in self.verdicts)

    @property
    def depends_on_phase(self):
        """The same continuous layout passes under some raster phases and
        fails under others."""
        v = set(self.verdicts)
        return "pass" in v and "fail" in v

    @property
    def depends_on_sweep(self):
        """Infeasible for every raster phase, so the failure is geometric,
        not a sampling artefact."""
        v = self.verdicts
        return bool(v) and all(x == "fail" for x in v)

    @property
    def stable_pass(self):
        v = self.verdicts
        return bool(v) and all(x == "pass" for x in v)

    def worst(self):
        """The failing result with the least clearance; used to name the
        contact when the failure is sweep-dependent."""
        fails = [r for r in self.results if r.failed]
        if not fails:
            return None
        return min(fails, key=lambda r: r.sweep.min_clearance)

    def as_dict(self):
        return {
            "fault": self.fault.name,
            "cell_size": self.cell_size,
            "verdicts": self.verdicts,
            "phases": [[r.scenario.phase_x, r.scenario.phase_y]
                       for r in self.results],
            "depends_on_phase": self.depends_on_phase,
            "depends_on_sweep": self.depends_on_sweep,
            "stable_pass": self.stable_pass,
            "per_phase": [r.summary() for r in self.results],
        }


def classify(scn, fault, cell_size=0.1, phases=DEFAULT_PHASES, snap_tol=1.6,
             drive=False):
    """Run the faulted scenario across rigid raster phases and decide whether
    the failure is a sampling-layout artefact or a swept-volume one.

    The driven pass is off by default here: the verdict is the independent
    sweep's, and a phase study only needs the geometry. Turn on ``drive`` when
    per-phase contact evidence is wanted."""
    results = []
    for dx, dy in phases:
        results.append(evaluate(phase_shift(scn, dx, dy), fault,
                                cell_size=cell_size, snap_tol=snap_tol,
                                drive=drive))
    return PhaseStudy(scn, fault, cell_size, results)


def diagnosis(study):
    """Turn a :class:`PhaseStudy` into the report the reducer must emit."""
    if study.all_invalid:
        return {
            "kind": "invalid",
            "detail": "the scenario did not produce a plannable channel at "
                      "any raster phase",
        }
    if study.stable_pass:
        return {"kind": "passes",
                "detail": "the planner's channel is feasible for the true "
                          "radius at every phase"}
    if study.depends_on_phase:
        # name the phases that flip so the sampling layout can be reproduced
        flips = [(r.scenario.phase_x, r.scenario.phase_y)
                 for r in study.results if r.failed]
        return {
            "kind": "phase_dependent",
            "detail": ("failure depends on raster phase: identical continuous "
                       "geometry passes under some phases and fails under "
                       "others"),
            "failing_phases": flips,
            "fault_cause": study.fault.describe(),
        }
    worst = study.worst()
    if worst is None:
        return {"kind": "passes", "detail": "no failing phase"}
    contact = worst.turn_anchor or worst.first_contact
    # name the turn nearest the offending contact, not merely the first bend
    # on the route: the scrape may sit at a later corner than the ramp entry
    turn_pt = None
    turn_angle = None
    if worst.turns and contact is not None:
        near = min(worst.turns, key=lambda t: (t[1] - contact.point).length())
        turn_pt, turn_angle = near[1], near[2]
    elif worst.turns:
        turn_pt, turn_angle = worst.turns[0][1], worst.turns[0][2]
    return {
        "kind": "sweep_dependent",
        "detail": ("the swept channel is too narrow for the true radius at "
                   "every phase; the body's radius crosses the turned-in "
                   "corner"),
        "fault_cause": study.fault.describe(),
        "turn_contact": contact.as_dict() if contact else None,
        "turn_point": ([round(turn_pt.x, 4), round(turn_pt.y, 4)]
                       if turn_pt is not None else None),
        "turn_angle_deg": round(turn_angle, 1) if turn_angle is not None else None,
        "contact_on_jamb": worst.contact_on_jamb,
        "min_clearance_m": round(worst.sweep.min_clearance, 5),
        "radius": worst.scenario.radius,
    }
