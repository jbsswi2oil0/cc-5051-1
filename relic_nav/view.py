"""Level view for reduced failure samples.

The reducer emits small, still-failing gate samples. This module is the
"one-click load" surface for them: a catalog of samples keyed by id, each of
which can be evaluated and drawn as a panel that overlays exactly the evidence
the reduction was justified with — the independent sweep's first-contact
position, the edges the planner accepted (portal spans), and, alongside the
faulted door, a same-width repaired control door that stays passable.

Panels are written as PNGs plus a small index JSON. Loading a sample is a
single ``load_sample`` call by id; the viewer never re-tunes the geometry.
"""

import json
import os

from .geometry import (
    Vec, closest_point_on_seg, segments_intersect, segment_intersection,
)
from .scene import (
    GateScenario, ErosionShortfall, RadiusMisread, repaired,
    build_gate_level, evaluate, phase_shift,
)

_SCENARIO_KEYS = ("door_width", "slant_deg", "door_x", "door_y", "radius",
                  "ramp_gap", "ramp_y0", "ramp_y1", "ramp_len", "ramp_rise",
                  "phase_x", "phase_y")
from .image import Canvas
from . import render


C_CONTACT = (255, 90, 90)
C_EDGE = (120, 255, 190)
C_CONTROL = (120, 190, 255)
C_TURN = (255, 210, 90)


class Sample:
    """One loadable failure sample: a scenario, the fault that breaks it, the
    reduced variant the reducer produced, and the control key."""

    def __init__(self, sid, title, scenario, fault, reduced=None, note=""):
        self.id = sid
        self.title = title
        self.scenario = scenario
        self.fault = fault
        self.reduced = reduced or scenario
        self.note = note

    def load(self, cell_size=0.1, drive=False):
        """One call: evaluate the reduced sample and its same-width control."""
        faulted = evaluate(self.reduced, self.fault, cell_size=cell_size,
                           drive=drive)
        control = evaluate(self.reduced, repaired(self.fault),
                           cell_size=cell_size, drive=drive)
        return faulted, control

    def as_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "note": self.note,
            "fault": self.fault.name,
            "fault_cause": self.fault.describe(),
            "scenario": _scenario_dict(self.scenario),
            "reduced": _scenario_dict(self.reduced),
        }


def _scenario_dict(scn):
    return {k: getattr(scn, k) for k in _SCENARIO_KEYS}


def _scenario_from_dict(d):
    return GateScenario(**{k: d[k] for k in _SCENARIO_KEYS if k in d})


def catalog(reduced_by_id=None):
    """The loadable samples. Widths here are chosen so each is a genuinely
    narrow door that the repaired control still passes — the control is the
    same width, so a pass there isolates the fault as the cause.

    ``reduced_by_id`` maps a sample id to the scenario the reducer shrank it
    to; when given, the panel and the one-click load show that smaller sample
    rather than the unreduced reference."""
    reduced_by_id = reduced_by_id or {}
    base = GateScenario()
    # radius misread: sweep-dependent, fails at every phase
    misread = Sample(
        "radius_misread", "RADIUS MISREAD / SWEEP DEPENDENT", base,
        RadiusMisread(0.15),
        reduced=reduced_by_id.get("radius_misread"),
        note="planner reads the collider small; every phase touches the "
             "jamb-high corner at the turn")
    # erosion shortfall, literal one cell: sweep-dependent
    erosion = Sample(
        "erosion_shortfall", "EROSION SHORTFALL (ONE CELL) / SWEEP DEPENDENT",
        base, ErosionShortfall(1.0),
        reduced=reduced_by_id.get("erosion_shortfall"),
        note="walkable surface eroded one cell short; best cell centre still "
             "inside the true radius at every phase")
    # marginal erosion: phase-dependent — the failure depends on sampling
    phase = Sample(
        "erosion_phase", "EROSION SHORTFALL 0.4 CELL / PHASE DEPENDENT",
        base, ErosionShortfall(0.4),
        reduced=reduced_by_id.get("erosion_phase"),
        note="a shallower erosion that only scrapes under some raster "
             "phases; the failure depends on the sampling layout")
    return [misread, erosion, phase]


def _overlay_edges(cv, cam, edges, poly, color=C_EDGE, band=0.28,
                   dedupe=0.6):
    """Show the accepted portal near the point the route crosses it.

    The rectangle partition emits portals that can span the room's full
    height; drawing them whole buries the map. Only the span actually used at
    the crossing matters, so each accepted portal is trimmed to a small window
    around its intersection with the route, and crossings packed closer than
    ``dedupe`` metres collapse into one mark."""
    if not poly:
        return
    hits = []
    for a, b in edges:
        ma, mb = Vec(*a), Vec(*b)
        for p, q in zip(poly, poly[1:]):
            if segments_intersect(ma, mb, p, q):
                hit = segment_intersection(ma, mb, p, q)
                if hit is not None:
                    hits.append((ma, mb, hit))
    hits.sort(key=lambda t: _along_route(t[2], poly))
    drawn = []
    for ma, mb, hit in hits:
        if any((hit - prev).length() < dedupe for prev in drawn):
            continue
        drawn.append(hit)
        d = (mb - ma).normalised()
        cv.line(cam.xy(hit - d * band), cam.xy(hit + d * band), color, 2)


def _along_route(p, poly):
    acc = 0.0
    for a, b in zip(poly, poly[1:]):
        q = closest_point_on_seg(p, a, b)
        if (q - p).length() < 1e-6:
            return acc + (q - a).length()
        acc += (b - a).length()
    return acc


def _overlay_contact(cv, cam, contact, color=C_CONTACT):
    if contact is None:
        return
    p = contact.point
    x, y = cam.xy(p)
    cv.circle(x, y, 7, outline=color, width=2)
    cv.line((x - 10, y), (x + 10, y), color, 1)
    cv.line((x, y - 10), (x, y + 10), color, 1)


def _overlay_route(cv, cam, poly, color=render.C_ROUTE):
    if not poly:
        return
    render.draw_route(cv, cam, poly, color=color, width=2)


def _focus_bounds(scn, pad=0.9):
    """Bounding box of the doorway, jamb span, ramp foot and turn, so the
    panel shows the corner detail at a useful scale rather than the whole
    room. The route is clipped by the camera; the evidence is local."""
    xs, ys = [], []
    dc = scn.door_center()
    xs.append(dc.x); ys.append(dc.y)
    for p in scn.jamb_points():
        xs.append(p.x); ys.append(p.y)
    rx0, ry0, rx1, ry1 = scn.ramp_rect()
    xs += [rx0, rx1]; ys += [ry0, ry1]
    tp = scn.turn_reference()
    xs.append(tp.x); ys.append(tp.y)
    return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)


def draw_sample_panel(sample, path, cell_size=0.1, panel_w=460, panel_h=380):
    """Faulted vs control, same width, side by side in one panel.

    Left: the faulted door with the planner route, accepted edges and the
    first-contact position. Right: the repaired control at identical
    continuous dimensions, with its (feasible) route. The comparison is the
    point — same width, only the fault differs."""
    faulted, control = sample.load(cell_size=cell_size, drive=True)
    lv = build_gate_level(sample.reduced)
    focus = _focus_bounds(sample.reduced)
    # half a navmesh cell of rigid translation, door and character together;
    # sizes are unchanged, so only the raster phase differs
    shifted = evaluate(phase_shift(sample.reduced, 0.05, 0.0), sample.fault,
                       cell_size=cell_size, drive=False)

    w, h = panel_w * 2, panel_h + 22
    cv = Canvas(w, h)
    cv.clear(render.C_BG)
    cv.text(sample.title, 6, 4, render.C_TEXT, scale=1)
    shift_txt = ("+0.5 CELL SHIFT: %s"
                 % ("STILL FAILS" if shifted.failed else "PASSES"))
    cv.text(shift_txt, 6, 15, C_TURN if shifted.failed else render.C_TEXT,
            scale=1)

    def draw_view(dx, res, title, verdict, vcolor, show_contact):
        sub = Canvas(panel_w, panel_h)
        sub.clear(render.C_BG)
        cam = render.Camera(focus, panel_w, panel_h, margin=14)
        render.draw_level(sub, cam, lv)
        _overlay_route(sub, cam, res.polyline,
                       render.C_ROUTE if show_contact else C_CONTROL)
        _overlay_edges(sub, cam, res.planner_edges, res.polyline)
        if show_contact:
            _overlay_contact(sub, cam, res.first_contact)
        turn_pt = (res.turn_anchor.point if res.turn_anchor is not None
                   else (res.turns[0][1] if res.turns else None))
        if turn_pt is not None:
            x, y = cam.xy(turn_pt)
            sub.circle(x, y, 4, outline=C_TURN, width=1)
            if show_contact:
                sub.text("TURN", x + 6, y + 3, C_TURN, scale=1)
        sub.text(title, 6, 4, render.C_TEXT, scale=1)
        sub.text(verdict, 6, 15, vcolor, scale=1)
        for yy in range(sub.h):
            base = (22 + yy) * w + dx
            cv.pixels[base:base + sub.w] = sub.pixels[yy * sub.w:(yy + 1) * sub.w]

    def verdict(res, good):
        if res.sweep is None:
            return "NO SWEEP", C_CONTACT
        state = "PASS" if res.sweep.feasible else "FAIL"
        col = C_CONTROL if good else C_CONTACT
        return ("%s  R=%.2f  CLEAR=%.2f" % (state, res.scenario.radius,
                                            res.sweep.min_clearance), col)

    draw_view(0, faulted, "FAULTED: " + sample.fault.name.upper(),
              *verdict(faulted, False), show_contact=True)
    draw_view(panel_w, control, "CONTROL: SAME WIDTH, REPAIRED",
              *verdict(control, True), show_contact=False)

    cv.save_png(path)
    return faulted, control


def _panel_summary(res):
    """Compact evidence record for the index: verdict, clearance, contact and
    the route/edges, without the full per-solid contact list (which belongs in
    the reducer report, not the loadable catalog)."""
    return {
        "fault": res.fault.name,
        "failed": res.failed,
        "passed": res.passed,
        "min_clearance_m": (round(res.sweep.min_clearance, 5)
                            if res.sweep else None),
        "radius": res.scenario.radius,
        "contact_on_jamb": res.contact_on_jamb,
        "first_contact": (res.first_contact.as_dict()
                          if res.first_contact else None),
        "turn_anchor": (res.turn_anchor.as_dict()
                        if res.turn_anchor else None),
        "route": [[round(p.x, 4), round(p.y, 4)] for p in res.polyline],
        "planner_edges": [list(e) for e in res.planner_edges],
        "driven_status": res.driven.status.value if res.driven else None,
        "driven_collisions": res.driven.collision_count if res.driven else None,
    }


def deliver_view(out_dir="artifacts", cell_size=0.1, reduced_by_id=None):
    """Write every sample's panel and the catalog index that the loader reads.

    ``reduced_by_id`` lets the caller pass in the scenarios produced by the
    reducer so the delivered panels and the on-disk index describe the shrunken
    samples, not the unreduced reference."""
    os.makedirs(out_dir, exist_ok=True)
    samples = catalog(reduced_by_id)
    index = {"samples": [], "cell_size": cell_size}
    for s in samples:
        png = os.path.join(out_dir, f"reduce_{s.id}.png")
        faulted, control = draw_sample_panel(s, png, cell_size=cell_size)
        entry = s.as_dict()
        entry["panel"] = os.path.basename(png)
        entry["faulted"] = _panel_summary(faulted)
        entry["control"] = _panel_summary(control)
        entry["control_passable"] = control.passed
        index["samples"].append(entry)
        print(f"{s.id:20s} faulted_failed={faulted.failed} "
              f"control_passed={control.passed} -> {png}")
    with open(os.path.join(out_dir, "reduce_samples.json"), "w") as fh:
        json.dump(index, fh, indent=2)
    return index


def load_sample(sid, out_dir="artifacts", cell_size=0.1):
    """One-click load: look up a sample by id from the delivered index (or the
    in-memory catalog), evaluate it, and return (sample, faulted, control).

    When an index is present the sample's reduced scenario is restored from
    the file, so a reload reproduces exactly the delivered shrunken geometry
    without re-running the reducer."""
    path = os.path.join(out_dir, "reduce_samples.json")
    if os.path.exists(path):
        with open(path) as fh:
            data = json.load(fh)
        by_id = {s.id: s for s in catalog()}
        for entry in data["samples"]:
            if entry["id"] != sid:
                continue
            s = by_id.get(sid)
            if s is None:
                raise KeyError(f"no reduced sample {sid!r}")
            if entry.get("reduced"):
                s = Sample(s.id, s.title, s.scenario, s.fault,
                           reduced=_scenario_from_dict(entry["reduced"]),
                           note=s.note)
            f, c = s.load(cell_size=cell_size, drive=True)
            return s, f, c
        raise KeyError(f"no reduced sample {sid!r}")
    for s in catalog():
        if s.id == sid:
            f, c = s.load(cell_size=cell_size, drive=True)
            return s, f, c
    raise KeyError(f"no reduced sample {sid!r}")
