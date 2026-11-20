"""Independent swept-volume feasibility check for a planned channel.

The navigator and the driven character share the same solid segments, so a
route the planner emits can still be physically impossible for the real body
if the planner eroded with the wrong radius, or if sampling placed walkable
cells a hair too close to a jamb. This module re-derives feasibility without
trusting the navmesh: it sweeps a disc of *true* radius along the planned
polyline and measures the exact segment-to-segment distance to every solid.

Nothing here consults walkable cells, regions, portals or the funnel; the only
inputs are the polyline and the solid segments, which is what makes it an
independent check. Two distinct questions are answered:

  * geometric clearance along the channel (distance from the swept centreline
    to the solids), and
  * whether that clearance leaves room for the body radius.

Contact records name the polyline segment, the solid segment and the point of
closest approach, so a failure can be attributed to "the character's turn at
the jar corner" rather than to an opaque boolean.
"""

import math

from .geometry import (
    Vec, closest_point_on_seg, dist2_point_seg, segments_intersect,
)


def seg_seg_closest(a, b, c, d):
    """Closest approach between two segments.

    Returns (distance, point_on_ab, point_on_cd). Intersecting segments report
    distance 0 and the crossing point on both. Endpoint-to-segment distances
    cover the parallel/clamped cases exactly, so the result is the true
    Euclidean distance between the two finite segments."""
    denom = (b - a).cross(d - c)
    if abs(denom) > 1e-12:
        t = ((c - a).cross(d - c)) / denom
        u = ((c - a).cross(b - a)) / denom
        if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
            p = Vec(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y))
            return 0.0, p, p.copy()
    best = float("inf")
    pa = pb = None
    for p, q0, q1, flip in (
        (a, c, d, False), (b, c, d, False),
        (c, a, b, True), (d, a, b, True),
    ):
        cp = closest_point_on_seg(p, q0, q1)
        d2 = (cp - p).length_sq()
        if d2 < best:
            best = d2
            if flip:
                pa, pb = cp, p
            else:
                pa, pb = p, cp
    return math.sqrt(best), pa, pb


class Contact:
    """A closest-approach event between the swept centreline and a solid."""

    __slots__ = ("distance", "seg_index", "point", "solid", "solid_index",
                 "along_t")

    def __init__(self, distance, seg_index, point, solid, solid_index,
                 along_t):
        self.distance = distance
        self.seg_index = seg_index      # which polyline segment was swept
        self.point = point              # centreline point of closest approach
        self.solid = solid              # (a, b) wall/ledge segment
        self.solid_index = solid_index
        self.along_t = along_t           # 0..1 position along that segment

    def as_dict(self):
        a, b = self.solid
        return {
            "distance_m": round(self.distance, 5),
            "seg_index": self.seg_index,
            "point": [round(self.point.x, 5), round(self.point.y, 5)],
            "solid": [[round(a.x, 5), round(a.y, 5)],
                      [round(b.x, 5), round(b.y, 5)]],
            "solid_index": self.solid_index,
            "along_segment": round(self.along_t, 4),
        }


class SweepResult:
    def __init__(self, radius, min_contact, contacts, length):
        self.radius = radius
        self.min_contact = min_contact
        self.contacts = contacts
        self.length = length

    @property
    def min_clearance(self):
        return self.min_contact.distance if self.min_contact else float("inf")

    @property
    def feasible(self):
        # tolerance matches the movement capsule test: a few microns of grid
        # quantisation must not read as a scrape
        return self.min_clearance + 1e-6 >= self.radius

    def as_dict(self):
        return {
            "radius": self.radius,
            "length_m": round(self.length, 5),
            "min_clearance_m": (round(self.min_clearance, 5)
                                if self.min_contact else None),
            "feasible": self.feasible,
            "min_contact": (self.min_contact.as_dict()
                            if self.min_contact else None),
            "contacts": [c.as_dict() for c in self.contacts],
        }


def _bbox_gap(a0x, a0y, a1x, a1y, b0x, b0y, b1x, b1y):
    dx = max(0.0, max(a0x, b0x) - min(a1x, b1x))
    dy = max(0.0, max(a0y, b0y) - min(a1y, b1y))
    return math.hypot(dx, dy)


def _contacts_for_segment(a, b, solids, cand, si, radius):
    seg_len = (b - a).length()
    if seg_len < 1e-12:
        return []
    out = []
    for wi in cand:
        w0, w1 = solids[wi]
        dist, pa, _pb = seg_seg_closest(a, b, w0, w1)
        t = max(0.0, min(1.0, (pa - a).dot(b - a) / (seg_len * seg_len)))
        out.append(Contact(dist, si, pa, (w0, w1), wi, t))
    return out


def sweep_polyline(poly, solids, radius, sample_step=None):
    """Sweep a disc of `radius` along `poly` against every solid segment.

    Uses the exact segment-to-segment distance, which is the true minimum
    clearance of the swept disc corridor (the disc radius is then compared
    against it). A bounding-box prefilter keeps the large ledge-segment list
    cheap; if no candidate lies within the query band it widens to a full
    scan, so the reported minimum is always exact."""
    if not poly or len(poly) < 2:
        return SweepResult(radius, None, [], 0.0)

    xs = [p.x for p in poly]
    ys = [p.y for p in poly]
    bx0, by0, bx1, by1 = min(xs), min(ys), max(xs), max(ys)

    contacts = []
    length = 0.0
    for si, (a, b) in enumerate(zip(poly, poly[1:])):
        length += (b - a).length()

    # widen the query until either the closest candidate is provably global
    # (its distance is below the query band) or we fall back to all solids
    for pad in (max(radius * 3.0, 1.5), max(radius * 10.0, 6.0), None):
        if pad is None:
            cand = range(len(solids))
        else:
            cand = [i for i, (a, b) in enumerate(solids)
                    if _bbox_gap(bx0, by0, bx1, by1,
                                 min(a.x, b.x), min(a.y, b.y),
                                 max(a.x, b.x), max(a.y, b.y)) <= pad]
        contacts = []
        for si, (a, b) in enumerate(zip(poly, poly[1:])):
            contacts.extend(_contacts_for_segment(a, b, solids, cand, si,
                                                  radius))
        if contacts and pad is not None:
            best = min(c.distance for c in contacts)
            if best <= pad:
                break
        elif contacts and pad is None:
            break
        elif not contacts:
            continue

    min_contact = min(contacts, key=lambda c: c.distance) if contacts else None
    return SweepResult(radius, min_contact, contacts, length)


def disc_clearance(poly, solids, radius, sample_step=None):
    """Discretised disc sweep: minimum clearance seen at sampled centreline
    points, plus the sampler that produced it. Used as a second opinion on the
    continuous segment distance and to expose any sampling dependence."""
    if not poly or len(poly) < 2:
        return float("inf"), None, 0
    if sample_step is None:
        sample_step = max(radius * 0.25, 1e-3)
    best = float("inf")
    best_at = None
    n_samples = 0
    for si, (a, b) in enumerate(zip(poly, poly[1:])):
        seg_len = (b - a).length()
        steps = max(1, int(math.ceil(seg_len / sample_step)))
        for k in range(steps + 1):
            t = k / steps
            p = Vec(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t)
            n_samples += 1
            for wi, (w0, w1) in enumerate(solids):
                d2 = dist2_point_seg(p.x, p.y, w0.x, w0.y, w1.x, w1.y)
                if d2 < best:
                    best = d2
                    best_at = (si, t, p, wi)
    return math.sqrt(best), best_at, n_samples


def turn_contact(sweep_result, turn_point, radius):
    """Closest-contact record nearest a nominated turn point.

    When a failure is driven by the character turning at a corner, the
    offending contact is the one whose centreline point is nearest the turn.
    Returning it explicitly lets the reducer say "this scrape is the turn at
    this jar corner", not merely "the route is infeasible"."""
    if sweep_result.min_contact is None:
        return None
    best = None
    best_d = float("inf")
    for c in sweep_result.contacts:
        if c.distance > radius + 1e-6:
            continue
        d = (c.point - turn_point).length()
        if d < best_d:
            best_d = d
            best = c
    return best


def polyline_turns(poly, min_angle_deg=20.0):
    """Index/point/angle of every interior vertex that changes heading by at
    least `min_angle_deg`. A ramp entry next to a slanted doorway produces one
    sharp turn; that is the vertex the fault analysis wants to name."""
    out = []
    for i in range(1, len(poly) - 1):
        u = (poly[i] - poly[i - 1])
        v = (poly[i + 1] - poly[i])
        if u.length() < 1e-9 or v.length() < 1e-9:
            continue
        cosang = max(-1.0, min(1.0, u.normalised().dot(v.normalised())))
        angle = math.degrees(math.acos(cosang))
        if angle >= min_angle_deg:
            out.append((i, poly[i], angle))
    return out
