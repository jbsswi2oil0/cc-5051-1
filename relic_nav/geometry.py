"""2D geometry primitives used by the navigation pipeline.

World units are metres. Everything is plain float arithmetic so the same
routines can be used both when building the navmesh and when the driven
character resolves collisions (a single source of geometric truth)."""

import math

EPS = 1e-9


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ---------------------------------------------------------------- vectors
class Vec:
    __slots__ = ("x", "y")

    def __init__(self, x=0.0, y=0.0):
        self.x = float(x)
        self.y = float(y)

    @staticmethod
    def of(p):
        if isinstance(p, Vec):
            return p
        return Vec(p[0], p[1])

    def __add__(self, o):
        return Vec(self.x + o.x, self.y + o.y)

    def __sub__(self, o):
        return Vec(self.x - o.x, self.y - o.y)

    def __mul__(self, s):
        return Vec(self.x * s, self.y * s)

    __rmul__ = __mul__

    def __truediv__(self, s):
        return Vec(self.x / s, self.y / s)

    def dot(self, o):
        return self.x * o.x + self.y * o.y

    def cross(self, o):
        return self.x * o.y - self.y * o.x

    def length(self):
        return math.hypot(self.x, self.y)

    def length_sq(self):
        return self.x * self.x + self.y * self.y

    def normalised(self):
        n = self.length()
        if n < EPS:
            return Vec(0.0, 0.0)
        return Vec(self.x / n, self.y / n)

    def perp_left(self):
        # 90deg counter-clockwise
        return Vec(-self.y, self.x)

    def copy(self):
        return Vec(self.x, self.y)

    def __neg__(self):
        return Vec(-self.x, -self.y)

    def __eq__(self, o):
        return isinstance(o, Vec) and abs(self.x - o.x) < EPS and abs(self.y - o.y) < EPS

    def __iter__(self):
        yield self.x
        yield self.y

    def __repr__(self):
        return f"Vec({self.x:.4f},{self.y:.4f})"

    def to_tuple(self):
        return (self.x, self.y)


def cross(ax, ay, bx, by):
    return ax * by - ay * bx


def dist2_point_seg(px, py, ax, ay, bx, by):
    dx = bx - ax
    dy = by - ay
    l2 = dx * dx + dy * dy
    if l2 < EPS:
        return (px - ax) ** 2 + (py - ay) ** 2
    t = clamp(((px - ax) * dx + (py - ay) * dy) / l2, 0.0, 1.0)
    qx = ax + t * dx
    qy = ay + t * dy
    return (px - qx) ** 2 + (py - qy) ** 2


def dist_point_seg(p, a, b):
    return math.sqrt(dist2_point_seg(p.x, p.y, a.x, a.y, b.x, b.y))


def closest_point_on_seg(p, a, b):
    dx = b.x - a.x
    dy = b.y - a.y
    l2 = dx * dx + dy * dy
    if l2 < EPS:
        return a.copy()
    t = clamp(((p.x - a.x) * dx + (p.y - a.y) * dy) / l2, 0.0, 1.0)
    return Vec(a.x + t * dx, a.y + t * dy)


def point_on_seg(p, a, b, tol=1e-6):
    return dist2_point_seg(p.x, p.y, a.x, a.y, b.x, b.y) <= tol * tol


def segments_intersect(a, b, c, d):
    """Strict-ish proper intersection; collinear overlaps handled separately."""
    r = b - a
    s = d - c
    rxs = r.cross(s)
    qpx = c.x - a.x
    qpy = c.y - a.y
    if abs(rxs) < EPS:
        if abs(qpx * r.y - qpy * r.x) > EPS:
            return False
        # collinear: test overlap of projections
        rr = r.dot(r)
        if rr < EPS:
            return point_on_seg(a, c, d)
        t0 = (c - a).dot(r) / rr
        t1 = (d - a).dot(r) / rr
        lo, hi = min(t0, t1), max(t0, t1)
        return hi >= 0.0 - 1e-9 and lo <= 1.0 + 1e-9
    t = (qpx * s.y - qpy * s.x) / rxs
    u = (qpx * r.y - qpy * r.x) / rxs
    return (0.0 - EPS <= t <= 1.0 + EPS) and (0.0 - EPS <= u <= 1.0 + EPS)


def segment_intersection(a, b, c, d):
    """Return intersection point or None (parallel/non-intersecting)."""
    r = b - a
    s = d - c
    rxs = r.cross(s)
    if abs(rxs) < EPS:
        return None
    t = ((c.x - a.x) * s.y - (c.y - a.y) * s.x) / rxs
    return Vec(a.x + t * r.x, a.y + t * r.y)


def point_in_polygon(p, poly):
    inside = False
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        if ((a.y > p.y) != (b.y > p.y)):
            xhit = a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y)
            if xhit > p.x:
                inside = not inside
    return inside


def point_in_rect(p, x0, y0, x1, y1, tol=0.0):
    return (x0 - tol <= p.x <= x1 + tol and y0 - tol <= p.y <= y1 + tol)


def polygon_area(poly):
    s = 0.0
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        s += a.x * b.y - b.x * a.y
    return 0.5 * s


def polyline_length(points):
    total = 0.0
    for a, b in zip(points, points[1:]):
        total += (b - a).length()
    return total


def line_polygon_intersections(p, d, poly):
    """Points where ray p + t*d crosses polygon edges (t real)."""
    out = []
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        e = b - a
        den = d.cross(e)
        if abs(den) < EPS:
            continue
        q = a - p
        t = q.cross(e) / den
        u = q.cross(d) / den
        if -EPS <= u <= 1.0 + EPS:
            out.append((t, Vec(p.x + t * d.x, p.y + t * d.y)))
    out.sort(key=lambda z: z[0])
    return out


# --------------------------------------------------------------- distance field
def edt_1d(f, n, d, v, z):
    """Felzenszwalb & Huttenlocher 1-D squared distance transform."""
    k = 0
    v[0] = 0
    z[0] = float("-inf")
    z[1] = float("inf")
    for q in range(1, n):
        while True:
            vk = v[k]
            s = ((f[q] + q * q) - (f[vk] + vk * vk)) / (2.0 * (q - vk))
            if s <= z[k]:
                k -= 1
                if k >= 0:
                    continue
            k += 1
            v[k] = q
            z[k] = s
            z[k + 1] = float("inf")
            break
    out = [0.0] * n
    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        vk = v[k]
        out[q] = (q - vk) ** 2 + f[vk]
    d[:] = out


def euclidean_distance_field(blocked, h, w):
    """Exact Euclidean distance (in cell units) of every free cell to the
    nearest blocked cell. Blocked cells receive 0."""
    INF = 1e18
    f = [0.0] * (h * w)
    for i in range(h * w):
        f[i] = 0.0 if blocked[i] else INF
    row_v = [0] * max(h, w)
    row_z = [0.0] * (max(h, w) + 1)
    row_d = [0.0] * max(h, w)
    # transform along columns (x fixed, over y)
    for x in range(w):
        col = [f[y * w + x] for y in range(h)]
        edt_1d(col, h, row_d, row_v, row_z)
        for y in range(h):
            f[y * w + x] = row_d[y]
    # transform along rows (y fixed, over x)
    for y in range(h):
        base = y * w
        row = f[base:base + w]
        edt_1d(row, w, row_d, row_v, row_z)
        f[base:base + w] = row_d
    return [math.sqrt(v) for v in f]


def merge_segments(segments, tol=1e-6):
    """Merge collinear connected segments so walls stored per-cell do not
    multiply the distance work."""
    segs = [(Vec.of(a), Vec.of(b)) for a, b in segments]
    used = [False] * len(segs)
    merged = []
    for i, (a, b) in enumerate(segs):
        if used[i]:
            continue
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j in range(len(segs)):
                if used[j]:
                    continue
                c, d = segs[j]
                d1 = b - a
                d2 = d - c
                if abs(d1.cross(d2)) > tol * max(1.0, d1.length(), d2.length()):
                    continue
                # collinear: attach if an endpoint coincides
                if point_on_seg(c, a, b, tol) and point_on_seg(d, a, b, tol):
                    used[j] = True
                    changed = True
                elif (c - b).length_sq() < tol * tol or (d - a).length_sq() < tol * tol:
                    if (c - b).length_sq() < tol * tol:
                        b = d
                    else:
                        a = c
                    used[j] = True
                    changed = True
                elif (d - b).length_sq() < tol * tol or (c - a).length_sq() < tol * tol:
                    if (d - b).length_sq() < tol * tol:
                        b = c
                    else:
                        a = d
                    used[j] = True
                    changed = True
        merged.append((a, b))
    return merged


def path_clears_corner(p0, p1, solids, radius, step=None):
    """A funnel corner is only accepted when the whole capsule between p0 and
    p1 keeps radius clearance. Sampling density scales with segment length so
    a long straight segment cannot leap through a doorway wall between checks.
    This is what stops a route cutting through a wall corner even if the two
    endpoints are individually valid."""
    length = (p1 - p0).length()
    if step is None:
        step = max(radius * 0.25, 1e-3)
    samples = max(6, int(math.ceil(length / step)))
    # tolerance absorbs cell-centre quantisation of eroded portal endpoints;
    # it is three orders of magnitude below the radius, never a wall shortcut
    limit = (radius - 1e-3) ** 2
    for k in range(samples + 1):
        t = k / samples
        p = Vec(p0.x + (p1.x - p0.x) * t, p0.y + (p1.y - p0.y) * t)
        for a, b in solids:
            if dist2_point_seg(p.x, p.y, a.x, a.y, b.x, b.y) < limit:
                return False
    return True
