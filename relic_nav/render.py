"""Top-down schematic rendering of levels, navmesh build stages and driven
movement. The deviation between the geometric route and the actual motion is
drawn explicitly: deviation spikes, the route line and the travelled trail
all appear together rather than a single ground line."""

from .geometry import Vec, dist_point_seg
from .image import Canvas
from . import font5x7

# palette
C_BG = (20, 22, 30)
C_FLOOR = (58, 62, 78)
C_RAMP = (96, 84, 52)
C_UPPER = (72, 92, 78)
C_STEP = (120, 70, 60)
C_VOID = (12, 12, 16)
C_WALL = (210, 200, 180)
C_WALK = (70, 140, 96)
C_ERODED = (170, 86, 60)
C_STEEP = (150, 70, 110)
C_WALL_CELL = (120, 100, 90)
C_REGION = [(70, 140, 96), (70, 110, 160), (150, 120, 70),
            (130, 90, 160), (160, 90, 110), (90, 160, 150)]
C_RECT_EDGE = (200, 220, 160)
C_PORTAL = (240, 200, 90)
C_ROUTE = (90, 200, 255)
C_TRAIL = (255, 160, 70)
C_DEV = (255, 80, 90)
C_SCOUT = (110, 220, 140)
C_HAULER = (230, 170, 90)
C_GOAL = (255, 230, 120)
C_SNAPPED = (255, 200, 80)
C_REJECT = (255, 90, 90)
C_TEXT = (235, 235, 240)
C_SOLID = (220, 215, 200)
C_GHOST = (255, 120, 130)


class Camera:
    def __init__(self, bounds, w, h, margin=26, top_band=0):
        x0, y0, x1, y1 = bounds
        # reserve a top band for the HUD so text never overlaps the map
        h_avail = max(1, h - top_band - margin)
        self.scale = min((w - 2 * margin) / (x1 - x0), h_avail / (y1 - y0))
        self.ox = margin - x0 * self.scale
        self.oy = top_band - y0 * self.scale

    def xy(self, p):
        if isinstance(p, Vec):
            return (int(p.x * self.scale + self.ox), int(p.y * self.scale + self.oy))
        return (int(p[0] * self.scale + self.ox), int(p[1] * self.scale + self.oy))

    def r(self, radius):
        return max(2, int(radius * self.scale))


def _surface_color(s):
    if s.kind == "ramp":
        return C_RAMP
    if s.z0 >= 1.9:
        return C_UPPER
    if s.z0 > 0.5:
        return C_STEP
    return C_FLOOR


def draw_level(cv, cam, level, alpha_marks=True):
    cv.clear(C_BG)
    for s in level.surfaces:
        pts = [cam.xy(p) for p in s.corners()]
        cv.polygon(pts, fill=_surface_color(s), outline=(30, 30, 36), width=1)
        if s.kind == "ramp":
            # chevrons pointing uphill
            n = 6
            for k in range(1, n):
                t = k / n
                if s.axis == "x":
                    x = s.x0 + t * (s.x1 - s.x0)
                    a = cam.xy(Vec(x, s.y0 + 0.6))
                    b = cam.xy(Vec(x + 0.35, (s.y0 + s.y1) / 2))
                    c = cam.xy(Vec(x, s.y1 - 0.6))
                else:
                    y = s.y0 + t * (s.y1 - s.y0)
                    a = cam.xy(Vec(s.x0 + 0.6, y))
                    b = cam.xy(Vec((s.x0 + s.x1) / 2, y + 0.35))
                    c = cam.xy(Vec(s.x1 - 0.6, y))
                cv.line(a, b, (180, 160, 100), 1)
                cv.line(b, c, (180, 160, 100), 1)
        label = f"{s.name} z={s.z0:.1f}" + (f"-{s.z1:.1f}" if s.kind == "ramp" else "")
        p = cam.xy(Vec(s.x0 + 0.15, s.y0 + 0.35))
        cv.text(label, p[0], p[1], (210, 210, 220), scale=1)
    for w in level.walls:
        cv.line(cam.xy(w.a), cam.xy(w.b), C_WALL, max(1, cam.r(w.thickness)))


def draw_solids(cv, cam, solids):
    for a, b in solids:
        cv.line(cam.xy(a), cam.xy(b), C_SOLID, 1)


def draw_grid_cells(cv, cam, nav, mode):
    cs = nav.cs
    from .navmesh import B_SLOPE, B_WALL
    for cy in range(nav.ny):
        for cx in range(nav.nx):
            i = nav._idx(cx, cy)
            p0 = cam.xy(Vec(nav.origin.x + cx * cs, nav.origin.y + cy * cs))
            p1 = cam.xy(Vec(nav.origin.x + (cx + 1) * cs, nav.origin.y + (cy + 1) * cs))
            if p1[0] <= p0[0] or p1[1] <= p0[1]:
                continue
            if mode == "blocked":
                r = nav.blocked_reason[i]
                c = None
                if r == B_SLOPE:
                    c = C_STEEP
                elif r == B_WALL:
                    c = C_WALL_CELL
                if c:
                    cv.fill_rect(p0[0], p0[1], p1[0], p1[1], c)
            elif mode == "distance":
                if nav.blocked_reason[i] not in (B_SLOPE, B_WALL) and nav.height[i] is not None:
                    d = nav.dist[i]
                    v = max(0, min(255, int(40 + d * 55)))
                    cv.fill_rect(p0[0], p0[1], p1[0], p1[1], (v // 3, v // 2, v))
            elif mode == "walkable":
                if nav.walkable[i]:
                    reg = nav.region[i] % len(C_REGION)
                    cv.fill_rect(p0[0], p0[1], p1[0], p1[1], C_REGION[reg])
                elif nav.height[i] is not None:
                    cv.fill_rect(p0[0], p0[1], p1[0], p1[1], C_ERODED)


def draw_walkable_outline(cv, cam, nav):
    """Cheap replay overlay: tint walkable cells with a 1px marker and trace
    the erosion boundary, instead of filling every raster cell."""
    cs = nav.cs
    tint = (46, 92, 66)
    for cy in range(0, nav.ny, 2):
        for cx in range(0, nav.nx, 2):
            i = nav._idx(cx, cy)
            if nav.walkable[i]:
                p = cam.xy(nav._cell_center(cx, cy))
                cv.put(p[0], p[1], tint)
    for cy in range(nav.ny):
        for cx in range(nav.nx):
            i = nav._idx(cx, cy)
            if not nav.walkable[i]:
                continue
            for side, dx, dy in (("N", 0, 1), ("S", 0, -1), ("W", -1, 0), ("E", 1, 0)):
                bx, by = cx + dx, cy + dy
                if not (0 <= bx < nav.nx and 0 <= by < nav.ny) or not nav.walkable[nav._idx(bx, by)]:
                    a, b = _cell_world_edge(nav, cx, cy, side)
                    cv.line(cam.xy(a), cam.xy(b), (110, 200, 140), 1)


def _cell_world_edge(nav, cx, cy, side):
    cs = nav.cs
    x0 = nav.origin.x + cx * cs
    y0 = nav.origin.y + cy * cs
    if side == "N":
        return Vec(x0, y0 + cs), Vec(x0 + cs, y0 + cs)
    if side == "S":
        return Vec(x0, y0), Vec(x0 + cs, y0)
    if side == "W":
        return Vec(x0, y0), Vec(x0, y0 + cs)
    return Vec(x0 + cs, y0), Vec(x0 + cs, y0 + cs)


def draw_rects(cv, cam, nav):
    for rc in nav.rects:
        x0 = nav.origin.x + rc.cx0 * nav.cs
        y0 = nav.origin.y + rc.cy0 * nav.cs
        x1 = nav.origin.x + (rc.cx1 + 1) * nav.cs
        y1 = nav.origin.y + (rc.cy1 + 1) * nav.cs
        pts = [cam.xy(Vec(x0, y0)), cam.xy(Vec(x1, y0)),
               cam.xy(Vec(x1, y1)), cam.xy(Vec(x0, y1))]
        cv.polygon(pts, fill=None, outline=C_RECT_EDGE, width=1)
    for ra, rb, a, b, mid in nav.portals:
        cv.line(cam.xy(a), cam.xy(b), C_PORTAL, 2)


def draw_route(cv, cam, poly, color=C_ROUTE, width=2):
    for a, b in zip(poly, poly[1:]):
        cv.line(cam.xy(a), cam.xy(b), color, width)
    for p in poly:
        x, y = cam.xy(p)
        cv.circle(x, y, 2, fill=color)


def draw_character(cv, cam, p, radius, color, z=None):
    x, y = cam.xy(p)
    r = cam.r(radius)
    cv.circle(x, y, r, outline=(255, 255, 255), width=1)
    cv.circle(x, y, max(1, r - 2), fill=color)


def draw_marker(cv, cam, p, color, label=None):
    x, y = cam.xy(p)
    cv.line((x - 5, y), (x + 5, y), color, 2)
    cv.line((x, y - 5), (x, y + 5), color, 2)
    if label:
        cv.text(label, x + 7, y - 6, color, scale=1)


def draw_deviation_spikes(cv, cam, frames, poly, stride=3):
    """For every sampled frame draw a line from the actual position to the
    closest point on the geometric route; red length is the real geometric
    deviation, so route-vs-motion disagreement is visible on the map."""
    for f in frames[::stride]:
        p = Vec(f.x, f.y)
        best_d, best_q = float("inf"), None
        n = len(poly) - 1
        lo = max(0, f.segment_index - 1)
        hi = min(n - 1, f.segment_index + 1)
        for k in range(lo, hi + 1):
            from .geometry import closest_point_on_seg
            q = closest_point_on_seg(p, poly[k], poly[k + 1])
            d = (q - p).length_sq()
            if d < best_d:
                best_d, best_q = d, q
        if best_q is not None and best_d > 1e-8:
            cv.line(cam.xy(best_q), cam.xy(p), C_DEV, 1)


def draw_trail(cv, cam, frames, stride=2, color=C_TRAIL):
    pts = [cam.xy(Vec(f.x, f.y)) for f in frames[::stride]]
    for a, b in zip(pts, pts[1:]):
        cv.line(a, b, color, 2)


def hud(cv, lines, scale=1, x=8, y=8):
    yy = y
    for line, color in lines:
        cv.text(line, x, yy, color, scale=scale)
        yy += (font5x7.GLYPH_H + 2) * scale
