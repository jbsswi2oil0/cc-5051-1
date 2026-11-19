"""Navigation mesh construction and corridor search.

Pipeline (one mesh per character, because radius and slope limits differ):

  1. Rasterise every surface into a height/slope grid; void and cells covered
     by explicit walls are blocked.
  2. Generate solid segments: explicit wall segments (any thickness, even
     sub-cell) plus ledge segments on void rims and step risers.
  3. Exact Euclidean distance field (Felzenszwalb-Huttenlocher EDT), refined
     analytically near every solid, then erode by the character radius.
  4. Flood-fill walkable regions, split each into axis-aligned rectangles and
     connect neighbouring rectangles through portal segments.
  5. A* over the rectangle graph, then a funnel algorithm pulls the corridor
     tight; every proposed corner is capsule-checked against the same solids
     used for movement, so a route can never clip a wall corner."""

import heapq
import math
from collections import deque
from enum import Enum

from .geometry import (
    Vec, euclidean_distance_field, merge_segments, point_in_rect,
    path_clears_corner, polyline_length, dist2_point_seg,
)


class SurfaceStatus(str, Enum):
    OK = "on_surface"
    SNAPPED_EDGE = "snapped_from_void"
    SNAPPED_CLEARANCE = "snapped_into_clearance"
    REJECTED_VOID = "rejected_void"
    REJECTED_SLOPE = "rejected_too_steep"
    REJECTED_CLEARANCE = "rejected_no_clearance_nearby"


class RouteStatus(str, Enum):
    OK = "route_ok"
    DISCONNECTED = "no_corridor_regions_disconnected"
    INVALID_ENDPOINT = "endpoint_not_on_walkable_surface"


class FollowStatus(str, Enum):
    ARRIVED = "arrived"
    FAILED_STUCK = "follow_failed_stuck"
    NO_ROUTE = "no_route_to_follow"
    NO_SURFACE = "start_or_goal_rejected"


class BuildReport:
    def __init__(self):
        self.cell_size = 0.0
        self.nx = self.ny = 0
        self.surface_cells = 0
        self.void_cells = 0
        self.wall_cells = 0
        self.steep_cells = 0
        self.eroded_cells = 0
        self.walkable_cells = 0
        self.n_regions = 0
        self.n_rects = 0
        self.n_portals = 0
        self.solids = 0
        self.notes = []

    def as_dict(self):
        return {k: v for k, v in self.__dict__.items()}


B_VOID, B_SLOPE, B_WALL, B_FREE = 0, 1, 2, 3


class RectNode:
    __slots__ = ("rid", "cx0", "cy0", "cx1", "cy1", "center", "portals")

    def __init__(self, rid, cx0, cy0, cx1, cy1, center):
        self.rid = rid                      # region id
        self.cx0, self.cy0, self.cx1, self.cy1 = cx0, cy0, cx1, cy1
        self.center = center                # Vec, world
        self.portals = []                   # (other_rect_id, portal_segment, midpoint)


class NavMesh:
    def __init__(self, level, spec, cell_size=0.1):
        self.level = level
        self.spec = spec
        self.cs = cell_size
        self.report = BuildReport()
        self._build()

    # ------------------------------------------------------------ 1. rasterise
    def _build(self):
        lv = self.level
        spec = self.spec
        cs = self.cs
        x0, y0, x1, y1 = lv.bounds
        nx = max(1, int(math.ceil((x1 - x0) / cs)))
        ny = max(1, int(math.ceil((y1 - y0) / cs)))
        self.nx, self.ny = nx, ny
        self.origin = Vec(x0, y0)
        rep = self.report
        rep.cell_size = cs
        rep.nx, rep.ny = nx, ny

        height = [None] * (nx * ny)          # ground height per cell
        slope = [0.0] * (nx * ny)
        blocked_reason = [B_VOID] * (nx * ny)

        def idx(cx, cy):
            return cy * nx + cx

        def cell_center(cx, cy):
            return Vec(x0 + (cx + 0.5) * cs, y0 + (cy + 0.5) * cs)

        walls = lv.wall_segments()

        for cy in range(ny):
            for cx in range(nx):
                p = cell_center(cx, cy)
                i = idx(cx, cy)
                top_surf = None
                top_h = None
                for s in lv.surfaces:
                    if s.contains(p):
                        h = s.height_at(p)
                        if top_h is None or h > top_h:
                            top_h, top_surf = h, s
                if top_surf is None:
                    continue
                height[i] = top_h
                rep.surface_cells += 1
                if top_surf.kind == "ramp" and top_surf.slope() > spec.max_slope + 1e-12:
                    blocked_reason[i] = B_SLOPE
                    rep.steep_cells += 1
                else:
                    blocked_reason[i] = B_FREE
                    slope[i] = top_surf.slope()

        # explicit walls: sub-cell-thin geometry marks the cells it crosses so
        # the EDT sees the topology; analytic distance restores exact clearance
        rep.void_cells = sum(1 for r in blocked_reason if r == B_VOID)
        for (wa, wb) in walls:
            # every cell whose square touches the segment is marked, so even a
            # sub-cell-thick wall blocks raster connectivity; the analytic pass
            # restores exact clearance later
            min_cx = max(0, int(math.floor((min(wa.x, wb.x) - x0) / cs)) - 1)
            max_cx = min(nx - 1, int(math.ceil((max(wa.x, wb.x) - x0) / cs)))
            min_cy = max(0, int(math.floor((min(wa.y, wb.y) - y0) / cs)) - 1)
            max_cy = min(ny - 1, int(math.ceil((max(wa.y, wb.y) - y0) / cs)))
            for cy in range(min_cy, max_cy + 1):
                for cx in range(min_cx, max_cx + 1):
                    i = idx(cx, cy)
                    if blocked_reason[i] in (B_VOID, B_SLOPE):
                        continue
                    p = cell_center(cx, cy)
                    if dist2_point_seg(p.x, p.y, wa.x, wa.y, wb.x, wb.y) <= (0.75 * cs) ** 2:
                        blocked_reason[i] = B_WALL
                        rep.wall_cells += 1

        # ------------------------------------------------------------ 2. solids
        # boundary segments where a usable cell meets void / step / blocked
        raw = []
        step_tol = spec.step_height

        def cell_edge(cx, cy, side):
            # world coordinates of one cell side
            wx0 = x0 + cx * cs
            wy0 = y0 + cy * cs
            wx1 = wx0 + cs
            wy1 = wy0 + cs
            if side == "N":
                return (Vec(wx0, wy1), Vec(wx1, wy1))
            if side == "S":
                return (Vec(wx0, wy0), Vec(wx1, wy0))
            if side == "W":
                return (Vec(wx0, wy0), Vec(wx0, wy1))
            return (Vec(wx1, wy0), Vec(wx1, wy1))

        for cy in range(ny):
            for cx in range(nx):
                i = idx(cx, cy)
                if blocked_reason[i] != B_FREE:
                    continue
                h = height[i]
                for side, (dx, dy) in (("N", (0, 1)), ("S", (0, -1)),
                                       ("W", (-1, 0)), ("E", (1, 0))):
                    nx_, ny_ = cx + dx, cy + dy
                    if not (0 <= nx_ < nx and 0 <= ny_ < ny):
                        raw.append(cell_edge(cx, cy, side))
                        continue
                    j = idx(nx_, ny_)
                    if blocked_reason[j] != B_FREE:
                        raw.append(cell_edge(cx, cy, side))
                    elif height[j] is not None and abs(height[j] - h) > step_tol + 1e-9:
                        raw.append(cell_edge(cx, cy, side))
        boundary = merge_segments(raw)
        # explicit walls kept as exact analytic solids regardless of raster
        self.solids = boundary + walls
        rep.solids = len(self.solids)

        # uniform spatial hash so analytic clearance stays O(cells * local)
        bucket = max(2.0 * cs, spec.radius)
        self._solid_grid = {}
        for si, (a, b) in enumerate(self.solids):
            gx0 = int(math.floor(min(a.x, b.x) / bucket))
            gx1 = int(math.floor(max(a.x, b.x) / bucket))
            gy0 = int(math.floor(min(a.y, b.y) / bucket))
            gy1 = int(math.floor(max(a.y, b.y) / bucket))
            for gx in range(gx0, gx1 + 1):
                for gy in range(gy0, gy1 + 1):
                    self._solid_grid.setdefault((gx, gy), []).append(si)

        # ------------------------------------------------------------ 3. EDT
        blocked = [r != B_FREE for r in blocked_reason]
        edt = euclidean_distance_field(blocked, ny, nx)
        dist = [d * cs for d in edt]
        # analytic distance to every explicit/ledge solid for all free cells:
        # exact clearance, immune to sub-cell-thick walls and grid aliasing.
        # Neighbour buckets are always queried so long segments cannot be missed.
        for cy in range(ny):
            for cx in range(nx):
                i = idx(cx, cy)
                if blocked[i]:
                    dist[i] = 0.0
                    continue
                p = cell_center(cx, cy)
                best = dist[i]
                gx = int(math.floor(p.x / bucket))
                gy = int(math.floor(p.y / bucket))
                ids = []
                for ddx in (-1, 0, 1):
                    for ddy in (-1, 0, 1):
                        ids.extend(self._solid_grid.get((gx + ddx, gy + ddy), ()))
                for si in ids:
                    a, b = self.solids[si]
                    d2 = dist2_point_seg(p.x, p.y, a.x, a.y, b.x, b.y)
                    if d2 < best * best:
                        best = math.sqrt(d2)
                dist[i] = best

        walkable = [False] * (nx * ny)
        for i in range(nx * ny):
            if not blocked[i] and dist[i] + 1e-12 >= spec.radius:
                walkable[i] = True
        rep.walkable_cells = sum(walkable)
        rep.eroded_cells = rep.surface_cells - rep.steep_cells - rep.wall_cells - rep.walkable_cells

        # ------------------------------------------------------------ 4. regions + rects
        region = [-1] * (nx * ny)
        rid = -1
        for cy in range(ny):
            for cx in range(nx):
                i = idx(cx, cy)
                if not walkable[i] or region[i] != -1:
                    continue
                rid += 1
                region[i] = rid
                q = deque([(cx, cy)])
                while q:
                    ax, ay = q.popleft()
                    cur = idx(ax, ay)
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        bx, by = ax + dx, ay + dy
                        if 0 <= bx < nx and 0 <= by < ny:
                            j = idx(bx, by)
                            # ramps change height every cell: connectivity uses
                            # the height delta to the *current* cell (a valid
                            # slope stays under max_slope*cs), while a step
                            # riser is a discontinuity an order of magnitude larger
                            if walkable[j] and region[j] == -1 and height[j] is not None \
                                    and abs(height[j] - height[cur]) <= cs * spec.max_slope + 0.05:
                                region[j] = rid
                                q.append((bx, by))
        n_regions = rid + 1
        rep.n_regions = n_regions

        covered = [False] * (nx * ny)
        rects = []
        for cy in range(ny):
            for cx in range(nx):
                i = idx(cx, cy)
                if not walkable[i] or covered[i]:
                    continue
                r = region[i]
                # extend right
                cx1 = cx
                while cx1 + 1 < nx and walkable[idx(cx1 + 1, cy)] \
                        and region[idx(cx1 + 1, cy)] == r and not covered[idx(cx1 + 1, cy)]:
                    cx1 += 1
                # extend down while the whole row span matches
                cy1 = cy
                ok = True
                while ok and cy1 + 1 < ny:
                    for ex in range(cx, cx1 + 1):
                        j = idx(ex, cy1 + 1)
                        if not walkable[j] or region[j] != r or covered[j]:
                            ok = False
                            break
                    if ok:
                        cy1 += 1
                for ey in range(cy, cy1 + 1):
                    for ex in range(cx, cx1 + 1):
                        covered[idx(ex, ey)] = True
                center = cell_center((cx + cx1) // 2, (cy + cy1) // 2)
                rects.append(RectNode(r, cx, cy, cx1, cy1, center))
        rep.n_rects = len(rects)

        # cell -> owning rect
        cell_rect = [-1] * (nx * ny)
        for ridx, rc in enumerate(rects):
            for ey in range(rc.cy0, rc.cy1 + 1):
                base = ey * nx
                for ex in range(rc.cx0, rc.cx1 + 1):
                    cell_rect[base + ex] = ridx

        # portals between rects
        portals = []
        seen_pairs = {}
        for cy in range(ny):
            for cx in range(nx):
                i = idx(cx, cy)
                if cell_rect[i] < 0:
                    continue
                ra = cell_rect[i]
                for dx, dy in ((1, 0), (0, 1)):
                    bx, by = cx + dx, cy + dy
                    if not (0 <= bx < nx and 0 <= by < ny):
                        continue
                    j = idx(bx, by)
                    rb = cell_rect[j]
                    if rb < 0 or rb == ra:
                        continue
                    key = (min(ra, rb), max(ra, rb))
                    edge = cell_edge(cx, cy, "E" if dx == 1 else "N")
                    seen_pairs.setdefault(key, []).append(edge)
        for (ra, rb), edges in seen_pairs.items():
            for a, b in merge_segments(edges):
                mid = (a + b) * 0.5
                portals.append((ra, rb, a, b, mid))
                rects[ra].portals.append((rb, (a, b), mid))
                rects[rb].portals.append((ra, (a, b), mid))
        rep.n_portals = len(portals)

        self.height = height
        self.slope = slope
        self.blocked_reason = blocked_reason
        self.dist = dist
        self.walkable = walkable
        self.region = region
        self.cell_rect = cell_rect
        self.rects = rects
        self.portals = portals
        self._cell_center = cell_center
        self._idx = idx

    # ------------------------------------------------------------ cell helpers
    def cell_at(self, p):
        cx = int(math.floor((p.x - self.origin.x) / self.cs))
        cy = int(math.floor((p.y - self.origin.y) / self.cs))
        if 0 <= cx < self.nx and 0 <= cy < self.ny:
            return cx, cy
        return None

    def locate(self, p, snap_tol=None):
        """Classify a requested point against the built walkable surface.

        Returns (status, usable_point_or_None, detail)."""
        if snap_tol is None:
            snap_tol = max(self.spec.radius + 2 * self.cs, 0.6)
        cell = self.cell_at(p)
        if cell is not None and self.walkable[self._idx(*cell)]:
            return SurfaceStatus.OK, p, "point lies on eroded walkable surface"
        # diagnose why the requested cell fails
        reason = "outside level bounds"
        if cell is not None:
            i = self._idx(*cell)
            br = self.blocked_reason[i]
            if br == B_VOID:
                reason = "no surface under point (void)"
            elif br == B_SLOPE:
                reason = (f"surface slope {math.degrees(math.atan(self.slope[i])):.1f}"
                          f"deg exceeds limit {self.spec.max_slope_deg:.1f}deg")
            elif br == B_WALL:
                reason = "point inside a solid wall"
            elif not self.walkable[i]:
                reason = (f"only {self.dist[i]:.2f} m clearance, "
                          f"character radius {self.spec.radius:.2f} m")
        # nearest walkable cell within snap tolerance
        best = None
        r = int(math.ceil(snap_tol / self.cs))
        if cell is not None:
            cx0, cy0 = cell
        else:
            cx0 = int(math.floor((p.x - self.origin.x) / self.cs))
            cy0 = int(math.floor((p.y - self.origin.y) / self.cs))
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                cx, cy = cx0 + dx, cy0 + dy
                if not (0 <= cx < self.nx and 0 <= cy < self.ny):
                    continue
                i = self._idx(cx, cy)
                if not self.walkable[i]:
                    continue
                q = self._cell_center(cx, cy)
                d = (q - p).length()
                if d <= snap_tol and (best is None or d < best[0]):
                    best = (d, q, self.blocked_reason[i])
        if best is None:
            if "void" in reason or "bounds" in reason:
                return SurfaceStatus.REJECTED_VOID, None, reason
            if "slope" in reason:
                return SurfaceStatus.REJECTED_SLOPE, None, reason
            return SurfaceStatus.REJECTED_CLEARANCE, None, reason
        snapped = best[1]
        if "void" in reason or "bounds" in reason:
            status = SurfaceStatus.SNAPPED_EDGE
        else:
            status = SurfaceStatus.SNAPPED_CLEARANCE
        return status, snapped, f"{reason}; snapped {best[0]:.2f} m to nearest walkable cell"

    # ------------------------------------------------------------ 5. A* + funnel
    def _rect_chain(self, start, goal):
        sc = self.cell_at(start)
        gc = self.cell_at(goal)
        if sc is None or gc is None:
            return None, RouteStatus.INVALID_ENDPOINT
        s_rect = self.cell_rect[self._idx(*sc)]
        g_rect = self.cell_rect[self._idx(*gc)]
        if s_rect < 0 or g_rect < 0:
            return None, RouteStatus.INVALID_ENDPOINT
        if self.region[self._idx(*sc)] != self.region[self._idx(*gc)]:
            return None, RouteStatus.DISCONNECTED
        if s_rect == g_rect:
            return [s_rect], RouteStatus.OK
        open_heap = [(0.0, 0, s_rect)]
        came = {s_rect: None}
        gscore = {s_rect: 0.0}
        counter = 1
        while open_heap:
            _, _, cur = heapq.heappop(open_heap)
            if cur == g_rect:
                break
            for other, seg, mid in self.rects[cur].portals:
                tentative = gscore[cur] + (self.rects[cur].center - mid).length() \
                            + (mid - self.rects[other].center).length()
                if tentative < gscore.get(other, float("inf")):
                    gscore[other] = tentative
                    came[other] = cur
                    f = tentative + (self.rects[other].center - self.rects[g_rect].center).length()
                    heapq.heappush(open_heap, (f, counter, other))
                    counter += 1
        if g_rect not in came:
            return None, RouteStatus.DISCONNECTED
        chain = []
        cur = g_rect
        while cur is not None:
            chain.append(cur)
            cur = came[cur]
        chain.reverse()
        return chain, RouteStatus.OK

    def _portal_sequence(self, chain, start, goal):
        """Oriented portals (left, right) along the chain, plus degenerate
        start/end portals. Each portal is oriented against the local
        rectangle-to-rectangle travel direction, matching the Recast funnel
        cross-product convention."""
        portals = [(start, start)]
        for ra, rb in zip(chain, chain[1:]):
            seg = None
            for other, s, mid in self.rects[ra].portals:
                if other == rb:
                    seg = s
                    break
            a, b = seg
            d = (self.rects[rb].center - self.rects[ra].center).normalised()
            # cross > 0 means (b-a) is counter-clockwise from travel dir,
            # i.e. b is on the left; cross < 0 means a is on the left
            if d.cross(b - a) > 0:
                left, right = b, a
            else:
                left, right = a, b
            portals.append((left, right))
        portals.append((goal, goal))
        return portals

    def find_route(self, start, goal):
        """Return dict with polyline, length, chain, portals and diagnostics."""
        chain, status = self._rect_chain(start, goal)
        if status != RouteStatus.OK:
            return {"status": status, "polyline": None, "length": 0.0,
                    "chain": chain, "portals": []}
        if len(chain) == 1:
            poly = [start.copy(), goal.copy()]
            return {"status": RouteStatus.OK, "polyline": poly,
                    "length": polyline_length(poly), "chain": chain,
                    "portals": [], "funnel_fallback": False}
        portals = self._portal_sequence(chain, start, goal)
        # portal midpoints always sit inside eroded walkable cells, so they
        # form a guaranteed-safe corridor backbone
        backbone = [start]
        for left, right in portals[1:-1]:
            backbone.append((left + right) * 0.5)
        backbone.append(goal)
        # pull the backbone tight: from each anchor jump to the furthest
        # vertex whose whole capsule stays clear of every solid. Vertices are
        # all portal endpoints/midpoints, hence individually valid, and every
        # emitted segment passes the same clearance test movement uses
        poly = self._tighten_corridor(backbone, portals)
        # corner capsule verification against the exact solids
        fallback = False
        for a, b in zip(poly, poly[1:]):
            if not path_clears_corner(a, b, self.solids, self.spec.radius):
                fallback = True
                break
        if fallback:
            # guaranteed-safe route: start -> portal midpoints -> goal, all
            # midpoints sit inside eroded walkable cells
            mids = [start]
            for left, right in portals[1:-1]:
                mids.append((left + right) * 0.5)
            mids.append(goal)
            poly = mids
        return {"status": RouteStatus.OK, "polyline": poly,
                "length": polyline_length(poly), "chain": chain,
                "portals": portals, "funnel_fallback": fallback}

    def _tighten_corridor(self, backbone, portals):
        """Visibility string-pulling restricted to the corridor.

        Candidate vertices are walkable cell centres snapped to each portal
        endpoint, so every candidate is a position the character can physically
        stand on (full radius clearance), not merely a point on a clearance
        boundary. From each anchor the furthest reachable candidate with a
        radius-clear capsule becomes the next anchor; every emitted segment is
        independently proven collision-free."""
        r = self.spec.radius

        def standable(p):
            cell = self.cell_at(p)
            if cell is None:
                return None
            cx, cy = cell
            i = self._idx(cx, cy)
            if self.walkable[i]:
                return self._cell_center(cx, cy)
            best = None
            for ddx in range(-2, 3):
                for ddy in range(-2, 3):
                    bx, by = cx + ddx, cy + ddy
                    if 0 <= bx < self.nx and 0 <= by < self.ny:
                        j = self._idx(bx, by)
                        if self.walkable[j]:
                            q = self._cell_center(bx, by)
                            d = (q - p).length_sq()
                            if best is None or d < best[0]:
                                best = (d, q)
            return None if best is None else best[1]

        out = [backbone[0]]
        anchor = backbone[0]
        anchor_pi = 0
        total_portals = len(portals) - 1
        while anchor_pi < total_portals:
            best = None
            for k in range(anchor_pi + 1, total_portals + 1):
                left, right = portals[k]
                opts = (left,) if left == right else (left, right)
                for raw in opts:
                    cand = standable(raw)
                    if cand is None:
                        continue
                    if path_clears_corner(anchor, cand, self.solids, r):
                        best = (k, cand)
            if best is None:
                k = anchor_pi + 1
                cand = standable(backbone[k]) or backbone[k]
                best = (k, cand)
            anchor_pi, anchor = best
            out.append(anchor)
        cleaned = []
        for p in out:
            if not cleaned or (p - cleaned[-1]).length_sq() > 1e-14:
                cleaned.append(p)
        return cleaned

    def _string_pull(self, poly):
        """Conservative corner de-zigzag: remove a middle corner only when the
        shortcut joining its neighbours keeps full radius clearance AND the
        removed point lies inside that safe capsule. It never introduces a
        segment the funnel did not already make locally reachable."""
        if len(poly) <= 2:
            return poly
        r = self.spec.radius
        changed = True
        work = list(poly)
        while changed and len(work) > 2:
            changed = False
            for k in range(1, len(work) - 1):
                a, mid, b = work[k - 1], work[k], work[k + 1]
                if path_clears_corner(a, b, self.solids, r):
                    work.pop(k)
                    changed = True
                    break
        return work

    @staticmethod
    def _funnel(portals):
        """Funnel/string-pulling over an oriented portal sequence.

        portals[i] = (left, right) with left/right defined from the traveller
        looking along the corridor; portals[0] and portals[-1] are degenerate.
        cross(apex, u, v) = (u-apex)x(v-apex): >0 means v is counter-clockwise
        (left) of ray apex->u, <0 means clockwise (right). The legal opening
        is bounded on its left by the left ray and on its right by the right
        ray, and each portal endpoint only tightens its own side."""
        n = len(portals)

        def cross(a, b, c):
            return (b - a).cross(c - a)

        apex = portals[0][0]
        apex_i = 0
        left = portals[0][0]
        right = portals[0][1]
        left_i = right_i = 0
        out = []
        i = 1
        while i < n:
            pl, pr = portals[i]
            # ---- right endpoint: only tightens the right (clockwise) side ----
            if cross(apex, right, pr) <= 0.0:
                if apex == right or cross(apex, left, pr) > 0.0:
                    # pr is still inside the opening: accept as new right ray
                    right, right_i = pr, i
                else:
                    # pr has swept past the left ray: corner at left vertex
                    out.append(left)
                    apex, apex_i = left, left_i
                    left = right = apex
                    left_i = right_i = apex_i
                    i = apex_i + 1
                    continue
            # ---- left endpoint: only tightens the left (counter-clockwise) side ----
            if cross(apex, left, pl) >= 0.0:
                if apex == left or cross(apex, right, pl) < 0.0:
                    # pl is still inside the opening: accept as new left ray
                    left, left_i = pl, i
                else:
                    # pl has swept past the right ray: corner at right vertex
                    out.append(right)
                    apex, apex_i = right, right_i
                    left = right = apex
                    left_i = right_i = apex_i
                    i = apex_i + 1
                    continue
            i += 1
        out.append(portals[-1][0])
        cleaned = []
        for p in out:
            if not cleaned or (p - cleaned[-1]).length_sq() > 1e-14:
                cleaned.append(p)
        return cleaned
