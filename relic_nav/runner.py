"""Driven character follow: the route is actually walked, not teleported.

The same solid segments that eroded the navmesh are used here as physical
colliders. Movement is integrated in substeps and resolved by pushing the
circle out of every solid; collisions, path deviation and stuck conditions are
all recorded so a broken mesh cannot hide behind a drawn line."""

import math
from dataclasses import dataclass, field

from .geometry import Vec, closest_point_on_seg, dist2_point_seg, polyline_length
from .navmesh import (
    NavMesh, SurfaceStatus, RouteStatus, FollowStatus,
)


@dataclass
class Frame:
    t: float
    x: float
    y: float
    z: float
    speed: float
    deviation: float
    segment_index: int
    collision: bool


@dataclass
class FollowResult:
    status: FollowStatus
    reason: str
    frames: list = field(default_factory=list)
    route: dict = None
    start_status: object = None
    start_detail: str = ""
    goal_status: object = None
    goal_detail: str = ""
    path_length: float = 0.0
    travelled: float = 0.0
    max_deviation: float = 0.0
    mean_deviation: float = 0.0
    collision_count: int = 0
    surface_failures: int = 0
    min_clearance: float = float("inf")
    duration: float = 0.0

    def as_dict(self):
        d = dict(self.__dict__)
        d["status"] = self.status.value
        if self.start_status is not None:
            d["start_status"] = self.start_status.value
        if self.goal_status is not None:
            d["goal_status"] = self.goal_status.value
        d["route"] = None if self.route is None else {
            "status": self.route["status"].value,
            "length": self.route["length"],
            "funnel_fallback": self.route.get("funnel_fallback", False),
            "polyline": [[p.x, p.y] for p in (self.route["polyline"] or [])],
        }
        return d


def _point_segment_distance(p, a, b):
    return math.sqrt(dist2_point_seg(p.x, p.y, a.x, a.y, b.x, b.y))


def _deviation_to_polyline(p, poly, seg_index):
    best = float("inf")
    best_i = seg_index
    n = len(poly) - 1
    for i in (seg_index - 1, seg_index, seg_index + 1):
        if 0 <= i < n:
            d = _point_segment_distance(p, poly[i], poly[i + 1])
            if d < best:
                best, best_i = d, i
    if best == float("inf"):
        return _point_segment_distance(p, poly[0], poly[-1]), max(0, n - 1)
    return best, best_i


def _min_clearance(p, solids):
    best = float("inf")
    for a, b in solids:
        d2 = dist2_point_seg(p.x, p.y, a.x, a.y, b.x, b.y)
        if d2 < best:
            best = d2
    return math.sqrt(best)


def move_with_collision(pos, delta, radius, solids):
    """Integrate one move with iterative circle-vs-segment push-out.

    The move is first attempted as one sub-stepped vector; if the collision
    resolution kills most of it, axis-separated sliding (x then y) is tried so
    the character skirts a wall instead of grinding into it. Returns
    (new_pos, collided, penetration_peak). Substeps guarantee a thin wall
    cannot be tunneled even when the per-frame move is larger than it."""
    n_sub = max(1, int(math.ceil(delta.length() / max(radius * 0.25, 1e-6))))
    step = delta / n_sub

    def relax(q):
        collided = False
        peak = 0.0
        for _sweep in range(2):
            for a, b in solids:
                cp = closest_point_on_seg(q, a, b)
                away = q - cp
                d2 = away.length_sq()
                if d2 < radius * radius:
                    d = math.sqrt(d2) if d2 > 1e-16 else 0.0
                    if d < 1e-9:
                        seg = (b - a).normalised()
                        q = cp + seg.perp_left() * radius
                    else:
                        q = q + away / d * (radius - d)
                    collided = True
                    peak = max(peak, radius - d)
        return q, collided, peak

    pos_direct = pos.copy()
    any_coll = False
    peak_all = 0.0
    for _ in range(n_sub):
        pos_direct = pos_direct + step
        pos_direct, c, pk = relax(pos_direct)
        any_coll = any_coll or c
        peak_all = max(peak_all, pk)

    if (pos_direct - pos).length_sq() >= 0.36 * delta.length_sq() or delta.length_sq() < 1e-12:
        return pos_direct, any_coll, peak_all

    # blocked: slide along the two world axes and keep the better projection
    candidates = []
    for sx, sy in ((step.x, 0.0), (0.0, step.y),
                   (step.x, step.y * 0.35), (step.x * 0.35, step.y)):
        q = pos + Vec(sx, sy)
        q, c, pk = relax(q)
        gain = (q - pos).dot(delta.normalised()) if delta.length_sq() > 1e-12 else 0.0
        candidates.append((gain, q, c, pk))
    gain, best, c, pk = max(candidates, key=lambda z: z[0])
    if gain > (pos_direct - pos).dot(delta.normalised()):
        return best, any_coll or c, max(peak_all, pk)
    return pos_direct, any_coll, peak_all


def follow_route(nav, route, start, goal, max_time=60.0, dt=1 / 60.0):
    spec = nav.spec
    solids = nav.solids
    poly = route["polyline"]
    frames = []
    pos = start.copy()
    wp_index = 1
    # half the tightest door gap still lets the waypoint trigger inside it;
    # the final goal uses a smaller tolerance separately
    arrive_r = max(0.25, 0.6 * spec.radius)
    stuck_time = 0.0
    last_progress_pos = start.copy()
    recovery_tangent = None
    travelled = 0.0
    deviations = []
    max_dev = 0.0
    collisions = 0
    surface_failures = 0
    min_clear = float("inf")
    seg_i = 0
    t = 0.0
    goal_tol = max(0.18, 0.5 * spec.radius)

    def record(collided, speed):
        nonlocal max_dev, seg_i
        dev, seg_i = _deviation_to_polyline(pos, poly, seg_i)
        max_dev = max(max_dev, dev)
        deviations.append(dev)
        h = nav.level.ground_height(pos)
        if h is None:
            h = -1.0
        frames.append(Frame(t, pos.x, pos.y, h, speed, dev, seg_i, collided))

    record(False, 0.0)
    min_clear = min(min_clear, _min_clearance(pos, solids))
    while t < max_time:
        if wp_index >= len(poly):
            break
        # skip every waypoint already within reach (handles door-gap funnel
        # points the circle cannot physically stand on), always keeping the
        # final goal as the active target until goal_tol is met
        while wp_index < len(poly) - 1 and (poly[wp_index] - pos).length() <= arrive_r:
            wp_index += 1
        target = poly[wp_index]
        is_final = wp_index == len(poly) - 1
        if is_final and (target - pos).length() <= goal_tol:
            break
        to_wp = target - pos
        dist = to_wp.length()
        if dist <= 1e-9:
            wp_index += 1
            continue
        # steering aims at the active waypoint, blended slightly with the
        # following segment to round corners; collision sliding does the rest
        steer = to_wp.normalised()
        if wp_index < len(poly) - 1:
            nxt = (poly[wp_index + 1] - target).normalised()
            steer = (steer + nxt * 0.35).normalised()
        # recovery: if the body has barely translated for 0.5 s the direct
        # line is pinned, so pick the best collision-free steering direction
        if (pos - last_progress_pos).length() < 0.12:
            stuck_time += dt
        else:
            stuck_time = 0.0
            recovery_tangent = None
            last_progress_pos = pos.copy()
        if stuck_time >= 0.5:
            # sample the full steering circle and keep the collision-free
            # direction with the largest forward projection toward the
            # waypoint; this resolves wall caps and narrow tip gaps without
            # committing to a tangent that points back into the obstacle
            best_dir, best_score = None, -1e9
            base = to_wp.normalised()
            for k in range(16):
                ang = 2 * math.pi * k / 16
                cand = Vec(math.cos(ang), math.sin(ang))
                probe = pos + cand * (spec.radius + 0.06)
                if _min_clearance(probe, solids) < spec.radius - 1e-3:
                    continue
                score = cand.dot(base) + 0.25 * cand.dot(steer)
                if recovery_tangent is not None:
                    score += 0.15 * cand.dot(recovery_tangent)
                if score > best_score:
                    best_score, best_dir = score, cand
            if best_dir is None:
                best_dir = base
            recovery_tangent = best_dir
            steer = (steer * 0.15 + best_dir).normalised()
        velocity = steer * spec.speed
        new_pos, collided, _peak = move_with_collision(pos, velocity * dt,
                                                       spec.radius, solids)
        if collided:
            collisions += 1
        ground = nav.level.ground_height(new_pos)
        if ground is None:
            surface_failures += 1
        moved = (new_pos - pos).length()
        travelled += moved
        pos = new_pos
        t += dt
        min_clear = min(min_clear, _min_clearance(pos, solids))
        record(collided, moved / dt)
        if stuck_time >= 2.0:
            return _finish(FollowStatus.FAILED_STUCK,
                           "character pinned against geometry for 2.0s; "
                           "follow-stage failure, not a teleport",
                           frames, poly, travelled, max_dev, deviations,
                           collisions, surface_failures, min_clear, t)

    reached = (pos - goal).length() <= max(arrive_r, 0.25)
    if reached:
        status = FollowStatus.ARRIVED
        reason = f"arrived within {max(arrive_r, 0.25):.2f} m of goal"
    else:
        status = FollowStatus.FAILED_STUCK
        reason = "follow timed out without reaching goal"
    return _finish(status, reason, frames, poly, travelled, max_dev,
                   deviations, collisions, surface_failures, min_clear, t)


def _finish(status, reason, frames, poly, travelled, max_dev, deviations,
            collisions, surface_failures, min_clear, t):
    return FollowResult(
        status=status, reason=reason, frames=frames,
        path_length=polyline_length(poly), travelled=travelled,
        max_deviation=max_dev,
        mean_deviation=sum(deviations) / len(deviations) if deviations else 0.0,
        collision_count=collisions, surface_failures=surface_failures,
        min_clearance=min_clear, duration=t,
    )


def run_scenario(level, spec, start_point, goal_point, cell_size=0.1,
                 snap_tol=0.7, max_time=60.0):
    """End-to-end: build mesh for the character, validate endpoints, search a
    corridor and drive the character along it. Failure stage is explicit."""
    nav = NavMesh(level, spec, cell_size=cell_size)
    s_status, s_point, s_detail = nav.locate(start_point, snap_tol=snap_tol)
    g_status, g_point, g_detail = nav.locate(goal_point, snap_tol=snap_tol)

    result = FollowResult(status=FollowStatus.NO_SURFACE, reason="",
                          start_status=s_status, start_detail=s_detail,
                          goal_status=g_status, goal_detail=g_detail)
    endpoint_bad = {
        SurfaceStatus.REJECTED_VOID, SurfaceStatus.REJECTED_SLOPE,
        SurfaceStatus.REJECTED_CLEARANCE,
    }
    if s_status in endpoint_bad or g_status in endpoint_bad:
        result.status = FollowStatus.NO_SURFACE
        result.reason = ("surface-stage rejection: "
                         f"start[{s_status.value}: {s_detail}] "
                         f"goal[{g_status.value}: {g_detail}]")
        return nav, result
    if s_point is None or g_point is None:
        result.status = FollowStatus.NO_SURFACE
        result.reason = f"endpoint unusable: {s_detail} / {g_detail}"
        return nav, result

    route = nav.find_route(s_point, g_point)
    result.route = route
    if route["status"] != RouteStatus.OK:
        result.status = FollowStatus.NO_ROUTE
        if route["status"] == RouteStatus.DISCONNECTED:
            result.reason = ("corridor-stage failure: walkable regions for "
                             f"{spec.name} are disconnected (door too narrow, "
                             "slope too steep or platform separated by void)")
        else:
            result.reason = f"corridor-stage failure: {route['status'].value}"
        return nav, result

    follow = follow_route(nav, route, s_point, g_point, max_time=max_time)
    follow.start_status = s_status
    follow.start_detail = s_detail
    follow.goal_status = g_status
    follow.goal_detail = g_detail
    follow.route = route
    return nav, follow
