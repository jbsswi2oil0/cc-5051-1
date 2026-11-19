"""End-to-end delivery: build-process sheets, replay GIFs and machine-readable
reports for every scenario, with explicit failure-stage attribution."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relic_nav.geometry import Vec
from relic_nav.level import make_ruin_level, make_characters, generate_random_ruin
from relic_nav.navmesh import NavMesh, SurfaceStatus, RouteStatus, FollowStatus
from relic_nav.runner import run_scenario
from relic_nav.image import Canvas, save_gif
from relic_nav import render
from relic_nav import font5x7


SCENARIOS = [
    ("S1_narrow_door", "door_narrow"),
    ("S2_room_b_far", "room_b_far"),
    ("S3_ramp_upper", "room_c"),
    ("S4_step_block", "step_top"),
    ("S5_disconnected_platform", "platform"),
    ("S6_void_click", "void"),
    ("S7_thin_wall_detour", "behind_thin"),
]


def build_stage_sheet(nav, path, w=1280, h=760):
    """Six-panel sheet documenting the navmesh construction for a character."""
    cv = Canvas(w, h)
    cam = render.Camera(nav.level.bounds, w // 3, h // 2, margin=24)
    panels = [
        ("1 SURFACES + SOLID WALLS", "level"),
        ("2 RASTER: SLOPE/WALL BLOCKED", "blocked"),
        ("3 EXACT CLEARANCE DISTANCE FIELD", "distance"),
        ("4 RADIUS-ERODED WALKABLE REGIONS", "walkable"),
        ("5 RECT PARTITIONS + PORTALS", "rects"),
        ("6 SOLIDS USED BY MOVEMENT", "solids"),
    ]
    for idx, (title, mode) in enumerate(panels):
        col, row = idx % 3, idx // 3
        ox, oy = col * (w // 3), row * (h // 2)
        sub = Canvas(w // 3, h // 2)
        render.draw_level(sub, cam, nav.level)
        if mode == "blocked":
            render.draw_grid_cells(sub, cam, nav, "blocked")
        elif mode == "distance":
            render.draw_grid_cells(sub, cam, nav, "distance")
        elif mode == "walkable":
            render.draw_grid_cells(sub, cam, nav, "walkable")
        elif mode == "rects":
            render.draw_grid_cells(sub, cam, nav, "walkable")
            render.draw_rects(sub, cam, nav)
        if mode in ("level", "solids", "rects"):
            render.draw_solids(sub, cam, nav.solids)
        sub.text(title, 6, 4, render.C_TEXT, scale=1)
        for yy in range(sub.h):
            base = (oy + yy) * w + ox
            cv.pixels[base:base + sub.w] = sub.pixels[yy * sub.w:(yy + 1) * sub.w]
    cv.save_png(path)


def _wrap(line, max_w):
    if font5x7.text_width(line) <= max_w:
        return [line]
    words = line.split(" ")
    out, cur = [], ""
    for word in words:
        trial = word if not cur else cur + " " + word
        if font5x7.text_width(trial) <= max_w:
            cur = trial
        else:
            if cur:
                out.append(cur)
            cur = word
    if cur:
        out.append(cur)
    return out


def render_scenario_frame(nav, result, frame, show_dev=True):
    w, h = 480, 360
    cv = Canvas(w, h)
    cam = render.Camera(nav.level.bounds, w, h, margin=24, top_band=78)
    render.draw_level(cv, cam, nav.level)
    render.draw_walkable_outline(cv, cam, nav)
    render.draw_solids(cv, cam, nav.solids)
    if result.route and result.route["polyline"]:
        render.draw_route(cv, cam, result.route["polyline"])
    if len(result.frames) > 1:
        upto = result.frames[:frame + 1]
        if show_dev:
            render.draw_deviation_spikes(cv, cam, upto, result.route["polyline"])
        render.draw_trail(cv, cam, upto)
    # start / goal markers with snap or rejection evidence
    if result.frames:
        start = Vec(result.frames[0].x, result.frames[0].y)
        render.draw_marker(cv, cam, start, (160, 255, 180), "START")
    else:
        start_p, _, _ = nav.locate(nav.level.markers["start"])
        render.draw_marker(cv, cam, nav.level.markers["start"],
                           (160, 255, 180), "START")
    goal_raw = nav.level.markers.get(getattr(result, "_goal_name", ""))
    color = render.C_REJECT if result.goal_status in (
        SurfaceStatus.REJECTED_VOID, SurfaceStatus.REJECTED_SLOPE,
        SurfaceStatus.REJECTED_CLEARANCE) else render.C_GOAL
    if goal_raw is not None:
        render.draw_marker(cv, cam, goal_raw, color, "CLICK")
    if result.frames:
        cur = result.frames[frame] if frame < len(result.frames) else result.frames[-1]
        render.draw_character(cv, cam, Vec(cur.x, cur.y), nav.spec.radius,
                              render.C_SCOUT if nav.spec.name == "scout" else render.C_HAULER)
        lines = [
            (f"{nav.spec.name.upper()} R={nav.spec.radius:.2f}M SLOPE<{nav.spec.max_slope_deg:.0f}DEG",
             render.C_TEXT),
            (f"T={cur.t:5.2f}S DEV={cur.deviation:.3f}M Z={cur.z:.2f}", render.C_TEXT),
            (f"STATUS {result.status.value}", render.C_DEV
             if result.status != FollowStatus.ARRIVED else render.C_SCOUT),
            (result.reason, render.C_SNAPPED),
        ]
    else:
        lines = [
            (f"{nav.spec.name.upper()} R={nav.spec.radius:.2f}M SLOPE<{nav.spec.max_slope_deg:.0f}DEG",
             render.C_TEXT),
            ("NO MOTION (ROUTE REJECTED)", render.C_TEXT),
            (f"STATUS {result.status.value}", render.C_DEV),
            (result.reason, render.C_SNAPPED),
        ]
    if result.route and result.route.get("polyline"):
        lines.append((f"ROUTE L={result.path_length:.2f}M TRAVELLED={result.travelled:.2f}M "
                      f"MAXDEV={result.max_deviation:.3f}M COLL={result.collision_count}",
                      render.C_TRAIL))
    if result.goal_status is not None and result.goal_status != SurfaceStatus.OK:
        lines.append((f"GOAL {result.goal_status.value}: {result.goal_detail}",
                      render.C_SNAPPED))
    wrapped = []
    for text, color in lines:
        for piece in _wrap(text, w - 16):
            wrapped.append((piece, color))
    render.hud(cv, wrapped)
    return cv


def make_replay(nav, result, path, max_frames=48, delay_cs=5):
    n = len(result.frames)
    if n <= 1:
        idxs = [0]
    else:
        idxs = sorted(set(int(i * (n - 1) / (max_frames - 1)) for i in range(max_frames)))
    if not result.frames:
        idxs = [0]
        canvases = [render_scenario_frame(nav, result, 0)]
        save_gif(canvases, path, delay_cs=delay_cs)
        canvases[0].save_png(path.replace(".gif", ".png"))
        return
    canvases = [render_scenario_frame(nav, result, i) for i in idxs]
    save_gif(canvases, path, delay_cs=delay_cs)
    # first frame also exported as PNG for quick inspection
    canvases[0].save_png(path.replace(".gif", ".png"))


def failure_stage(result):
    if result.status == FollowStatus.NO_SURFACE:
        return "surface_generation"
    if result.status == FollowStatus.NO_ROUTE:
        return "corridor_search"
    if result.status == FollowStatus.FAILED_STUCK:
        return "follow"
    return "none"


def deliver(out_dir="artifacts", cell_size=0.1):
    os.makedirs(out_dir, exist_ok=True)
    level = make_ruin_level()
    specs = make_characters()
    summary = {"level": level.name, "bounds": level.bounds, "cell_size": cell_size,
               "scenarios": []}

    # build-process sheet for each character
    navs = {}
    for spec in specs:
        nav = NavMesh(level, spec, cell_size=cell_size)
        navs[spec.name] = nav
        build_stage_sheet(nav, os.path.join(out_dir, f"build_{spec.name}.png"))

    for sname, marker in SCENARIOS:
        goal = level.markers[marker]
        for spec in specs:
            nav, result = run_scenario(level, spec, level.markers["start"], goal,
                                       cell_size=cell_size)
            result._goal_name = marker
            tag = f"{sname}__{spec.name}"
            gif = os.path.join(out_dir, f"{tag}.gif")
            make_replay(nav, result, gif)
            entry = {
                "scenario": sname,
                "character": spec.name,
                "radius": spec.radius,
                "max_slope_deg": spec.max_slope_deg,
                "goal_marker": marker,
                "goal_clicked": [goal.x, goal.y],
                "start_status": result.start_status.value,
                "start_detail": result.start_detail,
                "goal_status": result.goal_status.value if result.goal_status else None,
                "goal_detail": result.goal_detail,
                "failure_stage": failure_stage(result),
                "follow_status": result.status.value,
                "reason": result.reason,
                "reference_path_length_m": round(result.path_length, 4),
                "travelled_m": round(result.travelled, 4),
                "path_motion_ratio": (round(result.travelled / result.path_length, 4)
                                      if result.path_length else None),
                "max_deviation_m": round(result.max_deviation, 4),
                "mean_deviation_m": round(result.mean_deviation, 4),
                "collision_events": result.collision_count,
                "surface_gaps_detected": result.surface_failures,
                "min_clearance_m": (round(result.min_clearance, 4)
                                    if result.min_clearance != float("inf") else None),
                "duration_s": round(result.duration, 3),
                "replay": os.path.basename(gif),
            }
            summary["scenarios"].append(entry)
            print(f"{tag:38s} {result.status.value:22s} stage={entry['failure_stage']:18s} "
                  f"L={entry['reference_path_length_m']:7.2f} dev={entry['max_deviation_m']:.3f} "
                  f"coll={entry['collision_events']}")

    # one random-seed ruin as a reproducibility check
    rnd = generate_random_ruin(777)
    rnd_nav, rnd_res = run_scenario(rnd, specs[0], rnd.markers["start"],
                                    rnd.markers["upper"], cell_size=cell_size)
    rnd_res._goal_name = "upper"
    make_replay(rnd_nav, rnd_res, os.path.join(out_dir, "S8_random_ruin__scout.gif"))
    summary["random_ruin_seed_777"] = {
        "follow_status": rnd_res.status.value,
        "failure_stage": failure_stage(rnd_res),
        "regions": rnd_nav.report.n_regions,
        "rects": rnd_nav.report.n_rects,
        "reference_path_length_m": round(rnd_res.path_length, 3),
        "travelled_m": round(rnd_res.travelled, 3),
        "max_deviation_m": round(rnd_res.max_deviation, 3),
    }
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("report ->", os.path.join(out_dir, "report.json"))
    return summary


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "artifacts"
    deliver(out)
