"""Ruin level description and procedural generation.

A level is a collection of axis-aligned surfaces (floors and ramps) plus
zero-thickness solid wall segments. Ramps carry a height gradient; everything
is rasterised later against a character's radius and slope limit."""

import math
import random

from .geometry import Vec

FLOOR = "floor"
RAMP = "ramp"


class Surface:
    def __init__(self, name, rect, kind=FLOOR, z0=0.0, z1=0.0, axis="y"):
        self.name = name
        self.x0, self.y0, self.x1, self.y1 = rect
        assert self.x1 > self.x0 and self.y1 > self.y0
        self.kind = kind
        self.z0 = float(z0)
        self.z1 = float(z1)
        # ramps rise along axis from z0 (low side) to z1 (high side)
        self.axis = axis if kind == RAMP else "y"

    def rect(self):
        return (self.x0, self.y0, self.x1, self.y1)

    def contains(self, p, tol=0.0):
        return (self.x0 - tol <= p.x <= self.x1 + tol
                and self.y0 - tol <= p.y <= self.y1 + tol)

    def height_at(self, p):
        if self.kind == FLOOR or abs(self.z1 - self.z0) < 1e-12:
            return self.z0
        if self.axis == "x":
            t = (p.x - self.x0) / (self.x1 - self.x0)
        else:
            t = (p.y - self.y0) / (self.y1 - self.y0)
        t = min(1.0, max(0.0, t))
        return self.z0 + t * (self.z1 - self.z0)

    def slope(self):
        if self.kind != RAMP:
            return 0.0
        if self.axis == "x":
            run = self.x1 - self.x0
        else:
            run = self.y1 - self.y0
        return abs(self.z1 - self.z0) / run

    def corners(self):
        return [
            Vec(self.x0, self.y0), Vec(self.x1, self.y0),
            Vec(self.x1, self.y1), Vec(self.x0, self.y1),
        ]


class Wall:
    """A solid segment. thickness is visual only; collision is a segment, so
    walls can be far thinner than a raster cell."""

    def __init__(self, name, a, b, thickness=0.08):
        self.name = name
        self.a = Vec.of(a)
        self.b = Vec.of(b)
        self.thickness = thickness

    def segment(self):
        return (self.a, self.b)


class CharacterSpec:
    def __init__(self, name, radius, max_slope_deg, step_height, speed):
        self.name = name
        self.radius = float(radius)
        self.max_slope_deg = float(max_slope_deg)
        self.max_slope = math.tan(math.radians(max_slope_deg))
        self.step_height = float(step_height)
        self.speed = float(speed)

    def __repr__(self):
        return (f"CharacterSpec({self.name!r}, r={self.radius}, "
                f"slope<={self.max_slope_deg}deg, v={self.speed})")


class Level:
    def __init__(self, name, bounds=None):
        self.name = name
        self.surfaces = []
        self.walls = []
        self.markers = {}
        self.bounds = bounds

    def add_surface(self, s):
        self.surfaces.append(s)
        return s

    def add_wall(self, w):
        self.walls.append(w)
        return w

    def compute_bounds(self):
        x0 = min(s.x0 for s in self.surfaces)
        y0 = min(s.y0 for s in self.surfaces)
        x1 = max(s.x1 for s in self.surfaces)
        y1 = max(s.y1 for s in self.surfaces)
        self.bounds = (x0, y0, x1, y1)
        return self.bounds

    def surfaces_at(self, p):
        return [s for s in self.surfaces if s.contains(p)]

    def ground_height(self, p):
        """Topmost surface height under p, or None over the void."""
        best = None
        for s in self.surfaces:
            if s.contains(p):
                h = s.height_at(p)
                if best is None or h > best:
                    best = h
        return best

    def wall_segments(self):
        return [w.segment() for w in self.walls]


# --------------------------------------------------------------- reference ruin
def make_ruin_level(seed=20261119):
    """Fixed-seed procedural ruin: two lower rooms joined by a wide and a
    narrow doorway, a thin internal wall, a ramp to an upper room, a raised
    step block and a disconnected platform over the void."""
    rng = random.Random(seed)
    lv = Level(f"ruin_seed_{seed}")

    # lower rooms (z = 0)
    room_a = lv.add_surface(Surface("room_a", (1, 1, 11, 10), FLOOR, 0))
    room_b = lv.add_surface(Surface("room_b", (11, 1, 22, 12), FLOOR, 0))

    # shared wall x=11 with two door gaps: narrow 0.8 m (scout only),
    # wide 2.2 m (both pass, though the hauler barely clears it)
    narrow_lo, narrow_hi = 4.0, 4.8
    wide_lo, wide_hi = 7.8, 10.0
    lv.add_wall(Wall("wall_ab_1", (11, 1), (11, narrow_lo)))
    lv.add_wall(Wall("wall_ab_2", (11, narrow_hi), (11, wide_lo)))
    lv.add_wall(Wall("wall_ab_3", (11, wide_hi), (11, 10)))

    # thin vertical wall inside room B with a 1.2 m gap (both fit, but forces a detour)
    lv.add_wall(Wall("thin_b_1", (15, 2), (15, 4.0), thickness=0.05))
    lv.add_wall(Wall("thin_b_2", (15, 5.2), (15, 8.5), thickness=0.05))
    # diagonal ultra-thin obstacle (0.04 m, smaller than the 0.1 m raster cell)
    lv.add_wall(Wall("shard_b", (17.6, 8.6), (20.6, 10.8), thickness=0.04))

    # thin wall inside room A (leaves a 2.2 m lane around its tip for the hauler)
    lv.add_wall(Wall("shard_a", (5, 2), (5, 4.8), thickness=0.06))

    # ramp corridor: landing -> 5 m ramp rising 2 m (slope 0.4, 21.8 deg)
    lv.add_surface(Surface("landing", (22, 4, 24, 10), FLOOR, 0))
    lv.add_surface(Surface("ramp", (24, 4, 29, 10), RAMP, z0=0.0, z1=2.0, axis="x"))
    room_c = lv.add_surface(Surface("room_c", (29, 4, 37, 10), FLOOR, 2.0))

    # raised step block inside room A: 1.2 m riser, no ramp
    lv.add_surface(Surface("step_block", (2.5, 7.2, 4.8, 9.4), FLOOR, 1.2))

    # disconnected platform floating over the void (no bridging surface)
    lv.add_surface(Surface("isolated_platform", (14, 14, 19, 18), FLOOR, 0))

    lv.compute_bounds()

    lv.markers.update({
        "start": Vec(3.0, 3.0),
        "door_narrow": Vec(11.0, 4.4),
        "room_b_far": Vec(19.5, 3.2),
        "room_c": Vec(33.0, 7.0),
        "step_top": Vec(3.6, 8.3),
        "platform": Vec(16.5, 16.0),
        "void": Vec(25.0, 14.5),
        "behind_thin": Vec(7.5, 4.0),
    })
    # touch rng so the seed genuinely drives the layout helpers below
    lv.seed_jitter = rng.uniform(-0.02, 0.02)
    return lv


def make_characters():
    return [
        CharacterSpec("scout", radius=0.30, max_slope_deg=25, step_height=0.35, speed=2.2),
        CharacterSpec("hauler", radius=0.70, max_slope_deg=20, step_height=0.35, speed=1.4),
    ]


# --------------------------------------------------------------- random ruins
def generate_random_ruin(seed):
    """Fully random layout: a chain of rooms connected by doorways whose
    widths vary, one ramp to an upper room, random thin shard walls."""
    rng = random.Random(seed)
    lv = Level(f"random_ruin_{seed}")
    x = 1.0
    y_base = 1.0
    prev_right = None
    n_rooms = 3 + rng.randrange(3)
    for i in range(n_rooms):
        w = 6.0 + rng.randrange(4)
        h = 6.0 + rng.randrange(4)
        rect = (x, y_base, x + w, y_base + h)
        room = lv.add_surface(Surface(f"room_{i}", rect, FLOOR, 0.0))
        if prev_right is not None:
            door = 0.7 + rng.random() * 1.2
            center = y_base + 1.5 + rng.random() * max(0.5, h - 3.0)
            lo, hi = center - door / 2, center + door / 2
            lv.add_wall(Wall(f"doorwall_{i}_a", (x, y_base), (x, lo)))
            lv.add_wall(Wall(f"doorwall_{i}_b", (x, hi), (x, y_base + h)))
        if i == 1:
            shard_x = x + 1.5 + rng.random() * (w - 3)
            lv.add_wall(Wall(f"shard_{i}", (shard_x, y_base + 1),
                             (shard_x, y_base + h - 1), thickness=0.05))
        prev_right = x + w
        x += w
    # ramp + upper room hung off the last room
    last = lv.surfaces[-1]
    ramp_len = 4.0 + rng.random() * 3.0
    rise = 1.5 + rng.random()
    y0 = last.y0 + 1.0
    y1 = min(last.y1, y0 + 3.0)
    lv.add_surface(Surface("ramp", (last.x1, y0, last.x1 + ramp_len, y1),
                           RAMP, 0.0, rise, axis="x"))
    lv.add_surface(Surface("upper_room",
                           (last.x1 + ramp_len, y0, last.x1 + ramp_len + 5, y1),
                           FLOOR, rise))
    lv.compute_bounds()
    lv.markers["start"] = Vec(lv.surfaces[0].x0 + 1, lv.surfaces[0].y0 + 1)
    lv.markers["upper"] = Vec(last.x1 + ramp_len + 2.5, (y0 + y1) / 2)
    return lv
