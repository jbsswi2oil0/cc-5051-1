"""Ruin-exploration navigation: walkable-surface build, clearance corridors,
funnel routing and driven character follow. Pure standard library."""

from .level import Level, Surface, Wall, RAMP, FLOOR, CharacterSpec
from .navmesh import NavMesh, BuildReport, SurfaceStatus, RouteStatus, FollowStatus
from .runner import FollowResult, run_scenario
from .scene import GateScenario, ErosionShortfall, RadiusMisread
from .reducer import reduce_failure
from .sweep import sweep_polyline

__all__ = [
    "Level", "Surface", "Wall", "RAMP", "FLOOR", "CharacterSpec",
    "NavMesh", "BuildReport", "SurfaceStatus", "RouteStatus", "FollowStatus",
    "FollowResult", "run_scenario",
    "GateScenario", "ErosionShortfall", "RadiusMisread",
    "reduce_failure", "sweep_polyline",
]
