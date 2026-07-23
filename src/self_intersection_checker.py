"""
Conservative polygon-polygon self-intersection checks for realized polyhedra.

The checker is intentionally separate from realization construction and is only
used when explicitly requested by the command-line caller. Faces are regular
polygons, so each constructed face is treated as a convex planar polygon.

Numerical policy:
- Clear positive-length or positive-area intersections are rejected.
- The topologically expected contact between adjacent faces (a shared edge) or
  vertex-adjacent faces (a shared vertex) is allowed.
- Contacts that fall wholly inside the numerical ambiguity band are accepted
  rather than risking a false rejection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from data_structures import Edge, Face, RegularFacedPolyhedron


DEFAULT_DISTANCE_TOLERANCE = 1e-7
_PARALLEL_ANGLE_TOLERANCE = 1e-10


@dataclass(frozen=True)
class _FaceGeometry:
    face: Face
    points: np.ndarray
    normal: np.ndarray
    plane_constant: float
    projection_axis: int
    bbox_min: np.ndarray
    bbox_max: np.ndarray


def find_self_intersection(
    polyhedron: RegularFacedPolyhedron,
    *,
    distance_tolerance: float = DEFAULT_DISTANCE_TOLERANCE,
) -> Optional[str]:
    """
    Return a description of the first clear self-intersection, or ``None``.

    The test is pairwise over constructed faces. It distinguishes legitimate
    topological contact from geometric contact by comparing shared vertex and
    edge indices. Numerical edge cases inside the tolerance band are treated as
    non-intersections to avoid false failures.
    """
    if distance_tolerance <= 0.0:
        raise ValueError("distance_tolerance must be positive")

    geometries: list[_FaceGeometry] = []
    for face in polyhedron.faces:
        if not face.constructed or len(face.vertices) < 3:
            continue
        if not all(vertex.constructed for vertex in face.vertices):
            continue
        geometry = _build_face_geometry(face)
        if geometry is not None:
            geometries.append(geometry)

    for first_index, first in enumerate(geometries):
        for second in geometries[first_index + 1:]:
            message = _face_pair_intersection_message(
                first,
                second,
                distance_tolerance,
            )
            if message is not None:
                return message

    return None


def _build_face_geometry(face: Face) -> Optional[_FaceGeometry]:
    points = np.array([vertex.pos for vertex in face.vertices], dtype=float)

    # Newell's method is stable for ordered polygon vertices and does not depend
    # on any particular choice of three vertices.
    normal = np.zeros(3, dtype=float)
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        normal += np.cross(point, next_point)

    normal_length = float(np.linalg.norm(normal))
    if normal_length < 1e-14:
        # Degenerate geometry is handled by the existing realization validator.
        # Do not let this optional checker introduce a new failure category.
        return None

    normal /= normal_length
    plane_constant = float(np.dot(normal, points[0]))
    projection_axis = int(np.argmax(np.abs(normal)))
    return _FaceGeometry(
        face=face,
        points=points,
        normal=normal,
        plane_constant=plane_constant,
        projection_axis=projection_axis,
        bbox_min=np.min(points, axis=0),
        bbox_max=np.max(points, axis=0),
    )


def _face_pair_intersection_message(
    first: _FaceGeometry,
    second: _FaceGeometry,
    tolerance: float,
) -> Optional[str]:
    ambiguity_margin = 10.0 * tolerance

    if np.any(first.bbox_max < second.bbox_min - ambiguity_margin):
        return None
    if np.any(second.bbox_max < first.bbox_min - ambiguity_margin):
        return None

    normals_cross = np.cross(first.normal, second.normal)
    cross_length = float(np.linalg.norm(normals_cross))

    first_to_second = first.points @ second.normal - second.plane_constant
    second_to_first = second.points @ first.normal - first.plane_constant

    if cross_length <= _PARALLEL_ANGLE_TOLERANCE:
        return _parallel_face_intersection_message(
            first,
            second,
            first_to_second,
            second_to_first,
            tolerance,
        )

    # A polygon wholly and clearly on one side of the other face's plane cannot
    # intersect it. Values inside the ambiguity band are not used to reject.
    if _clearly_one_sided(first_to_second, tolerance):
        return None
    if _clearly_one_sided(second_to_first, tolerance):
        return None

    line_direction = normals_cross / cross_length
    line_origin = _plane_intersection_origin(first, second, normals_cross)

    first_interval = _polygon_plane_interval(
        first.points,
        first_to_second,
        line_origin,
        line_direction,
        tolerance,
    )
    second_interval = _polygon_plane_interval(
        second.points,
        second_to_first,
        line_origin,
        line_direction,
        tolerance,
    )

    if first_interval is None or second_interval is None:
        return None

    overlap_start = max(first_interval[0], second_interval[0])
    overlap_end = min(first_interval[1], second_interval[1])

    if overlap_end < overlap_start - ambiguity_margin:
        return None

    if overlap_end - overlap_start > ambiguity_margin:
        allowed_intervals = _shared_edge_intervals(
            first.face,
            second.face,
            line_origin,
            line_direction,
        )
        if _interval_covered_by_allowed_boundary(
            overlap_start,
            overlap_end,
            allowed_intervals,
            ambiguity_margin,
        ):
            return None

        return (
            f"Faces {first.face.index} and {second.face.index} intersect "
            "along a segment away from their shared boundary."
        )

    # Single-point contacts are tangential and numerically delicate. They are
    # intentionally accepted rather than risking a false rejection.
    return None


def _parallel_face_intersection_message(
    first: _FaceGeometry,
    second: _FaceGeometry,
    first_to_second: np.ndarray,
    second_to_first: np.ndarray,
    tolerance: float,
) -> Optional[str]:
    # Any consistently signed separation, however small, is a near-miss rather
    # than an intersection. This exact-sign guard is deliberately conservative:
    # a truly coplanar pair affected by one-sided roundoff may be accepted, but a
    # pair with a genuine gap will never be rejected as overlapping.
    if np.all(first_to_second > 0.0) or np.all(first_to_second < 0.0):
        return None
    if np.all(second_to_first > 0.0) or np.all(second_to_first < 0.0):
        return None

    coordinate_scale = max(
        1.0,
        float(np.max(np.abs(first.points))),
        float(np.max(np.abs(second.points))),
    )
    roundoff_tolerance = 256.0 * np.finfo(float).eps * coordinate_scale
    if (
        np.max(np.abs(first_to_second)) > roundoff_tolerance
        or np.max(np.abs(second_to_first)) > roundoff_tolerance
    ):
        return None

    first_2d = _project_points(first.points, first.projection_axis)
    second_2d = _project_points(second.points, first.projection_axis)

    intersection_polygon = _convex_polygon_intersection(first_2d, second_2d)
    overlap_area = abs(_signed_area(intersection_polygon))

    combined = np.vstack((first_2d, second_2d))
    planar_scale = float(np.linalg.norm(np.max(combined, axis=0) - np.min(combined, axis=0)))
    area_ambiguity = 10.0 * tolerance * max(1.0, planar_scale)

    if overlap_area > area_ambiguity:
        return (
            f"Faces {first.face.index} and {second.face.index} have "
            "overlapping coplanar interiors."
        )

    # A zero-area coplanar intersection can still be an overlapping edge. The
    # only allowed positive-length overlap is the same topological shared edge.
    first_edges = _projected_face_edges(first.face, first.projection_axis)
    second_edges = _projected_face_edges(second.face, first.projection_axis)
    length_ambiguity = 10.0 * tolerance

    for first_edge, first_start, first_end in first_edges:
        for second_edge, second_start, second_end in second_edges:
            overlap_length = _collinear_overlap_length(
                first_start,
                first_end,
                second_start,
                second_end,
                tolerance,
            )
            if overlap_length <= length_ambiguity:
                continue
            if first_edge.index == second_edge.index:
                continue
            return (
                f"Faces {first.face.index} and {second.face.index} overlap "
                "along a non-shared coplanar edge segment."
            )

    return None


def _clearly_one_sided(distances: np.ndarray, tolerance: float) -> bool:
    return bool(np.all(distances > tolerance) or np.all(distances < -tolerance))


def _plane_intersection_origin(
    first: _FaceGeometry,
    second: _FaceGeometry,
    normals_cross: np.ndarray,
) -> np.ndarray:
    denominator = float(np.dot(normals_cross, normals_cross))
    return (
        first.plane_constant * np.cross(second.normal, normals_cross)
        + second.plane_constant * np.cross(normals_cross, first.normal)
    ) / denominator


def _polygon_plane_interval(
    points: np.ndarray,
    distances: np.ndarray,
    line_origin: np.ndarray,
    line_direction: np.ndarray,
    tolerance: float,
) -> Optional[tuple[float, float]]:
    parameters: list[float] = []

    for index, first_point in enumerate(points):
        second_point = points[(index + 1) % len(points)]
        first_distance = float(distances[index])
        second_distance = float(distances[(index + 1) % len(points)])

        if abs(first_distance) <= tolerance:
            parameters.append(float(np.dot(first_point - line_origin, line_direction)))

        if first_distance * second_distance < 0.0:
            denominator = first_distance - second_distance
            if abs(denominator) <= 1e-30:
                continue
            fraction = first_distance / denominator
            intersection = first_point + fraction * (second_point - first_point)
            parameters.append(float(np.dot(intersection - line_origin, line_direction)))
        elif abs(first_distance) <= tolerance and abs(second_distance) <= tolerance:
            parameters.append(float(np.dot(second_point - line_origin, line_direction)))

    if not parameters:
        return None

    return min(parameters), max(parameters)


def _shared_edge_intervals(
    first_face: Face,
    second_face: Face,
    line_origin: np.ndarray,
    line_direction: np.ndarray,
) -> list[tuple[float, float]]:
    second_edge_indices = {edge.index for edge in second_face.edges}
    intervals: list[tuple[float, float]] = []

    for edge in first_face.edges:
        if edge.index not in second_edge_indices:
            continue
        first_parameter = float(np.dot(edge.vertices[0].pos - line_origin, line_direction))
        second_parameter = float(np.dot(edge.vertices[1].pos - line_origin, line_direction))
        intervals.append(
            (min(first_parameter, second_parameter), max(first_parameter, second_parameter))
        )

    return intervals


def _interval_covered_by_allowed_boundary(
    start: float,
    end: float,
    allowed_intervals: Sequence[tuple[float, float]],
    tolerance: float,
) -> bool:
    if not allowed_intervals:
        return False

    merged = _merge_intervals(allowed_intervals, tolerance)
    cursor = start
    for allowed_start, allowed_end in merged:
        if allowed_end < cursor - tolerance:
            continue
        if allowed_start > cursor + tolerance:
            return False
        cursor = max(cursor, allowed_end)
        if cursor >= end - tolerance:
            return True
    return cursor >= end - tolerance


def _merge_intervals(
    intervals: Sequence[tuple[float, float]],
    tolerance: float,
) -> list[tuple[float, float]]:
    ordered = sorted(intervals)
    merged: list[list[float]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1] + tolerance:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _project_points(points: np.ndarray, axis: int) -> np.ndarray:
    return np.delete(points, axis, axis=1)


def _signed_area(points: Sequence[np.ndarray] | np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    points_array = np.asarray(points, dtype=float)
    shifted = np.roll(points_array, -1, axis=0)
    return 0.5 * float(
        np.sum(points_array[:, 0] * shifted[:, 1] - shifted[:, 0] * points_array[:, 1])
    )


def _convex_polygon_intersection(
    subject_polygon: np.ndarray,
    clip_polygon: np.ndarray,
) -> list[np.ndarray]:
    output = [point.copy() for point in subject_polygon]
    clip_orientation = 1.0 if _signed_area(clip_polygon) >= 0.0 else -1.0

    for clip_index, clip_start in enumerate(clip_polygon):
        clip_end = clip_polygon[(clip_index + 1) % len(clip_polygon)]
        input_points = output
        output = []
        if not input_points:
            break

        previous = input_points[-1]
        previous_inside = _inside_half_plane(
            previous,
            clip_start,
            clip_end,
            clip_orientation,
        )

        for current in input_points:
            current_inside = _inside_half_plane(
                current,
                clip_start,
                clip_end,
                clip_orientation,
            )

            if current_inside:
                if not previous_inside:
                    intersection = _line_intersection_2d(
                        previous,
                        current,
                        clip_start,
                        clip_end,
                    )
                    if intersection is not None:
                        output.append(intersection)
                output.append(current)
            elif previous_inside:
                intersection = _line_intersection_2d(
                    previous,
                    current,
                    clip_start,
                    clip_end,
                )
                if intersection is not None:
                    output.append(intersection)

            previous = current
            previous_inside = current_inside

        output = _deduplicate_polygon_points(output)

    return output


def _inside_half_plane(
    point: np.ndarray,
    edge_start: np.ndarray,
    edge_end: np.ndarray,
    orientation: float,
) -> bool:
    return orientation * _cross_2d(edge_end - edge_start, point - edge_start) >= 0.0


def _line_intersection_2d(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> Optional[np.ndarray]:
    first_direction = first_end - first_start
    second_direction = second_end - second_start
    denominator = _cross_2d(first_direction, second_direction)
    if abs(denominator) < 1e-15:
        return None

    fraction = _cross_2d(second_start - first_start, second_direction) / denominator
    return first_start + fraction * first_direction


def _deduplicate_polygon_points(points: Iterable[np.ndarray]) -> list[np.ndarray]:
    deduplicated: list[np.ndarray] = []
    for point in points:
        if deduplicated and np.linalg.norm(point - deduplicated[-1]) < 1e-12:
            continue
        deduplicated.append(point)

    if (
        len(deduplicated) > 1
        and np.linalg.norm(deduplicated[0] - deduplicated[-1]) < 1e-12
    ):
        deduplicated.pop()
    return deduplicated


def _projected_face_edges(
    face: Face,
    projection_axis: int,
) -> list[tuple[Edge, np.ndarray, np.ndarray]]:
    edges: list[tuple[Edge, np.ndarray, np.ndarray]] = []
    for edge in face.edges:
        projected = _project_points(
            np.array([edge.vertices[0].pos, edge.vertices[1].pos]),
            projection_axis,
        )
        edges.append((edge, projected[0], projected[1]))
    return edges


def _collinear_overlap_length(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
    tolerance: float,
) -> float:
    first_direction = first_end - first_start
    second_direction = second_end - second_start
    first_length = float(np.linalg.norm(first_direction))
    second_length = float(np.linalg.norm(second_direction))
    if first_length <= tolerance or second_length <= tolerance:
        return 0.0

    first_unit = first_direction / first_length
    if abs(_cross_2d(first_unit, second_direction / second_length)) > tolerance:
        return 0.0

    if abs(_cross_2d(first_unit, second_start - first_start)) > tolerance:
        return 0.0
    if abs(_cross_2d(first_unit, second_end - first_start)) > tolerance:
        return 0.0

    second_parameters = (
        float(np.dot(second_start - first_start, first_unit)),
        float(np.dot(second_end - first_start, first_unit)),
    )
    second_min = min(second_parameters)
    second_max = max(second_parameters)

    overlap_start = max(0.0, second_min)
    overlap_end = min(first_length, second_max)
    return max(0.0, overlap_end - overlap_start)


def _cross_2d(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])