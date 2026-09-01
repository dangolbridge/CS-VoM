"""Final CS/mitral + generalized-capsule VoM workflow.

The validated CS -> CS-mitral elbow -> mitral geometry is reused when its
checkpoint is present and valid.  If that checkpoint is missing/incomplete
(or --rebuild-prefix is requested), those stages are rebuilt automatically
before the post-mitral generalized-capsule VoM is generated.

The VoM centreline still follows the LA surface toward LA Regions[3], is
offset outward by the requested tube radius + LA clearance, and is truncated
to the first 2/3 of its physical arc length.

Unlike the previous versions, the VoM surface is NOT made from sequential
rings.  The LA-offset VoM guide is preserved downstream, but its proximal portion is
smoothly fitted onto the exact last mitral-ring centre and exit tangent.  The
fit fades to zero over a short physical distance, so there is no separate large
connector.  A true distance-to-line-segment generalized capsule is then built
around the fitted guide.  Only a one-edge-length collar joins the exact saved
mitral ring to the matching proximal capsule opening.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from geometry_hybrid import (
    angle_between_vectors,
    apply_rigid_transform,
    build_sequential_tube_from_ring,
    combine_ring_sequences,
    compute_ring_normal,
    minimum_curvature_ratio,
    normalize_vector,
    offset_path_from_surface,
    nearest_vertex_distances,
    orient_path_start_near,
    order_ring_points,
    ordered_ring_normal,
    oriented_ring_normal_away_from_landmark,
    path_tangents,
    radius_profile_from_start_ring,
    resample_closed_ring,
    resample_open_path,
    rigid_align_path_to_start,
    ring_center,
    ring_radius,
    rotating_ring_elbow,
    rotation_matrix_from_vectors,
    smooth_path,
    surface_quality,
)
from mesh_io_hybrid import (
    load_regions,
    read_carp_mesh,
    read_hpts,
    read_pts,
    read_triangles,
    write_carp_surface,
    write_pts,
)


# -----------------------------------------------------------------------------
# Existing AtrialMTK / UAC extraction, kept close to the validated clean version
# -----------------------------------------------------------------------------


def resolve_relative(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def extract_cs_ring(
    target_directory: Path,
    mesh_name: str,
    region_points: np.ndarray,
    *,
    n_boundaries: int = 3,
) -> np.ndarray:
    """Extract the same boundary used as ``cs_ring`` by the legacy code."""
    if len(region_points) < 4:
        raise ValueError("Regions.txt must contain at least four coordinate rows")

    from uac import utils

    mesh_points, mesh_elements, *_ = read_carp_mesh(
        target_directory, mesh_name, return_surface=False
    )
    mesh_points = np.asarray(mesh_points, dtype=float)

    boundary_points: list[np.ndarray] = []
    boundary_nodes: list[np.ndarray] = []
    for boundary_index in range(n_boundaries):
        pts, nodes = utils.pv_boundary(boundary_index, mesh_elements, mesh_points)
        boundary_points.append(np.asarray(pts, dtype=float))
        boundary_nodes.append(np.asarray(nodes, dtype=int))

    remap_seeds = [
        np.asarray(region_points[0], dtype=float),
        np.asarray(region_points[1], dtype=float),
        np.asarray(region_points[3], dtype=float),
    ]
    seed_to_boundary = [
        utils.pv_index_remap(seed, boundary_points, np.mean) for seed in remap_seeds
    ]
    try:
        boundary_choice = [seed_to_boundary.index(i) for i in range(n_boundaries)]
        selected_boundary = boundary_choice.index(1)
    except ValueError as exc:
        raise RuntimeError(
            "Could not identify the CS boundary using the existing AtrialMTK region mapping"
        ) from exc

    nodes = boundary_nodes[selected_boundary]
    if len(nodes) < 3:
        raise ValueError("Selected CS boundary contains fewer than three points")
    return mesh_points[nodes]


def prepare_cs_ring(
    raw_ring: np.ndarray,
    *,
    spacing: float,
    ra_reference: np.ndarray,
) -> np.ndarray:
    """Order/resample CS ring with normal sign explicitly away from RA landmark."""
    raw_center = np.mean(raw_ring, axis=0)
    away_reference = raw_center - np.asarray(ra_reference, dtype=float)
    ordered = order_ring_points(
        raw_ring,
        reference_direction=away_reference,
        start_reference=ra_reference,
    )
    return resample_closed_ring(ordered, spacing=spacing)


def branch_limits(branch_cut: Sequence[float], tolerance: float) -> tuple[float, float, float, float]:
    branch_min, branch_max = map(float, branch_cut)
    branch_size = branch_max - branch_min
    return (
        branch_min + branch_size * tolerance,
        branch_max - branch_size * tolerance,
        branch_min,
        branch_size,
    )


def _build_uac_cloud(
    uac_points: np.ndarray,
    indices: np.ndarray,
    *,
    scale: np.ndarray,
    branch: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    cloud: list[np.ndarray] = []
    mapping: list[int] = []
    lower, upper, _, period = branch
    for index in indices:
        point = np.asarray(uac_points[index, :3], dtype=float) * scale
        cloud.append(point)
        mapping.append(int(index))
        if point[2] < lower or point[2] > upper:
            duplicate = point.copy()
            duplicate[2] += period if point[2] < lower else -period
            cloud.append(duplicate)
            mapping.append(int(index))
    if not cloud:
        raise ValueError("UAC search cloud is empty")
    return np.vstack(cloud), np.asarray(mapping, dtype=int)


def map_uac_targets_to_surface(
    uac_points: np.ndarray,
    surface_points: np.ndarray,
    targets: np.ndarray,
    *,
    rl_index: int = 3,
    rl_value: float = 0.0,
    hscale: Sequence[float] = (1.0, 1.0),
    branch_cut: Sequence[float] = (-np.pi, np.pi),
    branch_tolerance: float = 0.05,
) -> np.ndarray:
    uac_points = np.asarray(uac_points, dtype=float)
    surface_points = np.asarray(surface_points, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if len(uac_points) != len(surface_points):
        raise ValueError("UAC and LA surface point counts differ")

    scale = np.asarray([float(hscale[0]), float(hscale[1]), 1.0], dtype=float)
    branch = branch_limits(branch_cut, branch_tolerance)
    right_indices = np.flatnonzero(uac_points[:, rl_index] < rl_value)
    left_indices = np.flatnonzero(uac_points[:, rl_index] >= rl_value)

    right_tree = right_map = left_tree = left_map = None
    if len(right_indices):
        cloud, right_map = _build_uac_cloud(uac_points, right_indices, scale=scale, branch=branch)
        right_tree = cKDTree(cloud, compact_nodes=False, balanced_tree=False)
    if len(left_indices):
        cloud, left_map = _build_uac_cloud(uac_points, left_indices, scale=scale, branch=branch)
        left_tree = cKDTree(cloud, compact_nodes=False, balanced_tree=False)

    mapped_indices: list[int] = []
    for target in targets:
        query = target[:3] * scale
        if target[rl_index] >= rl_value:
            if left_tree is None or left_map is None:
                raise ValueError("Target requires an empty left UAC search cloud")
            _, local_index = left_tree.query(query, k=1)
            mapped_indices.append(int(left_map[int(local_index)]))
        else:
            if right_tree is None or right_map is None:
                raise ValueError("Target requires an empty right UAC search cloud")
            _, local_index = right_tree.query(query, k=1)
            mapped_indices.append(int(right_map[int(local_index)]))
    return surface_points[np.asarray(mapped_indices, dtype=int)]


def remove_duplicate_points(points: np.ndarray, tolerance: float = 1.0e-9) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    kept: list[np.ndarray] = []
    for p in points:
        if not kept or all(np.linalg.norm(p - q) > tolerance for q in kept):
            kept.append(p)
    if len(kept) < 2:
        raise ValueError("Fewer than two unique mapped points remain")
    return np.vstack(kept)


def filter_large_jumps(points: np.ndarray, threshold: float) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if threshold <= 0:
        return points.copy()
    kept = [points[0]]
    for p in points[1:]:
        if np.linalg.norm(p - kept[-1]) <= threshold:
            kept.append(p)
    if len(kept) < 2:
        raise ValueError("Jump filtering left fewer than two points")
    return np.vstack(kept)


def consecutive_midpoints(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        raise ValueError("At least two points required")
    return 0.5 * (points[:-1] + points[1:])


def scale_curve_about_centroid(points: np.ndarray, scale: float) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    center = np.mean(points, axis=0)
    return center + float(scale) * (points - center)


def smooth_open_path_with_endpoint_blend(
    path: np.ndarray,
    *,
    sigma: float,
) -> np.ndarray:
    """Smooth an open path while preserving endpoints without endpoint cusps.

    A plain Gaussian filter moves the endpoints.  Snapping only the first and
    last points back afterward can create a severe local fold.  Here the two
    endpoint corrections are distributed continuously along normalized arc
    length so the complete path absorbs the correction smoothly.
    """
    path = np.asarray(path, dtype=float)
    if len(path) < 2:
        raise ValueError("At least two points are required")
    if sigma <= 0.0:
        return path.copy()

    smoothed = gaussian_filter1d(path, sigma=float(sigma), axis=0, mode="nearest")

    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = float(cumulative[-1])
    if total_length <= 1.0e-12:
        raise ValueError("Cannot smooth a zero-length guide")
    s = cumulative / total_length

    start_correction = path[0] - smoothed[0]
    end_correction = path[-1] - smoothed[-1]
    smoothed = (
        smoothed
        + (1.0 - s)[:, None] * start_correction[None, :]
        + s[:, None] * end_correction[None, :]
    )

    # Numerical guard: the endpoints should be exact.
    smoothed[0] = path[0]
    smoothed[-1] = path[-1]
    return smoothed


def local_turn_angles_degrees(path: np.ndarray) -> np.ndarray:
    """Return unsigned turn angles between consecutive path segments."""
    path = np.asarray(path, dtype=float)
    if len(path) < 3:
        return np.empty(0, dtype=float)
    segments = np.diff(path, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    if np.any(lengths <= 1.0e-12):
        raise ValueError("Guide contains duplicate consecutive points")
    unit = segments / lengths[:, None]
    dots = np.sum(unit[:-1] * unit[1:], axis=1)
    return np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))


def process_vom_guide(
    mapped_points: np.ndarray,
    *,
    spacing: float,
    outlier_threshold: float,
    smoothing_sigma: float,
    curve_scale: float,
    use_midpoints: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """VoM-specific preprocessing with cusp-safe smoothing and re-resampling.

    Returns filtered mapped points, pre-smoothed uniformly sampled guide,
    endpoint-safe smoothed guide, re-resampled guide, and final scaled guide.
    """
    filtered = filter_large_jumps(remove_duplicate_points(mapped_points), outlier_threshold)
    working = consecutive_midpoints(filtered) if use_midpoints else filtered.copy()

    pre_smooth = resample_open_path(working, spacing)
    smoothed = smooth_open_path_with_endpoint_blend(
        pre_smooth, sigma=smoothing_sigma
    )

    # Smoothing changes local sample spacing.  Re-sample once more before the
    # legacy centroid scaling so the early VoM does not contain 50--60 um
    # segments adjacent to ~450 um segments.
    resampled = resample_open_path(smoothed, spacing)
    guide = scale_curve_about_centroid(resampled, curve_scale)
    return filtered, pre_smooth, smoothed, resampled, guide


def process_guide(
    mapped_points: np.ndarray,
    *,
    spacing: float,
    outlier_threshold: float,
    smoothing_sigma: float,
    curve_scale: float,
    use_midpoints: bool,
) -> tuple[np.ndarray, np.ndarray]:
    filtered = filter_large_jumps(remove_duplicate_points(mapped_points), outlier_threshold)
    working = consecutive_midpoints(filtered) if use_midpoints else filtered.copy()
    guide = resample_open_path(working, spacing)
    guide = smooth_path(guide, sigma=smoothing_sigma)
    guide = scale_curve_about_centroid(guide, curve_scale)
    return filtered, guide


def select_mitral_uac_targets(
    uac_points: np.ndarray,
    *,
    u_min: float,
    u_max: float,
    v_value: float,
    tolerance: float,
) -> np.ndarray:
    u = uac_points[:, 0]
    v = uac_points[:, 1]
    mask = (
        np.isclose(v, v_value, atol=tolerance, rtol=0.0)
        & (u >= u_min - tolerance)
        & (u < u_max - tolerance)
    )
    targets = uac_points[mask]
    if len(targets) < 2:
        raise ValueError("Mitral UAC selection produced fewer than two points")
    return targets[np.lexsort((targets[:, 1], targets[:, 0]))]


def select_vom_uac_targets(
    uac_points: np.ndarray,
    *,
    u_value: float,
    v_max: float,
    tolerance: float,
) -> np.ndarray:
    u = uac_points[:, 0]
    v = uac_points[:, 1]
    mask = np.isclose(u, u_value, atol=tolerance, rtol=0.0) & (v < v_max - tolerance)
    targets = uac_points[mask]
    if len(targets) < 2:
        raise ValueError("VoM UAC selection produced fewer than two points")
    return targets[np.argsort(targets[:, 1])]


# -----------------------------------------------------------------------------
# Automatic legacy-style elbow search
# -----------------------------------------------------------------------------


@dataclass
class ElbowChoice:
    score: float
    hinge_index: int
    angle_deg: float
    hinge_offset: float
    points: np.ndarray
    faces: np.ndarray
    last_ring: np.ndarray
    centerline: np.ndarray
    rings: np.ndarray
    exit_tangent: np.ndarray
    aligned_guide: np.ndarray
    guide_rotation: np.ndarray
    guide_translation: np.ndarray
    diagnostics: dict[str, float]


@dataclass
class VomConnectorChoice:
    score: float
    attachment_index: int
    start_handle_scale: float
    end_handle_scale: float
    centerline: np.ndarray
    points: np.ndarray
    faces: np.ndarray
    last_ring: np.ndarray
    rings: np.ndarray
    tangents: np.ndarray
    retained_vom: np.ndarray
    diagnostics: dict[str, float]


def _unit_vector(vector: np.ndarray, *, name: str = "vector") -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-12:
        raise ValueError(f"{name} has near-zero magnitude")
    return vector / norm


def robust_forward_tangent(path: np.ndarray, index: int, lookahead: int = 4) -> np.ndarray:
    """Estimate the forward tangent of the retained ORIGINAL VoM path.

    Several consecutive segment directions are averaged so one noisy mapped
    point does not control the elbow end orientation.
    """
    path = np.asarray(path, dtype=float)
    if index < 0 or index >= len(path) - 1:
        raise ValueError("attachment index must precede the last path point")
    stop = min(len(path) - 1, index + max(1, int(lookahead)))
    seg = np.diff(path[index : stop + 1], axis=0)
    lengths = np.linalg.norm(seg, axis=1)
    keep = lengths > 1.0e-12
    if not np.any(keep):
        raise ValueError("cannot estimate VoM tangent from duplicate points")
    directions = seg[keep] / lengths[keep, None]
    return _unit_vector(np.sum(directions, axis=0), name="VoM forward tangent")


def find_stable_vom_attachment_indices(
    path: np.ndarray,
    *,
    search_fraction: float,
    stable_window: int,
    max_turn_deg: float,
    min_segment_fraction: float,
    max_candidates: int,
) -> list[int]:
    """Return early indices after which the original VoM path is stable.

    The entrance fold seen in update2 had tiny 50--60 um segments followed by
    a ~160 degree reversal.  We do not repair those points.  We simply let the
    new elbow replace them and attach where several consecutive ORIGINAL VoM
    segments have recovered normal spacing and modest local turning.
    """
    path = np.asarray(path, dtype=float)
    if len(path) < stable_window + 3:
        raise ValueError("VoM guide is too short for stable attachment detection")

    seg_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    turn_angles = local_turn_angles_degrees(path)
    # Use the later path to estimate the normal sampling distance so the early
    # malformed entrance cannot bias the reference downward.
    later_start = min(max(3, len(seg_lengths) // 10), len(seg_lengths) - 1)
    reference_spacing = float(np.median(seg_lengths[later_start:]))
    if reference_spacing <= 1.0e-12:
        raise ValueError("VoM guide has zero reference spacing")

    max_index = min(
        len(path) - stable_window - 1,
        max(stable_window + 1, int(np.ceil(float(search_fraction) * len(path)))),
    )
    valid: list[int] = []
    for index in range(1, max_index + 1):
        local_lengths = seg_lengths[index : index + stable_window]
        if len(local_lengths) < stable_window:
            continue
        local_turns = turn_angles[index : index + stable_window - 1]
        spacing_ok = (
            float(np.min(local_lengths)) >= float(min_segment_fraction) * reference_spacing
            and float(np.max(local_lengths)) <= 1.8 * reference_spacing
        )
        turn_ok = len(local_turns) == 0 or float(np.max(local_turns)) <= float(max_turn_deg)
        if spacing_ok and turn_ok:
            valid.append(index)

    if not valid:
        # Conservative fallback: choose the earliest point after the largest
        # entrance turn, then search forward from there.
        if len(turn_angles):
            fallback = min(int(np.argmax(turn_angles)) + 2, len(path) - 2)
        else:
            fallback = 1
        valid = [fallback]

    # Preserve as much original VoM anatomy as possible: search only the first
    # stable attachment points.  A genuinely large elbow is achieved through
    # large Hermite handles, not by deleting a large fraction of the VoM path.
    return valid[: int(max_candidates)]


def hermite_connector_two_handles(
    start: np.ndarray,
    end: np.ndarray,
    start_tangent: np.ndarray,
    end_tangent: np.ndarray,
    *,
    start_handle_scale: float,
    end_handle_scale: float,
    resolution: float,
) -> np.ndarray:
    """Cubic Hermite connector with independent start/end handle lengths."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    t0 = _unit_vector(start_tangent, name="connector start tangent")
    t1 = _unit_vector(end_tangent, name="connector end tangent")
    chord = end - start
    distance = float(np.linalg.norm(chord))
    if distance < 1.0e-9:
        raise ValueError("connector endpoints are coincident")

    m0 = t0 * distance * float(start_handle_scale)
    m1 = t1 * distance * float(end_handle_scale)
    # Dense enough that endpoint finite-difference tangents closely represent
    # the prescribed Hermite derivatives, while keeping candidate search cheap.
    estimated_length = distance * (1.0 + 0.35 * (start_handle_scale + end_handle_scale))
    n_points = int(np.clip(np.ceil(estimated_length / max(float(resolution), 1.0)) + 1, 12, 120))
    t = np.linspace(0.0, 1.0, n_points)
    t2 = t * t
    t3 = t2 * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    path = (
        h00[:, None] * start[None, :]
        + h10[:, None] * m0[None, :]
        + h01[:, None] * end[None, :]
        + h11[:, None] * m1[None, :]
    )
    path[0] = start
    path[-1] = end
    return path


def search_vom_connector(
    *,
    start_ring: np.ndarray,
    start_tangent: np.ndarray,
    original_vom_path: np.ndarray,
    la_points: np.ndarray,
    landmark: np.ndarray,
    resolution: float,
    clearance: float,
    min_bend_radius_factor: float,
    attachment_indices: Sequence[int],
    handle_scales: Sequence[float],
    tangent_lookahead: int,
) -> tuple[VomConnectorChoice, list[dict[str, float]]]:
    """Search a large 3-D connector while preserving retained VoM points."""
    start_ring = np.asarray(start_ring, dtype=float)
    original_vom_path = np.asarray(original_vom_path, dtype=float)
    start_center = ring_center(start_ring)
    start_radius = ring_radius(start_ring)
    start_tangent = _unit_vector(start_tangent, name="mitral exit tangent")
    landmark = np.asarray(landmark, dtype=float)
    la_tree = cKDTree(np.asarray(la_points, dtype=float))

    rows: list[dict[str, float]] = []
    candidates: list[tuple[float, int, float, float, np.ndarray, dict[str, float]]] = []
    total = len(attachment_indices) * len(handle_scales) * len(handle_scales)
    evaluated = 0

    for attachment_index in attachment_indices:
        end = original_vom_path[int(attachment_index)]
        try:
            end_tangent = robust_forward_tangent(
                original_vom_path, int(attachment_index), lookahead=tangent_lookahead
            )
        except ValueError:
            continue

        target_vector = end - start_center
        landmark_vector = landmark - start_center
        target_landmark_angle = float(
            np.rad2deg(angle_between_vectors(target_vector, landmark_vector))
        ) if np.linalg.norm(landmark_vector) > 1.0e-12 else 0.0

        for start_scale in handle_scales:
            for end_scale in handle_scales:
                evaluated += 1
                try:
                    connector = hermite_connector_two_handles(
                        start_center,
                        end,
                        start_tangent,
                        end_tangent,
                        start_handle_scale=float(start_scale),
                        end_handle_scale=float(end_scale),
                        resolution=resolution,
                    )
                    tangents = path_tangents(connector)
                except (ValueError, FloatingPointError):
                    continue

                start_error = float(np.rad2deg(angle_between_vectors(tangents[0], start_tangent)))
                end_error = float(np.rad2deg(angle_between_vectors(tangents[-1], end_tangent)))
                curvature_ratio = minimum_curvature_ratio(connector, start_radius)

                center_distance, _ = la_tree.query(connector, k=1)
                wall_clearance = np.asarray(center_distance, dtype=float) - start_radius
                # Do not punish the inherited mitral interface point itself; it
                # already belongs to the accepted mitral surface.
                clearance_eval = wall_clearance[1:] if len(wall_clearance) > 1 else wall_clearance
                deficit = np.maximum(float(clearance) - clearance_eval, 0.0)
                mean_deficit = float(np.mean(deficit)) if len(deficit) else 0.0
                min_wall_clearance = float(np.min(clearance_eval)) if len(clearance_eval) else float(wall_clearance[0])

                retained = original_vom_path[int(attachment_index):]
                seam_segment = retained[1] - retained[0]
                seam_error = float(np.rad2deg(angle_between_vectors(tangents[-1], seam_segment)))
                connector_length = float(np.sum(np.linalg.norm(np.diff(connector, axis=0), axis=1)))
                chord_length = float(np.linalg.norm(end - start_center))
                length_ratio = connector_length / max(chord_length, 1.0e-12)

                # The 4th LA landmark is now only an anatomical side check.  A
                # candidate attaching on the opposite side receives a strong
                # penalty, but the final tangent is always the ORIGINAL VoM
                # tangent at the retained path.
                side_penalty = max(0.0, (target_landmark_angle - 90.0) / 90.0)
                curvature_penalty = max(0.0, float(min_bend_radius_factor) - curvature_ratio)
                score = (
                    0.25 * start_error
                    + 0.40 * end_error
                    + 0.40 * seam_error
                    + 8.0 * mean_deficit / max(float(clearance), 1.0)
                    + 25.0 * curvature_penalty
                    + 8.0 * side_penalty
                    + 0.35 * max(0.0, length_ratio - 1.0)
                    + 0.03 * float(attachment_index)
                )

                row = {
                    "score": float(score),
                    "attachment_index": float(attachment_index),
                    "start_handle_scale": float(start_scale),
                    "end_handle_scale": float(end_scale),
                    "start_tangent_error_deg": start_error,
                    "end_tangent_error_deg": end_error,
                    "seam_tangent_error_deg": seam_error,
                    "curvature_ratio": float(curvature_ratio),
                    "min_wall_clearance_estimate": min_wall_clearance,
                    "mean_clearance_deficit": mean_deficit,
                    "attachment_landmark_angle_deg": target_landmark_angle,
                    "connector_length": connector_length,
                    "length_ratio": length_ratio,
                }
                rows.append(row)
                candidates.append((float(score), int(attachment_index), float(start_scale), float(end_scale), connector, row))

                if evaluated % 100 == 0:
                    print(f"    mitral→VoM connector: {evaluated}/{total} candidates evaluated", flush=True)

    if not candidates:
        raise RuntimeError("No valid mitral→VoM smooth connector candidates were generated")

    candidates.sort(key=lambda item: item[0])
    rows.sort(key=lambda row: row["score"])

    # Build the full ring geometry only for the best-scoring candidates, and
    # prefer the first one whose actual tube wall has no severe LA penetration.
    chosen = None
    for score, attachment_index, start_scale, end_scale, connector, row in candidates[:25]:
        points, faces, last_ring, tangents, rings, _ = build_sequential_tube_from_ring(
            start_ring,
            connector,
            end_radius=start_radius,
            taper_start_fraction=0.5,
        )
        actual_dist, _ = la_tree.query(points, k=1)
        actual_clearance = np.asarray(actual_dist, dtype=float) - start_radius
        row["actual_min_wall_clearance"] = float(np.min(actual_clearance))
        row["actual_zero_area_triangles"] = float(surface_quality(points, faces).get("zero_area_triangles", 0))
        if chosen is None:
            chosen = (score, attachment_index, start_scale, end_scale, connector, points, faces, last_ring, tangents, rings, row)
        # Allow a small numerical/mesh-resolution deficit, but reject a true
        # penetration deeper than one requested clearance distance.
        if float(np.min(actual_clearance)) >= -float(clearance):
            chosen = (score, attachment_index, start_scale, end_scale, connector, points, faces, last_ring, tangents, rings, row)
            break

    assert chosen is not None
    score, attachment_index, start_scale, end_scale, connector, points, faces, last_ring, tangents, rings, row = chosen
    retained = original_vom_path[int(attachment_index):].copy()
    retained[0] = connector[-1]
    return (
        VomConnectorChoice(
            score=float(score),
            attachment_index=int(attachment_index),
            start_handle_scale=float(start_scale),
            end_handle_scale=float(end_scale),
            centerline=connector,
            points=points,
            faces=faces,
            last_ring=last_ring,
            rings=rings,
            tangents=tangents,
            retained_vom=retained,
            diagnostics=row,
        ),
        rows,
    )


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _sample_hinge_indices(n_points: int, max_hinges: int) -> np.ndarray:
    if n_points <= max_hinges:
        return np.arange(n_points, dtype=int)
    raw = np.linspace(0, n_points, max_hinges, endpoint=False)
    return np.unique(np.floor(raw).astype(int))


def search_elbow(
    *,
    start_ring: np.ndarray,
    desired_start_tangent: np.ndarray,
    target_guide: np.ndarray,
    la_points: np.ndarray,
    target_end_radius: float,
    resolution: float,
    clearance: float,
    angle_values: Sequence[float],
    offset_factors: Sequence[float],
    max_hinges: int,
    min_bend_radius_factor: float,
    clearance_ramp_fraction: float,
    label: str,
    end_normal_target_point: np.ndarray | None = None,
    end_normal_weight: float = 12.0,
) -> tuple[ElbowChoice, list[dict[str, float]]]:
    """Search hinge, angle and offset while preserving legacy ring rotation."""
    start_ring = np.asarray(start_ring, dtype=float)
    target_guide = np.asarray(target_guide, dtype=float)
    la_points = np.asarray(la_points, dtype=float)
    desired_start_tangent = np.asarray(desired_start_tangent, dtype=float)
    if end_normal_target_point is not None:
        end_normal_target_point = np.asarray(end_normal_target_point, dtype=float)
        if end_normal_target_point.shape != (3,):
            raise ValueError("end_normal_target_point must have shape (3,)")

    start_r = ring_radius(start_ring)
    target_tangent = path_tangents(target_guide)[0]
    hinge_indices = _sample_hinge_indices(len(start_ring), int(max_hinges))
    choices: list[ElbowChoice] = []
    diagnostics_table: list[dict[str, float]] = []

    for hinge_index in hinge_indices:
        local_hinge_radius = float(np.linalg.norm(start_ring[hinge_index] - ring_center(start_ring)))
        for offset_factor in offset_factors:
            hinge_offset = float(offset_factor) * start_r
            bend_radius = local_hinge_radius + hinge_offset
            bend_ratio = bend_radius / max(start_r, 1.0e-12)
            if bend_ratio < min_bend_radius_factor:
                continue

            for angle_deg in angle_values:
                try:
                    (
                        elbow_points,
                        elbow_faces,
                        last_ring,
                        elbow_centerline,
                        elbow_rings,
                        elbow_diag,
                    ) = rotating_ring_elbow(
                        start_ring,
                        hinge_index=int(hinge_index),
                        angle_deg=float(angle_deg),
                        hinge_offset=hinge_offset,
                        desired_start_tangent=desired_start_tangent,
                        resolution=resolution,
                    )
                except (ValueError, FloatingPointError):
                    continue

                exit_tangent = path_tangents(elbow_centerline)[-1]
                tangent_error_deg = float(np.rad2deg(angle_between_vectors(exit_tangent, target_tangent)))
                endpoint_distance = float(np.linalg.norm(elbow_centerline[-1] - target_guide[0]))

                # Optional anatomical constraint for the second elbow.
                # The last ring has an unsigned plane-normal axis, but the next
                # tube has a *directed* travel direction.  Therefore we use the
                # LA landmark (Regions.txt row 4) in two complementary ways:
                #   1) orient the fitted last-ring normal toward the landmark;
                #   2) require the actual elbow exit tangent to point that way.
                # The second condition prevents the VoM from leaving through
                # the opposite side of an otherwise correctly oriented plane.
                end_normal_error_deg = 0.0
                end_exit_target_error_deg = 0.0
                if end_normal_target_point is not None:
                    end_center = ring_center(last_ring)
                    target_direction = end_normal_target_point - end_center
                    if np.linalg.norm(target_direction) < 1.0e-12:
                        continue
                    target_direction = target_direction / np.linalg.norm(target_direction)
                    directed_end_normal = compute_ring_normal(
                        last_ring, reference_direction=target_direction
                    )
                    end_normal_error_deg = float(
                        np.rad2deg(
                            angle_between_vectors(directed_end_normal, target_direction)
                        )
                    )
                    end_exit_target_error_deg = float(
                        np.rad2deg(angle_between_vectors(exit_tangent, target_direction))
                    )

                try:
                    aligned_guide, guide_rotation, guide_translation = rigid_align_path_to_start(
                        target_guide,
                        elbow_centerline[-1],
                        exit_tangent,
                    )
                except ValueError:
                    continue

                # Clearance of the actual elbow wall.  For the CS connector the
                # allowed clearance ramps from zero near the ostium.
                elbow_dist = nearest_vertex_distances(elbow_points, la_points).reshape(
                    len(elbow_rings), len(start_ring)
                )
                ring_wall_distance = np.min(elbow_dist, axis=1)
                t = np.linspace(0.0, 1.0, len(elbow_rings))
                if clearance_ramp_fraction > 0.0:
                    required = clearance * _smoothstep(t / clearance_ramp_fraction)
                else:
                    required = np.full_like(t, clearance)
                elbow_deficit = np.maximum(required - ring_wall_distance, 0.0)

                # Fast centreline-based wall-clearance estimate for the next tube.
                guide_radii = radius_profile_from_start_ring(
                    len(aligned_guide),
                    start_r,
                    target_end_radius,
                    taper_start_fraction=0.5,
                )
                guide_center_distance = nearest_vertex_distances(aligned_guide, la_points)
                guide_wall_clearance = guide_center_distance - guide_radii
                guide_deficit = np.maximum(clearance - guide_wall_clearance, 0.0)

                # Penalize moving the anatomical guide unnecessarily far.  This
                # keeps the optimizer close to the original legacy placement.
                guide_displacement = float(
                    np.mean(np.linalg.norm(aligned_guide - target_guide, axis=1))
                )

                min_elbow_wall = float(np.min(ring_wall_distance))
                min_guide_wall = float(np.min(guide_wall_clearance))

                # Dimensionless objective. Clearance deficit is deliberately
                # dominant; tangent/translation terms then choose among safe bends.
                denom_clear = max(clearance, 250.0)
                denom_length = max(2.0 * start_r, 1000.0)
                score = (
                    8.0 * float(np.mean(elbow_deficit)) / denom_clear
                    + 7.0 * float(np.mean(guide_deficit)) / denom_clear
                    + 1.5 * tangent_error_deg / 30.0
                    + float(end_normal_weight) * end_exit_target_error_deg / 15.0
                    + 2.0 * end_normal_error_deg / 15.0
                    + 0.8 * endpoint_distance / denom_length
                    + 0.5 * guide_displacement / denom_length
                    + 0.15 * float(elbow_diag["arc_length"]) / denom_length
                )

                row = {
                    "score": float(score),
                    "hinge_index": float(hinge_index),
                    "angle_deg": float(angle_deg),
                    "hinge_offset": float(hinge_offset),
                    "bend_ratio": float(bend_ratio),
                    "start_tangent_error_deg": float(elbow_diag["start_tangent_error_deg"]),
                    "exit_tangent_error_deg": tangent_error_deg,
                    "end_ring_normal_target_error_deg": end_normal_error_deg,
                    "end_exit_target_error_deg": end_exit_target_error_deg,
                    "endpoint_distance": endpoint_distance,
                    "mean_guide_displacement": guide_displacement,
                    "min_elbow_wall_vertex_distance": min_elbow_wall,
                    "min_next_tube_wall_clearance_estimate": min_guide_wall,
                    "mean_elbow_clearance_deficit": float(np.mean(elbow_deficit)),
                    "mean_next_tube_clearance_deficit": float(np.mean(guide_deficit)),
                }
                diagnostics_table.append(row)
                choices.append(
                    ElbowChoice(
                        score=float(score),
                        hinge_index=int(hinge_index),
                        angle_deg=float(angle_deg),
                        hinge_offset=float(hinge_offset),
                        points=elbow_points,
                        faces=elbow_faces,
                        last_ring=last_ring,
                        centerline=elbow_centerline,
                        rings=elbow_rings,
                        exit_tangent=exit_tangent,
                        aligned_guide=aligned_guide,
                        guide_rotation=guide_rotation,
                        guide_translation=guide_translation,
                        diagnostics=row,
                    )
                )

    if not choices:
        raise RuntimeError(f"No valid {label} elbow candidates were generated")

    if end_normal_target_point is not None:
        # For the mitral→VoM elbow the actual *travel direction* is the
        # anatomical primary criterion.  The plane normal can always be given
        # either sign, so prioritising only its winding/sign can select an elbow
        # whose VoM leaves in the opposite direction.  First minimise the exit
        # tangent error toward the landmark, then use clearance/shape scoring
        # among candidates within a small angular tolerance of that optimum.
        min_exit_error = min(
            c.diagnostics["end_exit_target_error_deg"] for c in choices
        )
        exit_tolerance_deg = 3.0
        near_best = [
            c for c in choices
            if c.diagnostics["end_exit_target_error_deg"]
            <= min_exit_error + exit_tolerance_deg
        ]
        near_best.sort(key=lambda c: c.score)
        selected = near_best[0]
        diagnostics_table.sort(
            key=lambda row: (
                row["end_exit_target_error_deg"],
                row["end_ring_normal_target_error_deg"],
                row["score"],
            )
        )
    else:
        choices.sort(key=lambda c: c.score)
        diagnostics_table.sort(key=lambda row: row["score"])
        selected = choices[0]

    return selected, diagnostics_table


# -----------------------------------------------------------------------------
# Writing / inspection
# -----------------------------------------------------------------------------


def write_candidate_table(path: Path, rows: list[dict[str, float]], n_rows: int = 30) -> None:
    """Write heterogeneous candidate diagnostics safely.

    Some expensive post-selection diagnostics (for example
    ``actual_min_wall_clearance``) are computed only for the best connector
    candidates.  Therefore not every row has exactly the same dictionary
    keys.  Build the CSV header from the union of keys and emit ``nan`` when
    a diagnostic was not evaluated for a particular candidate.
    """
    if not rows:
        return

    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)

    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(keys) + "\n")
        for row in rows[:n_rows]:
            values = []
            for key in keys:
                value = row.get(key, float("nan"))
                values.append(f"{value:.10g}")
            handle.write(",".join(values) + "\n")


def inspect_hybrid(
    output_dir: Path,
    final_dir: Path,
    la_dir: Path,
    mesh_name: str,
) -> None:
    try:
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError("PyVista is required only for --inspect") from exc

    def surface(base: Path):
        pts = read_pts(base)
        tri, _ = read_triangles(base)
        faces = np.column_stack([np.full(len(tri), 3), tri]).reshape(-1)
        return pv.PolyData(pts, faces)

    p = pv.Plotter(title="Hybrid CS / VoM construction")
    p.add_axes()
    seen: set[Path] = set()
    for directory in (final_dir, la_dir):
        base = (directory / mesh_name).resolve()
        if base in seen:
            continue
        if Path(f"{base}.pts").exists() and Path(f"{base}.elem").exists():
            seen.add(base)
            p.add_mesh(surface(base), opacity=0.10, show_edges=False, label=f"{directory.name}/{mesh_name}")

    for name in ("elbow_cs_mitral_hybrid", "mitral_hybrid", "elbow_mitral_vom_hybrid", "vom_hybrid"):
        base = output_dir / name
        if Path(f"{base}.pts").exists():
            p.add_mesh(surface(base), show_edges=True, label=name)

    final_base = output_dir / "tube_cs_hybrid"
    if Path(f"{final_base}.pts").exists():
        p.add_mesh(surface(final_base), opacity=0.25, show_edges=False, label="tube_cs_hybrid")

    for name in (
        "cs_outward_tip",
        "la_vom_elbow_normal_target",
        "vom_attachment_point",
        "vom_attachment_tangent",
        "mitral_guide_aligned",
        "vom_guide_prior_after_mitral_transform",
        "vom_retained_original_path",
        "elbow_cs_mitral_centerline_hybrid",
        "elbow_mitral_vom_centerline_hybrid",
    ):
        file = output_dir / f"{name}.pts"
        if not file.exists():
            continue
        pts = read_pts(file)
        if len(pts) == 1:
            p.add_points(pts, point_size=12, render_points_as_spheres=True, label=name)
        elif len(pts) > 1:
            p.add_mesh(pv.Spline(pts, max(2, len(pts) * 2)), line_width=4, label=name)

    p.add_legend()
    p.show()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# Synthetic post-mitral VoM path: LA geodesic -> outward offset -> smooth blend
# -----------------------------------------------------------------------------


def surface_geodesic_path(
    surface_points: np.ndarray,
    triangles: np.ndarray,
    start_query: np.ndarray,
    target_query: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    """Shortest edge-geodesic on the LA surface between two nearby locations."""
    pts = np.asarray(surface_points, dtype=float)
    tri = np.asarray(triangles, dtype=int)
    if tri.ndim != 2 or tri.shape[1] != 3:
        raise ValueError("triangles must have shape (N, 3)")

    tree = cKDTree(pts)
    _, start_idx = tree.query(np.asarray(start_query, dtype=float), k=1)
    _, target_idx = tree.query(np.asarray(target_query, dtype=float), k=1)
    start_idx = int(start_idx)
    target_idx = int(target_idx)

    edges = np.vstack((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]))
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    weights = np.linalg.norm(pts[edges[:, 0]] - pts[edges[:, 1]], axis=1)

    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    cols = np.concatenate((edges[:, 1], edges[:, 0]))
    data = np.concatenate((weights, weights))
    graph = csr_matrix((data, (rows, cols)), shape=(len(pts), len(pts)))

    distances, predecessors = dijkstra(
        graph,
        directed=False,
        indices=start_idx,
        return_predecessors=True,
    )
    if not np.isfinite(distances[target_idx]):
        raise RuntimeError("No connected LA-surface route was found to Regions[3]")

    indices = [target_idx]
    current = target_idx
    while current != start_idx:
        current = int(predecessors[current])
        if current < 0:
            raise RuntimeError("Failed to reconstruct LA-surface route to Regions[3]")
        indices.append(current)
    indices.reverse()
    return pts[np.asarray(indices, dtype=int)], start_idx, target_idx


def prepare_landmark_offset_guide(
    *,
    start_center: np.ndarray,
    landmark: np.ndarray,
    la_points: np.ndarray,
    la_triangles: np.ndarray,
    start_radius: float,
    end_radius: float,
    clearance: float,
    resolution: float,
    smoothing_sigma: float,
    taper_start_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Build an LA-following, outward-offset guide from mitral to Regions[3]."""
    raw_surface_path, start_idx, target_idx = surface_geodesic_path(
        la_points, la_triangles, start_center, landmark
    )
    if len(raw_surface_path) < 2:
        raise ValueError("LA geodesic to Regions[3] contains fewer than two points")

    surface_guide = resample_open_path(raw_surface_path, spacing=float(resolution))
    if smoothing_sigma > 0.0 and len(surface_guide) > 4:
        # Here smoothing is safe because this is a synthetic route, not an
        # anatomical VoM path. offset_path_from_surface() projects it back to
        # the LA point cloud before applying the outward offset.
        surface_guide = smooth_path(surface_guide, sigma=float(smoothing_sigma))
        surface_guide = resample_open_path(surface_guide, spacing=float(resolution))

    guide_radii = radius_profile_from_start_ring(
        len(surface_guide),
        float(start_radius),
        float(end_radius),
        taper_start_fraction=float(taper_start_fraction),
    )
    offset_distance = guide_radii + float(clearance)
    offset_guide, anchors, normals = offset_path_from_surface(
        surface_guide,
        la_points,
        offset_distance,
        k_neighbors=30,
        normal_smoothing_sigma=1.5,
    )
    # Nearest-vertex anchoring can occasionally map consecutive samples to the
    # same LA vertex.  Remove those duplicates and restore approximately uniform
    # spacing before the inherited-ring sweep.
    offset_guide = remove_duplicate_points(offset_guide, tolerance=1.0e-6)
    offset_guide = resample_open_path(offset_guide, spacing=float(resolution))

    diagnostics = {
        "surface_start_vertex": float(start_idx),
        "surface_target_vertex": float(target_idx),
        "surface_route_length": float(np.sum(np.linalg.norm(np.diff(raw_surface_path, axis=0), axis=1))),
        "offset_guide_length": float(np.sum(np.linalg.norm(np.diff(offset_guide, axis=0), axis=1))),
        "target_center_to_region_distance": float(np.linalg.norm(offset_guide[-1] - np.asarray(landmark, dtype=float))),
    }
    return raw_surface_path, surface_guide, offset_guide, anchors, normals, diagnostics



def truncate_swept_tube_by_arc_fraction(
    centerline: np.ndarray,
    rings: np.ndarray,
    *,
    keep_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Keep an exact arc-length fraction of an already generated swept tube.

    The retained rings are copied unchanged.  If the requested cutoff lies
    between two centreline samples, one final ring is linearly interpolated
    between the corresponding neighbouring rings.  This avoids rebuilding the
    tube (and therefore avoids changing the geometry/radius profile of the
    retained section).

    Returns ``(trimmed_centerline, points, faces, trimmed_rings,
    original_length, retained_length)``.
    """
    path = np.asarray(centerline, dtype=float)
    ring_array = np.asarray(rings, dtype=float)
    fraction = float(keep_fraction)

    if path.ndim != 2 or path.shape[1] != 3 or len(path) < 2:
        raise ValueError("centerline must have shape (N, 3) with N >= 2")
    if ring_array.ndim != 3 or ring_array.shape[0] != len(path) or ring_array.shape[2] != 3:
        raise ValueError("rings must have shape (N, M, 3) and match the centerline length")
    if not (0.0 < fraction <= 1.0):
        raise ValueError("keep_fraction must be in (0, 1]")

    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    original_length = float(cumulative[-1])
    if original_length <= 1.0e-12:
        raise ValueError("VoM centerline has zero arc length")

    target_length = fraction * original_length
    if target_length >= original_length - 1.0e-9:
        points, faces, retained_rings = combine_ring_sequences(ring_array)
        return path.copy(), points, faces, retained_rings, original_length, original_length

    left = int(np.searchsorted(cumulative, target_length, side="right") - 1)
    left = max(0, min(left, len(path) - 2))
    seg_len = float(segment_lengths[left])
    if seg_len <= 1.0e-12:
        alpha = 0.0
    else:
        alpha = float((target_length - cumulative[left]) / seg_len)
    alpha = float(np.clip(alpha, 0.0, 1.0))

    tol = 1.0e-10
    if alpha <= tol:
        trimmed_path = path[: left + 1].copy()
        trimmed_rings = ring_array[: left + 1].copy()
    elif alpha >= 1.0 - tol:
        trimmed_path = path[: left + 2].copy()
        trimmed_rings = ring_array[: left + 2].copy()
    else:
        cut_point = (1.0 - alpha) * path[left] + alpha * path[left + 1]
        cut_ring = (1.0 - alpha) * ring_array[left] + alpha * ring_array[left + 1]
        trimmed_path = np.vstack((path[: left + 1], cut_point[None, :]))
        trimmed_rings = np.concatenate(
            (ring_array[: left + 1], cut_ring[None, :, :]),
            axis=0,
        )

    points, faces, retained_rings = combine_ring_sequences(trimmed_rings)
    retained_length = float(
        np.sum(np.linalg.norm(np.diff(trimmed_path, axis=0), axis=1))
    )
    return (
        trimmed_path,
        points,
        faces,
        retained_rings,
        original_length,
        retained_length,
    )


def search_blend_to_offset_guide(
    *,
    start_ring: np.ndarray,
    start_tangent: np.ndarray,
    offset_guide: np.ndarray,
    la_points: np.ndarray,
    end_radius: float,
    clearance: float,
    resolution: float,
    min_bend_radius_factor: float,
    handle_scales: Sequence[float],
    max_attachment_index: int,
    taper_start_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float], list[dict[str, float]]]:
    """Blend the exact last mitral ring onto the LA-offset guide.

    Returns final centreline, tube points/faces/rings/radii, last ring,
    selected diagnostics, and all candidate diagnostics.
    """
    start_ring = np.asarray(start_ring, dtype=float)
    start_center = ring_center(start_ring)
    start_tangent = _unit_vector(start_tangent, name="mitral exit tangent")
    guide = np.asarray(offset_guide, dtype=float)
    if len(guide) < 4:
        raise ValueError("Offset guide to Regions[3] is too short")

    # Do not attach at the first snapped LA-offset sample.  Search a few early
    # samples so the Hermite blend can absorb any seam mismatch smoothly.
    last_idx = min(int(max_attachment_index), len(guide) - 2)
    attachment_indices = list(range(2, last_idx + 1))
    if not attachment_indices:
        attachment_indices = [1]

    la_tree = cKDTree(np.asarray(la_points, dtype=float))
    rows: list[dict[str, float]] = []
    candidates: list[tuple[float, np.ndarray, dict[str, float]]] = []
    total = len(attachment_indices) * len(handle_scales) * len(handle_scales)
    evaluated = 0

    for attachment_idx in attachment_indices:
        end = guide[attachment_idx]
        end_tangent = robust_forward_tangent(guide, attachment_idx, lookahead=min(4, len(guide)-attachment_idx-1))
        for start_scale in handle_scales:
            for end_scale in handle_scales:
                evaluated += 1
                try:
                    blend = hermite_connector_two_handles(
                        start_center,
                        end,
                        start_tangent,
                        end_tangent,
                        start_handle_scale=float(start_scale),
                        end_handle_scale=float(end_scale),
                        resolution=float(resolution),
                    )
                except (ValueError, FloatingPointError):
                    continue

                full_path = np.vstack((blend, guide[attachment_idx + 1:]))
                if len(full_path) < 3:
                    continue
                tangents = path_tangents(full_path)
                radii = radius_profile_from_start_ring(
                    len(full_path),
                    ring_radius(start_ring),
                    float(end_radius),
                    taper_start_fraction=float(taper_start_fraction),
                )
                curvature_ratio = minimum_curvature_ratio(full_path, radii)
                center_dist, _ = la_tree.query(full_path, k=1)
                wall_clearance = np.asarray(center_dist, dtype=float) - radii
                # The first point is inherited from the accepted mitral tube.
                eval_clear = wall_clearance[1:] if len(wall_clearance) > 1 else wall_clearance
                deficit = np.maximum(float(clearance) - eval_clear, 0.0)
                mean_deficit = float(np.mean(deficit)) if len(deficit) else 0.0
                min_clearance = float(np.min(eval_clear)) if len(eval_clear) else float(wall_clearance[0])

                seam_tangent = robust_forward_tangent(guide, attachment_idx, lookahead=min(4, len(guide)-attachment_idx-1))
                seam_error = float(np.rad2deg(angle_between_vectors(tangents[len(blend)-1], seam_tangent)))
                start_error = float(np.rad2deg(angle_between_vectors(tangents[0], start_tangent)))
                curvature_penalty = max(0.0, float(min_bend_radius_factor) - float(curvature_ratio))

                score = (
                    0.30 * start_error
                    + 0.55 * seam_error
                    + 12.0 * mean_deficit / max(float(clearance), 1.0)
                    + 30.0 * curvature_penalty
                    + 0.02 * float(attachment_idx)
                )
                row = {
                    "score": float(score),
                    "attachment_index": float(attachment_idx),
                    "start_handle_scale": float(start_scale),
                    "end_handle_scale": float(end_scale),
                    "start_tangent_error_deg": start_error,
                    "seam_tangent_error_deg": seam_error,
                    "curvature_ratio": float(curvature_ratio),
                    "min_centerline_wall_clearance": min_clearance,
                    "mean_clearance_deficit": mean_deficit,
                }
                rows.append(row)
                candidates.append((float(score), full_path, row))
                if evaluated % 100 == 0:
                    print(f"    mitral→Region[3] blend: {evaluated}/{total} candidates evaluated", flush=True)

    if not candidates:
        raise RuntimeError("No valid mitral→Region[3] blend candidates were generated")
    candidates.sort(key=lambda item: item[0])
    rows.sort(key=lambda row: row["score"])

    chosen = None
    for _, path, row in candidates[:30]:
        points, faces, last_ring, tangents, rings, radii = build_sequential_tube_from_ring(
            start_ring,
            path,
            end_radius=float(end_radius),
            taper_start_fraction=float(taper_start_fraction),
        )
        actual_dist, _ = la_tree.query(points, k=1)
        # Approximate the requested wall clearance directly on tube vertices.
        actual_min = float(np.min(actual_dist))
        row["actual_min_tube_vertex_distance_to_LA"] = actual_min
        row["actual_zero_area_triangles"] = float(surface_quality(points, faces).get("zero_area_triangles", 0))
        if chosen is None:
            chosen = (path, points, faces, last_ring, tangents, rings, radii, row)
        # Prefer a candidate whose tube vertices do not penetrate the LA point
        # cloud more deeply than one mesh-scale tolerance.
        if actual_min >= max(0.0, float(clearance) * 0.25):
            chosen = (path, points, faces, last_ring, tangents, rings, radii, row)
            break

    assert chosen is not None
    path, points, faces, last_ring, tangents, rings, radii, row = chosen
    return path, points, faces, last_ring, tangents, rings, radii, row, rows



# -----------------------------------------------------------------------------
# Adaptive post-mitral VoM meshing
# -----------------------------------------------------------------------------


def polyline_arclength(path: np.ndarray) -> tuple[np.ndarray, float]:
    p = np.asarray(path, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 2:
        raise ValueError("path must have shape (N,3), N>=2")
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))))
    return cumulative, float(cumulative[-1])


def truncate_path_and_profile_by_arc_fraction(
    path: np.ndarray,
    values: np.ndarray,
    keep_fraction: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Truncate path and one scalar profile at an exact physical arc fraction."""
    p = np.asarray(path, dtype=float)
    v = np.asarray(values, dtype=float)
    if v.shape != (len(p),):
        raise ValueError("values must contain one scalar per path point")
    if not (0.0 < float(keep_fraction) <= 1.0):
        raise ValueError("keep_fraction must be in (0,1]")
    s, total = polyline_arclength(p)
    target = float(keep_fraction) * total
    if target >= total - 1.0e-9:
        return p.copy(), v.copy(), total, total
    left = int(np.searchsorted(s, target, side="right") - 1)
    left = max(0, min(left, len(p) - 2))
    ds = s[left + 1] - s[left]
    alpha = 0.0 if ds <= 1.0e-12 else float((target - s[left]) / ds)
    cut_p = (1.0 - alpha) * p[left] + alpha * p[left + 1]
    cut_v = (1.0 - alpha) * v[left] + alpha * v[left + 1]
    out_p = np.vstack((p[: left + 1], cut_p[None, :]))
    out_v = np.concatenate((v[: left + 1], [cut_v]))
    _, retained = polyline_arclength(out_p)
    return out_p, out_v, total, retained


def resample_path_and_profile(
    path: np.ndarray,
    values: np.ndarray,
    spacing: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly resample an open path and a scalar profile by arc length."""
    p = np.asarray(path, dtype=float)
    v = np.asarray(values, dtype=float)
    if spacing <= 0.0:
        raise ValueError("spacing must be positive")
    if v.shape != (len(p),):
        raise ValueError("values must contain one scalar per path point")
    s, total = polyline_arclength(p)
    if total <= 1.0e-12:
        raise ValueError("path has zero arc length")
    samples = np.arange(0.0, total, float(spacing))
    if len(samples) == 0 or total - samples[-1] > 1.0e-8:
        samples = np.concatenate((samples, [total]))
    elif abs(samples[-1] - total) > 1.0e-8:
        samples[-1] = total
    out = np.empty((len(samples), 3), dtype=float)
    for k in range(3):
        out[:, k] = np.interp(samples, s, p[:, k])
    out_v = np.interp(samples, s, v)
    return out, out_v


def ring_count_for_spacing(radius: float, target_edge: float, *, minimum: int = 6) -> int:
    """Choose ring vertex count so circumference spacing is near target_edge."""
    if radius <= 0.0 or target_edge <= 0.0:
        raise ValueError("radius and target_edge must be positive")
    return max(int(minimum), int(round(2.0 * np.pi * float(radius) / float(target_edge))))


def zipper_faces_between_rings(
    ring_a: np.ndarray,
    offset_a: int,
    ring_b: np.ndarray,
    offset_b: int,
) -> list[list[int]]:
    """Greedy shortest-edge zipper between two ordered closed rings.

    The geometric zipper is important when neighbouring rings are deliberately
    half-step staggered: an index-only zipper would connect the wrong angular
    neighbours and create ~600 um diagonals.  Here the next advance is chosen
    from the shorter of the two possible cross-ring edges.
    """
    A = np.asarray(ring_a, dtype=float)
    B = np.asarray(ring_b, dtype=float)
    n_a = len(A)
    n_b = len(B)
    if n_a < 3 or n_b < 3:
        raise ValueError("rings need at least three vertices")

    # Choose the cyclic B index nearest A[0] as a common zipper seam.
    shift_b = int(np.argmin(np.linalg.norm(B - A[0], axis=1)))

    def bj(local_j: int) -> int:
        return (local_j + shift_b) % n_b

    faces: list[list[int]] = []
    i = 0
    j = 0
    while i < n_a or j < n_b:
        ai = i % n_a
        bcur = bj(j % n_b)
        if i >= n_a:
            advance_a = False
        elif j >= n_b:
            advance_a = True
        else:
            ai1 = (i + 1) % n_a
            bnext = bj((j + 1) % n_b)
            d_if_a = float(np.linalg.norm(A[ai1] - B[bcur]))
            d_if_b = float(np.linalg.norm(A[ai] - B[bnext]))
            advance_a = d_if_a <= d_if_b

        if advance_a:
            ai1 = (i + 1) % n_a
            faces.append([offset_a + ai, offset_a + ai1, offset_b + bcur])
            i += 1
        else:
            bnext = bj((j + 1) % n_b)
            faces.append([offset_a + ai, offset_b + bnext, offset_b + bcur])
            j += 1
    return faces


def flatten_variable_rings(rings: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Flatten variable-size rings and zipper-stitch every consecutive pair."""
    clean = [np.asarray(r, dtype=float) for r in rings]
    if len(clean) < 2:
        raise ValueError("at least two rings are required")
    for r in clean:
        if r.ndim != 2 or r.shape[1] != 3 or len(r) < 3:
            raise ValueError("every ring must have shape (N,3), N>=3")
    offsets: list[int] = []
    total = 0
    for r in clean:
        offsets.append(total)
        total += len(r)
    points = np.vstack(clean)
    faces: list[list[int]] = []
    for k in range(len(clean) - 1):
        faces.extend(zipper_faces_between_rings(clean[k], offsets[k], clean[k+1], offsets[k+1]))
    return points, np.asarray(faces, dtype=int)


def build_adaptive_vom_rings(
    start_ring: np.ndarray,
    centerline: np.ndarray,
    radii: np.ndarray,
    *,
    target_edge: float,
    max_count_change: int = 1,
    stagger: bool = True,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Build variable-size, rotation-minimizing rings along the VoM centerline."""
    path = np.asarray(centerline, dtype=float)
    rr = np.asarray(radii, dtype=float)
    first = np.asarray(start_ring, dtype=float)
    if rr.shape != (len(path),):
        raise ValueError("radii must contain one value per centreline point")
    if len(path) < 2:
        raise ValueError("centerline must contain at least two points")
    # Force the path to inherit the accepted mitral endpoint exactly.
    start_center = ring_center(first)
    path = path.copy()
    path += start_center - path[0]
    path[0] = start_center
    rr = rr.copy()
    rr[0] = ring_radius(first)

    tangents = path_tangents(path)
    # Orient travel tangent consistently with the geometric first-ring normal.
    first_normal = compute_ring_normal(first, reference_direction=tangents[0])
    if np.dot(first_normal, tangents[0]) < 0.0:
        tangents = -tangents

    radial = first[0] - start_center
    radial -= tangents[0] * np.dot(radial, tangents[0])
    if np.linalg.norm(radial) < 1.0e-12:
        # Stable arbitrary perpendicular.
        basis = np.eye(3)[int(np.argmin(np.abs(tangents[0])))]
        radial = np.cross(tangents[0], basis)
    radial = normalize_vector(radial, name="initial VoM radial")

    rings: list[np.ndarray] = [first.copy()]
    counts = [len(first)]
    phases = [0.0]
    prev_radial = radial
    prev_count = len(first)

    for i in range(1, len(path)):
        rot = rotation_matrix_from_vectors(tangents[i-1], tangents[i])
        transported = prev_radial @ rot.T
        transported -= tangents[i] * np.dot(transported, tangents[i])
        if np.linalg.norm(transported) < 1.0e-12:
            basis = np.eye(3)[int(np.argmin(np.abs(tangents[i])))]
            transported = np.cross(tangents[i], basis)
        transported = normalize_vector(transported, name="transported VoM radial")
        binormal = normalize_vector(np.cross(tangents[i], transported), name="VoM binormal")

        target_n = ring_count_for_spacing(rr[i], target_edge)
        if max_count_change >= 0:
            target_n = int(np.clip(target_n, prev_count - max_count_change, prev_count + max_count_change))
            target_n = max(6, target_n)
        phase = (0.5 * (2.0 * np.pi / target_n)) if (stagger and i % 2 == 1) else 0.0
        angles = phase + np.linspace(0.0, 2.0 * np.pi, target_n, endpoint=False)
        ring = path[i][None, :] + float(rr[i]) * (
            np.cos(angles)[:, None] * transported[None, :]
            + np.sin(angles)[:, None] * binormal[None, :]
        )
        rings.append(ring)
        counts.append(target_n)
        phases.append(phase)
        prev_radial = transported
        prev_count = target_n

    return rings, np.asarray(counts, dtype=int), tangents


def triangle_mesh_metrics(points: np.ndarray, faces: np.ndarray, target_edge: float) -> dict[str, float | int]:
    """Edge/angle/area diagnostics for an arbitrary triangular surface."""
    p = np.asarray(points, dtype=float)
    f = np.asarray(faces, dtype=int)
    tri = p[f]
    a = np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1)
    b = np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1)
    c = np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1)
    all_edges = np.concatenate((a, b, c))
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)

    def angle(opposite: np.ndarray, side1: np.ndarray, side2: np.ndarray) -> np.ndarray:
        denom = np.maximum(2.0 * side1 * side2, 1.0e-12)
        cs = np.clip((side1*side1 + side2*side2 - opposite*opposite) / denom, -1.0, 1.0)
        return np.degrees(np.arccos(cs))

    A = angle(b, a, c)
    B = angle(c, a, b)
    C = 180.0 - A - B
    min_angles = np.minimum(np.minimum(A, B), C)
    max_angles = np.maximum(np.maximum(A, B), C)
    low = 0.75 * float(target_edge)
    high = 1.35 * float(target_edge)
    return {
        "triangles": int(len(f)),
        "edge_min": float(np.min(all_edges)),
        "edge_p05": float(np.percentile(all_edges, 5)),
        "edge_median": float(np.median(all_edges)),
        "edge_mean": float(np.mean(all_edges)),
        "edge_p95": float(np.percentile(all_edges, 95)),
        "edge_max": float(np.max(all_edges)),
        "edge_fraction_0.75_to_1.35_target": float(np.mean((all_edges >= low) & (all_edges <= high))),
        "area_min": float(np.min(areas)),
        "zero_or_near_zero_area": int(np.sum(areas <= 1.0e-8)),
        "minimum_angle_deg": float(np.min(min_angles)),
        "minimum_angle_p05_deg": float(np.percentile(min_angles, 5)),
        "maximum_angle_deg": float(np.max(max_angles)),
    }



def _vtk_polydata_from_triangles(points: np.ndarray, faces: np.ndarray):
    """Create vtkPolyData from an (N,3) triangle array."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    pts = np.asarray(points, dtype=float)
    tri = np.asarray(faces, dtype=np.int64)
    poly = vtk.vtkPolyData()
    vtk_pts = vtk.vtkPoints()
    vtk_pts.SetData(numpy_to_vtk(pts, deep=True))
    poly.SetPoints(vtk_pts)

    cells = vtk.vtkCellArray()
    legacy = np.column_stack((np.full(len(tri), 3, dtype=np.int64), tri)).reshape(-1)
    cells.SetCells(len(tri), numpy_to_vtk(legacy, deep=True, array_type=vtk.VTK_ID_TYPE))
    poly.SetPolys(cells)
    return poly


def _numpy_triangles_from_vtk(poly) -> tuple[np.ndarray, np.ndarray]:
    """Return points and triangles from triangular vtkPolyData."""
    from vtk.util.numpy_support import vtk_to_numpy

    if poly.GetNumberOfPoints() == 0 or poly.GetNumberOfPolys() == 0:
        raise RuntimeError("VTK produced an empty implicit VoM surface")
    points = vtk_to_numpy(poly.GetPoints().GetData()).astype(float, copy=True)
    data = vtk_to_numpy(poly.GetPolys().GetData()).astype(np.int64, copy=False)
    if len(data) % 4 != 0:
        raise RuntimeError("Implicit VTK surface is not purely triangular")
    vtk_faces = data.reshape(-1, 4)
    if not np.all(vtk_faces[:, 0] == 3):
        raise RuntimeError("Implicit VTK surface contains non-triangle cells")
    return points, vtk_faces[:, 1:4].astype(int, copy=True)


def build_implicit_capsule_surface(
    centerline: np.ndarray,
    radii: np.ndarray,
    *,
    voxel_size: float,
    sample_spacing: float,
    padding: float,
    max_voxels: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Build a tapered capsule-like envelope without sequential rings.

    The scalar field is min_i(||x-c_i|| - r_i) over densely sampled centreline
    spheres.  Dense sampling makes the envelope converge to a swept/capsule
    tube while avoiding all ring-frame rotation problems.
    """
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    path = np.asarray(centerline, dtype=float)
    rr = np.asarray(radii, dtype=float)
    if len(path) < 2 or rr.shape != (len(path),):
        raise ValueError("implicit VoM needs >=2 centerline points and one radius per point")
    voxel = float(voxel_size)
    sample = float(sample_spacing)
    if voxel <= 0 or sample <= 0:
        raise ValueError("implicit voxel/sample spacing must be positive")

    dense_path, dense_radii = resample_path_and_profile(path, rr, spacing=sample)
    max_r = float(np.max(dense_radii))
    extra = max(float(padding), 2.0 * voxel)
    lo = np.min(dense_path, axis=0) - (max_r + extra)
    hi = np.max(dense_path, axis=0) + (max_r + extra)
    dims = np.ceil((hi - lo) / voxel).astype(int) + 1
    nvox = int(np.prod(dims, dtype=np.int64))
    if nvox > int(max_voxels):
        raise MemoryError(
            f"Implicit grid would contain {nvox:,} points ({dims.tolist()}). "
            f"Increase --vm-implicit-voxel-size or --vm-implicit-max-voxels."
        )

    # VTK x-fastest ordering corresponds to a NumPy (z,y,x) C-order array.
    phi = np.full((dims[2], dims[1], dims[0]), np.float32(1.0e9), dtype=np.float32)
    margin = voxel * 1.5
    for c, rad in zip(dense_path, dense_radii):
        extent = float(rad) + margin
        ilo = np.maximum(np.floor((c - extent - lo) / voxel).astype(int), 0)
        ihi = np.minimum(np.ceil((c + extent - lo) / voxel).astype(int), dims - 1)
        xs = lo[0] + np.arange(ilo[0], ihi[0] + 1) * voxel
        ys = lo[1] + np.arange(ilo[1], ihi[1] + 1) * voxel
        zs = lo[2] + np.arange(ilo[2], ihi[2] + 1) * voxel
        zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
        local = np.sqrt((xx-c[0])**2 + (yy-c[1])**2 + (zz-c[2])**2) - float(rad)
        sl = np.s_[ilo[2]:ihi[2]+1, ilo[1]:ihi[1]+1, ilo[0]:ihi[0]+1]
        phi[sl] = np.minimum(phi[sl], local.astype(np.float32, copy=False))

    image = vtk.vtkImageData()
    image.SetDimensions(int(dims[0]), int(dims[1]), int(dims[2]))
    image.SetOrigin(float(lo[0]), float(lo[1]), float(lo[2]))
    image.SetSpacing(voxel, voxel, voxel)
    values = numpy_to_vtk(phi.ravel(order="C"), deep=True)
    values.SetName("implicit_distance")
    image.GetPointData().SetScalars(values)

    contour = vtk.vtkFlyingEdges3D()
    contour.SetInputData(image)
    contour.SetValue(0, 0.0)
    contour.Update()

    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(contour.GetOutput())
    clean.Update()
    tri_filter = vtk.vtkTriangleFilter()
    tri_filter.SetInputData(clean.GetOutput())
    tri_filter.Update()
    pts, faces = _numpy_triangles_from_vtk(tri_filter.GetOutput())
    diag = {
        "dense_samples": int(len(dense_path)),
        "grid_nx": int(dims[0]), "grid_ny": int(dims[1]), "grid_nz": int(dims[2]),
        "grid_points": int(nvox),
        "raw_surface_points": int(len(pts)),
        "raw_surface_triangles": int(len(faces)),
    }
    return pts, faces, diag



def build_generalized_capsule_surface(
    centerline: np.ndarray,
    radii: np.ndarray,
    *,
    voxel_size: float,
    sample_spacing: float,
    padding: float,
    max_voxels: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Build a tapered tube from distance to centreline *segments*.

    Unlike the older union-of-spheres field, every consecutive centreline pair
    is treated as a generalized capsule/frustum segment.  The zero level set is
    therefore cylindrical on a straight constant-radius proximal section and
    does not develop the sphere-union bulges that complicated the mitral seam.
    """
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    path = np.asarray(centerline, dtype=float)
    rr = np.asarray(radii, dtype=float)
    if len(path) < 2 or rr.shape != (len(path),):
        raise ValueError("generalized capsule needs >=2 centerline points and one radius per point")
    voxel = float(voxel_size)
    sample = float(sample_spacing)
    if voxel <= 0 or sample <= 0:
        raise ValueError("implicit voxel/sample spacing must be positive")

    # Resampling limits the geometric error of the piecewise-linear centreline,
    # but the field below is evaluated against line segments, not point spheres.
    dense_path, dense_radii = resample_path_and_profile(path, rr, spacing=sample)
    max_r = float(np.max(dense_radii))
    extra = max(float(padding), 2.0 * voxel)
    lo = np.min(dense_path, axis=0) - (max_r + extra)
    hi = np.max(dense_path, axis=0) + (max_r + extra)
    dims = np.ceil((hi - lo) / voxel).astype(int) + 1
    nvox = int(np.prod(dims, dtype=np.int64))
    if nvox > int(max_voxels):
        raise MemoryError(
            f"Generalized-capsule grid would contain {nvox:,} points ({dims.tolist()}). "
            f"Increase --vm-implicit-voxel-size or --vm-implicit-max-voxels."
        )

    phi = np.full((dims[2], dims[1], dims[0]), np.float32(1.0e9), dtype=np.float32)
    margin = voxel * 1.75
    for a, b, ra, rb in zip(dense_path[:-1], dense_path[1:], dense_radii[:-1], dense_radii[1:]):
        ab = b - a
        lab2 = float(np.dot(ab, ab))
        if lab2 < 1.0e-12:
            continue
        rad_max = max(float(ra), float(rb))
        seg_lo = np.minimum(a, b) - (rad_max + margin)
        seg_hi = np.maximum(a, b) + (rad_max + margin)
        ilo = np.maximum(np.floor((seg_lo - lo) / voxel).astype(int), 0)
        ihi = np.minimum(np.ceil((seg_hi - lo) / voxel).astype(int), dims - 1)
        if np.any(ihi < ilo):
            continue

        xs = lo[0] + np.arange(ilo[0], ihi[0] + 1) * voxel
        ys = lo[1] + np.arange(ilo[1], ihi[1] + 1) * voxel
        zs = lo[2] + np.arange(ilo[2], ihi[2] + 1) * voxel
        zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
        pax = xx - a[0]
        pay = yy - a[1]
        paz = zz - a[2]
        t = (pax*ab[0] + pay*ab[1] + paz*ab[2]) / lab2
        t = np.clip(t, 0.0, 1.0)
        qx = a[0] + t*ab[0]
        qy = a[1] + t*ab[1]
        qz = a[2] + t*ab[2]
        rad = float(ra) + t*(float(rb)-float(ra))
        local = np.sqrt((xx-qx)**2 + (yy-qy)**2 + (zz-qz)**2) - rad
        sl = np.s_[ilo[2]:ihi[2]+1, ilo[1]:ihi[1]+1, ilo[0]:ihi[0]+1]
        phi[sl] = np.minimum(phi[sl], local.astype(np.float32, copy=False))

    image = vtk.vtkImageData()
    image.SetDimensions(int(dims[0]), int(dims[1]), int(dims[2]))
    image.SetOrigin(float(lo[0]), float(lo[1]), float(lo[2]))
    image.SetSpacing(voxel, voxel, voxel)
    values = numpy_to_vtk(phi.ravel(order="C"), deep=True)
    values.SetName("generalized_capsule_distance")
    image.GetPointData().SetScalars(values)

    contour = vtk.vtkFlyingEdges3D()
    contour.SetInputData(image)
    contour.SetValue(0, 0.0)
    contour.Update()
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(contour.GetOutput())
    clean.Update()
    tri_filter = vtk.vtkTriangleFilter()
    tri_filter.SetInputData(clean.GetOutput())
    tri_filter.Update()
    pts, faces = _numpy_triangles_from_vtk(tri_filter.GetOutput())
    diag = {
        "dense_segment_samples": int(len(dense_path)),
        "grid_nx": int(dims[0]), "grid_ny": int(dims[1]), "grid_nz": int(dims[2]),
        "grid_points": int(nvox),
        "raw_surface_points": int(len(pts)),
        "raw_surface_triangles": int(len(faces)),
    }
    return pts, faces, diag

def clip_implicit_tube_open(
    points: np.ndarray,
    faces: np.ndarray,
    centerline: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Clip hemispherical implicit caps at the first/last centreline planes."""
    import vtk

    path = np.asarray(centerline, dtype=float)
    tang = path_tangents(path)
    poly = _vtk_polydata_from_triangles(points, faces)

    # Start: retain positive travel side of the plane through path[0].
    start_plane = vtk.vtkPlane()
    start_plane.SetOrigin(*map(float, path[0]))
    start_plane.SetNormal(*map(float, tang[0]))
    clip1 = vtk.vtkClipPolyData()
    clip1.SetInputData(poly)
    clip1.SetClipFunction(start_plane)
    clip1.SetValue(0.0)
    clip1.SetInsideOut(False)
    clip1.Update()

    # End: retain negative travel side of the plane through path[-1].
    end_plane = vtk.vtkPlane()
    end_plane.SetOrigin(*map(float, path[-1]))
    end_plane.SetNormal(*map(float, tang[-1]))
    clip2 = vtk.vtkClipPolyData()
    clip2.SetInputData(clip1.GetOutput())
    clip2.SetClipFunction(end_plane)
    clip2.SetValue(0.0)
    clip2.SetInsideOut(True)
    clip2.Update()

    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(clip2.GetOutput())
    clean.Update()
    tri_filter = vtk.vtkTriangleFilter()
    tri_filter.SetInputData(clean.GetOutput())
    tri_filter.Update()
    return _numpy_triangles_from_vtk(tri_filter.GetOutput())



def _compact_triangle_mesh(points: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop unused vertices and remap triangle indices."""
    pts = np.asarray(points, dtype=float)
    tri = np.asarray(faces, dtype=int)
    if len(tri) == 0:
        raise RuntimeError("Local cap removal deleted the complete implicit surface")
    used = np.unique(tri.reshape(-1))
    remap = np.full(len(pts), -1, dtype=int)
    remap[used] = np.arange(len(used), dtype=int)
    return pts[used].copy(), remap[tri]


def _triangle_component_count(faces: np.ndarray) -> tuple[int, np.ndarray]:
    """Connected triangle components through shared vertices."""
    tri = np.asarray(faces, dtype=int)
    if len(tri) == 0:
        return 0, np.empty(0, dtype=int)
    vertex_to_faces: dict[int, list[int]] = {}
    for fi, face in enumerate(tri):
        for v in face:
            vertex_to_faces.setdefault(int(v), []).append(fi)
    labels = np.full(len(tri), -1, dtype=int)
    comp = 0
    for seed in range(len(tri)):
        if labels[seed] >= 0:
            continue
        stack = [seed]
        labels[seed] = comp
        while stack:
            fi = stack.pop()
            for v in tri[fi]:
                for nb in vertex_to_faces[int(v)]:
                    if labels[nb] < 0:
                        labels[nb] = comp
                        stack.append(nb)
        comp += 1
    return comp, labels


def keep_largest_triangle_component(
    points: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Keep only the largest connected triangle component and compact vertices."""
    ncomp, labels = _triangle_component_count(faces)
    if ncomp <= 1:
        pts, tri = _compact_triangle_mesh(points, faces)
        return pts, tri, ncomp
    sizes = np.bincount(labels, minlength=ncomp)
    largest = int(np.argmax(sizes))
    tri = np.asarray(faces, dtype=int)[labels == largest]
    pts, tri = _compact_triangle_mesh(points, tri)
    return pts, tri, ncomp


def open_implicit_caps_locally(
    points: np.ndarray,
    faces: np.ndarray,
    centerline: np.ndarray,
    radii: np.ndarray,
    *,
    gate_factor: float = 1.75,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Open only the local proximal/distal implicit hemispherical caps.

    Unlike an infinite vtkPlane clip, this deletes cap triangles only inside a
    small neighbourhood of the first/last centreline point.  A distant part of
    a curved VoM therefore cannot be cut accidentally by the same plane.
    """
    pts = np.asarray(points, dtype=float)
    tri = np.asarray(faces, dtype=int)
    path = np.asarray(centerline, dtype=float)
    rr = np.asarray(radii, dtype=float)
    if len(path) < 2 or len(rr) != len(path):
        raise ValueError("local implicit-cap opening requires centreline radii")

    tang = path_tangents(path)
    cent = pts[tri].mean(axis=1)

    p0, t0, r0 = path[0], normalize_vector(tang[0]), float(rr[0])
    p1, t1, r1 = path[-1], normalize_vector(tang[-1]), float(rr[-1])

    rel0 = cent - p0
    rel1 = cent - p1
    proj0 = rel0 @ t0
    proj1 = rel1 @ t1
    dist0 = np.linalg.norm(rel0, axis=1)
    dist1 = np.linalg.norm(rel1, axis=1)

    gate0 = max(float(gate_factor) * r0, r0 + 1.0)
    gate1 = max(float(gate_factor) * r1, r1 + 1.0)

    # The implicit union-of-spheres produces a hemispherical cap behind the
    # first point and ahead of the last point.  Remove only those local faces.
    drop_start = (proj0 < 0.0) & (dist0 <= gate0)
    drop_end = (proj1 > 0.0) & (dist1 <= gate1)
    keep = ~(drop_start | drop_end)
    kept_faces = tri[keep]

    pts2, tri2 = _compact_triangle_mesh(pts, kept_faces)
    pts2, tri2, components_before_keep = keep_largest_triangle_component(pts2, tri2)
    diag = {
        "local_cap_removed_start_faces": int(np.count_nonzero(drop_start)),
        "local_cap_removed_end_faces": int(np.count_nonzero(drop_end)),
        "local_cap_components_before_largest_keep": int(components_before_keep),
    }
    return pts2, tri2, diag

def orient_triangles_outward_from_centerline(
    points: np.ndarray,
    faces: np.ndarray,
    centerline: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Globally orient a consistently-wound tube surface away from its centreline."""
    pts = np.asarray(points, dtype=float)
    tri = np.asarray(faces, dtype=int).copy()
    path = np.asarray(centerline, dtype=float)
    if len(tri) == 0 or len(path) == 0:
        return tri, False

    centroids = pts[tri].mean(axis=1)
    normals = np.cross(pts[tri[:, 1]] - pts[tri[:, 0]], pts[tri[:, 2]] - pts[tri[:, 0]])
    tree = cKDTree(path)
    _, nearest = tree.query(centroids, k=1)
    radial = centroids - path[np.asarray(nearest, dtype=int)]
    dots = np.einsum("ij,ij->i", normals, radial)
    finite = np.isfinite(dots)
    flip = bool(np.median(dots[finite]) < 0.0) if np.any(finite) else False
    if flip:
        tri[:, [1, 2]] = tri[:, [2, 1]]
    return tri, flip


def orient_collar_faces_outward(
    points: np.ndarray,
    faces: np.ndarray,
    start_center: np.ndarray,
    end_center: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Orient the short mitral-to-implicit collar away from its centre axis."""
    pts = np.asarray(points, dtype=float)
    tri = np.asarray(faces, dtype=int).copy()
    if len(tri) == 0:
        return tri, False
    a = np.asarray(start_center, dtype=float)
    b = np.asarray(end_center, dtype=float)
    axis = b - a
    denom = float(np.dot(axis, axis))
    centroids = pts[tri].mean(axis=1)
    if denom < 1.0e-12:
        nearest = np.repeat(a[None, :], len(centroids), axis=0)
    else:
        u = np.clip(((centroids - a) @ axis) / denom, 0.0, 1.0)
        nearest = a[None, :] + u[:, None] * axis[None, :]
    radial = centroids - nearest
    normals = np.cross(pts[tri[:, 1]] - pts[tri[:, 0]], pts[tri[:, 2]] - pts[tri[:, 0]])
    dots = np.einsum("ij,ij->i", normals, radial)
    finite = np.isfinite(dots)
    flip = bool(np.median(dots[finite]) < 0.0) if np.any(finite) else False
    if flip:
        tri[:, [1, 2]] = tri[:, [2, 1]]
    return tri, flip


def boundary_loops_from_triangles(faces: np.ndarray) -> list[np.ndarray]:
    """Return ordered vertex-index loops for manifold triangle boundary edges."""
    f = np.asarray(faces, dtype=int)
    counts: dict[tuple[int, int], int] = {}
    for a, b, c in f:
        for u, v in ((a,b),(b,c),(c,a)):
            key = (u,v) if u < v else (v,u)
            counts[key] = counts.get(key, 0) + 1
    boundary = [edge for edge, count in counts.items() if count == 1]
    adjacency: dict[int, list[int]] = {}
    for u, v in boundary:
        adjacency.setdefault(u, []).append(v)
        adjacency.setdefault(v, []).append(u)

    unused = {tuple(sorted(e)) for e in boundary}
    loops: list[np.ndarray] = []
    while unused:
        u, v = next(iter(unused))
        start = u
        loop = [u, v]
        unused.discard(tuple(sorted((u,v))))
        prev, cur = u, v
        for _ in range(len(boundary) + 2):
            candidates = [w for w in adjacency.get(cur, []) if w != prev]
            if not candidates:
                break
            nxt = candidates[0]
            edge = tuple(sorted((cur, nxt)))
            if nxt == start:
                unused.discard(edge)
                break
            if edge not in unused:
                # Try the other boundary neighbour if this one was consumed.
                alternatives = [w for w in candidates[1:] if tuple(sorted((cur,w))) in unused]
                if not alternatives:
                    break
                nxt = alternatives[0]
                edge = tuple(sorted((cur,nxt)))
            unused.discard(edge)
            loop.append(nxt)
            prev, cur = cur, nxt
        if len(loop) >= 3:
            loops.append(np.asarray(loop, dtype=int))
    return loops


def orient_and_shift_loop(loop_idx: np.ndarray, points: np.ndarray, reference_ring: np.ndarray) -> np.ndarray:
    """Choose cyclic seam/winding of an implicit boundary loop near a ring."""
    idx = np.asarray(loop_idx, dtype=int).copy()
    loop = np.asarray(points, dtype=float)[idx]
    shift = int(np.argmin(np.linalg.norm(loop - reference_ring[0], axis=1)))
    idx = np.roll(idx, -shift)
    loop = points[idx]
    ref_n = ordered_ring_normal(reference_ring)
    loop_n = ordered_ring_normal(loop)
    if np.dot(ref_n, loop_n) < 0.0:
        idx = np.concatenate((idx[:1], idx[:0:-1]))
    return idx



def refine_boundary_loop_to_count(
    points: np.ndarray,
    faces: np.ndarray,
    loop_idx: np.ndarray,
    target_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Refine an existing manifold boundary loop by splitting its longest edges.

    The newly inserted points lie exactly on the existing boundary edges, and the
    adjacent surface triangles are split at the same time.  Therefore the refined
    loop remains part of the implicit surface rather than becoming a duplicate
    geometric ring floating on top of it.

    This routine only refines (increases) the boundary vertex count.
    """
    pts = np.asarray(points, dtype=float).copy()
    face_list = [list(map(int, tri)) for tri in np.asarray(faces, dtype=int)]
    loop = [int(v) for v in np.asarray(loop_idx, dtype=int)]
    target_count = int(target_count)

    if target_count < 3:
        raise ValueError("target boundary count must be at least 3")
    if len(loop) > target_count:
        raise ValueError(
            f"Implicit start boundary has {len(loop)} vertices, which is already "
            f"greater than the requested {target_count}; this refinement routine "
            "only inserts vertices."
        )

    original_count = len(loop)
    inserted = 0

    while len(loop) < target_count:
        # Split the longest current boundary edge.  Repeating this produces a
        # nearly uniform arc-length sampling while preserving the original curve.
        lengths = np.array([
            np.linalg.norm(pts[loop[(i + 1) % len(loop)]] - pts[loop[i]])
            for i in range(len(loop))
        ])
        i = int(np.argmax(lengths))
        a = int(loop[i])
        b = int(loop[(i + 1) % len(loop)])

        # A true manifold boundary edge belongs to exactly one triangle.
        incident = [
            fi for fi, tri in enumerate(face_list)
            if a in tri and b in tri
        ]
        if len(incident) != 1:
            raise RuntimeError(
                f"Expected boundary edge ({a}, {b}) to have exactly one incident "
                f"triangle; found {len(incident)}"
            )

        fi = incident[0]
        tri = face_list[fi]
        new_idx = len(pts)
        new_point = 0.5 * (pts[a] + pts[b])
        pts = np.vstack((pts, new_point))

        # Preserve the original triangle winding when splitting the boundary edge.
        directed_ab = any(
            tri[k] == a and tri[(k + 1) % 3] == b for k in range(3)
        )
        directed_ba = any(
            tri[k] == b and tri[(k + 1) % 3] == a for k in range(3)
        )
        c = next(v for v in tri if v != a and v != b)

        if directed_ab:
            first = [a, new_idx, c]
            second = [new_idx, b, c]
        elif directed_ba:
            first = [b, new_idx, c]
            second = [new_idx, a, c]
        else:
            raise RuntimeError(f"Boundary edge ({a}, {b}) is not an edge of face {tri}")

        face_list[fi] = first
        face_list.append(second)
        loop.insert(i + 1, new_idx)
        inserted += 1

    refined_faces = np.asarray(face_list, dtype=int)
    refined_loop = np.asarray(loop, dtype=int)
    final_lengths = np.array([
        np.linalg.norm(pts[refined_loop[(i + 1) % len(refined_loop)]] - pts[refined_loop[i]])
        for i in range(len(refined_loop))
    ])
    diag = {
        "boundary_original_vertices": float(original_count),
        "boundary_refined_vertices": float(len(refined_loop)),
        "boundary_inserted_vertices": float(inserted),
        "boundary_edge_min": float(final_lengths.min()),
        "boundary_edge_median": float(np.median(final_lengths)),
        "boundary_edge_max": float(final_lengths.max()),
    }
    return pts, refined_faces, refined_loop, diag


def match_equal_boundary_loops(
    ring_a: np.ndarray,
    idx_b: np.ndarray,
    points_b: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Order an equal-size second loop for minimum one-to-one seam distance."""
    A = np.asarray(ring_a, dtype=float)
    base = np.asarray(idx_b, dtype=int)
    if len(A) != len(base):
        raise ValueError("match_equal_boundary_loops requires equal loop sizes")

    best_idx = None
    best_cost = np.inf
    best_shift = 0
    best_reversed = False

    # Evaluate both cyclic directions.  This handles arbitrary marching-cubes
    # boundary winding without allowing a local zipper seam to wander.
    for reversed_flag in (False, True):
        seq = base if not reversed_flag else np.concatenate((base[:1], base[:0:-1]))
        for shift in range(len(seq)):
            cand = np.roll(seq, -shift)
            B = np.asarray(points_b, dtype=float)[cand]
            cost = float(np.mean(np.sum((A - B) ** 2, axis=1)))
            if cost < best_cost:
                best_cost = cost
                best_idx = cand.copy()
                best_shift = int(shift)
                best_reversed = bool(reversed_flag)

    B = np.asarray(points_b, dtype=float)[best_idx]
    distances = np.linalg.norm(A - B, axis=1)
    return best_idx, {
        "one_to_one_shift": float(best_shift),
        "one_to_one_reversed": float(best_reversed),
        "one_to_one_edge_min": float(distances.min()),
        "one_to_one_edge_median": float(np.median(distances)),
        "one_to_one_edge_max": float(distances.max()),
    }


def one_to_one_loop_faces(idx_a: np.ndarray, idx_b: np.ndarray) -> np.ndarray:
    """Triangulate equal-size closed loops without zipper fans or pivot vertices."""
    a = np.asarray(idx_a, dtype=int)
    b = np.asarray(idx_b, dtype=int)
    if len(a) != len(b):
        raise ValueError("one_to_one_loop_faces requires equal loop sizes")
    faces: list[list[int]] = []
    n = len(a)
    for i in range(n):
        j = (i + 1) % n
        faces.append([int(a[i]), int(a[j]), int(b[i])])
        faces.append([int(a[j]), int(b[j]), int(b[i])])
    return np.asarray(faces, dtype=int)

def zipper_indexed_loops(
    points_a: np.ndarray, idx_a: np.ndarray,
    points_b: np.ndarray, idx_b: np.ndarray,
) -> np.ndarray:
    """Greedy shortest-edge zipper using arbitrary global vertex-index loops."""
    A = np.asarray(points_a, dtype=float)[np.asarray(idx_a, dtype=int)]
    B = np.asarray(points_b, dtype=float)[np.asarray(idx_b, dtype=int)]
    ia = np.asarray(idx_a, dtype=int); ib = np.asarray(idx_b, dtype=int)
    na, nb = len(A), len(B)
    faces: list[list[int]] = []
    i = j = 0
    while i < na or j < nb:
        ai = i % na; bj = j % nb
        if i >= na:
            advance_a = False
        elif j >= nb:
            advance_a = True
        else:
            ai1 = (i + 1) % na; bj1 = (j + 1) % nb
            advance_a = np.linalg.norm(A[ai1] - B[bj]) <= np.linalg.norm(A[ai] - B[bj1])
        if advance_a:
            ai1 = (i + 1) % na
            faces.append([int(ia[ai]), int(ia[ai1]), int(ib[bj])])
            i += 1
        else:
            bj1 = (j + 1) % nb
            faces.append([int(ia[ai]), int(ib[bj1]), int(ib[bj])])
            j += 1
    return np.asarray(faces, dtype=int)


def _smootherstep01(t: float) -> float:
    """Quintic smoothstep on [0,1] with zero slope at both ends."""
    x = float(np.clip(t, 0.0, 1.0))
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def build_boundary_loft_rings(
    start_ring: np.ndarray,
    end_ring: np.ndarray,
    *,
    n_sections: int,
) -> list[np.ndarray]:
    """Create intermediate cross-sections between two arbitrary closed loops.

    Endpoints are NOT returned: the caller keeps the exact mitral and implicit
    boundary vertices and only adds these intermediate rings.  Both loops must
    already have compatible winding/seam orientation.
    """
    a = np.asarray(start_ring, dtype=float)
    b = np.asarray(end_ring, dtype=float)
    if n_sections < 2:
        raise ValueError("n_sections must be >= 2")
    if len(a) < 3 or len(b) < 3:
        raise ValueError("loft boundary loops need at least 3 vertices")

    rings: list[np.ndarray] = []
    for k in range(1, int(n_sections)):
        t = k / float(n_sections)
        w = _smootherstep01(t)
        n = max(3, int(round((1.0 - w) * len(a) + w * len(b))))
        ar = resample_closed_ring(a, n_points=n)
        br = resample_closed_ring(b, n_points=n)
        rings.append((1.0 - w) * ar + w * br)
    return rings


def build_shared_boundary_loft(
    prefix_points: np.ndarray,
    mitral_idx: np.ndarray,
    vm_points: np.ndarray,
    start_loop: np.ndarray,
    *,
    n_sections: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Build a loft sharing the exact mitral and implicit boundary vertices.

    Returns only newly-created intermediate points plus global-index faces.  The
    returned faces assume final point ordering:
      [prefix_points, vm_points, new_loft_points].
    """
    pfx = np.asarray(prefix_points, dtype=float)
    vmp = np.asarray(vm_points, dtype=float)
    mi = np.asarray(mitral_idx, dtype=int)
    si = np.asarray(start_loop, dtype=int)
    a = pfx[mi]
    b = vmp[si]

    mids = build_boundary_loft_rings(a, b, n_sections=n_sections)
    base_new = len(pfx) + len(vmp)
    new_points = np.vstack(mids) if mids else np.empty((0, 3), dtype=float)

    loops_global: list[np.ndarray] = [mi]
    off = base_new
    for ring in mids:
        idx = np.arange(off, off + len(ring), dtype=int)
        loops_global.append(idx)
        off += len(ring)
    loops_global.append(len(pfx) + si)

    # Temporary complete coordinates so zipper distances use real positions.
    all_pts = np.vstack((pfx, vmp, new_points))
    faces_parts: list[np.ndarray] = []
    flips = 0
    centres: list[np.ndarray] = [np.mean(a, axis=0)]
    centres.extend(np.mean(r, axis=0) for r in mids)
    centres.append(np.mean(b, axis=0))

    for i in range(len(loops_global) - 1):
        f = zipper_indexed_loops(
            all_pts, loops_global[i],
            all_pts, loops_global[i + 1],
        )
        f, flipped = orient_collar_faces_outward(
            all_pts, f, centres[i], centres[i + 1]
        )
        flips += int(flipped)
        faces_parts.append(f)

    faces = np.vstack(faces_parts).astype(int) if faces_parts else np.empty((0, 3), dtype=int)
    diag = {
        "loft_sections": float(n_sections),
        "loft_intermediate_rings": float(len(mids)),
        "loft_start_vertices": float(len(a)),
        "loft_end_vertices": float(len(b)),
        "loft_new_vertices": float(len(new_points)),
        "loft_faces": float(len(faces)),
        "loft_strips_flipped": float(flips),
        "loft_endpoint_center_distance": float(np.linalg.norm(np.mean(b, axis=0) - np.mean(a, axis=0))),
    }
    return new_points, faces, diag


def simple_topology_quality(points: np.ndarray, faces: np.ndarray) -> dict[str, int | float]:
    """Topology/area checks for an (N,3) triangle mesh."""
    f = np.asarray(faces, dtype=int); p = np.asarray(points, dtype=float)
    counts: dict[tuple[int,int], int] = {}
    for a,b,c in f:
        for u,v in ((a,b),(b,c),(c,a)):
            key=(u,v) if u<v else (v,u)
            counts[key]=counts.get(key,0)+1
    tri=p[f]
    area=0.5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]), axis=1)
    return {
        "n_points": int(len(p)), "n_triangles": int(len(f)),
        "min_triangle_area": float(np.min(area)),
        "zero_area_triangles": int(np.sum(area <= 1.0e-8)),
        "boundary_edges": int(sum(v == 1 for v in counts.values())),
        "nonmanifold_edges": int(sum(v > 2 for v in counts.values())),
    }


def load_saved_prefix_rings(output_dir: Path) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Load already accepted CS->mitral elbow + mitral checkpoint geometry."""
    cs_path = output_dir / "cs_ring_hybrid.pts"
    elbow_path = output_dir / "elbow_cs_mitral_hybrid.pts"
    mitral_path = output_dir / "mitral_hybrid.pts"
    missing = [str(p) for p in (cs_path, elbow_path, mitral_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Resume mode needs the already-generated first-four-stage files. Missing:\n  "
            + "\n  ".join(missing)
        )
    cs_ring = read_pts(output_dir / "cs_ring_hybrid")
    n = len(cs_ring)
    elbow_points = read_pts(output_dir / "elbow_cs_mitral_hybrid")
    mitral_points = read_pts(output_dir / "mitral_hybrid")
    if len(elbow_points) % n != 0 or len(mitral_points) % n != 0:
        raise ValueError("Saved elbow/mitral point counts are not divisible by the CS ring point count")
    elbow_rings = elbow_points.reshape(-1, n, 3)
    mitral_rings = mitral_points.reshape(-1, n, 3)
    if not np.allclose(elbow_rings[-1], mitral_rings[0], atol=1.0e-5, rtol=0.0):
        raise ValueError("Saved elbow and mitral do not share the expected exact interface ring")
    prefix: list[np.ndarray] = [r.copy() for r in elbow_rings]
    prefix.extend(r.copy() for r in mitral_rings[1:])
    centers = np.asarray([ring_center(r) for r in mitral_rings], dtype=float)
    mitral_tangents = path_tangents(centers)
    return prefix, mitral_rings[-1].copy(), mitral_tangents[-1].copy(), mitral_rings



def build_cs_mitral_checkpoint(
    args: argparse.Namespace,
    final_dir: Path,
    la_dir: Path,
    output_dir: Path,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Build only the validated CS -> elbow -> mitral prefix and save it.

    This is the accepted first-four-stage construction used throughout the
    later VoM experiments.  No post-mitral geometry is built here.
    """
    if not args.mesh_name_uac:
        raise ValueError(
            "--mesh-name-uac is required when the CS/mitral checkpoint must be built"
        )

    regions = load_regions(resolve_relative(final_dir, args.seed_region))
    if len(regions) < 2:
        raise ValueError(
            "RA Regions.txt needs at least two rows; row 2 (index 1) is the CS reference"
        )
    ra_cs_reference = np.asarray(regions[1], dtype=float)

    print("[prefix 1/4] Extracting CS ring and orienting it away from RA Regions[1]")
    raw_cs = extract_cs_ring(final_dir, args.mesh_name_la, regions)
    cs_ring = prepare_cs_ring(
        raw_cs,
        spacing=float(args.resolution),
        ra_reference=ra_cs_reference,
    )
    cs_center = ring_center(cs_ring)
    cs_outward = oriented_ring_normal_away_from_landmark(
        cs_ring, ra_cs_reference
    )

    write_pts(output_dir / "cs_ring_raw", raw_cs)
    write_pts(output_dir / "cs_ring_hybrid", cs_ring)
    write_pts(output_dir / "ra_cs_reference", np.asarray([ra_cs_reference]))
    write_pts(
        output_dir / "cs_outward_tip",
        np.asarray([cs_center, cs_center + 5000.0 * cs_outward]),
    )

    print("[prefix 2/4] Mapping the mitral guide from UAC")
    uac = read_hpts(la_dir / args.mesh_name_uac)
    la_points = read_pts(la_dir / args.mesh_name_la)

    rl_index = int(args.RL[0])
    rl_value = float(args.RL[1])
    mitral_targets = select_mitral_uac_targets(
        uac,
        u_min=float(args.mitral_u_min),
        u_max=float(args.mitral_u_max),
        v_value=float(args.mitral_v),
        tolerance=float(args.uac_tolerance),
    )
    mitral_mapped = map_uac_targets_to_surface(
        uac,
        la_points,
        mitral_targets,
        rl_index=rl_index,
        rl_value=rl_value,
        hscale=args.Hscale,
        branch_cut=args.branch_cut,
        branch_tolerance=float(args.branch_tol),
    )
    _, mitral_guide = process_guide(
        mitral_mapped,
        spacing=float(args.resolution),
        outlier_threshold=float(args.outlier_threshold),
        smoothing_sigma=float(args.smoothing_sigma),
        curve_scale=float(args.mitral_curve_scale),
        use_midpoints=not bool(args.no_midpoints),
    )
    mitral_guide = orient_path_start_near(mitral_guide, cs_center)
    write_pts(output_dir / "mitral_guide_original", mitral_guide)

    angle_values = np.arange(
        float(args.elbow_angle_min),
        float(args.elbow_angle_max) + 0.5 * float(args.elbow_angle_step),
        float(args.elbow_angle_step),
    )

    print("[prefix 3/4] Automatically choosing the accepted CS→mitral elbow")
    elbow1, table1 = search_elbow(
        start_ring=cs_ring,
        desired_start_tangent=cs_outward,
        target_guide=mitral_guide,
        la_points=la_points,
        target_end_radius=float(args.mitral_end_radius),
        resolution=float(args.resolution),
        clearance=float(args.la_clearance),
        angle_values=angle_values,
        offset_factors=args.elbow_offset_factors,
        max_hinges=int(args.max_hinge_candidates),
        min_bend_radius_factor=float(args.min_bend_radius_factor),
        clearance_ramp_fraction=float(args.cs_clearance_ramp_fraction),
        label="CS→mitral",
    )
    write_candidate_table(
        output_dir / "elbow_cs_mitral_candidates.csv", table1
    )
    write_pts(
        output_dir / "elbow_cs_mitral_centerline_hybrid",
        elbow1.centerline,
    )
    write_pts(
        output_dir / "mitral_guide_aligned",
        elbow1.aligned_guide,
    )
    write_carp_surface(
        output_dir / "elbow_cs_mitral_hybrid",
        elbow1.points,
        elbow1.faces,
        region=5,
    )

    print("[prefix 4/4] Building the mitral sequentially from the elbow last ring")
    (
        mitral_points,
        mitral_faces,
        _mitral_last_ring,
        _mitral_tangents,
        _mitral_rings,
        mitral_radii,
    ) = build_sequential_tube_from_ring(
        elbow1.last_ring,
        elbow1.aligned_guide,
        end_radius=float(args.mitral_end_radius),
        taper_start_fraction=0.5,
    )
    write_carp_surface(
        output_dir / "mitral_hybrid",
        mitral_points,
        mitral_faces,
        region=5,
    )

    mitral_curvature_ratio = minimum_curvature_ratio(
        elbow1.aligned_guide, mitral_radii
    )
    with (output_dir / "cs_mitral_checkpoint_summary.txt").open(
        "w", encoding="utf-8"
    ) as handle:
        handle.write("Accepted CS -> mitral checkpoint\n")
        handle.write("================================\n\n")
        handle.write(f"CS ring vertices: {len(cs_ring)}\n")
        handle.write(f"CS outward normal: {cs_outward.tolist()}\n")
        handle.write(f"Selected elbow hinge: {elbow1.hinge_index}\n")
        handle.write(f"Selected elbow angle_deg: {elbow1.angle_deg:.10g}\n")
        handle.write(f"Selected elbow offset: {elbow1.hinge_offset:.10g}\n")
        handle.write(f"Selected elbow score: {elbow1.score:.10g}\n")
        handle.write(
            "Mitral minimum curvature-radius/tube-radius ratio: "
            f"{mitral_curvature_ratio:.10g}\n"
        )

    return load_saved_prefix_rings(output_dir)


def ensure_cs_mitral_checkpoint(
    args: argparse.Namespace,
    final_dir: Path,
    la_dir: Path,
    output_dir: Path,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Load the prefix when valid; otherwise rebuild it automatically."""
    required = (
        output_dir / "cs_ring_hybrid.pts",
        output_dir / "elbow_cs_mitral_hybrid.pts",
        output_dir / "mitral_hybrid.pts",
    )

    if not bool(args.rebuild_prefix) and all(path.exists() for path in required):
        try:
            loaded = load_saved_prefix_rings(output_dir)
            print(
                "[prefix] Valid CS→mitral checkpoint found; "
                "reusing it without recomputation"
            )
            return loaded
        except (ValueError, FileNotFoundError) as exc:
            print(
                "[prefix] Existing checkpoint is incomplete/inconsistent; "
                f"rebuilding it ({exc})"
            )

    elif bool(args.rebuild_prefix):
        print("[prefix] --rebuild-prefix requested; rebuilding CS→mitral")
    else:
        missing = [path.name for path in required if not path.exists()]
        print(
            "[prefix] CS→mitral checkpoint not found; rebuilding. "
            f"Missing: {', '.join(missing)}"
        )

    return build_cs_mitral_checkpoint(args, final_dir, la_dir, output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the validated CS->mitral prefix when needed, otherwise resume it, "
            "then generate the final LA Regions[3]-directed generalized-capsule VoM."
        )
    )
    parser.add_argument("--target-path-la", required=True)
    parser.add_argument("--target-path-final", required=True)
    parser.add_argument("--mesh-name-la", default="Labelled")
    parser.add_argument(
        "--mesh-name-uac",
        default=None,
        help="LA UAC basename; required only when the CS/mitral checkpoint must be built",
    )
    parser.add_argument("--seed-region", default="Regions.txt")
    parser.add_argument("--output-dir", default=None)

    # Validated CS -> mitral prefix parameters.
    parser.add_argument("--rebuild-prefix", action="store_true",
                        help="force regeneration of CS ring, CS→mitral elbow and mitral")
    parser.add_argument("--resolution", type=float, default=350.0)
    parser.add_argument("--outlier-threshold", type=float, default=2500.0)
    parser.add_argument("--smoothing-sigma", type=float, default=2.0)
    parser.add_argument("--mitral-curve-scale", type=float, default=1.3)
    parser.add_argument("--no-midpoints", action="store_true")

    parser.add_argument("--mitral-u-min", type=float, default=0.2)
    parser.add_argument("--mitral-u-max", type=float, default=0.9)
    parser.add_argument("--mitral-v", type=float, default=0.0)
    parser.add_argument("--uac-tolerance", type=float, default=1.0e-8)
    parser.add_argument(
        "--RL", type=float, nargs=2, default=(3.0, 0.0),
        metavar=("COORDINATE", "VALUE"),
    )
    parser.add_argument(
        "--Hscale", type=float, nargs=2, default=(1.0, 1.0),
        metavar=("Z", "RHO"),
    )
    parser.add_argument(
        "--branch-cut", type=float, nargs=2, default=(-np.pi, np.pi),
        metavar=("MIN", "MAX"),
    )
    parser.add_argument("--branch-tol", type=float, default=0.05)

    parser.add_argument("--mitral-end-radius", type=float, default=2500.0)
    parser.add_argument("--elbow-angle-min", type=float, default=5.0)
    parser.add_argument("--elbow-angle-max", type=float, default=60.0)
    parser.add_argument("--elbow-angle-step", type=float, default=5.0)
    parser.add_argument(
        "--elbow-offset-factors", type=float, nargs="+",
        default=(0.5, 0.75, 1.0, 1.25, 1.5),
    )
    parser.add_argument("--max-hinge-candidates", type=int, default=24)
    parser.add_argument("--cs-clearance-ramp-fraction", type=float, default=0.35)

    parser.add_argument("--vom-end-radius", type=float, default=700.0)
    parser.add_argument("--vom-taper-start-fraction", type=float, default=0.35)
    parser.add_argument("--la-clearance", type=float, default=500.0,
                        help="desired approximate tube-wall gap from LA")
    parser.add_argument("--min-bend-radius-factor", type=float, default=1.5)
    parser.add_argument(
        "--vm-blend-handle-scales", type=float, nargs="+",
        default=(0.30, 0.50, 0.75, 1.0, 1.35, 1.75, 2.25, 3.0),
    )
    parser.add_argument("--vm-max-blend-attachment-index", type=int, default=12)
    parser.add_argument("--vm-surface-guide-smoothing", type=float, default=1.0)

    parser.add_argument("--vm-target-edge", type=float, default=350.0,
                        help="target circumferential/triangle edge scale in micrometres")
    parser.add_argument("--vm-axial-spacing", type=float, default=None,
                        help="ring-centre spacing; default sqrt(3)/2 * vm-target-edge")
    parser.add_argument("--vm-max-ring-count-change", type=int, default=1,
                        help="maximum ring vertex-count change per consecutive ring")
    parser.add_argument("--no-vm-ring-stagger", action="store_true")
    parser.add_argument("--vm-keep-fraction", type=float, default=2.0/3.0)
    parser.add_argument(
        "--vm-proximal-neck-length", type=float, default=None,
        help="straight mitral-to-implicit collar length in micrometres; default 2x last-mitral radius",
    )
    parser.add_argument(
        "--vm-loft-length", type=float, default=None,
        help="distance from the mitral end ring to the implicit-body start / loft junction; default 0.8x proximal-neck length",
    )
    parser.add_argument(
        "--vm-loft-sections", type=int, default=16,
        help="number of intermediate loft strips between the mitral ring and implicit opening",
    )
    parser.add_argument(
        "--vm-proximal-fit-length", type=float, default=5000.0,
        help="arc length over which the VoM guide is fitted from the exact mitral end back to the original LA-offset guide",
    )
    parser.add_argument(
        "--vm-proximal-radius-hold-length", type=float, default=2500.0,
        help="physical distance from the mitral end over which the VoM radius stays equal to the mitral radius",
    )
    parser.add_argument(
        "--vm-seam-length", type=float, default=350.0,
        help="short direct collar length between the exact mitral ring and the generalized-capsule opening",
    )
    parser.add_argument("--vm-implicit-voxel-size", type=float, default=250.0,
                        help="implicit field voxel size in micrometres; 250 gives edges near the 350-um scale")
    parser.add_argument("--vm-implicit-sample-spacing", type=float, default=125.0,
                        help="spacing of variable-radius sphere samples along the VoM centreline")
    parser.add_argument("--vm-implicit-padding", type=float, default=750.0,
                        help="extra field padding outside the maximum local radius")
    parser.add_argument("--vm-implicit-max-voxels", type=int, default=50000000,
                        help="safety limit for implicit-grid point count")
    parser.add_argument("--inspect", action="store_true")
    return parser


def inspect_no_vm_path(output_dir: Path, final_dir: Path, la_dir: Path, mesh_name: str) -> None:
    try:
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError("PyVista is required only for --inspect") from exc

    def surface(base: Path):
        pts = read_pts(base)
        tri, _ = read_triangles(base)
        faces = np.column_stack([np.full(len(tri), 3), tri]).reshape(-1)
        return pv.PolyData(pts, faces)

    p = pv.Plotter(title="CS / mitral + landmark-directed VoM")
    p.add_axes()
    seen: set[Path] = set()
    for directory in (final_dir, la_dir):
        base = (directory / mesh_name).resolve()
        if base in seen:
            continue
        if Path(f"{base}.pts").exists() and Path(f"{base}.elem").exists():
            seen.add(base)
            p.add_mesh(surface(base), opacity=0.10, show_edges=False, label=f"{directory.name}/{mesh_name}")

    for name in ("elbow_cs_mitral_hybrid", "mitral_hybrid", "vom_no_vm_path"):
        base = output_dir / name
        if Path(f"{base}.pts").exists():
            p.add_mesh(surface(base), show_edges=True, label=name)

    final_base = output_dir / "tube_cs_no_vm_path"
    if Path(f"{final_base}.pts").exists():
        p.add_mesh(surface(final_base), opacity=0.25, show_edges=False, label="tube_cs_no_vm_path")

    for name in (
        "cs_outward_tip",
        "la_region3_target",
        "vm_surface_geodesic",
        "vm_surface_guide",
        "vm_offset_guide",
        "vm_centerline_no_path",
        "vm_target_offset",
        "mitral_guide_aligned",
        "elbow_cs_mitral_centerline_hybrid",
    ):
        file = output_dir / f"{name}.pts"
        if not file.exists():
            continue
        pts = read_pts(file)
        if len(pts) == 1:
            p.add_points(pts, point_size=12, render_points_as_spheres=True, label=name)
        elif len(pts) > 1:
            p.add_mesh(pv.Spline(pts, max(2, len(pts) * 2)), line_width=4, label=name)

    p.add_legend()
    p.show()



def _smoothstep01(values: np.ndarray) -> np.ndarray:
    u = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def fit_proximal_guide_to_mitral(
    guide: np.ndarray,
    start_center: np.ndarray,
    start_tangent: np.ndarray,
    fit_length: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit only the proximal guide onto the exact mitral position/direction.

    A rigidly aligned copy of the complete guide has the desired start centre and
    tangent.  Its displacement relative to the original anatomical LA-offset guide
    is blended to zero with a C1 smoothstep over ``fit_length``.  Downstream of
    that distance the original guide is exactly unchanged.
    """
    g = np.asarray(guide, dtype=float)
    if len(g) < 3:
        raise ValueError("VoM offset guide needs at least three points")
    fit_length = float(fit_length)
    if fit_length <= 0.0:
        raise ValueError("vm-proximal-fit-length must be positive")

    aligned, _, _ = rigid_align_path_to_start(
        g,
        np.asarray(start_center, dtype=float),
        normalize_vector(start_tangent),
    )
    s, total = polyline_arclength(g)
    effective = min(fit_length, total)
    u = s / max(effective, 1.0e-12)
    weight = 1.0 - _smoothstep01(u)
    weight[s >= effective] = 0.0
    fitted = g + weight[:, None] * (aligned - g)
    fitted[0] = np.asarray(start_center, dtype=float)

    start_error = float(np.rad2deg(angle_between_vectors(
        robust_forward_tangent(fitted, 0, lookahead=min(4, len(fitted)-1)),
        normalize_vector(start_tangent),
    )))
    downstream_mask = s >= effective
    downstream_error = (
        float(np.max(np.linalg.norm(fitted[downstream_mask] - g[downstream_mask], axis=1)))
        if np.any(downstream_mask) else 0.0
    )
    diag = {
        "proximal_fit_requested_length": fit_length,
        "proximal_fit_effective_length": effective,
        "proximal_fit_start_translation": float(np.linalg.norm(aligned[0] - g[0])),
        "proximal_fit_start_tangent_error_deg": start_error,
        "proximal_fit_downstream_max_displacement": downstream_error,
    }
    return fitted, diag


def radius_profile_by_arclength(
    path: np.ndarray,
    start_radius: float,
    end_radius: float,
    hold_length: float,
) -> np.ndarray:
    """Smooth physical-length radius taper with a constant-radius proximal zone."""
    s, total = polyline_arclength(path)
    hold = float(np.clip(hold_length, 0.0, max(total - 1.0e-9, 0.0)))
    if total <= hold + 1.0e-9:
        return np.full(len(path), float(start_radius), dtype=float)
    u = (s - hold) / (total - hold)
    q = _smoothstep01(u)
    q[s <= hold] = 0.0
    return float(start_radius) + (float(end_radius) - float(start_radius)) * q


def slice_path_profile_from_arclength(
    path: np.ndarray,
    values: np.ndarray,
    start_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a path/profile beginning at an exact physical arc-length location."""
    p = np.asarray(path, dtype=float)
    v = np.asarray(values, dtype=float)
    s, total = polyline_arclength(p)
    d = float(np.clip(start_distance, 0.0, total))
    if d <= 1.0e-12:
        return p.copy(), v.copy()
    if d >= total - 1.0e-12:
        raise ValueError("seam length reaches the end of the VoM centreline")
    left = int(np.searchsorted(s, d, side="right") - 1)
    left = max(0, min(left, len(p) - 2))
    ds = s[left + 1] - s[left]
    alpha = 0.0 if ds <= 1.0e-12 else float((d - s[left]) / ds)
    cut_p = (1.0 - alpha) * p[left] + alpha * p[left + 1]
    cut_v = (1.0 - alpha) * v[left] + alpha * v[left + 1]
    out_p = np.vstack((cut_p[None, :], p[left + 1:]))
    out_v = np.concatenate(([cut_v], v[left + 1:]))
    return out_p, out_v


def main() -> None:
    args = build_parser().parse_args()
    final_dir = Path(args.target_path_final).expanduser().resolve()
    la_dir = Path(args.target_path_la).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else final_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    axial_spacing = (
        float(args.vm_axial_spacing)
        if args.vm_axial_spacing is not None
        else (np.sqrt(3.0) / 2.0) * float(args.vm_target_edge)
    )

    prefix_rings, mitral_last_ring, mitral_exit_tangent, _ = (
        ensure_cs_mitral_checkpoint(args, final_dir, la_dir, output_dir)
    )

    la_regions = load_regions(resolve_relative(la_dir, args.seed_region))
    if len(la_regions) < 4:
        raise ValueError("The LA Regions.txt needs at least four rows; Regions[3] is the VoM target")
    region3_target = np.asarray(la_regions[3], dtype=float)
    la_points = read_pts(la_dir / args.mesh_name_la)
    la_triangles, _ = read_triangles(la_dir / args.mesh_name_la)
    write_pts(output_dir / "la_region3_target", np.asarray([region3_target]))

    print("[1/3] Building the accepted LA-offset route toward LA Regions[3]")
    (
        vm_surface_geodesic,
        vm_surface_guide,
        vm_offset_guide,
        vm_anchors,
        _vm_normals,
        _vm_guide_diag,
    ) = prepare_landmark_offset_guide(
        start_center=ring_center(mitral_last_ring),
        landmark=region3_target,
        la_points=la_points,
        la_triangles=la_triangles,
        start_radius=ring_radius(mitral_last_ring),
        end_radius=args.vom_end_radius,
        clearance=args.la_clearance,
        resolution=axial_spacing,
        smoothing_sigma=args.vm_surface_guide_smoothing,
        taper_start_fraction=args.vom_taper_start_fraction,
    )
    write_pts(output_dir / "vm_surface_geodesic", vm_surface_geodesic)
    write_pts(output_dir / "vm_surface_guide", vm_surface_guide)
    write_pts(output_dir / "vm_offset_guide", vm_offset_guide)
    write_pts(output_dir / "vm_surface_anchors", vm_anchors)
    write_pts(output_dir / "vm_target_offset", np.asarray([vm_offset_guide[-1]]))

    print("[2/3] Pulling only the proximal VoM guide onto the exact mitral end, then building one generalized capsule")
    mitral_center = ring_center(mitral_last_ring)
    mitral_radius = ring_radius(mitral_last_ring)
    mitral_dir = normalize_vector(mitral_exit_tangent)

    fitted_full, fit_diag = fit_proximal_guide_to_mitral(
        vm_offset_guide,
        mitral_center,
        mitral_dir,
        fit_length=float(args.vm_proximal_fit_length),
    )
    write_pts(output_dir / "vm_offset_guide_proximal_fitted_full", fitted_full)

    # Radius is constant near the mitral and then tapers smoothly.  Keep the
    # existing 2/3 physical-length rule after the proximal fit.
    full_radii = radius_profile_by_arclength(
        fitted_full,
        mitral_radius,
        float(args.vom_end_radius),
        hold_length=float(args.vm_proximal_radius_hold_length),
    )
    vm_trimmed, radii_trimmed, vm_full_length, vm_retained_length = truncate_path_and_profile_by_arc_fraction(
        fitted_full,
        full_radii,
        keep_fraction=float(args.vm_keep_fraction),
    )
    vm_centerline, vm_radii = resample_path_and_profile(
        vm_trimmed,
        radii_trimmed,
        spacing=min(axial_spacing, float(args.vm_implicit_sample_spacing)),
    )
    vm_centerline[0] = mitral_center
    vm_radii[0] = mitral_radius
    write_pts(output_dir / "vm_centerline_generalized_capsule_proximal_fit", vm_centerline)

    # Use only a one-edge-length direct collar.  The VoM itself has already been
    # moved onto the mitral, so this is not a geometric connector/blend.
    seam_length = min(
        float(args.vm_seam_length),
        0.25 * max(float(args.vm_proximal_fit_length), float(args.vm_target_edge)),
    )
    body_centerline, body_radii = slice_path_profile_from_arclength(
        vm_centerline, vm_radii, seam_length
    )
    seam_center = body_centerline[0].copy()
    write_pts(output_dir / "vm_generalized_capsule_body_centerline", body_centerline)

    raw_pts, raw_faces, implicit_diag = build_generalized_capsule_surface(
        body_centerline,
        body_radii,
        voxel_size=float(args.vm_implicit_voxel_size),
        sample_spacing=float(args.vm_implicit_sample_spacing),
        padding=float(args.vm_implicit_padding),
        max_voxels=int(args.vm_implicit_max_voxels),
    )
    vm_points, vm_faces, cap_diag = open_implicit_caps_locally(
        raw_pts, raw_faces, body_centerline, body_radii, gate_factor=1.75
    )
    implicit_diag.update(cap_diag)
    vm_faces, vm_faces_flipped = orient_triangles_outward_from_centerline(
        vm_points, vm_faces, body_centerline
    )

    loops = boundary_loops_from_triangles(vm_faces)
    if len(loops) != 2:
        centres_dbg = [np.mean(vm_points[L], axis=0) for L in loops]
        raise RuntimeError(
            f"Expected exactly two openings on generalized capsule; found {len(loops)}. "
            f"Loop sizes={[len(L) for L in loops]}, centres={centres_dbg}"
        )
    loop_centres = [np.mean(vm_points[L], axis=0) for L in loops]
    start_loop_i = int(np.argmin([np.linalg.norm(c - seam_center) for c in loop_centres]))
    start_loop_raw = orient_and_shift_loop(loops[start_loop_i], vm_points, mitral_last_ring)
    end_loop = loops[1 - start_loop_i]

    # Refine the *actual implicit boundary itself* to the exact mitral node count.
    # This is not a duplicate/resampled floating ring: the adjacent implicit
    # triangles are split simultaneously, so the new boundary remains part of
    # the capsule mesh.
    vm_points, vm_faces, start_loop, boundary_refine_diag = refine_boundary_loop_to_count(
        vm_points, vm_faces, start_loop_raw, len(mitral_last_ring)
    )
    start_loop, one_to_one_diag = match_equal_boundary_loops(
        mitral_last_ring, start_loop, vm_points
    )

    write_carp_surface(output_dir / "vom_generalized_capsule_proximal_fit", vm_points, vm_faces, region=5)
    write_pts(output_dir / "vm_generalized_start_boundary", vm_points[start_loop])
    write_pts(output_dir / "vm_generalized_end_boundary", vm_points[end_loop])

    print(
        "[3/3] Joining mitral→VoM with equal-node one-to-one seam "
        f"({len(mitral_last_ring)}↔{len(start_loop)}; no zipper fan)"
    )
    prefix_points, prefix_faces = flatten_variable_rings(prefix_rings)
    prefix_last_offset = len(prefix_points) - len(mitral_last_ring)
    mitral_idx = np.arange(prefix_last_offset, len(prefix_points), dtype=int)
    vm_offset = len(prefix_points)
    vm_idx = vm_offset + np.asarray(start_loop, dtype=int)
    combined_base = np.vstack((prefix_points, vm_points))
    collar_faces = one_to_one_loop_faces(mitral_idx, vm_idx)
    collar_faces, collar_flipped = orient_collar_faces_outward(
        combined_base,
        collar_faces,
        mitral_center,
        np.mean(vm_points[start_loop], axis=0),
    )
    final_points = combined_base
    final_faces = np.vstack((prefix_faces, vm_faces + vm_offset, collar_faces)).astype(int)

    write_carp_surface(output_dir / "tube_cs_generalized_capsule_proximal_fit", final_points, final_faces, region=5)

    collar_local_points = np.vstack((mitral_last_ring, vm_points[start_loop]))
    collar_a = np.arange(len(mitral_last_ring), dtype=int)
    collar_b = len(mitral_last_ring) + np.arange(len(start_loop), dtype=int)
    collar_local_faces = one_to_one_loop_faces(collar_a, collar_b)
    collar_local_faces, _ = orient_collar_faces_outward(
        collar_local_points,
        collar_local_faces,
        mitral_center,
        np.mean(vm_points[start_loop], axis=0),
    )
    write_carp_surface(output_dir / "vm_mitral_short_collar", collar_local_points, collar_local_faces, region=5)

    # Compatibility aliases for the existing viewer/workflow.
    write_carp_surface(output_dir / "vom_generalized_capsule", vm_points, vm_faces, region=5)
    write_carp_surface(output_dir / "tube_cs_generalized_capsule", final_points, final_faces, region=5)
    write_carp_surface(output_dir / "vom_implicit_capsule", vm_points, vm_faces, region=5)
    write_carp_surface(output_dir / "tube_cs_implicit_capsule", final_points, final_faces, region=5)
    write_carp_surface(output_dir / "vom_no_vm_path", vm_points, vm_faces, region=5)
    write_carp_surface(output_dir / "tube_cs_no_vm_path", final_points, final_faces, region=5)
    # Production alias used by the downstream biatrial workflow.
    write_carp_surface(output_dir / "tube_cs", final_points, final_faces, region=5)

    vm_quality = triangle_mesh_metrics(vm_points, vm_faces, float(args.vm_target_edge))
    topo = simple_topology_quality(final_points, final_faces)
    la_tree = cKDTree(la_points)
    vm_la_dist, _ = la_tree.query(vm_points, k=1)
    end_to_region = float(np.linalg.norm(np.mean(vm_points[end_loop], axis=0) - region3_target))
    start_center = np.mean(vm_points[start_loop], axis=0)
    start_center_error = float(np.linalg.norm(start_center - seam_center))
    start_radius_mean = float(np.mean(np.linalg.norm(vm_points[start_loop] - seam_center, axis=1)))

    summary = output_dir / "generalized_capsule_proximal_fit_summary.txt"
    with summary.open("w", encoding="utf-8") as h:
        h.write("Resume-only generalized-capsule VoM: proximal-fit version\n")
        h.write("=====================================================\n\n")
        h.write("Stages 1–4: loaded from accepted outputs; not recomputed.\n")
        h.write("VoM construction: proximal LA-offset guide fitted directly onto the exact mitral end; correction fades smoothly to zero downstream; no large connector.\n")
        h.write(f"LA target Regions[3]: {region3_target.tolist()}\n")
        h.write(f"Requested LA clearance: {args.la_clearance:.6g}\n")
        h.write(f"Full fitted centreline length: {vm_full_length:.6g}\n")
        h.write(f"Retained centreline length: {vm_retained_length:.6g}\n")
        h.write(f"Retained fraction: {vm_retained_length/vm_full_length:.10g}\n")
        h.write(f"Short collar length: {seam_length:.6g}\n")
        h.write(f"Mitral radius: {mitral_radius:.6g}\n")
        h.write(f"Generalized start-loop centre error from expected seam centre: {start_center_error:.6g}\n")
        h.write(f"Generalized start-loop mean radius from expected seam centre: {start_radius_mean:.6g}\n")
        h.write(f"Generalized faces globally flipped outward: {vm_faces_flipped}\n")
        h.write(f"Short collar faces globally flipped outward: {collar_flipped}\n")
        h.write(f"Implicit start boundary original vertices: {int(boundary_refine_diag['boundary_original_vertices'])}\n")
        h.write(f"Implicit start boundary refined vertices: {int(boundary_refine_diag['boundary_refined_vertices'])}\n")
        h.write(f"Boundary inserted vertices: {int(boundary_refine_diag['boundary_inserted_vertices'])}\n")
        h.write(f"Refined boundary edge min/median/max: {boundary_refine_diag['boundary_edge_min']:.6g} / {boundary_refine_diag['boundary_edge_median']:.6g} / {boundary_refine_diag['boundary_edge_max']:.6g}\n")
        h.write(f"One-to-one cyclic shift: {int(one_to_one_diag['one_to_one_shift'])}\n")
        h.write(f"One-to-one reversed: {bool(one_to_one_diag['one_to_one_reversed'])}\n")
        h.write(f"One-to-one seam edge min/median/max: {one_to_one_diag['one_to_one_edge_min']:.6g} / {one_to_one_diag['one_to_one_edge_median']:.6g} / {one_to_one_diag['one_to_one_edge_max']:.6g}\n")
        for key, value in fit_diag.items():
            h.write(f"{key}: {value}\n")
        for key, value in implicit_diag.items():
            h.write(f"{key}: {value}\n")
        h.write(f"Start boundary vertices: {len(start_loop)}\n")
        h.write(f"End boundary vertices: {len(end_loop)}\n")
        h.write(f"Min generalized VoM vertex distance to LA vertices: {float(np.min(vm_la_dist)):.6g}\n")
        h.write(f"Final VoM opening centre -> Regions[3]: {end_to_region:.6g}\n\n")
        h.write("Generalized VoM triangle quality\n")
        for key, value in vm_quality.items():
            h.write(f"  {key}: {value}\n")
        h.write("\nWhole-surface topology quality\n")
        for key, value in topo.items():
            h.write(f"  {key}: {value}\n")

    print(
        f"    proximal fit: {fit_diag['proximal_fit_effective_length']:.1f} µm; "
        f"start tangent error={fit_diag['proximal_fit_start_tangent_error_deg']:.2f}°"
    )
    print(
        f"    VoM shortened by arc length: kept {vm_retained_length:.1f} / {vm_full_length:.1f} "
        f"({100*vm_retained_length/vm_full_length:.1f}%)"
    )
    print(
        f"    proximal opening: centre error={start_center_error:.1f} µm, "
        f"mean radius={start_radius_mean:.1f} µm (mitral={mitral_radius:.1f})"
    )
    print(
        f"    short collar={seam_length:.1f} µm; "
        f"boundary_edges={topo['boundary_edges']}, nonmanifold_edges={topo['nonmanifold_edges']}"
    )
    print(f"Generalized VoM: {output_dir / 'vom_generalized_capsule_proximal_fit'}")
    print(f"Final surface: {output_dir / 'tube_cs_generalized_capsule_proximal_fit'}")
    print(f"Production alias: {output_dir / 'tube_cs'}")
    print(f"Diagnostics: {summary}")

    if args.inspect:
        inspect_no_vm_path(output_dir, final_dir, la_dir, args.mesh_name_la)


if __name__ == "__main__":
    main()
