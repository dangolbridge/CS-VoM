#!/usr/bin/env python3
"""Build the two Bachmann-bundle connector surfaces used by this workflow.

The completed AtrialMTK ``Bilayer_Combined`` mesh is treated as the upstream
input. It already contains the disconnected RA-side and LA-side BB surfaces.
This script builds only the two connector surfaces:

* closest-region bridge (default tag 70)
* anatomically guided bridge (default tag 80)

Both are written as standalone CARP surfaces for inspection and later merging.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from geometry_hybrid import (
    assign_two_mesh_components_by_references,
    boundary_vertex_set,
    faces_within_vertex_set,
    geodesic_line_through_centroid_toward_reference,
    geodesic_through_candidate_band,
    nearest_component_contact_pairs,
    nearest_reference_vertices,
    normalize_vector,
    orient_second_polyline_for_pairing,
    parallel_geodesic_line_through_centroid,
    polyline_length,
    principal_axis_from_paired_midpoints,
    ruled_surface_between_geodesics,
)
from mesh_io_hybrid import (
    load_regions,
    read_pts,
    read_triangles,
    write_carp_surface,
    write_pts,
    write_scalar_values,
)
from mesh_tags import (
    BB_ANATOMICAL_BRIDGE_TAG,
    BB_CLOSEST_BRIDGE_TAG,
    LA_EPI_REGIONS,
    RA_EPI_REGIONS,
)


def resolve_mesh_base(directory: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = directory / path
    return path.with_suffix("") if path.suffix in {".pts", ".elem", ".lon"} else path


def resolve_file(directory: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else directory / path


def vertices_from_regions(faces, regions, wanted_regions) -> np.ndarray:
    mask = np.isin(np.asarray(regions, dtype=int), np.asarray(wanted_regions, dtype=int))
    selected = np.asarray(faces, dtype=int)[mask]
    return np.unique(selected.reshape(-1)).astype(int) if len(selected) else np.empty(0, int)


def triangular_profile(cross_fraction, edge_value: float, center_value: float):
    t = np.asarray(cross_fraction, dtype=float)
    return edge_value + (center_value - edge_value) * (1.0 - 2.0 * np.abs(t - 0.5))


def build_surface_from_lines(line_ra, line_la, spacing: float):
    line_la, reversed_la, direct_score, reverse_score = orient_second_polyline_for_pairing(
        line_ra, line_la
    )
    points, faces, cross_fraction, common_fraction = ruled_surface_between_geodesics(
        line_ra, line_la, cross_spacing=float(spacing)
    )
    return {
        "ra_line": np.asarray(line_ra, float),
        "la_line": np.asarray(line_la, float),
        "bridge_points": points,
        "bridge_faces": faces,
        "cross_fraction": cross_fraction,
        "common_fraction": common_fraction,
        "la_reversed": reversed_la,
        "direct_score": direct_score,
        "reverse_score": reverse_score,
    }


def build_closest_bridge(points, ra_component, la_component, ra_faces, la_faces, *, max_pairs, spacing):
    pairs = nearest_component_contact_pairs(
        points, ra_component, la_component, max_pairs=max_pairs
    )
    axis = principal_axis_from_paired_midpoints(points, pairs)
    ra_candidates = np.asarray([ra for _, ra, _ in pairs], dtype=int)
    la_candidates = np.asarray([la for _, _, la in pairs], dtype=int)
    ra_start, ra_end, ra_idx = geodesic_through_candidate_band(
        points, ra_faces, ra_candidates, axis
    )
    la_start, la_end, la_idx = geodesic_through_candidate_band(
        points, la_faces, la_candidates, axis
    )
    result = build_surface_from_lines(points[ra_idx], points[la_idx], spacing)
    result.update(
        pairs=pairs,
        axis=axis,
        ra_start=ra_start,
        ra_end=ra_end,
        la_start=la_start,
        la_end=la_end,
    )
    return result


def build_anatomical_bridge(
    points,
    ra_component,
    la_component,
    ra_faces,
    la_faces,
    ra_boundary,
    la_boundary,
    svc_point,
    *,
    spacing,
    normal_neighbors,
):
    (
        ra_center,
        ra_svc_endpoint,
        ra_opposite_endpoint,
        ra_idx,
        ra_path_to_svc,
        ra_path_to_opposite,
    ) = geodesic_line_through_centroid_toward_reference(
        points, ra_faces, ra_component, ra_boundary, svc_point
    )
    ra_line = points[ra_idx]
    ra_direction = normalize_vector(
        ra_line[-1] - ra_line[0], name="overall RA BB attachment direction"
    )

    la = parallel_geodesic_line_through_centroid(
        points,
        la_faces,
        la_component,
        la_boundary,
        ra_direction,
        target_negative_vertices=len(ra_path_to_svc),
        target_positive_vertices=len(ra_path_to_opposite),
        normal_neighbors=normal_neighbors,
    )
    if la["negative_limited"]:
        print("Warning: LA negative side is shorter than the RA reference extent.", flush=True)
    if la["positive_limited"]:
        print("Warning: LA positive side is shorter than the RA reference extent.", flush=True)

    result = build_surface_from_lines(ra_line, points[la["line_indices"]], spacing)
    result.update(
        ra_center=ra_center,
        ra_svc_endpoint=ra_svc_endpoint,
        ra_opposite_endpoint=ra_opposite_endpoint,
        ra_path_to_svc=ra_path_to_svc,
        ra_path_to_opposite=ra_path_to_opposite,
        la=la,
    )
    return result


def write_bridge(output_base, result, *, tag, edge_thickness, center_thickness):
    regions = np.full(len(result["bridge_faces"]), int(tag), dtype=int)
    write_carp_surface(
        output_base,
        result["bridge_points"],
        result["bridge_faces"],
        region=regions,
        fiber=(1.0, 0.0, 0.0),
    )
    values = triangular_profile(result["cross_fraction"], edge_thickness, center_thickness)
    write_scalar_values(Path(f"{output_base}.f3d"), values)


def retention_error(line, result, side: str) -> float:
    n_along = len(result["common_fraction"])
    boundary = (
        result["bridge_points"][:n_along]
        if side == "ra"
        else result["bridge_points"][-n_along:]
    )
    distances, _ = nearest_reference_vertices(line, boundary, k=1)
    return float(np.max(distances))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build both BB connector surfaces")
    p.add_argument("--target-path", required=True)
    p.add_argument("--mesh-name", default="Bilayer_Combined")
    p.add_argument("--bb-mesh", default="BB")
    p.add_argument("--ra-path", default=None)
    p.add_argument("--ra-landmarks", default="Landmarks.txt")
    p.add_argument("--svc-landmark-index", type=int, default=4)
    p.add_argument("--svc-point", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"))
    p.add_argument("--scale-factor", type=float, default=1.0)

    p.add_argument("--closest-output", default="bb_closest_bridge")
    p.add_argument("--closest-tag", type=int, default=BB_CLOSEST_BRIDGE_TAG)
    p.add_argument("--closest-candidate-pairs", type=int, default=200)
    p.add_argument("--closest-spacing", type=float, default=350.0)
    p.add_argument("--closest-edge-thickness", type=float, default=2000.0)
    p.add_argument("--closest-center-thickness", type=float, default=3000.0)

    p.add_argument("--anatomical-output", default="bb_anatomical_bridge")
    p.add_argument("--anatomical-tag", type=int, default=BB_ANATOMICAL_BRIDGE_TAG)
    p.add_argument("--anatomical-spacing", type=float, default=350.0)
    p.add_argument("--anatomical-edge-thickness", type=float, default=2000.0)
    p.add_argument("--anatomical-center-thickness", type=float, default=5000.0)
    p.add_argument("--la-normal-neighbors", type=int, default=30)
    p.add_argument("--debug-prefix", default="bb_bridges")
    p.add_argument("--report", default="bb_bridges_report.txt")
    return p


def main() -> None:
    args = parser().parse_args()
    target_dir = Path(args.target_path).expanduser().resolve()
    ra_dir = Path(args.ra_path).expanduser().resolve() if args.ra_path else target_dir

    target_base = resolve_mesh_base(target_dir, args.mesh_name)
    target_points = read_pts(target_base)
    target_faces, target_regions = read_triangles(target_base)
    la_vertices = vertices_from_regions(target_faces, target_regions, LA_EPI_REGIONS)
    ra_vertices = vertices_from_regions(target_faces, target_regions, RA_EPI_REGIONS)
    if not len(la_vertices) or not len(ra_vertices):
        raise ValueError("Could not recover standard AtrialMTK LA/RA epicardial regions")

    bb_base = resolve_mesh_base(target_dir, args.bb_mesh)
    bb_points = read_pts(bb_base)
    bb_faces, _ = read_triangles(bb_base)
    ra_component, la_component, n_components = assign_two_mesh_components_by_references(
        bb_points,
        bb_faces,
        target_points[ra_vertices],
        target_points[la_vertices],
    )
    if n_components > 2:
        print(f"Warning: BB mesh has {n_components} components; using the two largest.", flush=True)

    ra_faces = faces_within_vertex_set(bb_faces, ra_component)
    la_faces = faces_within_vertex_set(bb_faces, la_component)
    ra_boundary, ra_irregular = boundary_vertex_set(ra_faces, warn_irregular=True)
    la_boundary, la_irregular = boundary_vertex_set(la_faces, warn_irregular=True)
    if ra_irregular or la_irregular:
        print(
            f"Warning: irregular BB boundary vertices: RA={len(ra_irregular)}, "
            f"LA={len(la_irregular)}. Continuing with one-sided-edge vertices.",
            flush=True,
        )

    if args.svc_point is not None:
        svc_point = np.asarray(args.svc_point, dtype=float)
    else:
        landmarks_path = resolve_file(ra_dir, args.ra_landmarks)
        landmarks = load_regions(landmarks_path) * float(args.scale_factor)
        if not 0 <= args.svc_landmark_index < len(landmarks):
            raise IndexError(
                f"SVC index {args.svc_landmark_index} outside {landmarks_path} "
                f"with {len(landmarks)} rows"
            )
        svc_point = np.asarray(landmarks[args.svc_landmark_index], dtype=float)

    closest = build_closest_bridge(
        bb_points,
        ra_component,
        la_component,
        ra_faces,
        la_faces,
        max_pairs=args.closest_candidate_pairs,
        spacing=args.closest_spacing,
    )
    anatomical = build_anatomical_bridge(
        bb_points,
        ra_component,
        la_component,
        ra_faces,
        la_faces,
        ra_boundary,
        la_boundary,
        svc_point,
        spacing=args.anatomical_spacing,
        normal_neighbors=args.la_normal_neighbors,
    )

    closest_base = resolve_mesh_base(target_dir, args.closest_output)
    anatomical_base = resolve_mesh_base(target_dir, args.anatomical_output)
    write_bridge(
        closest_base,
        closest,
        tag=args.closest_tag,
        edge_thickness=args.closest_edge_thickness,
        center_thickness=args.closest_center_thickness,
    )
    write_bridge(
        anatomical_base,
        anatomical,
        tag=args.anatomical_tag,
        edge_thickness=args.anatomical_edge_thickness,
        center_thickness=args.anatomical_center_thickness,
    )

    debug = resolve_file(target_dir, args.debug_prefix)
    write_pts(Path(f"{debug}_closest_ra_geodesic.pts"), closest["ra_line"])
    write_pts(Path(f"{debug}_closest_la_geodesic.pts"), closest["la_line"])
    write_pts(Path(f"{debug}_anatomical_ra_geodesic.pts"), anatomical["ra_line"])
    write_pts(Path(f"{debug}_anatomical_la_geodesic.pts"), anatomical["la_line"])
    write_pts(
        Path(f"{debug}_anatomical_ra_points.pts"),
        np.asarray([
            bb_points[anatomical["ra_svc_endpoint"]],
            bb_points[anatomical["ra_center"]],
            bb_points[anatomical["ra_opposite_endpoint"]],
            svc_point,
        ]),
    )
    la = anatomical["la"]
    write_pts(
        Path(f"{debug}_anatomical_la_points.pts"),
        np.asarray([
            bb_points[la["negative_attachment"]],
            bb_points[la["center"]],
            bb_points[la["positive_attachment"]],
        ]),
    )

    report_path = resolve_file(target_dir, args.report)
    with report_path.open("w", encoding="utf-8") as h:
        h.write("Bachmann-bundle bridge build report\n")
        h.write("==================================\n\n")
        h.write(f"input biatrial mesh: {target_base}\n")
        h.write(f"input BB mesh: {bb_base}\n")
        h.write(f"RA BB vertices: {len(ra_component)}\n")
        h.write(f"LA BB vertices: {len(la_component)}\n\n")
        h.write("closest-region bridge\n")
        h.write(f"  candidate pairs: {len(closest['pairs'])}\n")
        h.write(f"  RA geodesic vertices: {len(closest['ra_line'])}\n")
        h.write(f"  LA geodesic vertices: {len(closest['la_line'])}\n")
        h.write(f"  RA length: {polyline_length(closest['ra_line']):.8g}\n")
        h.write(f"  LA length: {polyline_length(closest['la_line']):.8g}\n")
        h.write(f"  RA retention error: {retention_error(closest['ra_line'], closest, 'ra'):.12g}\n")
        h.write(f"  LA retention error: {retention_error(closest['la_line'], closest, 'la'):.12g}\n")
        h.write(f"  tag: {args.closest_tag}\n\n")
        h.write("anatomical bridge\n")
        h.write(f"  RA center: {anatomical['ra_center']}\n")
        h.write(f"  RA SVC endpoint: {anatomical['ra_svc_endpoint']}\n")
        h.write(f"  RA opposite endpoint: {anatomical['ra_opposite_endpoint']}\n")
        h.write(f"  RA geodesic vertices: {len(anatomical['ra_line'])}\n")
        h.write(f"  RA side counts: {len(anatomical['ra_path_to_svc'])} / {len(anatomical['ra_path_to_opposite'])}\n")
        h.write(f"  LA center: {la['center']}\n")
        h.write(f"  LA geodesic vertices: {len(anatomical['la_line'])}\n")
        h.write(f"  RA retention error: {retention_error(anatomical['ra_line'], anatomical, 'ra'):.12g}\n")
        h.write(f"  LA retention error: {retention_error(anatomical['la_line'], anatomical, 'la'):.12g}\n")
        h.write(f"  tag: {args.anatomical_tag}\n")

    print("BB bridges created", flush=True)
    print(f"  closest:    {closest_base}.pts/.elem/.lon/.f3d", flush=True)
    print(f"  anatomical: {anatomical_base}.pts/.elem/.lon/.f3d", flush=True)
    print(f"  report:     {report_path}", flush=True)


if __name__ == "__main__":
    main()
