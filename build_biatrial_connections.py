#!/usr/bin/env python3
"""Build line-element connections for the simulation-ready biatrial surface.

This replaces the historical ``biatrial_iac_last.py`` workflow with a smaller,
function-based implementation that reuses ``geometry_hybrid.py`` and
``mesh_io_hybrid.py``.

The existing connection rules are retained:

* LA epicardium <-> LA endocardium: tag 32
* SAN -> RA surface: tag 33
* CT -> RA surface: tag 34
* PM -> RA surface: tag 35
* FO/interatrial RA <-> LA connections: tag 39
* BB -> closest atrial surface: tag 41
* CS -> LA connections: configurable positions, default tag 40
* VoM -> LA connections: configurable positions, default tag 50

CS/VoM placement
----------------
CS and VoM connection sites are specified as normalized arc positions along a
shortest surface path running from the CS opening to the distal VoM opening:

    0.0 = CS end
    1.0 = distal VoM end

For example::

    --cs-connection-positions 0.30 0.40 0.50 \
    --vom-connection-positions 0.70 0.80 0.90

Each requested site uses ``--connection-neighbors`` nearby tube vertices
(default 5), matching the neighbourhood concept used by the previous script.
Each selected tube vertex is connected to its nearest LA epicardial vertex.

The cleaned CS/VoM surface must already have been merged into the input
biatrial mesh.  A coordinate-mapping check verifies this before connections are
created.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from geometry_hybrid import (
    mesh_shortest_path_indices,
    nearest_reference_vertices,
    path_vertex_indices_at_fractions,
    triangle_boundary_loops,
)
from mesh_io_hybrid import (
    load_regions,
    read_carp_mesh,
    read_pts,
    read_triangles,
    write_carp_mesh,
    write_pts,
    write_scalar_values,
)


# -----------------------------------------------------------------------------
# Shared AtrialMTK region and generated connection tags
# -----------------------------------------------------------------------------

from mesh_tags import (
    BB_ATRIAL_CONNECTION_TAG,
    BB_REGION,
    CS_LA_CONNECTION_TAG,
    CT_CONNECTION_TAG,
    CT_REGION,
    FO_CONNECTION_TAG,
    LA_BILAYER_CONNECTION_TAG,
    LA_ENDO_REGIONS,
    LA_EPI_REGIONS,
    PM_CONNECTION_TAG,
    PM_REGION,
    RA_EPI_REGIONS,
    SAN_CONNECTION_TAG,
    SAN_REGION,
    VOM_LA_CONNECTION_TAG,
)

# Short local aliases keep the implementation readable while the numerical
# definitions remain centralized in mesh_tags.py.
TAG_LA_BILAYER = LA_BILAYER_CONNECTION_TAG
TAG_SAN = SAN_CONNECTION_TAG
TAG_CT = CT_CONNECTION_TAG
TAG_PM = PM_CONNECTION_TAG
TAG_FO = FO_CONNECTION_TAG
TAG_BB = BB_ATRIAL_CONNECTION_TAG


@dataclass(frozen=True)
class ConnectionRecord:
    kind: str
    requested_fraction: float | None
    anchor_tube_vertex: int | None
    source_vertex: int
    target_vertex: int
    tag: int
    line_length: float


# -----------------------------------------------------------------------------
# Generic workflow helpers
# -----------------------------------------------------------------------------


def resolve_file(directory: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else directory / path


def resolve_mesh_base(directory: Path, value: str | Path) -> Path:
    path = resolve_file(directory, value)
    if path.suffix in {".pts", ".elem", ".lon"}:
        path = path.with_suffix("")
    return path


def validate_fraction_list(values: Sequence[float], name: str) -> list[float]:
    result = [float(value) for value in values]
    if any(not np.isfinite(value) or value < 0.0 or value > 1.0 for value in result):
        raise ValueError(f"{name} values must all lie in [0, 1]")
    return result


def vertex_region_labels(n_points: int, elements: Iterable) -> np.ndarray:
    """Reproduce the old max-incident-region vertex label in one element pass."""
    labels = np.full(int(n_points), -1, dtype=int)
    for element in elements:
        region = int(element.reg)
        for vertex in element.n:
            vertex = int(vertex)
            if 0 <= vertex < n_points:
                labels[vertex] = max(labels[vertex], region)

    missing = np.flatnonzero(labels < 0)
    if len(missing):
        raise ValueError(
            f"{len(missing)} mesh vertices are not referenced by any input element; "
            "cannot reproduce Scalar_Labels.dat safely"
        )
    return labels


def vertices_with_regions(labels: np.ndarray, regions: Sequence[int]) -> np.ndarray:
    return np.flatnonzero(np.isin(labels, np.asarray(regions, dtype=int))).astype(int)


def _element_class():
    # Keep the UAC dependency local so ``--help``/static imports of helper modules
    # remain lightweight.
    from uac.element import Element

    return Element


def make_line(source: int, target: int, tag: int):
    Element = _element_class()
    return Element("Ln", [int(source), int(target)], int(tag))


def nearest_vertex_map(
    points: np.ndarray,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(source_indices) == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=int)
    if len(target_indices) == 0:
        raise ValueError("Target anatomical vertex set is empty")

    distances, local_indices = nearest_reference_vertices(
        points[source_indices], points[target_indices], k=1
    )
    return (
        np.asarray(distances, dtype=float).reshape(-1),
        target_indices[np.asarray(local_indices, dtype=int).reshape(-1)],
    )


def connect_nearest_sets(
    points: np.ndarray,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
    *,
    tag: int,
    kind: str,
) -> tuple[list, list[ConnectionRecord]]:
    distances, nearest_targets = nearest_vertex_map(points, source_indices, target_indices)
    lines = []
    records = []
    for source, target, distance in zip(source_indices, nearest_targets, distances):
        lines.append(make_line(int(source), int(target), int(tag)))
        records.append(
            ConnectionRecord(
                kind=kind,
                requested_fraction=None,
                anchor_tube_vertex=None,
                source_vertex=int(source),
                target_vertex=int(target),
                tag=int(tag),
                line_length=float(distance),
            )
        )
    return lines, records


def build_bb_connections(
    points: np.ndarray,
    bb_indices: np.ndarray,
    la_epi_indices: np.ndarray,
    ra_epi_indices: np.ndarray,
) -> tuple[list, list[ConnectionRecord]]:
    if len(bb_indices) == 0:
        return [], []

    la_dist, la_target = nearest_vertex_map(points, bb_indices, la_epi_indices)
    ra_dist, ra_target = nearest_vertex_map(points, bb_indices, ra_epi_indices)

    lines = []
    records = []
    for i, bb_vertex in enumerate(bb_indices):
        if la_dist[i] > ra_dist[i]:
            target = int(ra_target[i])
            distance = float(ra_dist[i])
        else:
            target = int(la_target[i])
            distance = float(la_dist[i])

        lines.append(make_line(int(bb_vertex), target, TAG_BB))
        records.append(
            ConnectionRecord(
                kind="BB",
                requested_fraction=None,
                anchor_tube_vertex=None,
                source_vertex=int(bb_vertex),
                target_vertex=target,
                tag=TAG_BB,
                line_length=distance,
            )
        )
    return lines, records


def build_fo_connections(
    points: np.ndarray,
    la_epi_indices: np.ndarray,
    ra_epi_indices: np.ndarray,
    landmark: np.ndarray,
    *,
    count: int,
) -> tuple[list, list[ConnectionRecord]]:
    if count <= 0:
        return [], []
    if len(la_epi_indices) == 0 or len(ra_epi_indices) == 0:
        raise ValueError("FO connections require non-empty LA and RA epicardial vertex sets")

    k = min(int(count), len(la_epi_indices))
    _, nearest_la_local = nearest_reference_vertices(
        np.asarray(landmark, dtype=float), points[la_epi_indices], k=k
    )
    nearest_la_local = np.asarray(nearest_la_local, dtype=int).reshape(-1)
    selected_la = la_epi_indices[nearest_la_local]

    distances, selected_ra = nearest_vertex_map(points, selected_la, ra_epi_indices)
    lines = []
    records = []
    for la_vertex, ra_vertex, distance in zip(selected_la, selected_ra, distances):
        # Preserve historical node order: RA first, LA second.
        lines.append(make_line(int(ra_vertex), int(la_vertex), TAG_FO))
        records.append(
            ConnectionRecord(
                kind="FO",
                requested_fraction=None,
                anchor_tube_vertex=None,
                source_vertex=int(ra_vertex),
                target_vertex=int(la_vertex),
                tag=TAG_FO,
                line_length=float(distance),
            )
        )
    return lines, records


# -----------------------------------------------------------------------------
# CS / VoM placement
# -----------------------------------------------------------------------------


def _loop_distance_to_seed(points: np.ndarray, loop: np.ndarray, seed: np.ndarray) -> float:
    return float(np.min(np.linalg.norm(points[loop] - np.asarray(seed, dtype=float), axis=1)))


def identify_cs_vom_ends(
    tube_points: np.ndarray,
    tube_faces: np.ndarray,
    cs_seed: np.ndarray,
    vom_seed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    loops = triangle_boundary_loops(tube_faces)
    if len(loops) < 2:
        raise ValueError(
            f"Expected at least two open CS/VoM boundary loops, found {len(loops)}"
        )

    cs_loop_index = int(
        np.argmin([_loop_distance_to_seed(tube_points, loop, cs_seed) for loop in loops])
    )
    cs_loop = loops[cs_loop_index]

    remaining = [i for i in range(len(loops)) if i != cs_loop_index]
    vom_loop_index = min(
        remaining,
        key=lambda i: _loop_distance_to_seed(tube_points, loops[i], vom_seed),
    )
    vom_loop = loops[vom_loop_index]

    cs_vertex = int(
        cs_loop[
            np.argmin(np.linalg.norm(tube_points[cs_loop] - np.asarray(cs_seed), axis=1))
        ]
    )
    vom_vertex = int(
        vom_loop[
            np.argmin(np.linalg.norm(tube_points[vom_loop] - np.asarray(vom_seed), axis=1))
        ]
    )
    return cs_loop, vom_loop, cs_vertex, vom_vertex


def map_tube_vertices_to_global(
    tube_points: np.ndarray,
    global_points: np.ndarray,
    *,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    distances, indices = nearest_reference_vertices(tube_points, global_points, k=1)
    distances = np.asarray(distances, dtype=float).reshape(-1)
    indices = np.asarray(indices, dtype=int).reshape(-1)

    if len(distances) and float(np.max(distances)) > float(tolerance):
        worst = int(np.argmax(distances))
        raise ValueError(
            "The cleaned CS/VoM mesh does not appear to be merged into the input "
            f"biatrial mesh. Worst tube->global point distance is {distances[worst]:.6g} "
            f"um at tube vertex {worst}, exceeding --cs-map-tolerance={tolerance}. "
            "Merge the same cleaned CS/VoM mesh into Bilayer_Combined before running "
            "this script."
        )
    return indices, distances


def nearby_tube_vertices(
    tube_points: np.ndarray,
    anchor_vertex: int,
    count: int,
) -> np.ndarray:
    """Return the nearest ``count`` tube vertices excluding the anchor itself."""
    count = int(count)
    if count < 1:
        raise ValueError("connection-neighbors must be at least 1")

    k = min(len(tube_points), count + 1)
    _, indices = nearest_reference_vertices(
        tube_points[int(anchor_vertex)], tube_points, k=k
    )
    indices = np.asarray(indices, dtype=int).reshape(-1)
    indices = indices[indices != int(anchor_vertex)]
    if len(indices) < count:
        raise ValueError(
            f"Only {len(indices)} non-anchor tube neighbours are available; requested {count}"
        )
    return indices[:count]


def build_cs_vom_connections(
    *,
    global_points: np.ndarray,
    la_epi_indices: np.ndarray,
    tube_points: np.ndarray,
    tube_faces: np.ndarray,
    tube_to_global: np.ndarray,
    cs_seed: np.ndarray,
    vom_seed: np.ndarray,
    cs_positions: Sequence[float],
    vom_positions: Sequence[float],
    connection_neighbors: int,
    cs_tag: int,
    vom_tag: int,
) -> tuple[list, list[ConnectionRecord], np.ndarray, list[tuple[str, float, int]]]:
    cs_loop, vom_loop, cs_end_vertex, vom_end_vertex = identify_cs_vom_ends(
        tube_points, tube_faces, cs_seed, vom_seed
    )

    path_indices = mesh_shortest_path_indices(
        tube_points, tube_faces, cs_end_vertex, vom_end_vertex
    )
    path_points = tube_points[path_indices]

    all_specs = [
        *(('CS', float(position), int(cs_tag)) for position in cs_positions),
        *(('VoM', float(position), int(vom_tag)) for position in vom_positions),
    ]

    if not all_specs:
        return [], [], path_indices, []

    fractions = [spec[1] for spec in all_specs]
    path_local_positions = path_vertex_indices_at_fractions(path_points, fractions)
    anchor_vertices = path_indices[path_local_positions]

    la_points = global_points[la_epi_indices]
    lines = []
    records = []
    anchor_records: list[tuple[str, float, int]] = []
    seen_lines: set[tuple[int, int, int]] = set()

    for (kind, requested_fraction, tag), anchor_vertex in zip(all_specs, anchor_vertices):
        anchor_vertex = int(anchor_vertex)
        anchor_records.append((kind, requested_fraction, anchor_vertex))

        source_local_vertices = nearby_tube_vertices(
            tube_points, anchor_vertex, int(connection_neighbors)
        )
        source_global_vertices = tube_to_global[source_local_vertices]

        la_distances, la_local = nearest_reference_vertices(
            global_points[source_global_vertices], la_points, k=1
        )
        la_distances = np.asarray(la_distances, dtype=float).reshape(-1)
        target_global_vertices = la_epi_indices[
            np.asarray(la_local, dtype=int).reshape(-1)
        ]

        for source_local, source_global, target_global, distance in zip(
            source_local_vertices,
            source_global_vertices,
            target_global_vertices,
            la_distances,
        ):
            key = (int(source_global), int(target_global), int(tag))
            if key in seen_lines:
                continue
            seen_lines.add(key)
            lines.append(make_line(*key))
            records.append(
                ConnectionRecord(
                    kind=kind,
                    requested_fraction=float(requested_fraction),
                    anchor_tube_vertex=anchor_vertex,
                    source_vertex=int(source_global),
                    target_vertex=int(target_global),
                    tag=int(tag),
                    line_length=float(distance),
                )
            )

    return lines, records, path_indices, anchor_records


# -----------------------------------------------------------------------------
# Fibres / thickness / audit
# -----------------------------------------------------------------------------


def extend_fibers(original_fibers, n_original_elements: int, n_new_lines: int) -> np.ndarray:
    fibers = np.asarray(original_fibers, dtype=float)
    if fibers.ndim == 1:
        if fibers.size == 3 and n_original_elements == 1:
            fibers = fibers.reshape(1, 3)
        else:
            raise ValueError("Input fibre array has an unexpected one-dimensional shape")
    if fibers.ndim != 2 or fibers.shape[1] < 3:
        raise ValueError("Input fibre array must contain at least three components per element")
    if len(fibers) != n_original_elements:
        raise ValueError(
            f"Input fibre count ({len(fibers)}) does not match element count "
            f"({n_original_elements})"
        )

    fibers = fibers[:, :3]
    if n_new_lines:
        line_fibers = np.tile(np.asarray([1.0, 0.0, 0.0]), (n_new_lines, 1))
        fibers = np.vstack((fibers, line_fibers))
    return fibers


def element_thickness(region: int) -> float:
    region = int(region)
    if region in {7, 6, 21, 22, 23, 24, 25, 26, 27, 28}:
        return 1500.0
    if region in {11, 12, 1, 13, 14, 2}:
        return 2000.0
    if region == 3:
        return 1500.0
    if region == 8:
        return 500.0
    if region == 5:
        return 2000.0
    if region == 9:
        return 15000.0
    return 2000.0


def write_connection_audit(
    path: Path,
    records: Sequence[ConnectionRecord],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "kind",
                "requested_fraction",
                "anchor_tube_vertex",
                "source_global_vertex",
                "target_global_vertex",
                "tag",
                "line_length_um",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.kind,
                    "" if record.requested_fraction is None else record.requested_fraction,
                    "" if record.anchor_tube_vertex is None else record.anchor_tube_vertex,
                    record.source_vertex,
                    record.target_vertex,
                    record.tag,
                    f"{record.line_length:.12g}",
                ]
            )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build bilayer/IAC/CS/VoM line connections for a biatrial mesh."
    )
    parser.add_argument("--la-path", required=True, help="LA source directory (landmarks/regions)")
    parser.add_argument("--ra-path", required=True, help="RA directory containing Regions.txt and CS/VoM mesh")
    parser.add_argument("--target-path", required=True, help="Directory containing the merged biatrial mesh")
    parser.add_argument(
        "--mesh-name",
        default="Bilayer_Combined",
        help="Input merged biatrial mesh basename (default: Bilayer_Combined)",
    )
    parser.add_argument(
        "--output-mesh",
        default="Reentry_surface_iac",
        help="Output mesh basename (default: Reentry_surface_iac)",
    )
    parser.add_argument(
        "--cs-mesh",
        default="tube_cs_cleaned",
        help="Cleaned CS/VoM mesh basename under --ra-path (default: tube_cs_cleaned)",
    )
    parser.add_argument(
        "--landmarks-file",
        default="Landmarks.txt",
        help="LA landmark filename under --la-path (default: Landmarks.txt)",
    )
    parser.add_argument(
        "--regions-file",
        default="Regions.txt",
        help="Region seed filename under LA/RA paths (default: Regions.txt)",
    )
    parser.add_argument("--scale-factor", type=float, default=1.0)
    parser.add_argument(
        "--fo-index",
        type=int,
        default=3,
        help="LA landmark index used for FO connection placement (default: 3)",
    )
    parser.add_argument(
        "--fo-connections",
        type=int,
        default=100,
        help="Number of FO RA-LA line elements (default: 100; 0 disables)",
    )

    parser.add_argument(
        "--cs-connection-positions",
        nargs="*",
        type=float,
        default=[],
        metavar="F",
        help="Normalized CS->VoM path positions for CS connections, each in [0,1]",
    )
    parser.add_argument(
        "--vom-connection-positions",
        nargs="*",
        type=float,
        default=[],
        metavar="F",
        help="Normalized CS->VoM path positions for VoM connections, each in [0,1]",
    )
    parser.add_argument(
        "--connection-neighbors",
        type=int,
        default=5,
        help="Nearby tube vertices connected per requested CS/VoM site (default: 5)",
    )
    parser.add_argument("--cs-connection-tag", type=int, default=CS_LA_CONNECTION_TAG)
    parser.add_argument("--vom-connection-tag", type=int, default=VOM_LA_CONNECTION_TAG)
    parser.add_argument(
        "--cs-end-region-index",
        type=int,
        default=0,
        help="RA Regions.txt row used to identify the CS opening (default: 0)",
    )
    parser.add_argument(
        "--vom-end-region-index",
        type=int,
        default=3,
        help="LA Regions.txt row used to identify the distal VoM opening (default: 3)",
    )
    parser.add_argument(
        "--cs-map-tolerance",
        type=float,
        default=2.0,
        help="Maximum tube->merged-mesh vertex mapping distance in um (default: 2)",
    )
    parser.add_argument(
        "--write-debug-points",
        action="store_true",
        help="Write CS->VoM path and requested anchor points for inspection",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    la_dir = Path(args.la_path).expanduser()
    ra_dir = Path(args.ra_path).expanduser()
    target_dir = Path(args.target_path).expanduser()

    for directory, name in ((la_dir, "LA"), (ra_dir, "RA"), (target_dir, "target")):
        if not directory.is_dir():
            raise FileNotFoundError(f"{name} directory does not exist: {directory}")

    cs_positions = validate_fraction_list(
        args.cs_connection_positions, "--cs-connection-positions"
    )
    vom_positions = validate_fraction_list(
        args.vom_connection_positions, "--vom-connection-positions"
    )
    if args.connection_neighbors < 1:
        raise ValueError("--connection-neighbors must be at least 1")
    if args.cs_map_tolerance < 0:
        raise ValueError("--cs-map-tolerance must be non-negative")

    print("[1/6] Reading merged biatrial mesh", flush=True)
    points, elements, fibers, _ = read_carp_mesh(
        target_dir, args.mesh_name, return_surface=False
    )
    points = np.asarray(points, dtype=float)
    elements = list(elements)

    print("[2/6] Building vertex anatomical labels", flush=True)
    labels = vertex_region_labels(len(points), elements)
    scalar_path = target_dir / "Scalar_Labels.dat"
    write_scalar_values(scalar_path, labels.astype(int))

    la_epi = vertices_with_regions(labels, LA_EPI_REGIONS)
    la_endo = vertices_with_regions(labels, LA_ENDO_REGIONS)
    ra_epi = vertices_with_regions(labels, RA_EPI_REGIONS)
    san = vertices_with_regions(labels, (SAN_REGION,))
    ct = vertices_with_regions(labels, (CT_REGION,))
    pm = vertices_with_regions(labels, (PM_REGION,))
    bb = vertices_with_regions(labels, (BB_REGION,))

    print(
        "    vertex sets: "
        f"LA_epi={len(la_epi)}, LA_endo={len(la_endo)}, "
        f"RA_epi={len(ra_epi)}, SAN={len(san)}, CT={len(ct)}, "
        f"PM={len(pm)}, BB={len(bb)}",
        flush=True,
    )

    all_lines: list = []
    all_records: list[ConnectionRecord] = []

    print("[3/6] Building existing bilayer/structure/FO connections", flush=True)
    for source, target, tag, kind in (
        (la_epi, la_endo, TAG_LA_BILAYER, "LA_bilayer"),
        (san, ra_epi, TAG_SAN, "SAN"),
        (ct, ra_epi, TAG_CT, "CT"),
        (pm, ra_epi, TAG_PM, "PM"),
    ):
        lines, records = connect_nearest_sets(
            points, source, target, tag=tag, kind=kind
        )
        all_lines.extend(lines)
        all_records.extend(records)

    lines, records = build_bb_connections(points, bb, la_epi, ra_epi)
    all_lines.extend(lines)
    all_records.extend(records)

    landmark_path = resolve_file(la_dir, args.landmarks_file)
    landmarks = np.loadtxt(landmark_path, delimiter=",", dtype=float) * float(args.scale_factor)
    if not (0 <= int(args.fo_index) < len(landmarks)):
        raise IndexError(
            f"--fo-index={args.fo_index} is outside {landmark_path} ({len(landmarks)} rows)"
        )
    lines, records = build_fo_connections(
        points,
        la_epi,
        ra_epi,
        landmarks[int(args.fo_index)],
        count=int(args.fo_connections),
    )
    all_lines.extend(lines)
    all_records.extend(records)

    print("[4/6] Building argument-controlled CS/VoM connections", flush=True)
    path_indices = np.empty(0, dtype=int)
    anchor_records: list[tuple[str, float, int]] = []

    if cs_positions or vom_positions:
        cs_mesh_base = resolve_mesh_base(ra_dir, args.cs_mesh)
        tube_points = read_pts(cs_mesh_base)
        tube_faces, _ = read_triangles(cs_mesh_base)

        tube_to_global, map_distances = map_tube_vertices_to_global(
            tube_points,
            points,
            tolerance=float(args.cs_map_tolerance),
        )
        print(
            f"    cleaned tube mapping max distance: {float(np.max(map_distances)):.6g} um",
            flush=True,
        )

        ra_regions_path = resolve_file(ra_dir, args.regions_file)
        la_regions_path = resolve_file(la_dir, args.regions_file)
        ra_regions = load_regions(ra_regions_path) * float(args.scale_factor)
        la_regions = load_regions(la_regions_path) * float(args.scale_factor)

        if not (0 <= int(args.cs_end_region_index) < len(ra_regions)):
            raise IndexError("--cs-end-region-index is outside RA Regions.txt")
        if not (0 <= int(args.vom_end_region_index) < len(la_regions)):
            raise IndexError("--vom-end-region-index is outside LA Regions.txt")

        lines, records, path_indices, anchor_records = build_cs_vom_connections(
            global_points=points,
            la_epi_indices=la_epi,
            tube_points=tube_points,
            tube_faces=tube_faces,
            tube_to_global=tube_to_global,
            cs_seed=ra_regions[int(args.cs_end_region_index)],
            vom_seed=la_regions[int(args.vom_end_region_index)],
            cs_positions=cs_positions,
            vom_positions=vom_positions,
            connection_neighbors=int(args.connection_neighbors),
            cs_tag=int(args.cs_connection_tag),
            vom_tag=int(args.vom_connection_tag),
        )
        all_lines.extend(lines)
        all_records.extend(records)

        print(
            f"    CS sites={len(cs_positions)}, VoM sites={len(vom_positions)}, "
            f"CS/VoM line elements={len(lines)}, surface path vertices={len(path_indices)}",
            flush=True,
        )

        if args.write_debug_points:
            write_pts(target_dir / "cs_vom_surface_path", tube_points[path_indices])
            if anchor_records:
                write_pts(
                    target_dir / "cs_vom_connection_anchors",
                    np.asarray([tube_points[record[2]] for record in anchor_records]),
                )
    else:
        print("    no CS/VoM positions requested; no tag 40/50 lines added", flush=True)

    print("[5/6] Assembling fibres and thickness", flush=True)
    output_elements = list(elements) + all_lines
    output_fibers = extend_fibers(fibers, len(elements), len(all_lines))

    thickness = [element_thickness(element.reg) for element in elements]
    thickness.extend([1.0] * len(all_lines))

    print("[6/6] Writing final mesh and audit", flush=True)
    output_base = target_dir / args.output_mesh
    write_carp_mesh(output_base, points, output_elements, output_fibers, None)
    write_scalar_values(Path(f"{output_base}.f3d"), thickness)

    audit_dir = target_dir / f"{args.output_mesh}_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    write_connection_audit(audit_dir / "line_connections.csv", all_records)

    counts: dict[str, int] = {}
    for record in all_records:
        counts[record.kind] = counts.get(record.kind, 0) + 1

    report_lines = [
        "Biatrial connection build report",
        "================================",
        "",
        f"input mesh: {args.mesh_name}",
        f"output mesh: {args.output_mesh}",
        f"input points: {len(points)}",
        f"input elements: {len(elements)}",
        f"new line elements: {len(all_lines)}",
        f"final elements: {len(output_elements)}",
        "",
        "CS / VoM placement",
        "------------------",
        f"cs mesh: {args.cs_mesh}",
        f"CS positions: {cs_positions}",
        f"VoM positions: {vom_positions}",
        f"connection neighbors per site: {args.connection_neighbors}",
        f"CS tag: {args.cs_connection_tag}",
        f"VoM tag: {args.vom_connection_tag}",
        f"surface path vertices: {len(path_indices)}",
        "",
        "Line counts by structure",
        "------------------------",
    ]
    for kind in sorted(counts):
        report_lines.append(f"{kind}: {counts[kind]}")
    report_lines.extend(
        [
            "",
            f"Scalar labels: {scalar_path}",
            f"Connection audit: {audit_dir / 'line_connections.csv'}",
        ]
    )
    (audit_dir / "build_report.txt").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    print("")
    print("Finished")
    print("--------")
    print(f"output mesh: {output_base}")
    print(f"new line elements: {len(all_lines)}")
    print(f"audit: {audit_dir / 'build_report.txt'}")


if __name__ == "__main__":
    main()
