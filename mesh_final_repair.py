#!/usr/bin/env python3
"""
Conservative final repair for the CS/mitral/VoM triangular surface.

Recommended input
-----------------
tube_cs_cleaned_stage3

This stage finally performs actual local mesh repair rather than only changing
the mitral->VoM connector geometry.

It repairs:
  * triangles with repeated vertex IDs;
  * true zero / numerically near-zero area triangles;
  * exact non-adjacent triangle self-intersections;
  * severe collapsing/sliver triangles ONLY when they are local to an
    intersecting patch.

Repair strategy
---------------
1. Detect exact self-intersections with VTK.
2. Mark both triangles of every intersecting pair.
3. Add true zero/degenerate triangles globally.
4. Add severe slivers only next to the intersection patch.
5. Remove the bad patch plus a small number of neighbouring face rings.
6. Preserve the two legitimate open tube ends.
7. Retriangulate only the newly-created local hole(s), using the existing
   boundary vertices (no fan-to-one-pivot connector).
8. Validate topology, triangle degeneracy, and exact self-intersections.
9. Write a NEW mesh; the input is never overwritten.

The script automatically tries the smallest patch first and expands it only if
the local repair does not validate.

Default output
--------------
tube_cs_repaired
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


# -----------------------------------------------------------------------------
# CARP I/O
# -----------------------------------------------------------------------------


def _with_suffix(path: str | Path, suffix: str) -> Path:
    path = Path(path)
    if path.suffix == suffix:
        return path
    return Path(f"{path}{suffix}")


def read_pts(path: str | Path) -> np.ndarray:
    file_path = _with_suffix(path, ".pts")
    with file_path.open("r", encoding="utf-8") as handle:
        first = handle.readline().split()
        if not first:
            raise ValueError(f"Empty point file: {file_path}")
        expected = int(first[0])

        rows = []
        for line_no, line in enumerate(handle, start=2):
            if not line.strip():
                continue
            p = line.split()
            if len(p) < 3:
                raise ValueError(f"{file_path}:{line_no}: expected 3 coordinates")
            rows.append([float(p[0]), float(p[1]), float(p[2])])

    points = np.asarray(rows, dtype=float)
    if len(points) != expected:
        raise ValueError(
            f"{file_path} declares {expected} points but contains {len(points)}"
        )
    return points


def read_elem(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    file_path = _with_suffix(path, ".elem")
    with file_path.open("r", encoding="utf-8") as handle:
        first = handle.readline().split()
        if not first:
            raise ValueError(f"Empty element file: {file_path}")
        expected = int(first[0])

        faces = []
        regions = []
        for line_no, line in enumerate(handle, start=2):
            if not line.strip():
                continue
            p = line.split()
            if len(p) < 5 or p[0] != "Tr":
                raise ValueError(
                    f"{file_path}:{line_no}: expected 'Tr i j k region'"
                )
            faces.append([int(p[1]), int(p[2]), int(p[3])])
            regions.append(int(p[4]))

    faces = np.asarray(faces, dtype=int)
    regions = np.asarray(regions, dtype=int)

    if len(faces) != expected:
        raise ValueError(
            f"{file_path} declares {expected} triangles but contains {len(faces)}"
        )

    return faces, regions


def read_lon_if_present(path: str | Path, n_faces: int) -> np.ndarray | None:
    file_path = _with_suffix(path, ".lon")
    if not file_path.exists():
        return None

    rows = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            p = line.split()
            if len(p) >= 3:
                rows.append([float(p[0]), float(p[1]), float(p[2])])

    data = np.asarray(rows, dtype=float)
    if len(data) != n_faces:
        print(
            f"Warning: {file_path} has {len(data)} rows for {n_faces} faces; "
            "output .lon will use 1 0 0.",
            flush=True,
        )
        return None
    return data


def write_mesh(
    base: Path,
    points: np.ndarray,
    faces: np.ndarray,
    regions: np.ndarray,
    lon: np.ndarray | None,
) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)

    with _with_suffix(base, ".pts").open("w", encoding="utf-8") as handle:
        handle.write(f"{len(points)}\n")
        np.savetxt(handle, points, fmt="%.16g")

    with _with_suffix(base, ".elem").open("w", encoding="utf-8") as handle:
        handle.write(f"{len(faces)} # number of faces\n")
        for face, region in zip(faces, regions):
            handle.write(
                f"Tr {int(face[0])} {int(face[1])} {int(face[2])} {int(region)}\n"
            )

    with _with_suffix(base, ".lon").open("w", encoding="utf-8") as handle:
        if lon is None:
            for _ in faces:
                handle.write("1 0 0\n")
        else:
            for vector in lon:
                handle.write(
                    f"{vector[0]:.16g} {vector[1]:.16g} {vector[2]:.16g}\n"
                )


# -----------------------------------------------------------------------------
# Triangle quality
# -----------------------------------------------------------------------------


def triangle_metrics(
    points: np.ndarray,
    faces: np.ndarray,
) -> dict[str, np.ndarray]:
    tri = points[faces]

    e01 = tri[:, 1] - tri[:, 0]
    e12 = tri[:, 2] - tri[:, 1]
    e20 = tri[:, 0] - tri[:, 2]

    l01 = np.linalg.norm(e01, axis=1)
    l12 = np.linalg.norm(e12, axis=1)
    l20 = np.linalg.norm(e20, axis=1)
    edges = np.column_stack((l01, l12, l20))

    cross = np.cross(e01, tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)

    eps = np.finfo(float).eps
    a = l12
    b = l20
    c = l01

    def angle(adj1, adj2, opp):
        den = np.maximum(2.0 * adj1 * adj2, eps)
        cos = (adj1 * adj1 + adj2 * adj2 - opp * opp) / den
        return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    a0 = angle(b, c, a)
    a1 = angle(c, a, b)
    a2 = 180.0 - a0 - a1
    angles = np.column_stack((a0, a1, a2))

    sum_sq = np.sum(edges * edges, axis=1)
    quality = np.zeros(len(faces), dtype=float)
    valid = sum_sq > eps
    quality[valid] = 4.0 * np.sqrt(3.0) * area[valid] / sum_sq[valid]

    repeated = np.asarray(
        [len({int(x), int(y), int(z)}) < 3 for x, y, z in faces],
        dtype=bool,
    )

    return {
        "area": area,
        "edge_min": edges.min(axis=1),
        "edge_max": edges.max(axis=1),
        "angle_min": angles.min(axis=1),
        "angle_max": angles.max(axis=1),
        "quality": quality,
        "repeated_vertex": repeated,
    }


# -----------------------------------------------------------------------------
# Topology helpers
# -----------------------------------------------------------------------------


def build_edge_face_map(
    faces: np.ndarray,
) -> dict[tuple[int, int], list[int]]:
    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_id, (a, b, c) in enumerate(faces):
        for u, v in ((a, b), (b, c), (c, a)):
            edge = (int(min(u, v)), int(max(u, v)))
            edge_faces[edge].append(face_id)
    return edge_faces


def build_face_adjacency(faces: np.ndarray) -> list[set[int]]:
    edge_faces = build_edge_face_map(faces)
    adjacency = [set() for _ in range(len(faces))]
    for ids in edge_faces.values():
        if len(ids) < 2:
            continue
        for face_id in ids:
            adjacency[face_id].update(j for j in ids if j != face_id)
    return adjacency


def expand_faces(
    seed_faces: set[int],
    adjacency: list[set[int]],
    rings: int,
) -> set[int]:
    expanded = set(seed_faces)
    frontier = set(seed_faces)

    for _ in range(max(0, int(rings))):
        nxt = set()
        for face_id in frontier:
            nxt.update(adjacency[face_id])
        nxt.difference_update(expanded)
        expanded.update(nxt)
        frontier = nxt
        if not frontier:
            break

    return expanded


def edge_topology(
    faces: np.ndarray,
) -> tuple[
    dict[tuple[int, int], list[tuple[int, int, int]]],
    list[tuple[int, int]],
    list[tuple[int, int]],
    list[tuple[int, int, tuple[int, int]]],
]:
    edge_rows: dict[
        tuple[int, int], list[tuple[int, int, int]]
    ] = defaultdict(list)

    for face_id, (a, b, c) in enumerate(faces):
        for u, v in ((a, b), (b, c), (c, a)):
            edge = (int(min(u, v)), int(max(u, v)))
            edge_rows[edge].append((face_id, int(u), int(v)))

    boundary = []
    nonmanifold = []
    conflicts = []

    for edge, rows in edge_rows.items():
        if len(rows) == 1:
            boundary.append(edge)
        elif len(rows) > 2:
            nonmanifold.append(edge)
        elif len(rows) == 2:
            f0, u0, v0 = rows[0]
            f1, u1, v1 = rows[1]
            if u0 == u1 and v0 == v1:
                conflicts.append((f0, f1, edge))

    return edge_rows, boundary, nonmanifold, conflicts


def ordered_boundary_loops(faces: np.ndarray) -> list[np.ndarray]:
    """
    Return oriented manifold boundary loops.

    The direction of each boundary edge is inherited from its incident triangle.
    """
    edge_rows, _, _, _ = edge_topology(faces)

    directed = [
        (rows[0][1], rows[0][2])
        for rows in edge_rows.values()
        if len(rows) == 1
    ]

    outgoing: dict[int, list[int]] = defaultdict(list)
    incoming: dict[int, list[int]] = defaultdict(list)

    for u, v in directed:
        outgoing[u].append(v)
        incoming[v].append(u)

    vertices = set(outgoing) | set(incoming)
    bad = [
        v
        for v in vertices
        if len(outgoing.get(v, [])) != 1
        or len(incoming.get(v, [])) != 1
    ]
    if bad:
        raise ValueError(
            "Boundary is not a collection of simple loops; "
            f"{len(bad)} boundary vertices have degree != 2"
        )

    next_vertex = {u: rows[0] for u, rows in outgoing.items()}
    unused = set(next_vertex)
    loops = []

    while unused:
        start = min(unused)
        current = start
        loop = [start]

        for _ in range(len(next_vertex) + 1):
            nxt = next_vertex[current]
            if nxt == start:
                break
            if nxt in loop:
                raise ValueError("Boundary loop self-revisited before closure")
            loop.append(nxt)
            current = nxt
        else:
            raise ValueError("Boundary loop failed to close")

        for vertex in loop:
            unused.discard(vertex)
        loops.append(np.asarray(loop, dtype=int))

    loops.sort(key=lambda x: (-len(x), int(np.min(x))))
    return loops


# -----------------------------------------------------------------------------
# Exact triangle self-intersection
# -----------------------------------------------------------------------------


def _vtk_polydata(points: np.ndarray, faces: np.ndarray):
    try:
        import vtk
    except ImportError as exc:
        raise RuntimeError("VTK is required for exact self-intersection repair") from exc

    vtk_points = vtk.vtkPoints()
    vtk_points.SetNumberOfPoints(len(points))
    for i, p in enumerate(points):
        vtk_points.SetPoint(i, float(p[0]), float(p[1]), float(p[2]))

    polys = vtk.vtkCellArray()
    for a, b, c in faces:
        tri = vtk.vtkTriangle()
        tri.GetPointIds().SetId(0, int(a))
        tri.GetPointIds().SetId(1, int(b))
        tri.GetPointIds().SetId(2, int(c))
        polys.InsertNextCell(tri)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetPolys(polys)
    return vtk, polydata


def find_self_intersections(
    points: np.ndarray,
    faces: np.ndarray,
    *,
    focus_face_ids: set[int] | None = None,
    progress_every: int = 5000,
) -> list[tuple[int, int]]:
    """
    Exact non-adjacent triangle intersections.

    If focus_face_ids is supplied, only pairs involving at least one focus face
    are tested.  Pairs sharing any vertex are ignored.
    """
    vtk, polydata = _vtk_polydata(points, faces)

    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(polydata)
    locator.BuildLocator()

    face_sets = [set(map(int, face)) for face in faces]
    tested: set[tuple[int, int]] = set()
    hits: list[tuple[int, int]] = []

    if focus_face_ids is None:
        scan_ids = range(len(faces))
    else:
        scan_ids = sorted(map(int, focus_face_ids))

    for scan_count, i in enumerate(scan_ids, start=1):
        tri_pts = points[faces[i]]
        mins = tri_pts.min(axis=0)
        maxs = tri_pts.max(axis=0)

        scale = max(float(np.max(maxs - mins)), 1.0)
        pad = max(1.0e-9 * scale, 1.0e-9)

        bounds = [
            float(mins[0] - pad), float(maxs[0] + pad),
            float(mins[1] - pad), float(maxs[1] + pad),
            float(mins[2] - pad), float(maxs[2] + pad),
        ]

        ids = vtk.vtkIdList()
        locator.FindCellsWithinBounds(bounds, ids)

        for k in range(ids.GetNumberOfIds()):
            j = int(ids.GetId(k))
            if j == i:
                continue

            pair = (min(i, j), max(i, j))
            if pair in tested:
                continue
            tested.add(pair)

            if face_sets[i].intersection(face_sets[j]):
                continue

            p = points[faces[i]]
            q = points[faces[j]]

            if vtk.vtkTriangle.TrianglesIntersect(
                p[0].tolist(), p[1].tolist(), p[2].tolist(),
                q[0].tolist(), q[1].tolist(), q[2].tolist(),
            ):
                hits.append(pair)

        if (
            focus_face_ids is None
            and progress_every > 0
            and scan_count % int(progress_every) == 0
        ):
            print(
                f"    self-intersection scan: {scan_count}/{len(faces)}, "
                f"hits={len(hits)}",
                flush=True,
            )

    return sorted(hits)


# -----------------------------------------------------------------------------
# Local polygon retriangulation
# -----------------------------------------------------------------------------


def project_loop_to_plane(
    points: np.ndarray,
    loop: np.ndarray,
) -> np.ndarray:
    coords = points[loop]
    center = np.mean(coords, axis=0)
    centered = coords - center

    if np.linalg.matrix_rank(centered) < 2:
        raise ValueError("Hole boundary is degenerate/collinear")

    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis_u = vh[0]
    axis_v = vh[1]
    return np.column_stack((centered @ axis_u, centered @ axis_v))


def signed_polygon_area(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(
        np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    )


def cross2(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = b - a
    ac = c - a
    return float(ab[0] * ac[1] - ab[1] * ac[0])


def point_in_triangle_2d(
    p: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    *,
    tol: float = 1.0e-12,
) -> bool:
    s1 = cross2(a, b, p)
    s2 = cross2(b, c, p)
    s3 = cross2(c, a, p)

    has_neg = (s1 < -tol) or (s2 < -tol) or (s3 < -tol)
    has_pos = (s1 > tol) or (s2 > tol) or (s3 > tol)
    return not (has_neg and has_pos)


def ear_clip_loop(
    points: np.ndarray,
    boundary_loop_directed: np.ndarray,
) -> np.ndarray:
    """
    Triangulate one local hole using only existing boundary vertices.

    The remaining mesh traverses each boundary edge in boundary_loop_directed.
    The patch must traverse that edge oppositely, so we reverse the loop before
    ear clipping.
    """
    patch_loop = boundary_loop_directed[::-1].copy()
    poly = project_loop_to_plane(points, patch_loop)

    area = signed_polygon_area(poly)
    if abs(area) < 1.0e-12:
        raise ValueError("Projected hole polygon has zero signed area")
    orientation = 1.0 if area > 0.0 else -1.0

    active = list(range(len(patch_loop)))
    triangles_local: list[tuple[int, int, int]] = []

    max_iterations = max(100, 10 * len(active))
    iterations = 0

    while len(active) > 3:
        iterations += 1
        if iterations > max_iterations:
            raise ValueError("Ear clipping did not converge")

        ear_found = False

        for pos in range(len(active)):
            i_prev = active[(pos - 1) % len(active)]
            i_curr = active[pos]
            i_next = active[(pos + 1) % len(active)]

            a = poly[i_prev]
            b = poly[i_curr]
            c = poly[i_next]

            turn = orientation * cross2(a, b, c)
            if turn <= 1.0e-10:
                continue

            contains_other = False
            for idx in active:
                if idx in (i_prev, i_curr, i_next):
                    continue
                if point_in_triangle_2d(poly[idx], a, b, c):
                    contains_other = True
                    break

            if contains_other:
                continue

            triangles_local.append((i_prev, i_curr, i_next))
            del active[pos]
            ear_found = True
            break

        if not ear_found:
            raise ValueError(
                "No valid ear found; local hole may be highly non-planar "
                "or its projected polygon may be non-simple"
            )

    triangles_local.append((active[0], active[1], active[2]))

    return np.asarray(
        [
            [
                int(patch_loop[i0]),
                int(patch_loop[i1]),
                int(patch_loop[i2]),
            ]
            for i0, i1, i2 in triangles_local
        ],
        dtype=int,
    )


# -----------------------------------------------------------------------------
# Candidate local repair
# -----------------------------------------------------------------------------


def loop_is_original_boundary(
    loop: np.ndarray,
    original_loop_sets: list[set[int]],
    *,
    min_overlap_fraction: float = 0.80,
) -> bool:
    loop_set = set(map(int, loop))
    for original in original_loop_sets:
        overlap = len(loop_set.intersection(original))
        if overlap / max(1, len(loop_set)) >= min_overlap_fraction:
            return True
    return False


def mode_region(values: np.ndarray, fallback: int = 5) -> int:
    if len(values) == 0:
        return fallback
    unique, counts = np.unique(values, return_counts=True)
    return int(unique[int(np.argmax(counts))])


def representative_fiber(
    lon: np.ndarray | None,
    removed_ids: np.ndarray,
) -> np.ndarray:
    if lon is None or len(removed_ids) == 0:
        return np.asarray([1.0, 0.0, 0.0], dtype=float)

    vector = np.mean(lon[removed_ids], axis=0)
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-12:
        return np.asarray([1.0, 0.0, 0.0], dtype=float)
    return vector / norm


def build_repair_candidate(
    *,
    points: np.ndarray,
    faces: np.ndarray,
    regions: np.ndarray,
    lon: np.ndarray | None,
    removal_face_ids: set[int],
    original_loop_sets: list[set[int]],
) -> dict:
    removal_ids = np.asarray(sorted(removal_face_ids), dtype=int)
    keep = np.ones(len(faces), dtype=bool)
    keep[removal_ids] = False

    kept_faces = faces[keep]
    kept_regions = regions[keep]
    kept_lon = lon[keep] if lon is not None else None

    loops_after_removal = ordered_boundary_loops(kept_faces)

    repair_loops = [
        loop
        for loop in loops_after_removal
        if not loop_is_original_boundary(loop, original_loop_sets)
    ]

    if not repair_loops:
        raise ValueError(
            "Removing the bad patch did not expose a separate local repair loop"
        )

    patch_faces_list = []
    for loop in repair_loops:
        if len(loop) < 3:
            raise ValueError("Local repair loop has fewer than three vertices")
        patch_faces_list.append(ear_clip_loop(points, loop))

    patch_faces = np.vstack(patch_faces_list).astype(int)

    patch_region = mode_region(regions[removal_ids], fallback=mode_region(regions))
    patch_regions = np.full(len(patch_faces), patch_region, dtype=int)

    candidate_faces = np.vstack((kept_faces, patch_faces)).astype(int)
    candidate_regions = np.concatenate((kept_regions, patch_regions))

    if lon is None:
        candidate_lon = None
    else:
        patch_fiber = representative_fiber(lon, removal_ids)
        patch_lon = np.tile(patch_fiber, (len(patch_faces), 1))
        candidate_lon = np.vstack((kept_lon, patch_lon))

    patch_start = len(kept_faces)
    patch_ids = set(range(patch_start, len(candidate_faces)))

    _, boundary, nonmanifold, conflicts = edge_topology(candidate_faces)

    remaining_loops = ordered_boundary_loops(candidate_faces)

    return {
        "faces": candidate_faces,
        "regions": candidate_regions,
        "lon": candidate_lon,
        "patch_faces": patch_faces,
        "patch_face_ids": patch_ids,
        "repair_loops": repair_loops,
        "remaining_loops": remaining_loops,
        "boundary_edges": len(boundary),
        "nonmanifold_edges": len(nonmanifold),
        "orientation_conflicts": len(conflicts),
        "removed_face_ids": removal_ids,
    }


def compact_mesh(
    points: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    used = np.unique(faces.ravel())
    mapping = np.full(len(points), -1, dtype=int)
    mapping[used] = np.arange(len(used), dtype=int)
    return points[used], mapping[faces]


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Actual conservative repair of zero/degenerate triangles and exact "
            "self-intersection patches."
        )
    )
    parser.add_argument(
        "--mesh-base",
        default="./tube_cs_cleaned_stage3",
        help="Input mesh basename (default: ./tube_cs_cleaned_stage3)",
    )
    parser.add_argument(
        "--output-base",
        default="./tube_cs_repaired",
        help="Output repaired mesh basename (default: ./tube_cs_repaired)",
    )
    parser.add_argument(
        "--target-edge",
        type=float,
        default=350.0,
        help="Reference edge length in um (default: 350)",
    )
    parser.add_argument(
        "--zero-area-relative",
        type=float,
        default=1.0e-8,
        help=(
            "True near-zero area threshold relative to an equilateral target "
            "triangle (default: 1e-8)"
        ),
    )
    parser.add_argument(
        "--local-short-edge",
        type=float,
        default=20.0,
        help=(
            "Severe sliver short-edge threshold in um. Applied only next to "
            "self-intersection faces (default: 20)"
        ),
    )
    parser.add_argument(
        "--local-min-angle",
        type=float,
        default=2.0,
        help=(
            "Severe sliver minimum-angle threshold in degrees. Applied only "
            "next to self-intersection faces (default: 2)"
        ),
    )
    parser.add_argument(
        "--max-patch-rings",
        type=int,
        default=3,
        help=(
            "Try removing 0..N shared-edge neighbour rings around the bad patch "
            "(default: 3)"
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5000,
        help="Full self-intersection scan progress interval",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    mesh_base = Path(args.mesh_base).expanduser()
    output_base = Path(args.output_base).expanduser()

    if mesh_base.resolve() == output_base.resolve():
        raise ValueError("Input and output basenames must differ")

    print("[1/6] Reading mesh", flush=True)
    points = read_pts(mesh_base)
    faces, regions = read_elem(mesh_base)
    lon = read_lon_if_present(mesh_base, len(faces))

    original_boundary_loops = ordered_boundary_loops(faces)
    original_loop_sets = [set(map(int, loop)) for loop in original_boundary_loops]

    print(
        f"    points={len(points)}, triangles={len(faces)}, "
        f"open boundary loops={[len(loop) for loop in original_boundary_loops]}",
        flush=True,
    )

    print("[2/6] Detecting hard degenerate / zero-area triangles", flush=True)
    metrics = triangle_metrics(points, faces)

    target_area = (math.sqrt(3.0) / 4.0) * float(args.target_edge) ** 2
    zero_area_threshold = target_area * float(args.zero_area_relative)

    hard_zero_ids = set(
        map(
            int,
            np.flatnonzero(
                metrics["repeated_vertex"]
                | (metrics["area"] <= zero_area_threshold)
            ),
        )
    )

    print(
        f"    true zero/degenerate faces={len(hard_zero_ids)} "
        f"(area threshold={zero_area_threshold:.6g} um^2)",
        flush=True,
    )

    print("[3/6] Detecting exact non-adjacent self-intersections", flush=True)
    intersections = find_self_intersections(
        points,
        faces,
        progress_every=int(args.progress_every),
    )

    self_faces: set[int] = set()
    for a, b in intersections:
        self_faces.add(int(a))
        self_faces.add(int(b))

    print(
        f"    self-intersection pairs={len(intersections)}, "
        f"unique intersecting faces={len(self_faces)}",
        flush=True,
    )

    adjacency = build_face_adjacency(faces)

    # Severe sliver cleanup is deliberately LOCAL to the intersection patch.
    local_neighborhood = expand_faces(self_faces, adjacency, 1) if self_faces else set()

    local_sliver_ids = set(
        int(i)
        for i in local_neighborhood
        if (
            metrics["edge_min"][i] < float(args.local_short_edge)
            and metrics["angle_min"][i] < float(args.local_min_angle)
        )
    )

    seed_faces = set(hard_zero_ids) | set(self_faces) | set(local_sliver_ids)

    print(
        f"    additional severe local slivers={len(local_sliver_ids)}",
        flush=True,
    )

    audit_dir = output_base.parent / f"{output_base.name}_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    with (audit_dir / "initial_self_intersections.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["face_a", "face_b"])
        writer.writerows(intersections)

    if not seed_faces:
        print("No repairable invalid faces were detected; copying mesh.", flush=True)
        write_mesh(output_base, points, faces, regions, lon)
        return

    print("[4/6] Trying minimal local retriangulation patches", flush=True)

    trials = []
    chosen = None

    for rings in range(0, int(args.max_patch_rings) + 1):
        removal = expand_faces(seed_faces, adjacency, rings)

        try:
            candidate = build_repair_candidate(
                points=points,
                faces=faces,
                regions=regions,
                lon=lon,
                removal_face_ids=removal,
                original_loop_sets=original_loop_sets,
            )
        except Exception as exc:
            trials.append(
                {
                    "rings": rings,
                    "removed": len(removal),
                    "status": f"build_failed: {exc}",
                }
            )
            print(f"    rings={rings}: build failed ({exc})", flush=True)
            continue

        candidate_metrics = triangle_metrics(points, candidate["faces"])
        zero_remaining = int(
            np.count_nonzero(
                candidate_metrics["repeated_vertex"]
                | (candidate_metrics["area"] <= zero_area_threshold)
            )
        )

        patch_intersections = find_self_intersections(
            points,
            candidate["faces"],
            focus_face_ids=candidate["patch_face_ids"],
            progress_every=0,
        )

        valid = (
            candidate["nonmanifold_edges"] == 0
            and candidate["orientation_conflicts"] == 0
            and len(candidate["remaining_loops"]) == len(original_boundary_loops)
            and zero_remaining == 0
            and len(patch_intersections) == 0
        )

        trial = {
            "rings": rings,
            "removed": len(removal),
            "patch_faces": len(candidate["patch_faces"]),
            "repair_loops": len(candidate["repair_loops"]),
            "nonmanifold": candidate["nonmanifold_edges"],
            "orientation_conflicts": candidate["orientation_conflicts"],
            "boundary_loops": len(candidate["remaining_loops"]),
            "zero_remaining": zero_remaining,
            "patch_intersections": len(patch_intersections),
            "status": "accepted" if valid else "rejected",
        }
        trials.append(trial)

        print(
            f"    rings={rings}: removed={len(removal)}, "
            f"patch_faces={len(candidate['patch_faces'])}, "
            f"zero={zero_remaining}, "
            f"patch_intersections={len(patch_intersections)}, "
            f"nonmanifold={candidate['nonmanifold_edges']}, "
            f"orientation_conflicts={candidate['orientation_conflicts']}, "
            f"boundary_loops={len(candidate['remaining_loops'])}",
            flush=True,
        )

        if valid:
            chosen = candidate
            chosen["patch_intersections"] = patch_intersections
            chosen["rings"] = rings
            break

    with (audit_dir / "repair_trials.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = [
            "rings",
            "removed",
            "patch_faces",
            "repair_loops",
            "nonmanifold",
            "orientation_conflicts",
            "boundary_loops",
            "zero_remaining",
            "patch_intersections",
            "status",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in trials:
            writer.writerow({key: row.get(key, "") for key in fields})

    if chosen is None:
        raise RuntimeError(
            "No conservative local repair candidate validated. "
            f"See {audit_dir / 'repair_trials.csv'}. "
            "The input mesh has not been modified."
        )

    print(
        f"[5/6] Candidate accepted with {chosen['rings']} neighbour ring(s); "
        "running full final intersection scan",
        flush=True,
    )

    final_intersections = find_self_intersections(
        points,
        chosen["faces"],
        progress_every=int(args.progress_every),
    )

    final_metrics = triangle_metrics(points, chosen["faces"])
    final_zero_ids = np.flatnonzero(
        final_metrics["repeated_vertex"]
        | (final_metrics["area"] <= zero_area_threshold)
    )

    _, final_boundary, final_nonmanifold, final_conflicts = edge_topology(
        chosen["faces"]
    )
    final_loops = ordered_boundary_loops(chosen["faces"])

    print("[6/6] Compacting and writing repaired mesh", flush=True)

    compact_points, compact_faces = compact_mesh(points, chosen["faces"])

    write_mesh(
        output_base,
        compact_points,
        compact_faces,
        chosen["regions"],
        chosen["lon"],
    )

    with (audit_dir / "removed_faces.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["original_face_id"])
        for face_id in chosen["removed_face_ids"]:
            writer.writerow([int(face_id)])

    with (audit_dir / "final_self_intersections.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["face_a_pre_compaction", "face_b_pre_compaction"])
        writer.writerows(final_intersections)

    patch_metrics = triangle_metrics(points, chosen["patch_faces"])

    report = f"""Final conservative mesh repair report
====================================

Input mesh: {mesh_base}
Output mesh: {output_base}

Input
-----
points: {len(points)}
triangles: {len(faces)}
original open boundary loops: {[len(loop) for loop in original_boundary_loops]}

Detection
---------
true zero/degenerate faces: {len(hard_zero_ids)}
zero-area threshold (um^2): {zero_area_threshold:.12g}
initial exact self-intersection pairs: {len(intersections)}
initial unique self-intersecting faces: {len(self_faces)}
severe local sliver faces added to repair: {len(local_sliver_ids)}

Selected repair
---------------
neighbour rings removed around seed patch: {chosen['rings']}
faces removed: {len(chosen['removed_face_ids'])}
local repair loops: {len(chosen['repair_loops'])}
new patch triangles: {len(chosen['patch_faces'])}

Patch quality
-------------
minimum edge (um): {patch_metrics['edge_min'].min():.12g}
maximum edge (um): {patch_metrics['edge_max'].max():.12g}
minimum angle (deg): {patch_metrics['angle_min'].min():.12g}
maximum angle (deg): {patch_metrics['angle_max'].max():.12g}
minimum area (um^2): {patch_metrics['area'].min():.12g}

Final validity
--------------
true zero/degenerate faces remaining: {len(final_zero_ids)}
exact self-intersection pairs remaining: {len(final_intersections)}
non-manifold edges: {len(final_nonmanifold)}
orientation-conflicting shared edges: {len(final_conflicts)}
open boundary loops: {[len(loop) for loop in final_loops]}
boundary edges: {len(final_boundary)}

Final compact mesh
------------------
points: {len(compact_points)}
triangles: {len(compact_faces)}

Acceptance target
-----------------
zero/degenerate faces remaining = 0
self-intersection pairs remaining = 0
non-manifold edges = 0
orientation conflicts = 0
same number of legitimate open boundary loops as input

The input mesh was not overwritten.
"""

    report_path = audit_dir / "final_repair_report.txt"
    report_path.write_text(report, encoding="utf-8")

    print("")
    print("Final repair result")
    print("-------------------")
    print(f"  zero/degenerate faces:   {len(final_zero_ids)}")
    print(f"  self-intersection pairs: {len(final_intersections)}")
    print(f"  non-manifold edges:      {len(final_nonmanifold)}")
    print(f"  orientation conflicts:   {len(final_conflicts)}")
    print(
        f"  open boundary loops:     "
        f"{[len(loop) for loop in final_loops]}"
    )
    print(f"  output:                  {output_base}")
    print(f"  report:                  {report_path}")

    if (
        len(final_zero_ids) != 0
        or len(final_intersections) != 0
        or len(final_nonmanifold) != 0
        or len(final_conflicts) != 0
        or len(final_loops) != len(original_boundary_loops)
    ):
        print("")
        print(
            "WARNING: the repaired mesh did not meet every acceptance target. "
            "Inspect the report before using it."
        )


if __name__ == "__main__":
    main()
