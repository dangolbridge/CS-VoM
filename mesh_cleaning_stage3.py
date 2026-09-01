#!/usr/bin/env python3
"""
Stage-3 mitral->VoM junction repair with a multi-ring transition.

This script assumes the stage-1 conservative weld has already been run.
It reuses the same provenance logic as stage 2, keeps the validated
2-layer terminal mitral peel, and replaces the old single-row stretched
collar with a short equal-node multi-ring transition.

Key rules:
  * input mesh is not overwritten;
  * generalized-capsule VoM is not regenerated;
  * mitral and VoM junction loops must retain the same node count;
  * correspondence is one-to-one, with best cyclic shift;
  * intermediate ring count is chosen from target axial spacing;
  * ring coordinates interpolate smoothly from mitral boundary to VoM boundary;
  * no zipper/fan triangulation is used.

Default output:
    tube_cs_cleaned_stage3.pts/.elem/.lon
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


# -----------------------------------------------------------------------------
# Basic CARP I/O
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
        expected = int(first[0])
        rows = []
        for line in handle:
            if not line.strip():
                continue
            p = line.split()
            rows.append([float(p[0]), float(p[1]), float(p[2])])
    pts = np.asarray(rows, dtype=float)
    if len(pts) != expected:
        raise ValueError(f"{file_path}: expected {expected} points, got {len(pts)}")
    return pts


def read_elem(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    file_path = _with_suffix(path, ".elem")
    with file_path.open("r", encoding="utf-8") as handle:
        first = handle.readline().split()
        expected = int(first[0])
        faces = []
        regions = []
        for line in handle:
            if not line.strip():
                continue
            p = line.split()
            if p[0] != "Tr":
                raise ValueError("Only triangular CARP elements are supported")
            faces.append([int(p[1]), int(p[2]), int(p[3])])
            regions.append(int(p[4]))
    faces = np.asarray(faces, dtype=int)
    regions = np.asarray(regions, dtype=int)
    if len(faces) != expected:
        raise ValueError(f"{file_path}: expected {expected} faces, got {len(faces)}")
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
    arr = np.asarray(rows, dtype=float)
    return arr if len(arr) == n_faces else None


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
        for f, r in zip(faces, regions):
            handle.write(f"Tr {f[0]} {f[1]} {f[2]} {int(r)}\n")

    with _with_suffix(base, ".lon").open("w", encoding="utf-8") as handle:
        if lon is None:
            for _ in faces:
                handle.write("1 0 0\n")
        else:
            for v in lon:
                handle.write(f"{v[0]:.16g} {v[1]:.16g} {v[2]:.16g}\n")


# -----------------------------------------------------------------------------
# Provenance recovery from stage-1
# -----------------------------------------------------------------------------


def read_stage1_vertex_mapping(path: Path) -> np.ndarray:
    rows = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append((int(row["original_vertex_id"]), int(row["cleaned_vertex_id"])))
    rows.sort()
    ids = np.asarray([x[0] for x in rows], dtype=int)
    if not np.array_equal(ids, np.arange(len(rows), dtype=int)):
        raise ValueError("Stage-1 vertex mapping is incomplete or unordered")
    return np.asarray([x[1] for x in rows], dtype=int)


def classify_cleaned_vertices(
    original_to_cleaned: np.ndarray,
    original_vm_offset: int,
    cleaned_vertex_count: int,
) -> np.ndarray:
    classes = np.full(cleaned_vertex_count, -1, dtype=np.int8)
    for old_id, cleaned_id in enumerate(original_to_cleaned):
        if cleaned_id < 0:
            continue
        cls = 0 if old_id < original_vm_offset else 1
        current = int(classes[cleaned_id])
        if current == -1:
            classes[cleaned_id] = cls
        elif current != cls:
            classes[cleaned_id] = 2
    return classes


def split_face_classes(
    faces: np.ndarray,
    vertex_classes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prefix = []
    vm = []
    mixed = []
    for i, face in enumerate(faces):
        cls = set(map(int, vertex_classes[face]))
        if -1 in cls or 2 in cls:
            raise ValueError(f"Face {i} has ambiguous provenance")
        if cls == {0}:
            prefix.append(i)
        elif cls == {1}:
            vm.append(i)
        elif cls == {0, 1}:
            mixed.append(i)
    return (
        np.asarray(prefix, dtype=int),
        np.asarray(vm, dtype=int),
        np.asarray(mixed, dtype=int),
    )


# -----------------------------------------------------------------------------
# Boundary loops
# -----------------------------------------------------------------------------


def ordered_boundary_loops(faces: np.ndarray) -> list[np.ndarray]:
    edge_rows: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            key = (int(min(u, v)), int(max(u, v)))
            edge_rows[key].append((int(u), int(v)))

    boundary_edges = [rows[0] for rows in edge_rows.values() if len(rows) == 1]
    outgoing: dict[int, list[int]] = defaultdict(list)
    incoming: dict[int, list[int]] = defaultdict(list)
    for u, v in boundary_edges:
        outgoing[u].append(v)
        incoming[v].append(u)

    for v in set(outgoing) | set(incoming):
        if len(outgoing.get(v, [])) != 1 or len(incoming.get(v, [])) != 1:
            raise ValueError("Boundary is not a simple oriented loop collection")

    next_v = {u: vs[0] for u, vs in outgoing.items()}
    unused = set(next_v)
    loops = []

    while unused:
        start = min(unused)
        current = start
        loop = [start]
        while True:
            nxt = next_v[current]
            if nxt == start:
                break
            loop.append(nxt)
            current = nxt
        for v in loop:
            unused.discard(v)
        loops.append(np.asarray(loop, dtype=int))

    loops.sort(key=lambda x: (-len(x), int(np.min(x))))
    return loops


def loop_center(points: np.ndarray, loop: np.ndarray) -> np.ndarray:
    return np.mean(points[loop], axis=0)


def closest_boundary_pair(
    points: np.ndarray,
    loops_a: list[np.ndarray],
    loops_b: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    best = None
    for a in loops_a:
        ca = loop_center(points, a)
        for b in loops_b:
            cb = loop_center(points, b)
            d = float(np.linalg.norm(ca - cb))
            if best is None or d < best[0]:
                best = (d, a, b)
    assert best is not None
    return best[1], best[2]


def choose_new_terminal_loop(
    points: np.ndarray,
    previous_loop: np.ndarray,
    loops: list[np.ndarray],
) -> np.ndarray:
    c = loop_center(points, previous_loop)
    distances = [
        float(np.linalg.norm(loop_center(points, loop) - c))
        for loop in loops
    ]
    return loops[int(np.argmin(distances))]


def peel_prefix_layers(
    points: np.ndarray,
    prefix_faces: np.ndarray,
    prefix_original_ids: np.ndarray,
    initial_loop: np.ndarray,
    layers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    faces = prefix_faces.copy()
    ids = prefix_original_ids.copy()
    loop = initial_loop.copy()

    for _ in range(layers):
        terminal = set(map(int, loop))
        keep = np.asarray(
            [not any(int(v) in terminal for v in face) for face in faces],
            dtype=bool,
        )
        if np.all(keep):
            raise ValueError("Peel removed no faces")
        faces = faces[keep]
        ids = ids[keep]
        loops = ordered_boundary_loops(faces)
        loop = choose_new_terminal_loop(points, loop, loops)

    return faces, ids, loop


# -----------------------------------------------------------------------------
# Ring correspondence and multi-ring transition
# -----------------------------------------------------------------------------


def best_loop_alignment(
    points: np.ndarray,
    mitral_loop: np.ndarray,
    vm_loop_directed: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    if len(mitral_loop) != len(vm_loop_directed):
        raise ValueError("Mitral and VoM loops must have equal node count")

    # reverse VoM boundary direction for compatible connector winding
    vm = vm_loop_directed[::-1].copy()
    a = points[mitral_loop]

    best_shift = 0
    best_loop = vm.copy()
    best_error = np.inf

    for shift in range(len(vm)):
        candidate = np.roll(vm, -shift)
        b = points[candidate]
        err = float(np.mean(np.sum((a - b) ** 2, axis=1)))
        if err < best_error:
            best_error = err
            best_shift = shift
            best_loop = candidate.copy()

    return best_loop, best_shift, float(np.sqrt(best_error))


def smoothstep(t: np.ndarray) -> np.ndarray:
    return t * t * (3.0 - 2.0 * t)


def build_intermediate_rings(
    points: np.ndarray,
    mitral_loop: np.ndarray,
    vm_loop: np.ndarray,
    *,
    target_axial_spacing: float,
    min_intervals: int = 2,
) -> tuple[list[np.ndarray], int, float]:
    """
    Return coordinate rings including endpoints.

    We choose interval count from mean point-to-point bridge distance so the
    axial separation is close to target_axial_spacing.
    """
    a = points[mitral_loop]
    b = points[vm_loop]
    bridge_lengths = np.linalg.norm(b - a, axis=1)
    mean_bridge = float(np.mean(bridge_lengths))

    n_intervals = max(
        int(min_intervals),
        int(np.ceil(mean_bridge / float(target_axial_spacing))),
    )

    ts = np.linspace(0.0, 1.0, n_intervals + 1)
    weights = smoothstep(ts)

    rings = []
    for w in weights:
        rings.append((1.0 - w) * a + w * b)

    return rings, n_intervals, mean_bridge


def flatten_transition(
    rings: list[np.ndarray],
    mitral_loop_global: np.ndarray,
    vm_loop_global: np.ndarray,
    base_point_count: int,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """
    Create only intermediate points and connector faces.

    Endpoint rings reference existing global mitral/VoM vertices.
    """
    n = len(mitral_loop_global)
    if any(len(r) != n for r in rings):
        raise ValueError("All transition rings must have equal node count")

    global_ring_ids: list[np.ndarray] = [mitral_loop_global.copy()]
    new_points = []

    for ring in rings[1:-1]:
        start = base_point_count + len(new_points)
        ids = np.arange(start, start + n, dtype=int)
        global_ring_ids.append(ids)
        new_points.extend(ring)

    global_ring_ids.append(vm_loop_global.copy())

    faces = []
    for r0, r1 in zip(global_ring_ids[:-1], global_ring_ids[1:]):
        for i in range(n):
            j = (i + 1) % n
            a0 = int(r0[i])
            a1 = int(r0[j])
            b0 = int(r1[i])
            b1 = int(r1[j])

            faces.append([a1, a0, b0])
            faces.append([a1, b0, b1])

    return (
        np.asarray(new_points, dtype=float).reshape(-1, 3),
        np.asarray(faces, dtype=int),
        global_ring_ids,
    )


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------


def triangle_metrics(points: np.ndarray, faces: np.ndarray) -> dict[str, float]:
    tri = points[faces]
    l0 = np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1)
    l1 = np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1)
    l2 = np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1)
    edges = np.column_stack((l0, l1, l2))

    area = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
        axis=1,
    )

    eps = np.finfo(float).eps
    a, b, c = l1, l2, l0

    def angle(x, y, opposite):
        den = np.maximum(2.0 * x * y, eps)
        cos = (x*x + y*y - opposite*opposite) / den
        return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    A = angle(b, c, a)
    B = angle(c, a, b)
    C = 180.0 - A - B
    angles = np.column_stack((A, B, C))

    return {
        "edge_min": float(edges.min()),
        "edge_median": float(np.median(edges)),
        "edge_max": float(edges.max()),
        "angle_min": float(angles.min()),
        "angle_max": float(angles.max()),
        "area_min": float(area.min()),
    }


def edge_topology_counts(faces: np.ndarray) -> tuple[int, int, int]:
    edge_rows: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            key = (int(min(u, v)), int(max(u, v)))
            edge_rows[key].append((int(u), int(v)))

    boundary = 0
    nonmanifold = 0
    conflicts = 0
    for rows in edge_rows.values():
        if len(rows) == 1:
            boundary += 1
        elif len(rows) > 2:
            nonmanifold += 1
        elif len(rows) == 2 and rows[0] == rows[1]:
            conflicts += 1

    return boundary, nonmanifold, conflicts


def local_intersections(
    points: np.ndarray,
    faces: np.ndarray,
    focus_ids: np.ndarray,
) -> list[tuple[int, int]]:
    try:
        import vtk
    except ImportError as exc:
        raise RuntimeError("VTK is required for stage-3 intersection checking") from exc

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

    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(polydata)
    locator.BuildLocator()

    face_sets = [set(map(int, f)) for f in faces]
    tested = set()
    hits = []

    for i in map(int, focus_ids):
        tri_pts = points[faces[i]]
        mins = tri_pts.min(axis=0)
        maxs = tri_pts.max(axis=0)
        scale = max(float(np.max(maxs - mins)), 1.0)
        pad = max(1e-9 * scale, 1e-9)

        bounds = [
            float(mins[0]-pad), float(maxs[0]+pad),
            float(mins[1]-pad), float(maxs[1]+pad),
            float(mins[2]-pad), float(maxs[2]+pad),
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

    return sorted(hits)


def compact_mesh(
    points: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    used = np.unique(faces.ravel())
    mapping = np.full(len(points), -1, dtype=int)
    mapping[used] = np.arange(len(used), dtype=int)
    return points[used], mapping[faces]


def mode_region(regions: np.ndarray, fallback: int = 5) -> int:
    if len(regions) == 0:
        return fallback
    vals, counts = np.unique(regions, return_counts=True)
    return int(vals[int(np.argmax(counts))])


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage-3 multi-ring repair of the mitral-to-VoM junction."
    )
    parser.add_argument(
        "--mesh-base",
        default="./tube_cs_cleaned_stage1",
        help="Input stage-1 mesh basename",
    )
    parser.add_argument(
        "--output-base",
        default="./tube_cs_cleaned_stage3",
        help="Output mesh basename",
    )
    parser.add_argument(
        "--stage1-mapping",
        default=None,
        help="Path to original_to_cleaned_vertex.csv",
    )
    parser.add_argument(
        "--vm-surface-base",
        default="./vom_generalized_capsule_proximal_fit",
        help="Saved VoM surface basename",
    )
    parser.add_argument(
        "--peel-layers",
        type=int,
        default=2,
        help="Fixed terminal mitral peel depth (default: 2)",
    )
    parser.add_argument(
        "--target-axial-spacing",
        type=float,
        default=325.0,
        help="Target spacing between transition rings in um (default: 325)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    mesh_base = Path(args.mesh_base).expanduser()
    output_base = Path(args.output_base).expanduser()

    mapping_path = (
        Path(args.stage1_mapping).expanduser()
        if args.stage1_mapping
        else mesh_base.parent / f"{mesh_base.name}_audit" / "original_to_cleaned_vertex.csv"
    )

    print("[1/5] Reading stage-1 mesh and provenance", flush=True)
    points = read_pts(mesh_base)
    faces, regions = read_elem(mesh_base)
    lon = read_lon_if_present(mesh_base, len(faces))

    mapping = read_stage1_vertex_mapping(mapping_path)
    vm_saved_points = read_pts(Path(args.vm_surface_base).expanduser())

    original_vm_offset = len(mapping) - len(vm_saved_points)
    classes = classify_cleaned_vertices(mapping, original_vm_offset, len(points))

    ambiguous = np.flatnonzero(classes == 2)
    if len(ambiguous):
        raise ValueError(
            f"{len(ambiguous)} cross-class welded vertices found; provenance unsafe"
        )

    prefix_face_ids, vm_face_ids, mixed_face_ids = split_face_classes(faces, classes)

    print(
        f"    prefix={len(prefix_face_ids)}, VoM={len(vm_face_ids)}, old collar={len(mixed_face_ids)}",
        flush=True,
    )

    print("[2/5] Recovering junction loops and peeling mitral", flush=True)
    prefix_faces0 = faces[prefix_face_ids]
    vm_faces = faces[vm_face_ids]

    prefix_loops = ordered_boundary_loops(prefix_faces0)
    vm_loops = ordered_boundary_loops(vm_faces)

    mitral_initial, vm_start = closest_boundary_pair(points, prefix_loops, vm_loops)

    print(
        f"    recovered pre-peel junction: "
        f"mitral={len(mitral_initial)}, VoM={len(vm_start)} nodes",
        flush=True,
    )

    peeled_prefix_faces, peeled_prefix_original_ids, mitral_loop = peel_prefix_layers(
        points,
        prefix_faces0,
        prefix_face_ids,
        mitral_initial,
        int(args.peel_layers),
    )

    if len(mitral_loop) != len(vm_start):
        raise ValueError(
            "Junction node-count mismatch after the mitral peel: "
            f"pre-peel mitral={len(mitral_initial)}, VoM={len(vm_start)}; "
            f"post-peel mitral={len(mitral_loop)}, VoM={len(vm_start)}. "
            "Do not use zipper/fan triangulation to hide this mismatch. "
            "Re-run Stage 1 with --vm-surface-base so old collar/junction "
            "vertices are protected from near-coincident welding."
        )

    vm_aligned, shift, rms = best_loop_alignment(points, mitral_loop, vm_start)

    print(
        f"    junction nodes: {len(mitral_loop)}↔{len(vm_aligned)}, "
        f"alignment shift={shift}, RMS bridge={rms:.1f} um",
        flush=True,
    )

    print("[3/5] Building multi-ring transition", flush=True)
    rings, n_intervals, mean_bridge = build_intermediate_rings(
        points,
        mitral_loop,
        vm_aligned,
        target_axial_spacing=float(args.target_axial_spacing),
    )

    new_points, transition_faces, ring_ids = flatten_transition(
        rings,
        mitral_loop,
        vm_aligned,
        len(points),
    )

    all_points = np.vstack((points, new_points))

    base_faces = np.vstack((peeled_prefix_faces, vm_faces)).astype(int)
    transition_start = len(base_faces)
    all_faces = np.vstack((base_faces, transition_faces)).astype(int)
    transition_ids = np.arange(
        transition_start,
        transition_start + len(transition_faces),
        dtype=int,
    )

    connector_region = mode_region(regions[mixed_face_ids], fallback=5)
    all_regions = np.concatenate(
        (
            regions[peeled_prefix_original_ids],
            regions[vm_face_ids],
            np.full(len(transition_faces), connector_region, dtype=int),
        )
    )

    if lon is None:
        all_lon = None
    else:
        collar_fiber = np.mean(lon[mixed_face_ids], axis=0)
        norm = np.linalg.norm(collar_fiber)
        if norm <= 1e-12:
            collar_fiber = np.asarray([1.0, 0.0, 0.0])
        else:
            collar_fiber = collar_fiber / norm

        all_lon = np.vstack(
            (
                lon[peeled_prefix_original_ids],
                lon[vm_face_ids],
                np.tile(collar_fiber, (len(transition_faces), 1)),
            )
        )

    axial_step = mean_bridge / n_intervals
    print(
        f"    mean bridge={mean_bridge:.1f} um, intervals={n_intervals}, "
        f"mean axial step≈{axial_step:.1f} um",
        flush=True,
    )

    print("[4/5] Checking transition geometry/topology", flush=True)
    metrics = triangle_metrics(all_points, transition_faces)
    boundary_edges, nonmanifold, conflicts = edge_topology_counts(all_faces)
    intersections = local_intersections(all_points, all_faces, transition_ids)

    print(
        f"    transition intersections={len(intersections)}, "
        f"nonmanifold={nonmanifold}, orientation conflicts={conflicts}",
        flush=True,
    )
    print(
        f"    transition edges median={metrics['edge_median']:.1f} um, "
        f"min/max={metrics['edge_min']:.1f}/{metrics['edge_max']:.1f} um",
        flush=True,
    )

    print("[5/5] Compacting and writing stage-3 mesh", flush=True)
    compact_points, compact_faces = compact_mesh(all_points, all_faces)

    write_mesh(
        output_base,
        compact_points,
        compact_faces,
        all_regions,
        all_lon,
    )

    audit_dir = output_base.parent / f"{output_base.name}_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    with (audit_dir / "transition_intersections.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["face_a_pre_compaction", "face_b_pre_compaction"])
        writer.writerows(intersections)

    report = f"""Stage-3 multi-ring mitral-to-VoM repair
======================================

Input mesh: {mesh_base}
Output mesh: {output_base}

Junction setup
--------------
peel layers: {int(args.peel_layers)}
junction node count: {len(mitral_loop)}
best cyclic shift: {shift}
RMS mitral-to-VoM bridge distance (um): {rms:.12g}
mean bridge distance (um): {mean_bridge:.12g}

Transition
----------
target axial spacing (um): {float(args.target_axial_spacing):.12g}
number of axial intervals: {n_intervals}
number of inserted intermediate rings: {max(0, n_intervals - 1)}
approximate mean axial step (um): {axial_step:.12g}
new transition faces: {len(transition_faces)}

Transition triangle geometry
----------------------------
minimum edge (um): {metrics['edge_min']:.12g}
median edge (um): {metrics['edge_median']:.12g}
maximum edge (um): {metrics['edge_max']:.12g}
minimum angle (deg): {metrics['angle_min']:.12g}
maximum angle (deg): {metrics['angle_max']:.12g}
minimum area (um^2): {metrics['area_min']:.12g}

Topology
--------
boundary edges: {boundary_edges}
non-manifold edges: {nonmanifold}
orientation-conflicting shared edges: {conflicts}
transition-related exact self-intersections: {len(intersections)}

Final mesh
----------
points: {len(compact_points)}
triangles: {len(compact_faces)}

The generalized-capsule VoM surface was not regenerated.
The input stage-1 mesh was not overwritten.
"""
    report_path = audit_dir / "cleaning_stage3_report.txt"
    report_path.write_text(report, encoding="utf-8")

    print("")
    print("Stage-3 repair complete")
    print("-----------------------")
    print(f"  intervals:                 {n_intervals}")
    print(f"  inserted rings:            {max(0, n_intervals - 1)}")
    print(f"  transition faces:          {len(transition_faces)}")
    print(f"  transition intersections:  {len(intersections)}")
    print(f"  non-manifold edges:        {nonmanifold}")
    print(f"  orientation conflicts:     {conflicts}")
    print(f"  output:                    {output_base}")
    print(f"  report:                    {report_path}")


if __name__ == "__main__":
    main()
