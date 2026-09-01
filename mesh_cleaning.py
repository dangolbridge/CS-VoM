#!/usr/bin/env python3
"""
Stage-1 conservative cleaner for CARP triangular surface meshes.

This script modifies ONLY:
  1. near-coincident vertices -> welded away from the protected junction;
  2. triangles collapsed by that weld -> removed;
  3. duplicate / colocated triangles after welding -> removed;
  4. unused vertices -> compacted away.

It DOES NOT repair:
  * self-intersections;
  * bad angles / sliver triangles;
  * non-manifold topology;
  * large-scale geometry.

The input mesh is never overwritten unless the user explicitly chooses the same
output basename.

Example
-------
python3 mesh_cleaning_update1.py \
    --mesh-base ./tube_cs \
    --output-base ./tube_cs_cleaned_stage1
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


# -----------------------------------------------------------------------------
# I/O
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
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"{file_path}:{line_no}: expected 3 coordinates")
            rows.append([float(parts[0]), float(parts[1]), float(parts[2])])

    points = np.asarray(rows, dtype=float)
    if len(points) != expected:
        raise ValueError(
            f"{file_path} declares {expected} points but contains {len(points)} rows"
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
            parts = line.split()
            if parts[0] != "Tr" or len(parts) < 5:
                raise ValueError(
                    f"{file_path}:{line_no}: expected 'Tr i j k region'"
                )
            faces.append([int(parts[1]), int(parts[2]), int(parts[3])])
            regions.append(int(parts[4]))

    faces = np.asarray(faces, dtype=int)
    regions = np.asarray(regions, dtype=int)

    if len(faces) != expected:
        raise ValueError(
            f"{file_path} declares {expected} elements but contains {len(faces)} rows"
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
            parts = line.split()
            if len(parts) < 3:
                continue
            rows.append([float(parts[0]), float(parts[1]), float(parts[2])])

    data = np.asarray(rows, dtype=float)
    if len(data) != n_faces:
        print(
            f"Warning: {file_path} has {len(data)} rows but mesh has {n_faces} faces; "
            "cleaned .lon will use default '1 0 0'.",
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
# Welding
# -----------------------------------------------------------------------------


class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=int)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = int(self.parent[x])
        return int(x)

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return

        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def build_weld_map(
    points: np.ndarray,
    tolerance: float,
    *,
    protected_vertices: np.ndarray | None = None,
) -> tuple[np.ndarray, list[list[int]], int]:
    """
    Return old-vertex -> representative-old-vertex map.

    ``protected_vertices`` are never welded to any other vertex. In this
    workflow these are the old mixed-collar vertices, whose topology must stay
    intact until the dedicated junction-reconstruction stage.

    The lowest original vertex ID in each weld group is used as representative,
    so welding does not move any surviving coordinate.
    """
    if tolerance < 0:
        raise ValueError("coincident tolerance must be non-negative")

    protected = np.zeros(len(points), dtype=bool)
    if protected_vertices is not None:
        ids = np.asarray(protected_vertices, dtype=int).reshape(-1)
        if len(ids):
            if np.any(ids < 0) or np.any(ids >= len(points)):
                raise ValueError("protected vertex ID lies outside the input mesh")
            protected[ids] = True

    uf = UnionFind(len(points))
    skipped_protected_pairs = 0

    if tolerance > 0 and len(points):
        tree = cKDTree(points)
        for a, b in tree.query_pairs(r=float(tolerance)):
            a = int(a)
            b = int(b)
            if protected[a] or protected[b]:
                skipped_protected_pairs += 1
                continue
            uf.union(a, b)

    groups_by_root: dict[int, list[int]] = defaultdict(list)
    for vertex_id in range(len(points)):
        groups_by_root[uf.find(vertex_id)].append(vertex_id)

    old_to_rep = np.arange(len(points), dtype=int)
    welded_groups: list[list[int]] = []

    for members in groups_by_root.values():
        members = sorted(members)
        representative = members[0]
        old_to_rep[members] = representative
        if len(members) > 1:
            welded_groups.append(members)

    return old_to_rep, welded_groups, skipped_protected_pairs


def junction_vertices_from_vm_split(
    points: np.ndarray,
    faces: np.ndarray,
    vm_surface_base: str | Path,
) -> tuple[np.ndarray, int, int]:
    """Recover old collar vertices from the known prefix/VoM point split."""
    vm_points = read_pts(vm_surface_base)
    vm_count = int(len(vm_points))
    prefix_count = int(len(points) - vm_count)

    if vm_count <= 0 or prefix_count <= 0:
        raise ValueError(
            "Could not recover a positive prefix/VoM point split from "
            f"{vm_surface_base}"
        )

    is_prefix = faces < prefix_count
    is_vm = faces >= prefix_count
    mixed_mask = np.any(is_prefix, axis=1) & np.any(is_vm, axis=1)

    if not np.any(mixed_mask):
        raise ValueError(
            "No mixed prefix<->VoM collar faces were found in the raw tube. "
            "Refusing to guess which junction vertices should be protected."
        )

    protected = np.unique(faces[mixed_mask].reshape(-1)).astype(int)
    return protected, prefix_count, int(np.count_nonzero(mixed_mask))


def remove_collapsed_faces(
    faces: np.ndarray,
    original_face_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.array(
        [len({int(a), int(b), int(c)}) == 3 for a, b, c in faces],
        dtype=bool,
    )
    removed = original_face_ids[~valid]
    return faces[valid], original_face_ids[valid], removed


def remove_duplicate_faces(
    faces: np.ndarray,
    original_face_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove geometrically duplicate triangles regardless of winding.

    The first occurrence is retained.
    """
    seen: dict[tuple[int, int, int], int] = {}
    keep = np.ones(len(faces), dtype=bool)
    removed_original_ids = []

    for local_id, face in enumerate(faces):
        key = tuple(sorted(map(int, face)))
        if key in seen:
            keep[local_id] = False
            removed_original_ids.append(int(original_face_ids[local_id]))
        else:
            seen[key] = local_id

    return (
        faces[keep],
        original_face_ids[keep],
        np.asarray(removed_original_ids, dtype=int),
    )


def compact_vertices(
    points: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove unused points.

    Returns new points, remapped faces, and old-current-ID -> compact-ID map
    (-1 for unused vertices).
    """
    used = np.unique(faces.ravel())
    old_to_new = np.full(len(points), -1, dtype=int)
    old_to_new[used] = np.arange(len(used), dtype=int)
    return points[used], old_to_new[faces], old_to_new


def count_duplicate_faces(faces: np.ndarray) -> int:
    seen = set()
    duplicates = 0
    for face in faces:
        key = tuple(sorted(map(int, face)))
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)
    return duplicates


def count_collapsed_faces(faces: np.ndarray) -> int:
    return sum(
        len({int(a), int(b), int(c)}) < 3
        for a, b, c in faces
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stage-1 conservative CARP surface cleaner: weld near-coincident "
            "vertices and remove only resulting collapsed/duplicate triangles."
        )
    )
    parser.add_argument(
        "--mesh-base",
        default="./tube_cs",
        help="Input mesh basename (default: ./tube_cs)",
    )
    parser.add_argument(
        "--output-base",
        default="./tube_cs_cleaned_stage1",
        help="Output mesh basename (default: ./tube_cs_cleaned_stage1)",
    )
    parser.add_argument(
        "--coincident-tol",
        type=float,
        default=2.0,
        help="Maximum vertex distance in um to weld (default: 2.0)",
    )
    parser.add_argument(
        "--vm-surface-base",
        default=None,
        help=(
            "Generalized-capsule VoM mesh basename. When supplied, vertices "
            "belonging to the old prefix<->VoM collar are protected from "
            "near-coincident welding."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    input_base = Path(args.mesh_base).expanduser()
    output_base = Path(args.output_base).expanduser()

    if input_base.resolve() == output_base.resolve():
        raise ValueError(
            "Input and output basenames must differ; this stage intentionally "
            "does not overwrite the original mesh."
        )

    print("[1/4] Reading raw mesh", flush=True)
    points = read_pts(input_base)
    faces, regions = read_elem(input_base)
    lon = read_lon_if_present(input_base, len(faces))

    if len(points) == 0 or len(faces) == 0:
        raise ValueError("Input mesh is empty")

    original_point_count = len(points)
    original_face_count = len(faces)
    original_face_ids = np.arange(len(faces), dtype=int)
    protected_vertices = np.empty(0, dtype=int)
    prefix_point_count = None
    mixed_collar_face_count = 0

    if args.vm_surface_base is not None:
        (
            protected_vertices,
            prefix_point_count,
            mixed_collar_face_count,
        ) = junction_vertices_from_vm_split(
            points,
            faces,
            Path(args.vm_surface_base).expanduser(),
        )
        print(
            f"    protecting {len(protected_vertices)} old junction vertices "
            f"from {mixed_collar_face_count} mixed-collar faces",
            flush=True,
        )

    print(
        f"[2/4] Welding vertices within {float(args.coincident_tol):.6g} um",
        flush=True,
    )

    old_to_rep, welded_groups, skipped_protected_pairs = build_weld_map(
        points,
        float(args.coincident_tol),
        protected_vertices=protected_vertices,
    )

    # Keep the exact coordinate of the lowest-index representative.
    representative_ids = np.unique(old_to_rep)
    rep_to_intermediate = np.full(len(points), -1, dtype=int)
    rep_to_intermediate[representative_ids] = np.arange(
        len(representative_ids), dtype=int
    )

    intermediate_points = points[representative_ids]
    intermediate_faces = rep_to_intermediate[old_to_rep[faces]]

    print("[3/4] Removing weld-induced collapsed and duplicate triangles", flush=True)

    faces_after_collapse, kept_ids, collapsed_face_ids = remove_collapsed_faces(
        intermediate_faces,
        original_face_ids,
    )

    faces_after_duplicates, kept_ids, duplicate_face_ids = remove_duplicate_faces(
        faces_after_collapse,
        kept_ids,
    )

    kept_regions = regions[kept_ids]
    kept_lon = lon[kept_ids] if lon is not None else None

    final_points, final_faces, intermediate_to_final = compact_vertices(
        intermediate_points,
        faces_after_duplicates,
    )

    # Build original vertex -> final vertex mapping.
    original_to_intermediate = rep_to_intermediate[old_to_rep]
    original_to_final = np.full(len(points), -1, dtype=int)
    valid = original_to_intermediate >= 0
    original_to_final[valid] = intermediate_to_final[
        original_to_intermediate[valid]
    ]

    # Final guards.
    final_collapsed = count_collapsed_faces(final_faces)
    final_duplicates = count_duplicate_faces(final_faces)
    if final_collapsed != 0 or final_duplicates != 0:
        raise RuntimeError(
            "Internal cleaning error: collapsed/duplicate triangles remain "
            f"(collapsed={final_collapsed}, duplicate={final_duplicates})."
        )

    print("[4/4] Writing cleaned mesh and audit files", flush=True)
    write_mesh(
        output_base,
        final_points,
        final_faces,
        kept_regions,
        kept_lon,
    )

    audit_dir = output_base.parent / f"{output_base.name}_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    if len(protected_vertices):
        with (audit_dir / "protected_junction_vertices.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["original_vertex_id"])
            for vertex_id in protected_vertices:
                writer.writerow([int(vertex_id)])

    with (audit_dir / "welded_vertex_groups.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["group_id", "representative_vertex", "member_vertex"])
        for group_id, group in enumerate(welded_groups):
            representative = group[0]
            for member in group:
                writer.writerow([group_id, representative, member])

    with (audit_dir / "original_to_cleaned_vertex.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["original_vertex_id", "cleaned_vertex_id"])
        for old_id, new_id in enumerate(original_to_final):
            writer.writerow([old_id, int(new_id)])

    with (audit_dir / "removed_faces.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["original_face_id", "reason"])
        for face_id in collapsed_face_ids:
            writer.writerow([int(face_id), "collapsed_after_vertex_weld"])
        for face_id in duplicate_face_ids:
            writer.writerow([int(face_id), "duplicate_after_vertex_weld"])

    report = f"""Stage-1 mesh cleaning report
============================

Input mesh: {input_base}
Output mesh: {output_base}
Coincident tolerance (um): {float(args.coincident_tol):.12g}

Before
------
points: {original_point_count}
triangles: {original_face_count}

Junction protection
-------------------
VoM reference supplied: {args.vm_surface_base is not None}
original prefix points: {prefix_point_count if prefix_point_count is not None else "n/a"}
protected junction vertices: {len(protected_vertices)}
mixed collar faces identified: {mixed_collar_face_count}
near-coincident pairs skipped because protected: {skipped_protected_pairs}

Vertex welding
--------------
welded groups: {len(welded_groups)}
vertices participating in welded groups: {sum(len(g) for g in welded_groups)}
points after representative welding: {len(intermediate_points)}

Face cleanup
------------
collapsed faces removed after welding: {len(collapsed_face_ids)}
duplicate faces removed after welding: {len(duplicate_face_ids)}

After
-----
points: {len(final_points)}
triangles: {len(final_faces)}
collapsed faces remaining: {final_collapsed}
duplicate faces remaining: {final_duplicates}

Not modified in this stage
--------------------------
self-intersections
triangle shape/slivers
extreme angles
non-manifold topology unrelated to vertex welding
large-scale geometry

The original input mesh was not overwritten.
"""
    report_path = audit_dir / "cleaning_stage1_report.txt"
    report_path.write_text(report, encoding="utf-8")

    print("")
    print("Stage-1 cleaning complete")
    print("-------------------------")
    print(f"  protected junction verts: {len(protected_vertices)}")
    print(f"  protected weld pairs:     {skipped_protected_pairs}")
    print(f"  welded groups:            {len(welded_groups)}")
    print(f"  collapsed faces removed:  {len(collapsed_face_ids)}")
    print(f"  duplicate faces removed:  {len(duplicate_face_ids)}")
    print(f"  points:                   {original_point_count} -> {len(final_points)}")
    print(f"  triangles:                {original_face_count} -> {len(final_faces)}")
    print(f"  cleaned mesh:             {output_base}")
    print(f"  audit report:             {report_path}")


if __name__ == "__main__":
    main()
