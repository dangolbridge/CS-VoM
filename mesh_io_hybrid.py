"""I/O helpers for atrial tube construction.

This module intentionally contains no anatomical or geometry logic.  It only
handles the small subset of CARP/openCARP files needed by the tube-building
workflow.
"""

from __future__ import annotations

from pathlib import Path
import os
from typing import Iterable, Sequence

import numpy as np


def _with_suffix(path: str | Path, suffix: str) -> Path:
    """Return *path* with *suffix*, without duplicating an existing suffix."""
    path = Path(path)
    if path.suffix == suffix:
        return path
    return Path(f"{path}{suffix}")


def read_points(
    path: str | Path,
    *,
    dimensions: int = 3,
    suffix: str = ".pts",
    skip_metadata_line: bool = False,
) -> np.ndarray:
    """Read a CARP-style point file.

    Parameters
    ----------
    path
        File path with or without the requested suffix.
    dimensions
        Number of coordinates to read from each row.  Use 3 for ``.pts`` and
        4 for the UAC ``.hpts`` files used by the current workflow.
    suffix
        Expected file suffix.
    skip_metadata_line
        Set to ``True`` for ``.vtx``-style files that contain one metadata
        line after the point count.

    Returns
    -------
    numpy.ndarray
        Array with shape ``(N, dimensions)``.
    """
    if dimensions < 1:
        raise ValueError("dimensions must be at least 1")

    file_path = _with_suffix(path, suffix)
    with file_path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline().split()
        if not first_line:
            raise ValueError(f"Empty point file: {file_path}")

        expected_count = int(first_line[0])
        if skip_metadata_line:
            handle.readline()

        points: list[list[float]] = []
        for line_number, line in enumerate(handle, start=2):
            if not line.strip():
                continue
            values = line.split()
            if len(values) < dimensions:
                raise ValueError(
                    f"{file_path}:{line_number} has {len(values)} values; "
                    f"expected at least {dimensions}."
                )
            points.append([float(value) for value in values[:dimensions]])

    if len(points) != expected_count:
        raise ValueError(
            f"{file_path} declares {expected_count} points but contains "
            f"{len(points)} point rows."
        )

    return np.asarray(points, dtype=float)


def read_pts(path: str | Path) -> np.ndarray:
    """Read a three-dimensional ``.pts`` file."""
    return read_points(path, dimensions=3, suffix=".pts")


def read_hpts(path: str | Path) -> np.ndarray:
    """Read a four-coordinate UAC ``.hpts`` file."""
    return read_points(path, dimensions=4, suffix=".hpts")


def read_vtx(path: str | Path) -> np.ndarray:
    """Read vertex indices from a CARP ``.vtx`` file."""
    values = read_points(
        path,
        dimensions=1,
        suffix=".vtx",
        skip_metadata_line=True,
    )
    return values[:, 0].astype(int)


def write_pts(path: str | Path, points: Sequence[Sequence[float]]) -> Path:
    """Write points in CARP ``.pts`` format and return the written path."""
    file_path = _with_suffix(path, ".pts")
    file_path.parent.mkdir(parents=True, exist_ok=True)

    points_array = np.asarray(points, dtype=float)
    if points_array.ndim != 2 or points_array.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")

    with file_path.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(points_array)}\n")
        np.savetxt(handle, points_array, fmt="%.16g")

    return file_path


def triangles_from_faces(faces: np.ndarray | Sequence[Sequence[int]]) -> np.ndarray:
    """Convert supported triangle representations to an ``(N, 3)`` array.

    Accepted inputs are:

    - ``(N, 3)`` triangle indices;
    - ``(N, 4)`` VTK-style rows ``[3, i0, i1, i2]``;
    - a flat VTK-style array ``[3, i0, i1, i2, 3, ...]``.
    """
    array = np.asarray(faces, dtype=int)

    if array.ndim == 1:
        if len(array) % 4 != 0:
            raise ValueError("Flat VTK triangle array length must be divisible by 4")
        array = array.reshape(-1, 4)

    if array.ndim != 2:
        raise ValueError("faces must be a 1-D flat array or a 2-D triangle array")

    if array.shape[1] == 4:
        if not np.all(array[:, 0] == 3):
            raise ValueError("VTK-style triangle rows must start with 3")
        array = array[:, 1:4]
    elif array.shape[1] != 3:
        raise ValueError("Triangle faces must have 3 indices per face")

    if np.any(array < 0):
        raise ValueError("Triangle indices must be non-negative")

    return array.astype(int, copy=False)


def read_triangles(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read triangular elements from a CARP ``.elem`` file.

    Returns
    -------
    triangles
        ``(N, 3)`` integer vertex indices.
    regions
        ``(N,)`` integer region labels.
    """
    file_path = _with_suffix(path, ".elem")
    with file_path.open("r", encoding="utf-8") as handle:
        header = handle.readline().split()
        if not header:
            raise ValueError(f"Empty element file: {file_path}")
        expected_count = int(header[0])

        triangles: list[list[int]] = []
        regions: list[int] = []
        for line_number, line in enumerate(handle, start=2):
            if not line.strip():
                continue
            parts = line.split()
            if parts[0] != "Tr":
                raise ValueError(
                    f"{file_path}:{line_number}: expected triangular element 'Tr', "
                    f"got {parts[0]!r}."
                )
            if len(parts) < 5:
                raise ValueError(f"Malformed triangle at {file_path}:{line_number}")
            triangles.append([int(parts[1]), int(parts[2]), int(parts[3])])
            regions.append(int(parts[4]))

    if len(triangles) != expected_count:
        raise ValueError(
            f"{file_path} declares {expected_count} elements but contains "
            f"{len(triangles)} rows."
        )

    return np.asarray(triangles, dtype=int), np.asarray(regions, dtype=int)



def read_mixed_elements(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read ``Tr`` and ``Ln`` elements from a CARP ``.elem`` file.

    This is intended for final biatrial meshes that contain both triangular
    tissue surfaces and one-dimensional connector elements.

    Returns
    -------
    triangles, triangle_regions, lines, line_regions
        Triangle indices have shape ``(N, 3)`` and line indices ``(M, 2)``.
        Empty element classes are returned with the corresponding empty shape.
    """
    file_path = _with_suffix(path, ".elem")
    with file_path.open("r", encoding="utf-8") as handle:
        header = handle.readline().split()
        if not header:
            raise ValueError(f"Empty element file: {file_path}")
        expected_count = int(header[0])

        triangles: list[list[int]] = []
        triangle_regions: list[int] = []
        lines: list[list[int]] = []
        line_regions: list[int] = []
        parsed_count = 0

        for line_number, line in enumerate(handle, start=2):
            if not line.strip():
                continue
            parts = line.split()
            etype = parts[0]

            if etype == "Tr":
                if len(parts) < 5:
                    raise ValueError(
                        f"Malformed triangle at {file_path}:{line_number}"
                    )
                triangles.append([int(parts[1]), int(parts[2]), int(parts[3])])
                triangle_regions.append(int(parts[4]))
            elif etype == "Ln":
                if len(parts) < 4:
                    raise ValueError(
                        f"Malformed line element at {file_path}:{line_number}"
                    )
                lines.append([int(parts[1]), int(parts[2])])
                line_regions.append(int(parts[3]))
            else:
                raise ValueError(
                    f"{file_path}:{line_number}: unsupported CARP element type "
                    f"{etype!r}; visualization currently supports Tr and Ln."
                )
            parsed_count += 1

    if parsed_count != expected_count:
        raise ValueError(
            f"{file_path} declares {expected_count} elements but contains "
            f"{parsed_count} parsed rows."
        )

    tri = (
        np.asarray(triangles, dtype=int).reshape(-1, 3)
        if triangles else np.empty((0, 3), dtype=int)
    )
    tri_reg = np.asarray(triangle_regions, dtype=int)
    line_arr = (
        np.asarray(lines, dtype=int).reshape(-1, 2)
        if lines else np.empty((0, 2), dtype=int)
    )
    line_reg = np.asarray(line_regions, dtype=int)

    return tri, tri_reg, line_arr, line_reg

def write_triangles(
    path: str | Path,
    faces: np.ndarray | Sequence[Sequence[int]],
    *,
    region: int | Sequence[int] = 5,
) -> Path:
    """Write triangular faces to a CARP ``.elem`` file."""
    file_path = _with_suffix(path, ".elem")
    file_path.parent.mkdir(parents=True, exist_ok=True)

    triangles = triangles_from_faces(faces)
    if np.isscalar(region):
        regions = np.full(len(triangles), int(region), dtype=int)
    else:
        regions = np.asarray(region, dtype=int)
        if regions.shape != (len(triangles),):
            raise ValueError("region must be a scalar or contain one value per face")

    with file_path.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(triangles)} # number of faces\n")
        for (i0, i1, i2), reg in zip(triangles, regions):
            handle.write(f"Tr {i0} {i1} {i2} {reg}\n")

    return file_path


def write_lon(
    path: str | Path,
    n_elements: int,
    *,
    fiber: Iterable[float] = (1.0, 0.0, 0.0),
) -> Path:
    """Write one three-component fibre vector per element.

    This preserves the header-free ``.lon`` convention used by the current
    tube-generation script.
    """
    file_path = _with_suffix(path, ".lon")
    file_path.parent.mkdir(parents=True, exist_ok=True)

    fiber_array = np.asarray(tuple(fiber), dtype=float)
    if fiber_array.shape != (3,):
        raise ValueError("fiber must contain exactly three values")

    with file_path.open("w", encoding="utf-8") as handle:
        for _ in range(int(n_elements)):
            handle.write(" ".join(f"{value:.16g}" for value in fiber_array) + "\n")

    return file_path


def write_carp_surface(
    base_path: str | Path,
    points: Sequence[Sequence[float]],
    faces: np.ndarray | Sequence[Sequence[int]],
    *,
    region: int | Sequence[int] = 5,
    fiber: Iterable[float] = (1.0, 0.0, 0.0),
) -> tuple[Path, Path, Path]:
    """Write ``.pts``, ``.elem`` and ``.lon`` files for a triangular surface."""
    triangles = triangles_from_faces(faces)
    pts_path = write_pts(base_path, points)
    elem_path = write_triangles(base_path, triangles, region=region)
    lon_path = write_lon(base_path, len(triangles), fiber=fiber)
    return pts_path, elem_path, lon_path


def load_regions(path: str | Path, *, delimiter: str = ",") -> np.ndarray:
    """Load the anatomical seed/region coordinate file used by the workflow."""
    return np.loadtxt(Path(path), delimiter=delimiter, dtype=float)


def read_carp_mesh(
    directory: str | Path,
    mesh_name: str,
    *,
    return_surface: bool = False,
):
    """Thin wrapper around ``uac.utils.read_carp``.

    The import is intentionally local so that the generic I/O helpers can be
    imported even when AtrialMTK/UAC is not installed.
    """
    from uac import utils  # local import: optional external dependency

    directory = str(Path(directory))
    if not directory.endswith(os.sep):
        directory += os.sep
    return utils.read_carp(directory, mesh_name, return_surface=return_surface)

# -----------------------------------------------------------------------------
# Generic mixed-element I/O used by build_biatrial_connections.py
# -----------------------------------------------------------------------------


def write_scalar_values(path: str | Path, values) -> Path:
    """Write one scalar value per line and return the written path."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(values)
    with file_path.open("w", encoding="utf-8") as handle:
        for value in array.reshape(-1):
            if np.issubdtype(array.dtype, np.integer):
                handle.write(f"{int(value)}\n")
            else:
                handle.write(f"{float(value):.16g}\n")
    return file_path


def read_scalar_values(path: str | Path, *, dtype=float) -> np.ndarray:
    """Read a one-value-per-line scalar file."""
    return np.loadtxt(Path(path), dtype=dtype)


def write_carp_mesh(
    base_path: str | Path,
    points,
    elements,
    fibers=None,
    data=None,
):
    """Write a CARP mesh that may contain mixed element types such as Tr/Ln.

    This intentionally delegates mixed-element formatting to ``uac.utils`` so
    the output convention remains identical to the existing AtrialMTK/UAC
    workflow.
    """
    from uac import utils

    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)

    pts = np.asarray(points, dtype=float)
    elems = list(elements)

    if fibers is None:
        fiber_array = np.tile(np.asarray([1.0, 0.0, 0.0]), (len(elems), 1))
    else:
        fiber_array = np.asarray(fibers, dtype=float)
        if fiber_array.ndim != 2 or fiber_array.shape[1] != 3:
            raise ValueError("fibers must have shape (N, 3)")
        if len(fiber_array) != len(elems):
            raise ValueError(
                f"fibers has {len(fiber_array)} rows but mesh has {len(elems)} elements"
            )

    utils.write_carp(pts, elems, fiber_array, data, str(base))
    return (
        _with_suffix(base, ".pts"),
        _with_suffix(base, ".elem"),
        _with_suffix(base, ".lon"),
    )
