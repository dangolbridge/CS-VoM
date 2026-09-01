#!/usr/bin/env python3
"""General visualization for CARP/openCARP meshes used by this project.

The default mode writes a self-contained interactive Plotly HTML file, so a
user can inspect the mesh in an ordinary web browser without Meshalyzer or any
other dedicated mesh viewer.

Both triangular tissue elements (``Tr``) and connector line elements (``Ln``)
are supported.  Element tags are shown as separate legend entries, and
diagnostic ``.pts`` files can be overlaid.

Examples
--------
Final biatrial mesh::

    python3 visualization.py \
        --mesh Reentry_surface_iac \
        --output-html Reentry_surface_iac.html \
        --open-browser

Overlay BB geodesics::

    python3 visualization.py \
        --mesh BB \
        --point-file bb_bridges_anatomical_ra_geodesic.pts \
        --point-file bb_bridges_anatomical_la_geodesic.pts \
        --point-lines \
        --output-html bb_check.html

PyVista remains available as an optional desktop mode when installed.
Meshalyzer is also retained as an optional external viewer.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import webbrowser

import numpy as np

from mesh_io_hybrid import read_mixed_elements, read_pts


def _mesh_base(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.with_suffix("") if path.suffix in {".pts", ".elem", ".lon"} else path


def _require_plotly():
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError(
            "Browser visualization requires Plotly. Install it with "
            "'python -m pip install plotly'."
        ) from exc
    return go


def _require_pyvista():
    try:
        import pyvista as pv
    except ImportError as exc:
        raise RuntimeError(
            "PyVista visualization is optional. Install it with "
            "'python -m pip install pyvista'."
        ) from exc
    return pv


def _filter_by_tags(elements, regions, tags):
    if tags is None:
        return elements, regions
    tags = np.asarray(sorted(set(map(int, tags))), dtype=int)
    mask = np.isin(regions, tags)
    return elements[mask], regions[mask]


def _deterministic_triangle_limit(triangles, regions, max_triangles):
    """Visualization-only deterministic thinning, preserving every tag."""
    if max_triangles is None or len(triangles) <= max_triangles:
        return triangles, regions
    if max_triangles < 1:
        raise ValueError("--max-triangles must be positive")

    selected = []
    unique = np.unique(regions)
    # Give each tag a proportional quota, with at least one triangle per tag.
    for tag in unique:
        ids = np.flatnonzero(regions == tag)
        quota = max(1, int(round(max_triangles * len(ids) / len(triangles))))
        if quota >= len(ids):
            selected.extend(ids.tolist())
        else:
            take = np.linspace(0, len(ids) - 1, quota, dtype=int)
            selected.extend(ids[take].tolist())

    selected = np.asarray(sorted(set(selected)), dtype=int)
    if len(selected) > max_triangles:
        selected = selected[np.linspace(0, len(selected) - 1, max_triangles, dtype=int)]
    return triangles[selected], regions[selected]


def _edge_segments(points, triangles):
    edges = set()
    for a, b, c in np.asarray(triangles, dtype=int):
        for u, v in ((a, b), (b, c), (c, a)):
            edges.add((min(int(u), int(v)), max(int(u), int(v))))
    x, y, z = [], [], []
    for u, v in edges:
        p, q = points[u], points[v]
        x.extend((p[0], q[0], None))
        y.extend((p[1], q[1], None))
        z.extend((p[2], q[2], None))
    return x, y, z


def build_plotly_figure(
    mesh_base: str | Path,
    *,
    point_files=(),
    point_lines: bool = False,
    tags=None,
    surface_opacity: float = 1.0,
    show_edges: bool = False,
    show_axes: bool = True,
    max_triangles: int | None = None,
    title: str | None = None,
):
    """Create an interactive Plotly figure for a mixed CARP mesh."""
    go = _require_plotly()
    base = _mesh_base(mesh_base)
    points = read_pts(base)
    triangles, tri_regions, lines, line_regions = read_mixed_elements(base)

    triangles, tri_regions = _filter_by_tags(triangles, tri_regions, tags)
    lines, line_regions = _filter_by_tags(lines, line_regions, tags)
    triangles, tri_regions = _deterministic_triangle_limit(
        triangles, tri_regions, max_triangles
    )

    fig = go.Figure()

    # One surface trace per region makes tags independently hideable in legend.
    for tag in np.unique(tri_regions):
        tri = triangles[tri_regions == tag]
        fig.add_trace(
            go.Mesh3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                i=tri[:, 0],
                j=tri[:, 1],
                k=tri[:, 2],
                name=f"Tr tag {int(tag)}",
                opacity=float(surface_opacity),
                flatshading=True,
                hoverinfo="name",
                showlegend=True,
            )
        )

        if show_edges:
            x, y, z = _edge_segments(points, tri)
            fig.add_trace(
                go.Scatter3d(
                    x=x, y=y, z=z,
                    mode="lines",
                    name=f"edges tag {int(tag)}",
                    line={"width": 1},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    # Connector line elements, grouped by tag.
    for tag in np.unique(line_regions):
        elems = lines[line_regions == tag]
        x, y, z = [], [], []
        for u, v in elems:
            p, q = points[int(u)], points[int(v)]
            x.extend((p[0], q[0], None))
            y.extend((p[1], q[1], None))
            z.extend((p[2], q[2], None))
        fig.add_trace(
            go.Scatter3d(
                x=x, y=y, z=z,
                mode="lines",
                name=f"Ln tag {int(tag)}",
                line={"width": 5},
                hoverinfo="name",
            )
        )

    # Optional diagnostic point/geodesic files.
    for value in point_files:
        path = Path(value).expanduser()
        pts = read_pts(path)
        mode = "lines+markers" if point_lines and len(pts) > 1 else "markers"
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode=mode,
                name=path.stem,
                marker={"size": 4},
                line={"width": 5},
            )
        )

    axis = {
        "visible": bool(show_axes),
        "showbackground": False,
        "showgrid": bool(show_axes),
        "zeroline": False,
    }
    fig.update_layout(
        title=title or base.name,
        scene={
            "xaxis": dict(axis, title="x"),
            "yaxis": dict(axis, title="y"),
            "zaxis": dict(axis, title="z"),
            "aspectmode": "data",
        },
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "t": 45, "b": 0},
    )

    return fig


def write_interactive_html(
    mesh_base: str | Path,
    output_html: str | Path,
    *,
    point_files=(),
    point_lines: bool = False,
    tags=None,
    surface_opacity: float = 1.0,
    show_edges: bool = False,
    show_axes: bool = True,
    max_triangles: int | None = None,
    title: str | None = None,
    open_browser: bool = False,
) -> Path:
    """Write a self-contained interactive HTML visualization."""
    fig = build_plotly_figure(
        mesh_base,
        point_files=point_files,
        point_lines=point_lines,
        tags=tags,
        surface_opacity=surface_opacity,
        show_edges=show_edges,
        show_axes=show_axes,
        max_triangles=max_triangles,
        title=title,
    )

    output = Path(output_html).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    # include_plotlyjs=True makes the HTML independent of an internet connection.
    fig.write_html(
        str(output),
        include_plotlyjs=True,
        full_html=True,
        auto_open=False,
    )
    print(f"Interactive visualization: {output}", flush=True)

    if open_browser:
        webbrowser.open(output.resolve().as_uri())
    return output


def show_pyvista(
    mesh_base: str | Path,
    *,
    point_files=(),
    tags=None,
    surface_opacity: float = 1.0,
):
    """Optional desktop visualization using PyVista."""
    pv = _require_pyvista()
    base = _mesh_base(mesh_base)
    points = read_pts(base)
    triangles, tri_regions, lines, line_regions = read_mixed_elements(base)
    triangles, tri_regions = _filter_by_tags(triangles, tri_regions, tags)
    lines, line_regions = _filter_by_tags(lines, line_regions, tags)

    plotter = pv.Plotter(title=base.name)
    plotter.add_axes()

    for tag in np.unique(tri_regions):
        tri = triangles[tri_regions == tag]
        vtk_faces = np.column_stack(
            (np.full(len(tri), 3, dtype=np.int64), tri)
        ).reshape(-1)
        poly = pv.PolyData(points, vtk_faces)
        plotter.add_mesh(
            poly,
            opacity=float(surface_opacity),
            show_edges=True,
            label=f"Tr tag {int(tag)}",
        )

    for tag in np.unique(line_regions):
        for u, v in lines[line_regions == tag]:
            segment = pv.Line(points[int(u)], points[int(v)])
            plotter.add_mesh(segment, line_width=4)

    for value in point_files:
        pts = read_pts(value)
        if len(pts) == 1:
            plotter.add_points(pts, point_size=8, render_points_as_spheres=True)
        elif len(pts) > 1:
            plotter.add_mesh(pv.lines_from_points(pts), line_width=4)

    plotter.add_legend()
    plotter.show()


def launch_meshalyzer(
    mesh_base: str | Path,
    *point_files: str | Path,
    executable: str = "meshalyzer",
    wait: bool = False,
):
    """Launch Meshalyzer when the external viewer is available."""
    if shutil.which(executable) is None:
        raise FileNotFoundError(
            f"Could not find {executable!r} on PATH. Use browser mode instead "
            "if Meshalyzer is not installed."
        )

    mesh_arg = str(_mesh_base(mesh_base))
    if not mesh_arg.endswith("."):
        mesh_arg += "."
    args = [str(Path(p).expanduser()) for p in point_files]
    command = [executable, mesh_arg, *args]
    print("Launching:", " ".join(command), flush=True)
    if wait:
        return subprocess.run(command, check=False)
    return subprocess.Popen(command)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Visualize CARP Tr/Ln meshes in a web browser (default), PyVista, "
            "or Meshalyzer."
        )
    )
    p.add_argument("--mesh", required=True, help="CARP mesh basename")
    p.add_argument(
        "--viewer",
        choices=("browser", "pyvista", "meshalyzer"),
        default="browser",
    )
    p.add_argument(
        "--point-file",
        action="append",
        default=[],
        help="Overlay a CARP .pts file; repeat for multiple files",
    )
    p.add_argument(
        "--point-lines",
        action="store_true",
        help="Connect consecutive points in --point-file overlays",
    )
    p.add_argument(
        "--tags",
        nargs="+",
        type=int,
        default=None,
        help="Show only these Tr/Ln tags",
    )
    p.add_argument("--surface-opacity", type=float, default=1.0)
    p.add_argument("--show-edges", action="store_true")
    p.add_argument("--hide-axes", action="store_true")
    p.add_argument(
        "--max-triangles",
        type=int,
        default=None,
        help=(
            "Visualization-only deterministic triangle limit for very large "
            "meshes; the source mesh is never modified"
        ),
    )
    p.add_argument("--title", default=None)
    p.add_argument(
        "--output-html",
        default=None,
        help="Browser output path; default: <mesh>_view.html",
    )
    p.add_argument("--open-browser", action="store_true")
    return p


def main() -> None:
    args = parser().parse_args()
    if not (0.0 < args.surface_opacity <= 1.0):
        raise ValueError("--surface-opacity must lie in (0, 1]")

    if args.viewer == "browser":
        output = (
            Path(args.output_html)
            if args.output_html
            else Path(f"{_mesh_base(args.mesh)}_view.html")
        )
        write_interactive_html(
            args.mesh,
            output,
            point_files=args.point_file,
            point_lines=args.point_lines,
            tags=args.tags,
            surface_opacity=args.surface_opacity,
            show_edges=args.show_edges,
            show_axes=not args.hide_axes,
            max_triangles=args.max_triangles,
            title=args.title,
            open_browser=args.open_browser,
        )
    elif args.viewer == "pyvista":
        show_pyvista(
            args.mesh,
            point_files=args.point_file,
            tags=args.tags,
            surface_opacity=args.surface_opacity,
        )
    else:
        launch_meshalyzer(args.mesh, *args.point_file, wait=True)


if __name__ == "__main__":
    main()
