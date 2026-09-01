#!/usr/bin/env python3
"""Master assembly pipeline for the CS/VoM biatrial model workflow.

This script intentionally starts from two already-validated inputs:

1. an AtrialMTK ``Bilayer_Combined`` mesh that already contains the RA-side
   and LA-side Bachmann-bundle surfaces; and
2. the accepted cleaned CS/VoM surface, normally ``tube_cs_cleaned``.

It orchestrates only the downstream assembly steps:

    Bilayer_Combined
        -> extract BB (tag 10)
        -> merge cleaned CS/VoM
        -> add bilayer/IAC/CS/VoM line elements
        -> build the two BB connector surfaces
        -> merge both BB connectors
        -> rebuild the final element-wise .f3d file
        -> validate final CARP element/fibre/scalar counts
        -> optionally generate a self-contained browser visualization

Geometry is not reimplemented here.  ``meshtool`` performs mesh extraction and
merging, while the project-specific Python scripts perform anatomical geometry.
No accepted input mesh is overwritten.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
from typing import Iterable, Sequence

import numpy as np

from mesh_io_hybrid import read_mixed_elements, read_pts
from mesh_tags import (
    BB_ANATOMICAL_BRIDGE_TAG,
    BB_CLOSEST_BRIDGE_TAG,
    BB_REGION,
    CS_LA_CONNECTION_TAG,
    VOM_LA_CONNECTION_TAG,
)


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def mesh_base(directory: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = directory / path
    if path.suffix in {".pts", ".elem", ".lon", ".f3d"}:
        path = path.with_suffix("")
    return path


def complete_mesh(base: Path, *, require_lon: bool = True) -> bool:
    required = [Path(f"{base}.pts"), Path(f"{base}.elem")]
    if require_lon:
        required.append(Path(f"{base}.lon"))
    return all(path.is_file() for path in required)


def any_mesh_output_exists(base: Path) -> bool:
    return any(Path(f"{base}{suffix}").exists() for suffix in (".pts", ".elem", ".lon", ".f3d"))


def require_mesh(base: Path, *, description: str, require_lon: bool = True) -> None:
    missing = [suffix for suffix in (".pts", ".elem") if not Path(f"{base}{suffix}").is_file()]
    if require_lon and not Path(f"{base}.lon").is_file():
        missing.append(".lon")
    if missing:
        raise FileNotFoundError(
            f"{description} is incomplete: {base} (missing {', '.join(missing)})"
        )


def remove_mesh_outputs(base: Path, *, include_f3d: bool = True) -> None:
    suffixes = [".pts", ".elem", ".lon"]
    if include_f3d:
        suffixes.append(".f3d")
    for suffix in suffixes:
        path = Path(f"{base}{suffix}")
        if path.exists():
            path.unlink()


def count_nonempty_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def declared_element_count(elem_path: Path) -> int:
    with elem_path.open("r", encoding="utf-8") as handle:
        first = handle.readline().split()
    if not first:
        raise ValueError(f"Empty element file: {elem_path}")
    return int(first[0])


def element_signature(base: Path) -> list[tuple[str, int]]:
    """Return element type/tag order exactly as stored in a CARP .elem file."""
    path = Path(f"{base}.elem")
    signature: list[tuple[str, int]] = []
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().split()
        if not header:
            raise ValueError(f"Empty element file: {path}")
        expected = int(header[0])
        for line_no, line in enumerate(handle, start=2):
            if not line.strip():
                continue
            parts = line.split()
            etype = parts[0]
            if etype == "Tr":
                if len(parts) < 5:
                    raise ValueError(f"Malformed Tr element at {path}:{line_no}")
                tag = int(parts[4])
            elif etype == "Ln":
                if len(parts) < 4:
                    raise ValueError(f"Malformed Ln element at {path}:{line_no}")
                tag = int(parts[3])
            else:
                raise ValueError(
                    f"Unsupported element type {etype!r} at {path}:{line_no}"
                )
            signature.append((etype, tag))
    if len(signature) != expected:
        raise ValueError(
            f"{path} declares {expected} elements but contains {len(signature)} rows"
        )
    return signature


def verify_merge_element_order(final_base: Path, ordered_inputs: Sequence[Path]) -> None:
    expected: list[tuple[str, int]] = []
    for base in ordered_inputs:
        expected.extend(element_signature(base))
    actual = element_signature(final_base)
    if actual != expected:
        mismatch = next(
            (i for i, (a, b) in enumerate(zip(actual, expected)) if a != b),
            min(len(actual), len(expected)),
        )
        raise ValueError(
            "meshtool merge did not preserve the expected element type/tag order; "
            f"first mismatch is at element {mismatch}. Refusing to concatenate .f3d."
        )


def read_scalar_file(path: Path) -> np.ndarray:
    values = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                values.append(float(line.split()[0]))
            except (ValueError, IndexError) as exc:
                raise ValueError(f"Malformed scalar row at {path}:{line_no}") from exc
    return np.asarray(values, dtype=float)


def concatenate_scalar_files(inputs: Sequence[Path], output: Path) -> int:
    arrays = []
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"Required scalar file does not exist: {path}")
        arrays.append(read_scalar_file(path))
    values = np.concatenate(arrays) if arrays else np.empty(0, dtype=float)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(f"{float(value):.16g}\n")
    return len(values)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Runner:
    def __init__(self, *, dry_run: bool, log_path: Path):
        self.dry_run = bool(dry_run)
        self.log_path = log_path
        self.commands: list[list[str]] = []
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.dry_run:
            self.log_path.write_text("", encoding="utf-8")

    def run(self, command: Iterable[object], *, cwd: Path | None = None) -> None:
        cmd = [str(x) for x in command]
        self.commands.append(cmd)
        rendered = shlex.join(cmd)
        prefix = f"(cd {shlex.quote(str(cwd))} && " if cwd else ""
        suffix = ")" if cwd else ""
        print(f"+ {prefix}{rendered}{suffix}", flush=True)
        if self.dry_run:
            return
        with self.log_path.open("a", encoding="utf-8") as handle:
            if cwd:
                handle.write(f"# cwd: {cwd}\n")
            handle.write(rendered + "\n")
        subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


# -----------------------------------------------------------------------------
# Pipeline stages
# -----------------------------------------------------------------------------


def prepare_output(base: Path, *, resume: bool, force: bool, description: str) -> bool:
    """Return True when an existing complete output should be reused."""
    exists = complete_mesh(base)
    partial = any_mesh_output_exists(base) and not exists
    if resume and exists:
        print(f"    resume: reusing {description}: {base}", flush=True)
        return True
    if resume and partial:
        raise FileExistsError(
            f"Cannot resume {description}: partial output files exist for {base}. "
            "Use --force to regenerate the stage."
        )
    if force:
        remove_mesh_outputs(base)
        return False
    if exists or partial:
        state = "complete" if exists else "partial"
        raise FileExistsError(
            f"{description} {state} output already exists: {base}. Use --resume "
            "for complete outputs or --force to regenerate generated outputs."
        )
    return False


def meshtool_extract_bb(
    runner: Runner,
    *,
    meshtool: str,
    input_mesh: Path,
    output_mesh: Path,
    resume: bool,
    force: bool,
) -> None:
    if prepare_output(output_mesh, resume=resume, force=force, description="BB working mesh"):
        return
    runner.run(
        [
            meshtool,
            "extract",
            "mesh",
            f"-msh={input_mesh}",
            f"-tags={BB_REGION}",
            f"-submsh={output_mesh}",
            "-ofmt=carp_txt",
        ]
    )


def meshtool_merge(
    runner: Runner,
    *,
    meshtool: str,
    mesh1: Path,
    mesh2: Path,
    output: Path,
    resume: bool,
    force: bool,
    description: str,
) -> None:
    if prepare_output(output, resume=resume, force=force, description=description):
        return
    runner.run(
        [
            meshtool,
            "merge",
            "meshes",
            f"-msh1={mesh1}",
            f"-msh2={mesh2}",
            "-ofmt=carp_txt",
            f"-outmsh={output}",
        ]
    )


def run_connection_builder(
    runner: Runner,
    *,
    python: str,
    script: Path,
    la_dir: Path,
    ra_dir: Path,
    work_dir: Path,
    input_mesh_name: str,
    output_mesh_name: str,
    cs_mesh: Path,
    fo_index: int,
    fo_connections: int,
    cs_positions: Sequence[float],
    vom_positions: Sequence[float],
    connection_neighbors: int,
    scale_factor: float,
    resume: bool,
    force: bool,
) -> None:
    output = work_dir / output_mesh_name
    if prepare_output(output, resume=resume, force=force, description="connection mesh"):
        # The builder's f3d is essential for final assembly.
        if not Path(f"{output}.f3d").is_file():
            raise FileNotFoundError(
                f"Cannot resume connection stage because {output}.f3d is missing"
            )
        return

    command = [
        python,
        script,
        "--la-path", la_dir,
        "--ra-path", ra_dir,
        "--target-path", work_dir,
        "--mesh-name", input_mesh_name,
        "--output-mesh", output_mesh_name,
        "--cs-mesh", cs_mesh,
        "--fo-index", fo_index,
        "--fo-connections", fo_connections,
        "--connection-neighbors", connection_neighbors,
        "--scale-factor", scale_factor,
        "--write-debug-points",
    ]
    if cs_positions:
        command.extend(["--cs-connection-positions", *cs_positions])
    if vom_positions:
        command.extend(["--vom-connection-positions", *vom_positions])
    runner.run(command)


def run_bb_builder(
    runner: Runner,
    *,
    python: str,
    script: Path,
    work_dir: Path,
    base_mesh_name: str,
    bb_mesh_name: str,
    ra_dir: Path,
    svc_landmark_index: int,
    scale_factor: float,
    closest_output: str,
    anatomical_output: str,
    closest_spacing: float,
    anatomical_spacing: float,
    resume: bool,
    force: bool,
) -> None:
    closest = work_dir / closest_output
    anatomical = work_dir / anatomical_output
    complete = (
        complete_mesh(closest)
        and complete_mesh(anatomical)
        and Path(f"{closest}.f3d").is_file()
        and Path(f"{anatomical}.f3d").is_file()
    )
    any_existing = any_mesh_output_exists(closest) or any_mesh_output_exists(anatomical)
    if resume and complete:
        print("    resume: reusing both BB connector surfaces", flush=True)
        return
    if resume and any_existing and not complete:
        raise FileExistsError(
            "Cannot resume BB bridge stage because its outputs are partial. "
            "Use --force to regenerate both bridges."
        )
    if not force and any_existing:
        raise FileExistsError(
            "BB bridge outputs already exist or are partial. Use --resume for a "
            "complete pair or --force to regenerate them."
        )
    if force:
        remove_mesh_outputs(closest)
        remove_mesh_outputs(anatomical)

    runner.run(
        [
            python,
            script,
            "--target-path", work_dir,
            "--mesh-name", base_mesh_name,
            "--bb-mesh", bb_mesh_name,
            "--ra-path", ra_dir,
            "--svc-landmark-index", svc_landmark_index,
            "--scale-factor", scale_factor,
            "--closest-output", closest_output,
            "--closest-tag", BB_CLOSEST_BRIDGE_TAG,
            "--closest-spacing", closest_spacing,
            "--anatomical-output", anatomical_output,
            "--anatomical-tag", BB_ANATOMICAL_BRIDGE_TAG,
            "--anatomical-spacing", anatomical_spacing,
        ]
    )


def validate_final_mesh(base: Path) -> dict:
    require_mesh(base, description="final mesh", require_lon=True)
    elem_path = Path(f"{base}.elem")
    lon_path = Path(f"{base}.lon")
    f3d_path = Path(f"{base}.f3d")
    if not f3d_path.is_file():
        raise FileNotFoundError(f"Final .f3d does not exist: {f3d_path}")

    n_declared = declared_element_count(elem_path)
    # .lon and .f3d are header-free in this workflow.
    n_lon = count_nonempty_rows(lon_path)
    n_f3d = count_nonempty_rows(f3d_path)
    triangles, tri_regions, lines, line_regions = read_mixed_elements(base)
    n_parsed = len(triangles) + len(lines)

    if not (n_declared == n_parsed == n_lon == n_f3d):
        raise ValueError(
            "Final mesh cardinality mismatch: "
            f"elem header={n_declared}, parsed elements={n_parsed}, "
            f"lon rows={n_lon}, f3d rows={n_f3d}"
        )

    points = read_pts(base)
    if len(triangles) and int(np.max(triangles)) >= len(points):
        raise ValueError("Final triangle mesh contains an out-of-range vertex index")
    if len(lines) and int(np.max(lines)) >= len(points):
        raise ValueError("Final line elements contain an out-of-range vertex index")

    tri_counts = {str(int(tag)): int(np.sum(tri_regions == tag)) for tag in np.unique(tri_regions)}
    line_counts = {str(int(tag)): int(np.sum(line_regions == tag)) for tag in np.unique(line_regions)}

    return {
        "points": int(len(points)),
        "elements": int(n_declared),
        "triangles": int(len(triangles)),
        "lines": int(len(lines)),
        "lon_rows": int(n_lon),
        "f3d_rows": int(n_f3d),
        "triangle_tag_counts": tri_counts,
        "line_tag_counts": line_counts,
    }


# -----------------------------------------------------------------------------
# CLI / orchestration
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Assemble the final biatrial CS/VoM model from an AtrialMTK "
            "Bilayer_Combined mesh and an accepted tube_cs_cleaned surface."
        )
    )
    p.add_argument("--la-path", required=True, help="AtrialMTK LA/Bia result directory")
    p.add_argument("--ra-path", required=True, help="AtrialMTK RA/result directory")
    p.add_argument(
        "--work-dir",
        default=None,
        help="Generated-output directory; default: --ra-path",
    )
    p.add_argument("--mesh-name", default="Bilayer_Combined")
    p.add_argument("--cs-mesh", default="tube_cs_cleaned")
    p.add_argument("--bb-mesh", default="BB")

    p.add_argument("--cs-connection-positions", nargs="*", type=float, default=[])
    p.add_argument("--vom-connection-positions", nargs="*", type=float, default=[])
    p.add_argument("--connection-neighbors", type=int, default=5)
    p.add_argument("--fo-index", type=int, default=3)
    p.add_argument("--fo-connections", type=int, default=100)
    p.add_argument(
        "--svc-landmark-index",
        type=int,
        required=True,
        help="Row in RA Landmarks.txt representing the SVC reference",
    )
    p.add_argument("--scale-factor", type=float, default=1.0)

    p.add_argument("--closest-bb-spacing", type=float, default=350.0)
    p.add_argument("--anatomical-bb-spacing", type=float, default=350.0)

    p.add_argument("--merged-mesh", default="Biatrial_with_CSVoM")
    p.add_argument("--connections-mesh", default="Reentry_surface_iac_base")
    p.add_argument("--closest-bridge", default="bb_closest_bridge")
    p.add_argument("--anatomical-bridge", default="bb_anatomical_bridge")
    p.add_argument("--output-mesh", default="Reentry_surface_iac")

    p.add_argument(
        "--scripts-dir",
        default=None,
        help="Directory containing project scripts; default: directory of this script",
    )
    p.add_argument("--meshtool", default="meshtool")

    p.add_argument("--visualize", action="store_true", help="Write interactive final HTML")
    p.add_argument("--open-browser", action="store_true", help="Open HTML after creation")
    p.add_argument("--visualization-html", default=None)
    p.add_argument("--visualization-max-triangles", type=int, default=None)
    p.add_argument("--visualization-opacity", type=float, default=0.8)
    p.add_argument("--visualization-show-edges", action="store_true")

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="Reuse complete generated stages")
    mode.add_argument("--force", action="store_true", help="Regenerate generated stages")
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    return p


def validate_fraction_list(values: Sequence[float], option: str) -> None:
    for value in values:
        if not np.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"{option} values must all lie in [0, 1]")


def main() -> None:
    args = build_parser().parse_args()

    validate_fraction_list(args.cs_connection_positions, "--cs-connection-positions")
    validate_fraction_list(args.vom_connection_positions, "--vom-connection-positions")
    if args.connection_neighbors < 1:
        raise ValueError("--connection-neighbors must be at least 1")
    if args.fo_connections < 0:
        raise ValueError("--fo-connections must be non-negative")
    if args.closest_bb_spacing <= 0 or args.anatomical_bb_spacing <= 0:
        raise ValueError("BB spacings must be positive")
    if not (0.0 < args.visualization_opacity <= 1.0):
        raise ValueError("--visualization-opacity must lie in (0, 1]")

    la_dir = Path(args.la_path).expanduser().resolve()
    ra_dir = Path(args.ra_path).expanduser().resolve()
    work_dir = (
        Path(args.work_dir).expanduser().resolve()
        if args.work_dir is not None
        else ra_dir
    )
    scripts_dir = (
        Path(args.scripts_dir).expanduser().resolve()
        if args.scripts_dir is not None
        else Path(__file__).resolve().parent
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    if not la_dir.is_dir():
        raise FileNotFoundError(f"LA directory does not exist: {la_dir}")
    if not ra_dir.is_dir():
        raise FileNotFoundError(f"RA directory does not exist: {ra_dir}")

    input_mesh = mesh_base(ra_dir, args.mesh_name)
    cs_mesh = mesh_base(ra_dir, args.cs_mesh)
    require_mesh(input_mesh, description="AtrialMTK Bilayer_Combined")
    require_mesh(cs_mesh, description="cleaned CS/VoM surface")

    for required in (
        la_dir / "Regions.txt",
        la_dir / "Landmarks.txt",
        ra_dir / "Regions.txt",
        ra_dir / "Landmarks.txt",
    ):
        if not required.is_file():
            raise FileNotFoundError(f"Required AtrialMTK landmark/region file missing: {required}")

    connection_script = scripts_dir / "build_biatrial_connections.py"
    bb_script = scripts_dir / "build_bb_bridges.py"
    visualization_script = scripts_dir / "visualization.py"
    for script in (connection_script, bb_script):
        if not script.is_file():
            raise FileNotFoundError(f"Required workflow script missing: {script}")
    if args.visualize and not visualization_script.is_file():
        raise FileNotFoundError(f"Visualization script missing: {visualization_script}")

    meshtool_executable = shutil.which(args.meshtool)
    if meshtool_executable is None:
        if args.dry_run:
            meshtool_executable = args.meshtool
            print(f"Warning: {args.meshtool!r} not found on PATH during dry run", flush=True)
        else:
            raise FileNotFoundError(
                f"Could not find {args.meshtool!r} on PATH. meshtool is required."
            )

    bb_base = work_dir / args.bb_mesh
    merged_base = work_dir / args.merged_mesh
    connections_base = work_dir / args.connections_mesh
    closest_base = work_dir / args.closest_bridge
    anatomical_base = work_dir / args.anatomical_bridge
    merge1_base = work_dir / f"{args.output_mesh}_with_closest_BB"
    final_base = work_dir / args.output_mesh

    audit_dir = work_dir / f"{args.output_mesh}_pipeline"
    audit_dir.mkdir(parents=True, exist_ok=True)
    runner = Runner(dry_run=args.dry_run, log_path=audit_dir / "commands.log")

    print("Biatrial model assembly", flush=True)
    print("=======================", flush=True)
    print(f"AtrialMTK mesh: {input_mesh}", flush=True)
    print(f"cleaned CS/VoM: {cs_mesh}", flush=True)
    print(f"work directory: {work_dir}", flush=True)
    print(f"final mesh: {final_base}", flush=True)

    print("\n[1/7] Extracting BB working surface", flush=True)
    meshtool_extract_bb(
        runner,
        meshtool=meshtool_executable,
        input_mesh=input_mesh,
        output_mesh=bb_base,
        resume=args.resume,
        force=args.force,
    )

    print("\n[2/7] Merging cleaned CS/VoM with AtrialMTK mesh", flush=True)
    meshtool_merge(
        runner,
        meshtool=meshtool_executable,
        mesh1=input_mesh,
        mesh2=cs_mesh,
        output=merged_base,
        resume=args.resume,
        force=args.force,
        description="biatrial + CS/VoM intermediate",
    )

    print("\n[3/7] Building line-element connections", flush=True)
    run_connection_builder(
        runner,
        python=sys.executable,
        script=connection_script,
        la_dir=la_dir,
        ra_dir=ra_dir,
        work_dir=work_dir,
        input_mesh_name=args.merged_mesh,
        output_mesh_name=args.connections_mesh,
        cs_mesh=cs_mesh,
        fo_index=args.fo_index,
        fo_connections=args.fo_connections,
        cs_positions=args.cs_connection_positions,
        vom_positions=args.vom_connection_positions,
        connection_neighbors=args.connection_neighbors,
        scale_factor=args.scale_factor,
        resume=args.resume,
        force=args.force,
    )

    print("\n[4/7] Building both Bachmann-bundle connector surfaces", flush=True)
    run_bb_builder(
        runner,
        python=sys.executable,
        script=bb_script,
        work_dir=work_dir,
        base_mesh_name=str(input_mesh),
        bb_mesh_name=args.bb_mesh,
        ra_dir=ra_dir,
        svc_landmark_index=args.svc_landmark_index,
        scale_factor=args.scale_factor,
        closest_output=args.closest_bridge,
        anatomical_output=args.anatomical_bridge,
        closest_spacing=args.closest_bb_spacing,
        anatomical_spacing=args.anatomical_bb_spacing,
        resume=args.resume,
        force=args.force,
    )

    print("\n[5/7] Merging BB connector surfaces", flush=True)
    meshtool_merge(
        runner,
        meshtool=meshtool_executable,
        mesh1=connections_base,
        mesh2=closest_base,
        output=merge1_base,
        resume=args.resume,
        force=args.force,
        description="connection mesh + closest BB bridge",
    )
    meshtool_merge(
        runner,
        meshtool=meshtool_executable,
        mesh1=merge1_base,
        mesh2=anatomical_base,
        output=final_base,
        resume=args.resume,
        force=args.force,
        description="final mesh",
    )

    print("\n[6/7] Rebuilding final f3d and validating", flush=True)
    final_f3d = Path(f"{final_base}.f3d")
    if not args.dry_run:
        verify_merge_element_order(
            final_base,
            [connections_base, closest_base, anatomical_base],
        )
        print("    meshtool element order verified for final f3d assembly", flush=True)
        n_f3d = concatenate_scalar_files(
            [
                Path(f"{connections_base}.f3d"),
                Path(f"{closest_base}.f3d"),
                Path(f"{anatomical_base}.f3d"),
            ],
            final_f3d,
        )
        print(f"    final f3d rows: {n_f3d}", flush=True)
        validation = validate_final_mesh(final_base)
        print(
            "    validated: "
            f"points={validation['points']}, elements={validation['elements']}, "
            f"Tr={validation['triangles']}, Ln={validation['lines']}",
            flush=True,
        )
    else:
        validation = {"status": "dry-run; final files not generated"}

    print("\n[7/7] Writing provenance and optional visualization", flush=True)
    html_path = None
    if args.visualize:
        html_path = (
            Path(args.visualization_html).expanduser()
            if args.visualization_html
            else Path(f"{final_base}_view.html")
        )
        command: list[object] = [
            sys.executable,
            visualization_script,
            "--mesh", final_base,
            "--output-html", html_path,
            "--surface-opacity", args.visualization_opacity,
        ]
        if args.visualization_max_triangles is not None:
            command.extend(["--max-triangles", args.visualization_max_triangles])
        if args.visualization_show_edges:
            command.append("--show-edges")
        if args.open_browser:
            command.append("--open-browser")
        runner.run(command)

    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "build_biatrial_model.py",
        "python": sys.version,
        "platform": platform.platform(),
        "software_versions": {
            "numpy": package_version("numpy"),
            "scipy": package_version("scipy"),
            "scikit-learn": package_version("scikit-learn"),
            "plotly": package_version("plotly"),
        },
        "executables": {
            "python": sys.executable,
            "meshtool": str(meshtool_executable),
        },
        "inputs": {
            "atrialmtk_mesh": str(input_mesh),
            "cleaned_cs_vom": str(cs_mesh),
            "atrialmtk_elem_sha256": sha256_file(Path(f"{input_mesh}.elem")),
            "cs_vom_elem_sha256": sha256_file(Path(f"{cs_mesh}.elem")),
        },
        "parameters": {
            "cs_connection_positions": list(map(float, args.cs_connection_positions)),
            "vom_connection_positions": list(map(float, args.vom_connection_positions)),
            "connection_neighbors": int(args.connection_neighbors),
            "fo_index": int(args.fo_index),
            "fo_connections": int(args.fo_connections),
            "svc_landmark_index": int(args.svc_landmark_index),
            "scale_factor": float(args.scale_factor),
            "closest_bb_spacing": float(args.closest_bb_spacing),
            "anatomical_bb_spacing": float(args.anatomical_bb_spacing),
            "generated_tags": {
                "cs": CS_LA_CONNECTION_TAG,
                "vom": VOM_LA_CONNECTION_TAG,
                "bb_closest": BB_CLOSEST_BRIDGE_TAG,
                "bb_anatomical": BB_ANATOMICAL_BRIDGE_TAG,
            },
        },
        "outputs": {
            "bb_mesh": str(bb_base),
            "biatrial_with_cs_vom": str(merged_base),
            "connections_mesh": str(connections_base),
            "closest_bb_bridge": str(closest_base),
            "anatomical_bb_bridge": str(anatomical_base),
            "final_mesh": str(final_base),
            "visualization_html": str(html_path) if html_path is not None else None,
        },
        "validation": validation,
        "commands": [shlex.join(command) for command in runner.commands],
        "resume": bool(args.resume),
        "force": bool(args.force),
        "dry_run": bool(args.dry_run),
    }

    provenance_path = audit_dir / "pipeline_parameters.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    report = [
        "Biatrial model pipeline report",
        "==============================",
        "",
        f"input AtrialMTK mesh: {input_mesh}",
        f"input cleaned CS/VoM: {cs_mesh}",
        f"final mesh: {final_base}",
        f"CS connection positions: {args.cs_connection_positions}",
        f"VoM connection positions: {args.vom_connection_positions}",
        f"SVC landmark index: {args.svc_landmark_index}",
        f"closest BB spacing: {args.closest_bb_spacing}",
        f"anatomical BB spacing: {args.anatomical_bb_spacing}",
        "",
        "Validation",
        "----------",
        json.dumps(validation, indent=2),
        "",
        f"Provenance JSON: {provenance_path}",
        f"Command log: {audit_dir / 'commands.log'}",
    ]
    (audit_dir / "pipeline_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\nFinished", flush=True)
    print("--------", flush=True)
    print(f"final mesh: {final_base}", flush=True)
    if html_path is not None:
        print(f"interactive HTML: {html_path}", flush=True)
    print(f"pipeline report: {audit_dir / 'pipeline_report.txt'}", flush=True)
    print(f"provenance: {provenance_path}", flush=True)


if __name__ == "__main__":
    main()
