#!/usr/bin/env python3
"""End-to-end CLI for building a biatrial CS/VoM model from AtrialMTK outputs.

This is the top-level orchestration command for the public workflow.

Starting inputs
---------------
The command expects completed AtrialMTK result directories:

* ``--ra-path`` containing the RA outputs, including ``Bilayer_Combined`` and
  ``Regions.txt``;
* ``--la-path`` containing the LA/Bia outputs and the UAC coordinate file.

If ``tube_cs_cleaned`` does not already exist, the CLI performs the complete
validated CS/VoM preparation sequence:

    tube_making.py
        -> mesh_cleaning.py (Stage-1 conservative weld)
        -> mesh_cleaning_stage3.py (junction reconstruction)
        -> mesh_final_repair.py (local final repair)
        -> tube_cs_cleaned

It then delegates the downstream assembly to ``build_biatrial_model.py``:

    Bilayer_Combined + tube_cs_cleaned
        -> CS/VoM merge
        -> line-element connections
        -> two BB connector surfaces
        -> final Reentry_surface_iac
        -> validation / .f3d reconstruction
        -> optional browser visualization

Geometry algorithms are intentionally not duplicated here.  This file is only
an orchestration layer around the validated project scripts and ``meshtool``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Iterable


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def mesh_complete(base: Path, *, require_lon: bool = True) -> bool:
    required = [Path(f"{base}.pts"), Path(f"{base}.elem")]
    if require_lon:
        required.append(Path(f"{base}.lon"))
    return all(path.is_file() for path in required)


def junction_safe_stage1_complete(base: Path) -> bool:
    """True only for a complete Stage-1 mesh made with junction protection."""
    if not mesh_complete(base):
        return False
    audit_dir = base.parent / f"{base.name}_audit"
    return (
        (audit_dir / "original_to_cleaned_vertex.csv").is_file()
        and (audit_dir / "protected_junction_vertices.csv").is_file()
    )


def quote_command(command: Iterable[object]) -> str:
    return " ".join(shlex.quote(str(x)) for x in command)


def resolve_script(
    scripts_dir: Path,
    explicit: str | None,
    default_name: str,
    *,
    description: str,
) -> Path:
    """Resolve one workflow script by its canonical filename or explicit override."""
    value = explicit if explicit is not None else default_name
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = scripts_dir / path
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {description}: {path}\n"
            f"Expected canonical filename: {default_name}"
        )
    return path.resolve()


def verify_script_cli(path: Path, required_options: Iterable[str], description: str) -> None:
    """Check the real argparse interface instead of relying on source-code markers."""
    result = subprocess.run(
        [sys.executable, str(path), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        details = (result.stderr or result.stdout).strip()
        raise RuntimeError(
            f"Could not load {description}: {path}\n"
            f"Running '--help' failed with exit code {result.returncode}.\n{details}"
        )
    help_text = result.stdout + result.stderr
    missing = [option for option in required_options if option not in help_text]
    if missing:
        raise RuntimeError(
            f"{description} has an incompatible command-line interface: {path}\n"
            f"Missing options: {', '.join(missing)}"
        )


class Runner:
    def __init__(self, *, dry_run: bool, log_path: Path):
        self.dry_run = bool(dry_run)
        self.log_path = log_path
        self.commands: list[str] = []

    def run(self, command: list[object], *, cwd: Path | None = None) -> None:
        cmd = [str(x) for x in command]
        printable = quote_command(cmd)
        self.commands.append(printable)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(printable + "\n")

        print(f"\n$ {printable}", flush=True)
        if self.dry_run:
            return

        subprocess.run(
            cmd,
            check=True,
            cwd=str(cwd) if cwd is not None else None,
        )


def remove_mesh(base: Path) -> None:
    for suffix in (".pts", ".elem", ".lon", ".f3d"):
        path = Path(f"{base}{suffix}")
        if path.exists():
            path.unlink()


def remove_directory(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)


def ensure_meshtool(executable: str, *, dry_run: bool) -> None:
    if dry_run:
        return
    if Path(executable).is_file():
        return
    if shutil.which(executable) is None:
        raise FileNotFoundError(
            f"Could not find meshtool executable {executable!r} on PATH."
        )


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")


def require_mesh(base: Path, description: str) -> None:
    missing = [
        suffix for suffix in (".pts", ".elem", ".lon")
        if not Path(f"{base}{suffix}").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Incomplete {description}: {base} "
            f"(missing {', '.join(missing)})"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run the complete AtrialMTK -> CS/VoM tube -> cleaning -> biatrial "
            "assembly workflow."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # AtrialMTK inputs / output location.
    p.add_argument("--ra-path", required=True, help="AtrialMTK RA result directory")
    p.add_argument("--la-path", required=True, help="AtrialMTK LA/Bia result directory")
    p.add_argument(
        "--work-dir",
        default=None,
        help="Generated-output directory; defaults to --ra-path",
    )
    p.add_argument("--bilayer-mesh", default="Bilayer_Combined")
    p.add_argument("--mesh-name-la", default="Labelled")
    p.add_argument(
        "--mesh-name-uac",
        default="Labelled_Coords_2D_Rescaling_v3_C",
        help="LA UAC .hpts basename used to construct the CS/mitral prefix",
    )
    p.add_argument("--seed-region", default="Regions.txt")

    # Tube geometry: expose the main scientific controls, leave the validated
    # lower-level defaults inside the tube builder.
    tube = p.add_argument_group("CS/VoM tube construction")
    tube.add_argument("--resolution", type=float, default=350.0)
    tube.add_argument("--la-clearance", type=float, default=500.0)
    tube.add_argument("--vom-end-radius", type=float, default=700.0)
    tube.add_argument("--vm-keep-fraction", type=float, default=2.0 / 3.0)
    tube.add_argument("--vm-target-edge", type=float, default=350.0)
    tube.add_argument("--vm-implicit-voxel-size", type=float, default=250.0)
    tube.add_argument("--vm-implicit-sample-spacing", type=float, default=125.0)
    tube.add_argument("--vm-proximal-fit-length", type=float, default=5000.0)
    tube.add_argument("--vm-proximal-radius-hold-length", type=float, default=2500.0)
    tube.add_argument("--vm-seam-length", type=float, default=350.0)

    # Conservative cleaning parameters.
    clean = p.add_argument_group("tube cleaning")
    clean.add_argument("--coincident-tol", type=float, default=2.0)
    clean.add_argument("--peel-layers", type=int, default=2)
    clean.add_argument("--target-axial-spacing", type=float, default=325.0)
    clean.add_argument("--repair-target-edge", type=float, default=350.0)
    clean.add_argument("--repair-max-patch-rings", type=int, default=3)

    # Final model assembly.
    assembly = p.add_argument_group("biatrial assembly")
    assembly.add_argument(
        "--svc-landmark-index",
        type=int,
        required=True,
        help="Row in RA Landmarks.txt representing the SVC reference",
    )
    assembly.add_argument(
        "--cs-connection-positions",
        nargs="*",
        type=float,
        default=[0.30, 0.40, 0.50],
    )
    assembly.add_argument(
        "--vom-connection-positions",
        nargs="*",
        type=float,
        default=[0.70, 0.80, 0.90],
    )
    assembly.add_argument("--connection-neighbors", type=int, default=5)
    assembly.add_argument("--fo-index", type=int, default=3)
    assembly.add_argument("--fo-connections", type=int, default=100)
    assembly.add_argument("--scale-factor", type=float, default=1.0)
    assembly.add_argument("--closest-bb-spacing", type=float, default=350.0)
    assembly.add_argument("--anatomical-bb-spacing", type=float, default=350.0)
    assembly.add_argument("--output-mesh", default="Reentry_surface_iac")
    assembly.add_argument("--meshtool", default="meshtool")

    # Visualization.
    viz = p.add_argument_group("visualization")
    viz.add_argument(
        "--visualize",
        action="store_true",
        help="Create a self-contained browser visualization of the final mesh",
    )
    viz.add_argument("--open-browser", action="store_true")
    viz.add_argument("--visualization-max-triangles", type=int, default=None)
    viz.add_argument("--visualization-opacity", type=float, default=0.8)
    viz.add_argument("--visualization-show-edges", action="store_true")
    viz.add_argument(
        "--visualize-cleaned-tube",
        action="store_true",
        help="Also create an HTML visualization of tube_cs_cleaned before assembly",
    )

    # Script discovery / version-proofing.
    files = p.add_argument_group("workflow script locations")
    files.add_argument(
        "--scripts-dir",
        default=None,
        help="Directory containing the workflow scripts; defaults to this file's directory",
    )
    files.add_argument("--tube-script", default=None, help="Override tube_making.py")
    files.add_argument("--stage1-cleaner-script", default=None, help="Override mesh_cleaning.py")
    files.add_argument("--stage3-cleaner-script", default=None, help="Override mesh_cleaning_stage3.py")
    files.add_argument("--final-repair-script", default=None, help="Override mesh_final_repair.py")
    files.add_argument("--assembly-script", default=None, help="Override build_biatrial_model.py")
    files.add_argument("--visualization-script", default=None, help="Override visualization.py")

    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume",
        action="store_true",
        help="Reuse complete intermediate/final stages and continue missing work",
    )
    mode.add_argument(
        "--force",
        action="store_true",
        help="Regenerate tube, cleaning stages, and final assembly outputs",
    )
    p.add_argument("--dry-run", action="store_true", help="Print commands only")

    return p


def validate_args(args: argparse.Namespace) -> None:
    for name in ("cs_connection_positions", "vom_connection_positions"):
        values = getattr(args, name)
        for value in values:
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"--{name.replace('_', '-')} values must lie in [0, 1]")

    positives = {
        "--resolution": args.resolution,
        "--la-clearance": args.la_clearance,
        "--vom-end-radius": args.vom_end_radius,
        "--vm-target-edge": args.vm_target_edge,
        "--vm-implicit-voxel-size": args.vm_implicit_voxel_size,
        "--vm-implicit-sample-spacing": args.vm_implicit_sample_spacing,
        "--vm-proximal-fit-length": args.vm_proximal_fit_length,
        "--vm-proximal-radius-hold-length": args.vm_proximal_radius_hold_length,
        "--vm-seam-length": args.vm_seam_length,
        "--coincident-tol": args.coincident_tol,
        "--target-axial-spacing": args.target_axial_spacing,
        "--repair-target-edge": args.repair_target_edge,
        "--closest-bb-spacing": args.closest_bb_spacing,
        "--anatomical-bb-spacing": args.anatomical_bb_spacing,
    }
    for option, value in positives.items():
        if float(value) <= 0.0:
            raise ValueError(f"{option} must be positive")

    if not 0.0 < float(args.vm_keep_fraction) <= 1.0:
        raise ValueError("--vm-keep-fraction must lie in (0, 1]")
    if int(args.peel_layers) < 1:
        raise ValueError("--peel-layers must be at least 1")
    if int(args.repair_max_patch_rings) < 0:
        raise ValueError("--repair-max-patch-rings cannot be negative")
    if int(args.connection_neighbors) < 1:
        raise ValueError("--connection-neighbors must be at least 1")


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

    script_self = Path(__file__).resolve()
    scripts_dir = (
        Path(args.scripts_dir).expanduser().resolve()
        if args.scripts_dir
        else script_self.parent
    )
    ra_dir = Path(args.ra_path).expanduser().resolve()
    la_dir = Path(args.la_path).expanduser().resolve()
    work_dir = (
        Path(args.work_dir).expanduser().resolve()
        if args.work_dir
        else ra_dir
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the canonical workflow scripts.  Compatibility is checked from
    # each script's actual ``--help`` output, so an old file with the same name
    # fails with a clear interface error rather than a source-marker mismatch.
    tube_script = resolve_script(
        scripts_dir, args.tube_script, "tube_making.py",
        description="full CS/VoM tube builder",
    )
    stage1_script = resolve_script(
        scripts_dir, args.stage1_cleaner_script, "mesh_cleaning.py",
        description="Stage-1 conservative cleaner",
    )
    stage3_script = resolve_script(
        scripts_dir, args.stage3_cleaner_script, "mesh_cleaning_stage3.py",
        description="Stage-3 junction reconstruction",
    )
    final_repair_script = resolve_script(
        scripts_dir, args.final_repair_script, "mesh_final_repair.py",
        description="final local mesh repair",
    )
    assembly_script = resolve_script(
        scripts_dir, args.assembly_script, "build_biatrial_model.py",
        description="biatrial assembly master",
    )

    visualization_script: Path | None = None
    if args.visualize_cleaned_tube:
        visualization_script = resolve_script(
            scripts_dir, args.visualization_script, "visualization.py",
            description="browser visualization helper",
        )

    # Verify the interfaces actually used below.
    verify_script_cli(
        tube_script,
        ("--target-path-la", "--target-path-final", "--output-dir", "--rebuild-prefix"),
        "tube builder",
    )
    verify_script_cli(
        stage1_script,
        ("--mesh-base", "--output-base", "--coincident-tol", "--vm-surface-base"),
        "Stage-1 cleaner",
    )
    verify_script_cli(
        stage3_script,
        ("--mesh-base", "--output-base", "--stage1-mapping",
         "--vm-surface-base", "--peel-layers", "--target-axial-spacing"),
        "Stage-3 cleaner",
    )
    verify_script_cli(
        final_repair_script,
        ("--mesh-base", "--output-base", "--target-edge", "--max-patch-rings"),
        "final repair",
    )
    verify_script_cli(
        assembly_script,
        ("--la-path", "--ra-path", "--cs-mesh", "--svc-landmark-index",
         "--scripts-dir"),
        "biatrial assembly master",
    )
    if visualization_script is not None:
        verify_script_cli(
            visualization_script,
            ("--mesh", "--output-html"),
            "visualization helper",
        )

    # Inputs that should come from completed AtrialMTK.
    bilayer_base = ra_dir / args.bilayer_mesh
    if not args.dry_run:
        require_mesh(bilayer_base, "AtrialMTK Bilayer_Combined mesh")
        require_file(ra_dir / args.seed_region, "RA Regions file")
        require_file(ra_dir / "Landmarks.txt", "RA Landmarks file")

        la_mesh_base = la_dir / args.mesh_name_la
        require_file(Path(f"{la_mesh_base}.pts"), "LA labelled points")
        require_file(Path(f"{la_mesh_base}.elem"), "LA labelled elements")

        uac_file = la_dir / f"{args.mesh_name_uac}.hpts"
        # The UAC file is only strictly needed when the tube must be built.
        if args.force or not mesh_complete(work_dir / "tube_cs_cleaned"):
            require_file(uac_file, "LA UAC coordinate file")

    ensure_meshtool(args.meshtool, dry_run=args.dry_run)

    audit_dir = work_dir / "Bilayer_CS_VoM_pipeline"
    if args.force and not args.dry_run:
        remove_directory(audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    command_log = audit_dir / "commands.log"
    if args.force and command_log.exists() and not args.dry_run:
        command_log.unlink()

    runner = Runner(dry_run=args.dry_run, log_path=command_log)

    # Standard generated mesh names.
    raw_tube = work_dir / "tube_cs"
    stage1 = work_dir / "tube_cs_cleaned_stage1"
    stage3 = work_dir / "tube_cs_cleaned_stage3"
    cleaned = work_dir / "tube_cs_cleaned"
    vm_surface = work_dir / "vom_generalized_capsule_proximal_fit"

    if args.force and not args.dry_run:
        for base in (raw_tube, stage1, stage3, cleaned):
            remove_mesh(base)
            remove_directory(base.parent / f"{base.name}_audit")

    print("\nBilayer_CS_VoM end-to-end workflow", flush=True)
    print("==================================", flush=True)
    print(f"RA path:          {ra_dir}", flush=True)
    print(f"LA path:          {la_dir}", flush=True)
    print(f"work directory:   {work_dir}", flush=True)
    print(f"tube builder:     {tube_script.name}", flush=True)
    print(f"assembly master:  {assembly_script.name}", flush=True)

    # ------------------------------------------------------------------
    # Stage 1: full CS/VoM construction
    # ------------------------------------------------------------------
    stage1_rebuilt = False
    stage3_rebuilt = False

    final_tube_already_available = mesh_complete(cleaned)

    if final_tube_already_available and not args.force:
        print("\n[1/5] Cleaned CS/VoM already exists; reusing tube_cs_cleaned.", flush=True)
    else:
        if args.resume and mesh_complete(raw_tube):
            print("\n[1/5] Reusing existing raw tube_cs.", flush=True)
        else:
            print("\n[1/5] Building complete CS/VoM tube.", flush=True)
            tube_command: list[object] = [
                sys.executable,
                tube_script,
                "--target-path-la", la_dir,
                "--target-path-final", ra_dir,
                "--mesh-name-la", args.mesh_name_la,
                "--mesh-name-uac", args.mesh_name_uac,
                "--seed-region", args.seed_region,
                "--output-dir", work_dir,
                "--resolution", args.resolution,
                "--la-clearance", args.la_clearance,
                "--vom-end-radius", args.vom_end_radius,
                "--vm-keep-fraction", args.vm_keep_fraction,
                "--vm-target-edge", args.vm_target_edge,
                "--vm-implicit-voxel-size", args.vm_implicit_voxel_size,
                "--vm-implicit-sample-spacing", args.vm_implicit_sample_spacing,
                "--vm-proximal-fit-length", args.vm_proximal_fit_length,
                "--vm-proximal-radius-hold-length", args.vm_proximal_radius_hold_length,
                "--vm-seam-length", args.vm_seam_length,
            ]
            if args.force:
                tube_command.append("--rebuild-prefix")
            runner.run(tube_command, cwd=work_dir)
            if not args.dry_run:
                require_mesh(raw_tube, "generated raw CS/VoM tube")
                require_mesh(vm_surface, "generated generalized-capsule VoM surface")

        # --------------------------------------------------------------
        # Stage 2: conservative weld
        # --------------------------------------------------------------
        if args.resume and junction_safe_stage1_complete(stage1):
            print(
                "\n[2/5] Reusing junction-safe Stage-1 cleaned tube.",
                flush=True,
            )
        else:
            if args.resume and mesh_complete(stage1):
                print(
                    "\n[2/5] Existing Stage-1 mesh was made without the "
                    "junction-protection audit; rebuilding it.",
                    flush=True,
                )
            else:
                print(
                    "\n[2/5] Conservative near-coincident vertex cleaning.",
                    flush=True,
                )
            runner.run(
                [
                    sys.executable,
                    stage1_script,
                    "--mesh-base", raw_tube,
                    "--output-base", stage1,
                    "--coincident-tol", args.coincident_tol,
                    "--vm-surface-base", vm_surface,
                ],
                cwd=work_dir,
            )
            stage1_rebuilt = True
            if not args.dry_run:
                require_mesh(stage1, "Stage-1 cleaned tube")
                require_file(
                    stage1.parent
                    / f"{stage1.name}_audit"
                    / "original_to_cleaned_vertex.csv",
                    "Stage-1 vertex provenance map",
                )

        # --------------------------------------------------------------
        # Stage 3: validated junction reconstruction
        # --------------------------------------------------------------
        if args.resume and (not stage1_rebuilt) and mesh_complete(stage3):
            print("\n[3/5] Reusing Stage-3 junction-reconstructed tube.", flush=True)
        else:
            print("\n[3/5] Reconstructing mitral/VoM junction.", flush=True)
            mapping = (
                stage1.parent
                / f"{stage1.name}_audit"
                / "original_to_cleaned_vertex.csv"
            )
            runner.run(
                [
                    sys.executable,
                    stage3_script,
                    "--mesh-base", stage1,
                    "--output-base", stage3,
                    "--stage1-mapping", mapping,
                    "--vm-surface-base", vm_surface,
                    "--peel-layers", args.peel_layers,
                    "--target-axial-spacing", args.target_axial_spacing,
                ],
                cwd=work_dir,
            )
            stage3_rebuilt = True
            if not args.dry_run:
                require_mesh(stage3, "Stage-3 junction-reconstructed tube")

        # --------------------------------------------------------------
        # Stage 4: final local repair -> production cleaned alias
        # --------------------------------------------------------------
        if args.resume and (not stage3_rebuilt) and mesh_complete(cleaned):
            print("\n[4/5] Reusing final tube_cs_cleaned.", flush=True)
        else:
            print("\n[4/5] Final local intersection/degeneracy repair.", flush=True)
            runner.run(
                [
                    sys.executable,
                    final_repair_script,
                    "--mesh-base", stage3,
                    "--output-base", cleaned,
                    "--target-edge", args.repair_target_edge,
                    "--max-patch-rings", args.repair_max_patch_rings,
                ],
                cwd=work_dir,
            )
            if not args.dry_run:
                require_mesh(cleaned, "final cleaned CS/VoM tube")

    # Optional browser view of the cleaned tube.
    if args.visualize_cleaned_tube:
        assert visualization_script is not None
        tube_html = work_dir / "tube_cs_cleaned_view.html"
        print("\n[optional] Writing cleaned-tube browser visualization.", flush=True)
        runner.run(
            [
                sys.executable,
                visualization_script,
                "--mesh", cleaned,
                "--output-html", tube_html,
            ],
            cwd=work_dir,
        )

    # ------------------------------------------------------------------
    # Stage 5: downstream biatrial assembly
    # ------------------------------------------------------------------
    print("\n[5/5] Building final biatrial model.", flush=True)
    assembly_command: list[object] = [
        sys.executable,
        assembly_script,
        "--la-path", la_dir,
        "--ra-path", ra_dir,
        "--work-dir", work_dir,
        "--mesh-name", args.bilayer_mesh,
        "--cs-mesh", cleaned,
        "--svc-landmark-index", args.svc_landmark_index,
        "--connection-neighbors", args.connection_neighbors,
        "--fo-index", args.fo_index,
        "--fo-connections", args.fo_connections,
        "--scale-factor", args.scale_factor,
        "--closest-bb-spacing", args.closest_bb_spacing,
        "--anatomical-bb-spacing", args.anatomical_bb_spacing,
        "--output-mesh", args.output_mesh,
        "--scripts-dir", scripts_dir,
        "--meshtool", args.meshtool,
    ]

    assembly_command.extend(
        ["--cs-connection-positions", *args.cs_connection_positions]
    )
    assembly_command.extend(
        ["--vom-connection-positions", *args.vom_connection_positions]
    )

    if args.visualize:
        assembly_command.append("--visualize")
        assembly_command.extend(
            ["--visualization-opacity", args.visualization_opacity]
        )
        if args.visualization_max_triangles is not None:
            assembly_command.extend(
                ["--visualization-max-triangles", args.visualization_max_triangles]
            )
        if args.visualization_show_edges:
            assembly_command.append("--visualization-show-edges")
        if args.open_browser:
            assembly_command.append("--open-browser")

    if args.resume:
        assembly_command.append("--resume")
    elif args.force:
        assembly_command.append("--force")
    if args.dry_run:
        assembly_command.append("--dry-run")

    runner.run(assembly_command, cwd=work_dir)

    final_base = work_dir / args.output_mesh
    if not args.dry_run:
        require_mesh(final_base, "final biatrial model")
        require_file(Path(f"{final_base}.f3d"), "final element-wise .f3d")

    # ------------------------------------------------------------------
    # Top-level provenance
    # ------------------------------------------------------------------
    parameters = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "ra_path": str(ra_dir),
        "la_path": str(la_dir),
        "work_dir": str(work_dir),
        "bilayer_mesh": args.bilayer_mesh,
        "mesh_name_la": args.mesh_name_la,
        "mesh_name_uac": args.mesh_name_uac,
        "seed_region": args.seed_region,
        "tube": {
            "resolution_um": args.resolution,
            "la_clearance_um": args.la_clearance,
            "vom_end_radius_um": args.vom_end_radius,
            "vm_keep_fraction": args.vm_keep_fraction,
            "vm_target_edge_um": args.vm_target_edge,
            "vm_implicit_voxel_size_um": args.vm_implicit_voxel_size,
            "vm_implicit_sample_spacing_um": args.vm_implicit_sample_spacing,
            "vm_proximal_fit_length_um": args.vm_proximal_fit_length,
            "vm_proximal_radius_hold_length_um": args.vm_proximal_radius_hold_length,
            "vm_seam_length_um": args.vm_seam_length,
        },
        "cleaning": {
            "coincident_tol_um": args.coincident_tol,
            "peel_layers": args.peel_layers,
            "target_axial_spacing_um": args.target_axial_spacing,
            "repair_target_edge_um": args.repair_target_edge,
            "repair_max_patch_rings": args.repair_max_patch_rings,
        },
        "connections": {
            "cs_positions": list(args.cs_connection_positions),
            "vom_positions": list(args.vom_connection_positions),
            "neighbors": args.connection_neighbors,
            "fo_index": args.fo_index,
            "fo_connections": args.fo_connections,
        },
        "bb": {
            "closest_spacing_um": args.closest_bb_spacing,
            "anatomical_spacing_um": args.anatomical_bb_spacing,
        },
        "scripts": {
            "tube": str(tube_script),
            "stage1_cleaner": str(stage1_script),
            "stage3_cleaner": str(stage3_script),
            "final_repair": str(final_repair_script),
            "assembly": str(assembly_script),
            "visualization": str(visualization_script) if visualization_script else None,
        },
        "commands": runner.commands,
        "mode": {
            "resume": bool(args.resume),
            "force": bool(args.force),
            "dry_run": bool(args.dry_run),
        },
        "final_mesh": str(final_base),
    }

    parameter_path = audit_dir / "pipeline_parameters.json"
    parameter_path.write_text(json.dumps(parameters, indent=2) + "\n", encoding="utf-8")

    report = f"""Bilayer_CS_VoM pipeline
=======================

RA AtrialMTK directory: {ra_dir}
LA AtrialMTK directory: {la_dir}
Working directory:      {work_dir}

Production cleaned tube:
  {cleaned}

Final biatrial model:
  {final_base}

Commands:
  {command_log}

Parameters:
  {parameter_path}

The top-level CLI does not reimplement geometry.  It orchestrates the validated
tube construction, conservative cleaning, and downstream biatrial assembly
scripts in a reproducible order.
"""
    (audit_dir / "pipeline_report.txt").write_text(report, encoding="utf-8")

    if args.dry_run:
        print("\nDry run complete. No workflow commands were executed.", flush=True)
    else:
        print("\nPipeline complete.", flush=True)
        print(f"  cleaned CS/VoM: {cleaned}", flush=True)
        print(f"  final mesh:      {final_base}", flush=True)
        print(f"  audit:           {audit_dir}", flush=True)


if __name__ == "__main__":
    main()
