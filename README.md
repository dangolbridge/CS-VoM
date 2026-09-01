# CS–VoM Biatrial Mesh Workflow

Research software for constructing a coronary sinus (CS)–vein of Marshall (VoM) surface, integrating it into an AtrialMTK biatrial model, adding inter-atrial and CS/VoM connections, constructing Bachmann's bundle (BB) connector surfaces, and preparing a CARP/openCARP-compatible final mesh.

> **Version 0.1.0**
>
> This is an early research release. The workflow is intentionally conservative: when a geometry cannot be repaired without making a large or poorly controlled modification, the pipeline stops instead of silently producing a questionable mesh. Every generated model should be inspected before simulation.

## Overview

The main entry point is:

```bash
python3 Bilayer_CS_VoM.py ...
```

The complete workflow is:

```text
Completed AtrialMTK model
        |
        |-- Bilayer_Combined
        |-- RA / LA labelled surfaces
        |-- Regions / Landmarks
        |-- LA UAC coordinates
        |
        v
tube_making.py
        |
        v
tube_cs
        |
        v
mesh_cleaning.py
Stage 1: conservative near-coincident vertex cleaning
        |
        v
tube_cs_cleaned_stage1
        |
        v
mesh_cleaning_stage3.py
Mitral–VoM junction reconstruction
        |
        v
tube_cs_cleaned_stage3
        |
        v
mesh_final_repair.py
Conservative local degeneracy / self-intersection repair
        |
        v
tube_cs_cleaned
        |
        v
build_biatrial_model.py
        |
        |-- merge CS–VoM
        |-- add line-element connections
        |-- construct BB connector surfaces
        |-- merge BB connectors
        |-- rebuild / validate .f3d
        |
        v
Reentry_surface_iac
        |
        v
optional interactive HTML visualization
```

## Repository files

```text
Bilayer_CS_VoM.py
tube_making.py
mesh_cleaning.py
mesh_cleaning_stage3.py
mesh_final_repair.py
build_biatrial_model.py
build_biatrial_connections.py
build_bb_bridges.py
geometry_hybrid.py
mesh_io_hybrid.py
mesh_tags.py
visualization.py
```

### Main scripts

| File | Purpose |
| --- | --- |
| `Bilayer_CS_VoM.py` | End-to-end command-line workflow. |
| `tube_making.py` | Builds the CS → mitral prefix and generalized-capsule VoM surface. |
| `mesh_cleaning.py` | Conservative Stage-1 welding/cleanup while protecting the CS–VoM junction. |
| `mesh_cleaning_stage3.py` | Reconstructs the mitral–VoM junction using equal-node multi-ring interpolation. |
| `mesh_final_repair.py` | Detects and conservatively repairs local degeneracies and exact self-intersections. |
| `build_biatrial_model.py` | Downstream assembly of the final biatrial model. |
| `build_biatrial_connections.py` | Adds bilayer, inter-atrial, CS, and VoM line-element connections. |
| `build_bb_bridges.py` | Builds the two BB connector surfaces. |
| `visualization.py` | Browser-first visualization for mixed CARP `Tr` and `Ln` meshes. |

### Helper modules

| File | Purpose |
| --- | --- |
| `geometry_hybrid.py` | Geometry, ring, path, geodesic, and ruled-surface utilities. |
| `mesh_io_hybrid.py` | CARP/openCARP and UAC file I/O. |
| `mesh_tags.py` | Centralized anatomical and generated element tags. |

## Requirements

### Python

Python **3.10 or newer** is recommended.

Core Python packages used by the workflow include:

```bash
python -m pip install numpy scipy vtk plotly
```

Optional visualization support:

```bash
python -m pip install pyvista
```

### External software

The workflow also requires:

- **AtrialMTK / UAC**: the Python environment must provide the `uac` package used by the AtrialMTK workflow.
- **meshtool**: must be available on `PATH`, or supplied with `--meshtool`.
- **CARP/openCARP-style mesh files**: `.pts`, `.elem`, `.lon`, and generated `.f3d` files.

The code has primarily been developed for Linux-based research environments.

## Expected input data

The top-level workflow expects completed AtrialMTK RA and LA/Bia result directories.

A typical layout is:

```text
Data/
├── RA_XX/
│   ├── Bilayer_Combined.pts
│   ├── Bilayer_Combined.elem
│   ├── Bilayer_Combined.lon
│   ├── Labelled.pts
│   ├── Labelled.elem
│   ├── Regions.txt
│   └── Landmarks.txt
│
└── Bia_XX/
    ├── Labelled.pts
    ├── Labelled.elem
    ├── Regions.txt
    ├── Landmarks.txt
    └── Labelled_Coords_2D_Rescaling_v3_C.hpts
```

`Bilayer_Combined` is expected to already contain the RA and LA surfaces and the disconnected RA-side and LA-side BB surfaces (tag `10`).

The exact upstream AtrialMTK naming can be changed through CLI options when necessary.

## Quick start

From the RA result directory:

```bash
python3 /path/to/CS_VoM/Bilayer_CS_VoM.py \
    --ra-path . \
    --la-path ../Bia_XX \
    --svc-landmark-index 6 \
    --visualize
```

`--svc-landmark-index` must correspond to the SVC reference row in the RA `Landmarks.txt` file. It is deliberately not hard-coded because landmark ordering may differ between datasets.

### Recommended first run

Before executing the full workflow, inspect the commands with:

```bash
python3 /path/to/CS_VoM/Bilayer_CS_VoM.py \
    --ra-path . \
    --la-path ../Bia_XX \
    --svc-landmark-index 6 \
    --visualize \
    --dry-run
```

Then rerun without `--dry-run`.

### Resume an interrupted run

```bash
python3 /path/to/CS_VoM/Bilayer_CS_VoM.py \
    --ra-path . \
    --la-path ../Bia_XX \
    --svc-landmark-index 6 \
    --visualize \
    --resume
```

The pipeline reuses compatible completed stages when possible.

### Force regeneration

```bash
python3 /path/to/CS_VoM/Bilayer_CS_VoM.py \
    --ra-path . \
    --la-path ../Bia_XX \
    --svc-landmark-index 6 \
    --visualize \
    --force
```

Use `--force` when generated geometry should be rebuilt rather than reused.

## Default geometric parameters

The current top-level defaults are:

| Parameter | Default |
| --- | ---: |
| Main resolution | `350 µm` |
| LA clearance | `500 µm` |
| Distal VoM radius | `700 µm` |
| Retained VoM arc-length fraction | `2/3` |
| VoM target edge scale | `350 µm` |
| Implicit voxel size | `250 µm` |
| Implicit sample spacing | `125 µm` |
| Proximal guide-fit length | `5000 µm` |
| Proximal radius-hold length | `2500 µm` |
| Short mitral–VoM seam length | `350 µm` |
| Stage-1 coincident tolerance | `2 µm` |
| Mitral peel depth | `2 layers` |
| Transition axial spacing | `325 µm` |
| Final-repair target edge | `350 µm` |
| Maximum final-repair patch expansion | `3 rings` |
| Closest BB bridge spacing | `350 µm` |
| Anatomical BB bridge spacing | `350 µm` |

All of these can be overridden through `Bilayer_CS_VoM.py --help`.

## CS–VoM construction

`tube_making.py` constructs the CS–mitral–VoM surface in two main parts.

### CS → mitral prefix

The script:

1. extracts the CS boundary from the labelled RA surface;
2. orients the CS ring away from the RA anatomical reference;
3. maps the mitral guide using UAC coordinates;
4. automatically constructs the CS-to-mitral elbow;
5. builds the mitral section sequentially.

### VoM

The VoM guide:

1. starts from the accepted terminal mitral ring;
2. follows the LA surface toward `LA Regions[3]`;
3. is offset from the LA surface using the requested clearance;
4. is proximally fitted to the mitral end;
5. is truncated according to physical arc length;
6. is converted into a generalized capsule;
7. is joined to the terminal mitral ring with an equal-node one-to-one seam.

The default retained VoM length is:

```text
2/3 of the physical guide arc length
```

## CS → mitral checkpoint reuse

When available, the validated CS-to-mitral prefix can be reused.

The checkpoint is recognized from:

```text
cs_ring_hybrid.pts
elbow_cs_mitral_hybrid.pts
mitral_hybrid.pts
```

If these files are present and internally consistent, `tube_making.py` reuses them. Otherwise, the CS-to-mitral geometry is rebuilt automatically.

To explicitly rebuild the prefix when calling `tube_making.py` directly:

```bash
python3 tube_making.py ... --rebuild-prefix
```

## Mesh cleaning

### Stage 1

`mesh_cleaning.py` performs conservative cleanup of the raw tube:

- welds near-coincident vertices using a default tolerance of `2 µm`;
- removes triangles collapsed by the weld;
- removes duplicate/co-located triangles;
- compacts unused vertices;
- protects the old CS–VoM collar/junction from welding when the VoM reference surface is supplied;
- writes an original-to-cleaned vertex map for downstream provenance.

Typical output:

```text
tube_cs_cleaned_stage1.pts
tube_cs_cleaned_stage1.elem
tube_cs_cleaned_stage1.lon
tube_cs_cleaned_stage1_audit/
```

### Junction reconstruction

`mesh_cleaning_stage3.py` reconstructs the mitral–VoM transition.

Current design rules:

- default mitral peel depth: `2`;
- mitral and VoM junction loops must have the same node count;
- one-to-one correspondence;
- best cyclic alignment;
- multiple intermediate rings;
- no zipper/fan-to-pivot triangulation.

Typical output:

```text
tube_cs_cleaned_stage3.pts
tube_cs_cleaned_stage3.elem
tube_cs_cleaned_stage3.lon
tube_cs_cleaned_stage3_audit/
```

### Final repair

`mesh_final_repair.py` performs a conservative final repair.

It checks for:

- repeated-vertex triangles;
- true zero or near-zero area triangles;
- exact non-adjacent self-intersections;
- severe local slivers adjacent to an intersection patch;
- non-manifold topology;
- orientation conflicts.

It tries the smallest local patch first and expands only within the configured limit.

The input mesh is not silently replaced when no validated conservative repair can be found.

## Biatrial assembly

After obtaining `tube_cs_cleaned`, `build_biatrial_model.py`:

1. extracts the disconnected BB surfaces (tag `10`);
2. merges the cleaned CS–VoM mesh into `Bilayer_Combined`;
3. adds line-element connections;
4. builds the closest-region BB connector;
5. builds the anatomically guided BB connector;
6. merges both BB surfaces;
7. reconstructs the final element-wise `.f3d`;
8. validates element counts, indices, tags, `.lon`, and `.f3d`;
9. optionally produces an interactive browser visualization.

The default final basename is:

```text
Reentry_surface_iac
```

## Generated connections and tags

The current shared tag definitions are:

| Tag | Meaning |
| ---: | --- |
| `32` | LA epicardium ↔ LA endocardium |
| `33` | SAN → RA |
| `34` | CT → RA |
| `35` | PM → RA |
| `39` | FO / inter-atrial RA ↔ LA |
| `40` | CS → LA |
| `41` | BB → atrial surface |
| `50` | VoM → LA |
| `70` | Closest-region BB connector surface |
| `80` | Anatomically guided BB connector surface |

Relevant structural tags used by the workflow include:

| Tag | Structure |
| ---: | --- |
| `3` | SAN |
| `8` | CT |
| `9` | PM |
| `10` | BB |

The public tag definitions intentionally do not include fibrosis-specific labels.

## CS and VoM connection positions

CS and VoM connections are positioned on a normalized shortest surface path:

```text
0.0 = CS opening
1.0 = distal VoM opening
```

Defaults:

```text
CS:  0.30 0.40 0.50
VoM: 0.70 0.80 0.90
```

with:

```text
5 nearby tube vertices per requested position
```

Example:

```bash
python3 Bilayer_CS_VoM.py \
    --ra-path ./RA_XX \
    --la-path ./Bia_XX \
    --svc-landmark-index 6 \
    --cs-connection-positions 0.30 0.40 0.50 \
    --vom-connection-positions 0.70 0.80 0.90 \
    --connection-neighbors 5
```

## Visualization

The default visualization mode uses Plotly and writes a self-contained HTML file.

Example:

```bash
python3 visualization.py \
    --mesh Reentry_surface_iac \
    --output-html Reentry_surface_iac_view.html \
    --open-browser
```

Display selected tags only:

```bash
python3 visualization.py \
    --mesh Reentry_surface_iac \
    --tags 10 40 41 50 70 80 \
    --surface-opacity 0.5 \
    --output-html connections.html
```

Overlay diagnostic point files:

```bash
python3 visualization.py \
    --mesh BB \
    --point-file bb_bridges_anatomical_ra_geodesic.pts \
    --point-file bb_bridges_anatomical_la_geodesic.pts \
    --point-lines \
    --output-html bb_check.html
```

The viewer supports both triangular (`Tr`) and line (`Ln`) CARP elements.

## Audit and provenance

The top-level workflow creates:

```text
Bilayer_CS_VoM_pipeline/
├── commands.log
├── pipeline_parameters.json
└── pipeline_report.txt
```

These files are generated per run and are useful for reproducibility.

They should normally **not be committed to the source repository** because they contain machine- and dataset-specific paths.

Individual cleaning/building stages also write diagnostic and audit files alongside their generated meshes.

## Known limitations — v0.1.0

This is an early research release.

1. **The final local repair is intentionally conservative.**  
   If the self-intersection region cannot be represented as a small valid local repair patch, the pipeline stops instead of making a large uncontrolled modification.

2. **Some geometries may require further improvement of the mitral–VoM transition.**  
   `mesh_cleaning_stage3.py` reports transition self-intersections. A non-zero intersection count should be inspected carefully, and the final model should not be treated as accepted merely because an intermediate mesh was written.

3. **Landmark indexing is dataset-dependent.**  
   In particular, `--svc-landmark-index` must be verified for each input convention.

4. **The workflow assumes the AtrialMTK region convention used by this project.**  
   If your AtrialMTK labels differ, update `mesh_tags.py` and verify all anatomical selections.

5. **Manual visual inspection remains part of the acceptance process.**  
   Automated checks complement, but do not replace, inspection of the CS–mitral–VoM transition, BB connectors, and final connection layout.

6. **The software is research code and is not intended for clinical use.**

## Recommended validation before simulation

For each generated model, verify at minimum:

```text
zero / degenerate triangles = 0
self-intersection pairs      = 0
non-manifold edges           = 0
orientation conflicts        = 0
```

Also inspect:

- CS orientation and CS opening;
- CS-to-mitral elbow;
- terminal mitral region;
- mitral–VoM transition;
- VoM course relative to the LA;
- distal VoM opening;
- CS→LA and VoM→LA connection sites;
- both BB connector surfaces;
- final line-element tags.

## Development status

The code is being actively developed. File formats, CLI options, and geometry heuristics may change between early releases.

For reproducible studies, record the Git commit or release tag used to generate each model.

## Suggested citation

A formal software citation will be added in a future release. Until then, please cite the associated methodology/publication when available and record the repository version or Git commit used for model generation.

## License

Add the repository license before public release and ensure that it is compatible with all incorporated or adapted upstream code, including AtrialMTK/UAC dependencies.

