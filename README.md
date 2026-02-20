# PolyBuilder

A tool for constructing and analyzing asymmetric regular-faced polyhedra. This program takes planar graphs as input and computes valid 3D realizations where all faces are regular polygons.

## Overview

PolyBuilder uses spherical geometry and dihedral angle constraints to find all possible realizations of polyhedra with regular faces from a given 3-connected planar graph.

## Requirements

- Python 3.6+
- networkx (for graph operations)

## Usage

### Basic Command

```bash
python polybuilder.py --F <face_count> --g6_path <input_file> --export_objs --output_path <output_dir>
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--F` | Yes | Number of faces in the polyhedron |
| `--g6_path` | Yes | Path to input graph file in .g6 format (graph6 format) |
| `--output_path` | No | Directory where results and OBJ files will be saved |
| `--graph_subset_range` | No | Restrict analysis to a subset of graphs by index range (e.g., `0 100` analyzes the first 100 graphs). |
| `--combination_limit` | No | Maximum combinations to explore per graph (default: unlimited). Graphs exceeding this limit are rejected. |
| `--allow_coplanar_dihedrals` | No | Allow dihedral angles of 180° (coplanar faces). By default these are rejected to avoid creating faces that are not regular polygons. |
| `--disable_overlap_check` | No | Skip the overlap detection check during realization validation. |
| `--perform_asymmetry_check` | No | Perform a check for non-trivial graph and dihedral symmetries, and display realizations where none were found. |
| `--export_objs` | No | Export valid realizations as OBJ files for 3D visualization |
| `--export_invalid_objs` | No | Export invalid realizations as OBJ files for 3D visualization |
| `--display_dihedral_solutions` | No | Print detailed dihedral angle solutions found during solving |
| `--show_progress_details` | No | Display verbose progress information during computation |
| `--save_progress` | No | Path to save progress file after every graph (e.g., `checkpoint.json`). Automatically saves progress to the same file every time a graph is processed. If `--resume_from` is specified without `--save_progress`, new progress will be saved back to the resume path by default. |
| `--resume_from` | No | Resume from a previous progress file. Restores all progress and results from the last saved state. If `--save_progress` is not specified, progress updates will be saved to this path. |

### Examples

#### Export polyhedra with 9 faces:
```bash
python polybuilder.py --F 9 --g6_path ../input/input_graphs_f9.g6 --export_objs --output_path ../output/out_f9
```

#### Check for asymmetry in realizations:
```bash
python polybuilder.py --F 9 --g6_path ../input/input_graphs_f9.g6 --export_objs --output_path ../output/out_f9
```

#### Explore graphs while viewing solution details:
```bash
python polybuilder.py --F 10 --g6_path ../input/input_graphs_f10.g6 --display_dihedral_solutions --show_progress_details
```

#### Long-running analysis with checkpointing:
```bash
python polybuilder.py --F 13 --g6_path ../input/input_graphs_f13.g6 --save_progress checkpoint.json --graph_subset_range 0 1000000 --export_objs --output_path ../output/out_f13
```

#### Resume from checkpoint after interruption or to continue with a different subset:
```bash
python polybuilder.py --F 13 --g6_path ../input/input_graphs_f13.g6 --resume_from checkpoint.json --graph_subset_range 1000000 2000000 --export_objs --output_path ../output/out_f13
```

#### Limit computational effort when solution space is too large:
```bash
python polybuilder.py --F 18 --g6_path ../input/input_graphs_f18.g6 --allow_coplanar_dihedrals --combination_limit 16
```

**How it works:**
- Progress saves: run settings, current progress index, and all accumulated results (unsolved graphs, dihedral solutions, realizations, asymmetric realizations)
- On resume: validates that F and g6_path match the F and g6_path specified in a progress file, then continues from the saved index
- `--graph_subset_range` start index is ignored when resuming from a progress file if the start index is earlier than the checkpoint index to prevent redundant data
- Progress file is JSON format and human-readable

### Input: Generating Graph Files

Input graphs must be in **graph6 format** (.g6 files). Generate them using [plantri](http://users.cecs.anu.edu.au/~bdm/plantri/):

```bash
plantri -pg <face_count> <output_file>.g6
```

For example, to generate all 3-connected planar graphs with 10 nodes:
```bash
plantri -pg 10 input_graphs_f10.g6
```

### Output

Results are saved to the specified output directory containing:
- OBJ files (if `--export_objs` or `--export_invalid_objs` is specified) for 3D model visualization

MODULE BREAKDOWN
================

### 1. `data_structures.py` (~550 lines)
   - **Geometry helpers**: `v_add()`, `v_sub()`, `v_scale()`, `v_norm()`, `v_normalize()`, `v_dot()`, `v_cross()`
   - **Core classes**: `Vertex`, `Edge`, `Face`, `RegularFacedPolyhedron`
   - **Graph utilities**: `enumerate_embedding_faces()`, `read_g6_graphs()`, `create_polyhedron_from_graph()`

### 2. `dihedral_solver.py` (~1125 lines)
   - **Spherical triangle solving**: `SphericalTriangle` class with `compute_SSS()` and `compute_SAS()`
   - **Triangulation**: `SphericalTriangulation` class with `solve_triangulation()` and `get_dihedral()`
   - **Main orchestrator**: `extend_solved_vertices()` - branches solutions based on vertex states
   - **Helpers**: `num_solutions()`, `has_valid_dihedrals()`, `calculate_dihedral()`, `calculate_possible_dihedrals()`

### 3. `realization_constructor.py` (~483 lines)
   - **3D coordinate construction**: `construct_polyhedron_realization()` with 4 embedded transforms
   - **Edge alignment**: `rotate_face_about_normal_to_align_edge()` with precision fix (rtol=0)
   - **Validation**: `is_valid_realization()` - checks unit edges and face regularity
   - **Export**: `export_regular_faced_polyhedron_to_OBJ()` - writes OBJ format files

### 4. `symmetry_checker.py` (~52 lines)
   - **Graph construction**: `build_vertex_graph_with_dihedrals()` with dihedral attributes
   - **Detection**: `has_nontrivial_automorphism_with_dihedrals()` - finds non-identity isomorphisms
   - **Helpers**: `TAU`, `_wrap_0_tau()`, `_ang_dist()` - angle normalization and circular distance

### 5. `checkpoint.py` (~80 lines)
   - **Data class**: `CheckpointData` - container for run state (settings, progress index, results)
   - **Serialization**: `save_checkpoint()` - writes checkpoint to JSON file
   - **Deserialization**: `load_checkpoint()` - loads checkpoint from JSON file
   - **Special handling**: Manages float edge cases (infinity, NaN) for JSON compatibility

### 6. `utilities.py` (~60 lines)
   - **Formatting**: `format_face_types()` - converts face type lists to human-readable summaries (e.g., `[3 triangles, 2 squares, 1 pentagon]`)
   - **Display**: `format_dihedral_degrees()` - converts dihedral angles from radians to degrees with reduced precision (4 decimal places by default)

## Import Structure

`polybuilder.py` imports all modules:
```python
from data_structures import *
from dihedral_solver import *
from realization_constructor import *
from symmetry_checker import *
from checkpoint import *
from utilities import *
```

**Dependency graph**:
```
polybuilder.py (orchestration) → All modules
data_structures.py → networkx
dihedral_solver.py → data_structures
realization_constructor.py → data_structures
symmetry_checker.py → data_structures + networkx
checkpoint.py → json
utilities.py → collections
```

## Author

Created by Julian Spencer ([@hoolyan](https://github.com/hoolyan))