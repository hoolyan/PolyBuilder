# PolyBuilder

PolyBuilder is a computational search tool for constructing and analyzing polyhedra whose faces are regular polygons with a common edge length. It takes the dual graph of a candidate polyhedron, solves for compatible dihedral angles using spherical geometry, folds the faces into 3D, and checks the resulting realization for symmetry.

The program was developed as part of a search for the regular-faced polyhedron with the fewest faces and no nontrivial symmetry.

## What PolyBuilder Does

For each 3-connected simple planar input graph, PolyBuilder:

1. Interprets each graph node as a polyhedron face.
2. Uses the degree of each node to determine the number of sides of that face.
3. Applies spherical-triangle constraints at the polyhedron's vertices.
4. Propagates known dihedral angles and branches when two discrete local solutions are possible.
5. Constructs 3D coordinates for each complete dihedral assignment.
6. Rejects assignments that do not close with unit-length edges and regular face geometry.
7. Searches for nontrivial graph automorphisms that preserve face data and dihedral angles.
8. Optionally exports constructed realizations as OBJ files.

The method supports convex and nonconvex dihedral configurations. It is intended for exhaustive computational searches over graphs that become discretely solvable through the vertex-dihedral propagation process.

## Scope and Limitations

PolyBuilder is a numerical research program, not a formal proof assistant. Its results should be interpreted with the following limitations in mind:

- **Propagation must become discrete.** Some graphs may remain locally flexible because no vertex initially has a finite set of dihedral completions. PolyBuilder reports these graphs as unsolved rather than declaring them impossible. Among the supplied graph sets with at most nine faces, the octahedral graph is the only graph with this behavior.
- **Self-intersection is not checked exhaustively.** The current strict validation detects coincident vertices, edge midpoints, and face centers, but it does not perform a complete polygon-polygon intersection test. A geometrically closed output may therefore still represent a self-intersecting surface and should be checked separately when simplicity matters.
- **Calculations use floating-point tolerances.** Closure, regularity, equality of dihedral angles, and symmetry preservation are tested numerically.
- **Optional limits can make a run incomplete.** Using `--combination_limit`, a graph subset, or a face-set filter intentionally restricts the search.
- **Checkpoint compatibility is only partially enforced.** The program currently verifies `F` and `g6_path` when resuming. Use the same remaining search settings unless you deliberately intend to combine different runs.

For the supplied `F = 4` through `F = 9` graph sets, the propagation method solves every graph except the octahedral graph. After separately excluding self-intersecting outputs, the search produces two distinct simple asymmetric realizations with nine faces and none with fewer than nine among the graphs solved by the method.

## Requirements

- Python 3.10+
- [NetworkX](https://networkx.org/)
- [NumPy](https://numpy.org/)
- [tqdm](https://tqdm.github.io/)
- [Matplotlib](https://matplotlib.org/)

Install the Python dependencies with:

```bash
python -m pip install networkx numpy tqdm matplotlib
```

## Input Graphs

Input files use the [graph6](https://users.cecs.anu.edu.au/~bdm/data/formats.html) format, with one graph per line.

PolyBuilder treats the input as the **dual graph** of the polyhedron:

- graph nodes correspond to polyhedron faces;
- graph edges correspond to shared polyhedron edges;
- planar faces of the embedded graph correspond to polyhedron vertices;
- the degree of a graph node is the number of sides of the corresponding regular face.

Input graphs should be 3-connected, simple, and planar. They can be generated with [plantri](https://users.cecs.anu.edu.au/~bdm/plantri/):

```bash
plantri -pg <face_count> <output_file>.g6
```

For example:

```bash
plantri -pg 9 input_graphs_f9.g6
```

## Usage

### Basic command

```bash
python polybuilder.py --F <face_count> --g6_path <input_file>
```

To export constructed realizations:

```bash
python polybuilder.py \
  --F 9 \
  --g6_path input_graphs_f9.g6 \
  --export_objs \
  --output_path output_f9
```

### Arguments

| Argument | Description |
|---|---|
| `--F` | Number of polyhedron faces. This must equal the number of nodes in each input graph. |
| `--g6_path` | Path to a graph6 input file. |
| `--output_path` | Directory for OBJ exports. Required with `--export_objs` or `--export_invalid_objs`. |
| `--graph_subset_range START END` | Process graph indices in the half-open interval `[START, END)`. |
| `--combination_limit N` | Stop further branching for a graph when the number of partial solutions exceeds `N`. This may leave that graph incompletely searched. |
| `--specify_face_set SPEC` | Process only graphs with a specified multiset of face types, such as `3:8,4:3,5:2`. |
| `--allow_coplanar_dihedrals` | Permit dihedral angles of 180 degrees. These degenerate coplanar configurations are rejected by default. |
| `--disable_overlap_check` | Disable the limited coincidence check used during realization validation. This flag does not refer to a complete self-intersection test. |
| `--perform_asymmetry_check` | Display the final summary of asymmetric realizations. Symmetry classification is currently performed internally for all constructed realizations regardless of this flag. |
| `--export_objs` | Export accepted realizations as OBJ files. |
| `--export_invalid_objs` | Export rejected constructed realizations for debugging. |
| `--display_dihedral_solutions` | Print the dihedral angles associated with reported solutions. |
| `--show_progress_details` | Print detailed propagation and validation information. |
| `--save_progress PATH` | Save a JSON checkpoint after each processed graph and at the end of the run. |
| `--resume_from PATH` | Resume from a checkpoint. If `--save_progress` is omitted, updates are written back to the same file. |

## Examples

### Search all supplied nine-face graphs

```bash
python polybuilder.py \
  --F 9 \
  --g6_path input_graphs_f9.g6 \
  --perform_asymmetry_check \
  --display_dihedral_solutions
```

### Export accepted and rejected constructions

```bash
python polybuilder.py \
  --F 9 \
  --g6_path input_graphs_f9.g6 \
  --export_objs \
  --export_invalid_objs \
  --output_path output_f9
```

### Search only a particular face multiset

```bash
python polybuilder.py \
  --F 13 \
  --g6_path input_graphs_f13.g6 \
  --specify_face_set 3:8,4:3,5:2
```

### Run a subset with checkpointing

```bash
python polybuilder.py \
  --F 13 \
  --g6_path input_graphs_f13.g6 \
  --graph_subset_range 0 1000000 \
  --save_progress checkpoint_f13.json
```

### Resume a checkpoint

```bash
python polybuilder.py \
  --F 13 \
  --g6_path input_graphs_f13.g6 \
  --resume_from checkpoint_f13.json
```

### Limit branching on large searches

```bash
python polybuilder.py \
  --F 18 \
  --g6_path input_graphs_f18.g6 \
  --combination_limit 16
```

## Results and Output

During a run, PolyBuilder reports graphs in four categories:

- **Unsolved graphs:** the propagation method could not fully determine the dihedrals.
- **Graphs with dihedral solutions:** complete locally compatible dihedral assignments were found.
- **Graphs with realizations:** at least one dihedral assignment produced a closed 3D construction that passed the current numerical validation.
- **Graphs with asymmetric realizations:** at least one constructed realization had no nontrivial automorphism preserving the tested face and dihedral data.

OBJ files are written only when an export flag is supplied. Checkpoint files contain run settings, the next graph index, and the accumulated result categories in human-readable JSON.

## Project Structure

### `polybuilder.py`
Main command-line entry point. Streams input graphs, coordinates the dihedral search, constructs realizations, classifies symmetry, exports OBJ files, and reports results.

### `data_structures.py`
Defines the `Vertex`, `Edge`, `Face`, and `RegularFacedPolyhedron` classes, vector helpers, graph6 streaming, planar embedding utilities, and conversion from a dual graph to the internal polyhedron representation.

### `dihedral_solver.py`
Implements spherical triangles, spherical triangulations, local solvability tests, dihedral calculation, propagation, and branching.

### `realization_constructor.py`
Builds regular polygon faces in 3D, folds them according to assigned dihedrals, validates closure and regular face geometry, and exports OBJ files.

### `symmetry_checker.py`
Uses NetworkX graph isomorphism to search for nonidentity automorphisms that preserve vertex data and edge dihedral angles.

### `checkpoint.py`
Saves and loads resumable JSON checkpoints.

### `utilities.py`
Contains formatting, face-set parsing, and object-size helpers.

## Author

Created by Julian Spencer ([@hoolyan](https://github.com/hoolyan)).