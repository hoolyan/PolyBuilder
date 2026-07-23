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
7. Optionally rejects clear polygon-polygon self-intersections while allowing the shared edges and vertices required by the polyhedron topology.
8. Searches for nontrivial graph automorphisms that preserve face data and dihedral angles.
9. Optionally exports constructed realizations as OBJ files.

The method supports convex and nonconvex dihedral configurations. It is intended for exhaustive computational searches over graphs that become discretely solvable through the vertex-dihedral propagation process.

## Scope and Limitations

PolyBuilder is a numerical research program, not a formal proof assistant. Its results should be interpreted with the following limitations in mind:

- **Propagation must become discrete.** Some graphs may remain locally flexible because no vertex initially has a finite set of dihedral completions. PolyBuilder reports these graphs as unsolved rather than declaring them impossible. Among the supplied graph sets with at most nine faces, the octahedral graph is the only graph with this behavior.
- **Self-intersection checking is optional and numerical.** With `--perform-self-intersection-check`, PolyBuilder performs pairwise convex polygon checks, allowing only the shared edges and vertices implied by the topology. Clear crossings and coplanar overlaps are rejected. Contacts wholly inside the numerical ambiguity band are accepted rather than risking a false rejection, so exact tangencies or extremely near contacts may still warrant independent inspection.
- **Calculations use floating-point tolerances.** Closure, regularity, equality of dihedral angles, and symmetry preservation are tested numerically.
- **Optional limits can make a run incomplete.** Using `--combination-limit`, a graph subset, or a face-set filter intentionally restricts the search.
- **Checkpoint compatibility is only partially enforced.** The program verifies `face-count`, `g6-path`, and the self-intersection setting when resuming. Use the same remaining search settings unless you deliberately intend to combine different runs.

For the supplied `face-count = 4` through `face-count = 9` graph sets, the propagation method solves every graph except the octahedral graph. With the optional self-intersection check enabled, the search leaves two distinct simple asymmetric realizations with nine faces and none with fewer than nine among the graphs solved by the method.

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
plantri -pg <face-count> <output-file>.g6
```

For example:

```bash
plantri -pg 9 input_graphs_f9.g6
```

## Usage

### Basic command

```bash
python polybuilder.py --face-count <face-count> --g6-path <input-file>
```

To export constructed realizations:

```bash
python polybuilder.py --face-count 9 --g6-path input_graphs_f9.g6 --export-objs --output-path output_f9
```

### Arguments

| Argument | Description |
|---|---|
| `--face-count N` | Number of polyhedron faces. This must equal the number of nodes in each input graph. |
| `--g6-path PATH` | Path to a graph6 input file. |
| `--output-path PATH` | Directory for OBJ exports. Required with `--export-objs` or `--export-invalid-objs`. |
| `--graph-subset-range START END` | Process graph indices in the half-open interval `[START, END)`. |
| `--combination-limit N` | Stop further branching for a graph when the number of partial solutions exceeds `N`. This may leave that graph incompletely searched. |
| `--specify-face-set SPEC` | Process only graphs with a specified multiset of face types, such as `3:8,4:3,5:2`. |
| `--allow-coplanar-dihedrals` | Permit dihedral angles of 180 degrees. These degenerate coplanar configurations are rejected by default. |
| `--disable-overlap-check` | Disable the limited coincidence check used during realization validation. This is separate from the optional polygon-polygon check. |
| `--perform-self-intersection-check` | Reject realizations with clear polygon-polygon self-intersections. Disabled by default, so existing runs are unchanged unless this flag is supplied. |
| `--perform-asymmetry-check` | Display the final summary of asymmetric realizations. Symmetry classification is currently performed internally for all constructed realizations regardless of this flag. |
| `--export-objs` | Export accepted realizations as OBJ files. |
| `--export-invalid-objs` | Export rejected constructed realizations for debugging. |
| `--display-dihedral-solutions` | Print the dihedral angles associated with reported solutions. |
| `--show-progress-details` | Print detailed propagation and validation information. |
| `--save-progress PATH` | Save a JSON checkpoint to `PATH` after each processed graph and at the end of the run. |
| `--resume-from PATH` | Resume from a checkpoint. If `--save-progress` is omitted, updates are written back to the same file. |

## Examples

### Search all supplied nine-face graphs

```bash
python polybuilder.py --face-count 9 --g6-path input_graphs_f9.g6 --perform-asymmetry-check --display-dihedral-solutions
```

### Reject self-intersecting realizations

```bash
python polybuilder.py --face-count 9 --g6-path input_graphs_f9.g6 --perform-self-intersection-check --perform-asymmetry-check
```

### Export accepted and rejected constructions

```bash
python polybuilder.py --face-count 9 --g6-path input_graphs_f9.g6 --export-objs --export-invalid-objs --output-path output_f9
```

### Search only a particular face multiset

```bash
python polybuilder.py --face-count 13 --g6-path input_graphs_f13.g6 --specify-face-set 3:8,4:3,5:2
```

### Run a subset with checkpointing

```bash
python polybuilder.py --face-count 13 --g6-path input_graphs_f13.g6 --graph-subset-range 0 1000000 --save-progress checkpoint_f13.json
```

### Resume a checkpoint

```bash
python polybuilder.py --face-count 13 --g6-path input_graphs_f13.g6 --resume-from checkpoint_f13.json
```

### Limit branching on large searches

```bash
python polybuilder.py --face-count 18 --g6-path input_graphs_f18.g6 --combination-limit 16
```

## Results and Output

During a run, PolyBuilder reports graphs in four categories:

- **Unsolved graphs:** the propagation method could not fully determine the dihedrals.
- **Graphs with dihedral solutions:** complete locally compatible dihedral assignments were found.
- **Graphs with realizations:** at least one dihedral assignment produced a closed 3D construction that passed the current numerical validation.
- **Graphs with asymmetric realizations:** at least one constructed realization had no nontrivial automorphism preserving the tested face and dihedral data.

OBJ files are written only when an export flag is supplied. Checkpoint files contain run settings, the next graph index, and the accumulated result categories in human-readable JSON.

## Tests

Run the unit and integration tests from the project root with:

```bash
python -m unittest discover -s tests -v
```

## Project Structure

### `polybuilder.py`
Main command-line entry point. Streams input graphs, coordinates the dihedral search, constructs realizations, classifies symmetry, exports OBJ files, and reports results.

### `data_structures.py`
Defines the `Vertex`, `Edge`, `Face`, and `RegularFacedPolyhedron` classes, vector helpers, graph6 streaming, planar embedding utilities, and conversion from a dual graph to the internal polyhedron representation.

### `dihedral_solver.py`
Implements spherical triangles, spherical triangulations, local solvability tests, dihedral calculation, propagation, and branching.

### `realization_constructor.py`
Builds regular polygon faces in 3D, folds them according to assigned dihedrals, validates closure and regular face geometry, and exports OBJ files.

### `self_intersection_checker.py`
Performs the optional conservative pairwise polygon-polygon intersection test, distinguishing geometric crossings from topologically shared edges and vertices.

### `symmetry_checker.py`
Uses NetworkX graph isomorphism to search for nonidentity automorphisms that preserve vertex data and edge dihedral angles.

### `checkpoint.py`
Saves and loads resumable JSON checkpoints.

### `utilities.py`
Contains formatting and face-set parsing helpers.

## Author

Created by Julian Spencer ([@hoolyan](https://github.com/hoolyan)).