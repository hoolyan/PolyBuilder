"""
Data structures for regular-faced polyhedra.

Defines:
- Basic geometry helpers (vector operations)
- Core polyhedron data classes: Vertex, Edge, Face, RegularFacedPolyhedron
- Graph reading and conversion from planar graphs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Iterator, Tuple, Dict, Hashable
import sys
import time
import mmap
import shutil

import networkx as nx
import numpy as np

# ============================================================
# Basic geometry helpers
# ============================================================

def v_add(a, b):
    return np.array(a) + np.array(b)


def v_sub(a, b):
    return np.array(a) - np.array(b)


def v_scale(a, s: float):
    return np.array(a) * float(s)


def v_norm(a) -> float:
    return float(np.linalg.norm(a))


def v_normalize(a):
    n = v_norm(a)
    if n == 0.0:
        raise ValueError("Cannot normalize zero vector")
    return a / n


def v_dot(a, b) -> float:
    return float(np.dot(a, b))


def v_cross(a, b):
    return np.cross(a, b)

# ============================================================
# Core data structures
# ============================================================

@dataclass
class Vertex:
    index: int
    faces: List["Face"] = field(default_factory=list)
    edges: List["Edge"] = field(default_factory=list)

    vertex_dihedrals_valid: bool = False

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    constructed: bool = False

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    @pos.setter
    def pos(self, v: np.ndarray):
        self.x = float(v[0])
        self.y = float(v[1])
        self.z = float(v[2])


@dataclass
class Edge:
    index: int
    vertices: Tuple[Vertex, Vertex]  # should be length 2
    faces: List["Face"] = field(default_factory=list)  # should be length 2

    has_assigned_dihedral: bool = False
    dihedral: float = 0.0  # radians

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    constructed: bool = False

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    @pos.setter
    def pos(self, v: np.ndarray):
        self.x = float(v[0])
        self.y = float(v[1])
        self.z = float(v[2])


@dataclass
class Face:
    index: int
    vertices: List[Vertex] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    normal_x: float = 0.0
    normal_y: float = 1.0
    normal_z: float = 0.0
    constructed: bool = False

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)

    @pos.setter
    def pos(self, v: np.ndarray):
        self.x = float(v[0])
        self.y = float(v[1])
        self.z = float(v[2])


@dataclass
class RegularFacedPolyhedron:
    faces: List[Face]
    edges: List[Edge]
    vertices: List[Vertex]

    def deep_copy(self) -> "RegularFacedPolyhedron":
        """
        Deep copy creating an exact replica of the RegularFacedPolyhedron.
        
        Ensures all scalar fields, nested data structures, and inter-object 
        references are properly copied and remapped. No shared state between 
        original and copy.

        Note: This function creates tons of circular references between vertices, edges, and faces.
        A more efficient implementation could involve storing additional indices instead of direct references.
        """
        v_map: Dict[int, Vertex] = {}
        e_map: Dict[int, Edge] = {}
        f_map: Dict[int, Face] = {}

        # === STEP 1: Copy all Vertex scalar fields ===
        for v in self.vertices:
            v_copy = Vertex(
                index=v.index,
                faces=[],  # will be populated in step 3
                edges=[],  # will be populated in step 3
                vertex_dihedrals_valid=v.vertex_dihedrals_valid,
                x=v.x,
                y=v.y,
                z=v.z,
                constructed=v.constructed
            )
            v_map[v.index] = v_copy

        # === STEP 2: Copy all Face scalar fields ===
        for f in self.faces:
            f_copy = Face(
                index=f.index,
                vertices=[],  # will be populated in step 3
                edges=[],     # will be populated in step 3
                x=f.x,
                y=f.y,
                z=f.z,
                constructed=f.constructed
            )
            f_map[f.index] = f_copy

        # === STEP 3: Copy all Edge scalar fields, with remapped Vertex references ===
        for e in self.edges:
            v0_copy = v_map[e.vertices[0].index]
            v1_copy = v_map[e.vertices[1].index]
            e_copy = Edge(
                index=e.index,
                vertices=(v0_copy, v1_copy),
                faces=[],  # will be populated in step 4
                has_assigned_dihedral=e.has_assigned_dihedral,  # copy dihedral assignment status
                dihedral=e.dihedral,  # copy dihedral value
                x=e.x,
                y=e.y,
                z=e.z,
                constructed=e.constructed
            )
            e_map[e.index] = e_copy

        # === STEP 4: Rebuild all adjacency relationships with remapped references ===
        # Rebuild Face -> Vertex/Edge references
        for f in self.faces:
            f_copy = f_map[f.index]
            f_copy.vertices = [v_map[v.index] for v in f.vertices]
            f_copy.edges = [e_map[e.index] for e in f.edges]

        # Rebuild Vertex -> Face/Edge references
        for v in self.vertices:
            v_copy = v_map[v.index]
            v_copy.faces = [f_map[f.index] for f in v.faces]
            v_copy.edges = [e_map[e.index] for e in v.edges]

        # Rebuild Edge -> Face references
        for e in self.edges:
            e_copy = e_map[e.index]
            e_copy.faces = [f_map[f.index] for f in e.faces]

        # === STEP 5: Return new RegularFacedPolyhedron with copied lists ===
        return RegularFacedPolyhedron(
            faces=[f_map[i] for i in sorted(f_map.keys())],
            edges=[e_map[i] for i in sorted(e_map.keys())],
            vertices=[v_map[i] for i in sorted(v_map.keys())],
        )

    def solutions_match(self, other: "RegularFacedPolyhedron") -> bool:
        if len(self.edges) != len(other.edges):
            return False
        for e_self, e_other in zip(self.edges, other.edges):
            if e_self.has_assigned_dihedral != e_other.has_assigned_dihedral:
                return False
            if e_self.has_assigned_dihedral:
                if abs(e_self.dihedral - e_other.dihedral) > 1e-12:
                    return False
                
        for v_self, v_other in zip(self.vertices, other.vertices):
            if v_self.vertex_dihedrals_valid != v_other.vertex_dihedrals_valid:
                return False
        return True


# ============================================================
# Graph reading / conversion
# ============================================================

def enumerate_embedding_faces(embedding: nx.algorithms.planarity.PlanarEmbedding):
    """
    Return a list of faces of the embedding.
    Each face is a list of nodes (the nodes of the original graph).
    """
    faces = []
    seen_half_edges: set[tuple[Hashable, Hashable]] = set()

    # Embedding is an undirected graph with a combinatorial rotation system.
    # Walk each half-edge (u, v) once and traverse the face to its right.
    for u in embedding.nodes():
        for v in embedding.neighbors_cw_order(u):
            if (u, v) in seen_half_edges:
                continue
            face = embedding.traverse_face(u, v, seen_half_edges)
            faces.append(face)

    return faces

def scout_g6_graph_count(path: str, start_index: int, end_index: int) -> int:
    """
    Count a specific subset of graphs from a .g6 file by index range.
    
    Uses mmap to handle large files safely—Windows pages on demand
    instead of thrashing when skipping to high indices.
    
    Args:
        path: Path to .g6 file
        start_index: Starting graph index (0-based)
        end_index: Ending graph index (exclusive); None for end of file
    
    Returns:
        The number of graphs read in the specified range.
    """
    
    graph_count = 0
    terminal_width = shutil.get_terminal_size().columns
    
    sys.stdout.write("\r" + " " * terminal_width + "\r")
    sys.stdout.flush()
    sys.stdout.write(f"Scouting graphs...")
    sys.stdout.flush()
    with open(path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
            # Count lines until we reach start_index
            line_count = 0
            byte_pos = 0
            
            # Skip to start_index
            while line_count < start_index:
                byte_pos = m.find(b'\n', byte_pos)
                if byte_pos == -1:
                    return graph_count  # start_index beyond file
                byte_pos += 1
                line_count += 1

            sys.stdout.write("\r" + " " * terminal_width + "\r")
            sys.stdout.flush()
            
            # Count from start_index to end_index
            next_update_time = time.time()
            while line_count < (end_index if end_index is not None else float('inf')):
                line_end = m.find(b'\n', byte_pos)
                if line_end == -1:
                    line_end = len(m)
                
                if line_end == byte_pos:
                    break
                
                graph_count += 1
                
                # Update progress at 60 Hz (every 1/60th of a second)
                current_time = time.time()
                if current_time >= next_update_time:
                    sys.stdout.write(f"\rScouted {graph_count} graphs...")
                    sys.stdout.flush()
                    next_update_time = next_update_time + (int((current_time - next_update_time) * 60.0) + 1.0) / 60.0
                
                byte_pos = line_end + 1
                line_count += 1
    
    # Clear the progress line using terminal width to avoid wrapping
    if graph_count > 0:
        sys.stdout.write("\r" + " " * terminal_width + "\r")
        sys.stdout.flush()
    
    return graph_count

def stream_g6_graphs(path: str, start_index: int = 0, end_index: int = None) -> Iterator[Tuple[nx.Graph, int]]:
    """
    Generator that streams graphs from a .g6 file one at a time.
    
    Yields (graph, index) tuples. Memory usage stays constant (only 1 graph at a time).
    Uses mmap for efficient seeking to start_index.
    
    Args:
        path: Path to .g6 file
        start_index: Starting graph index (0-based)
        end_index: Ending graph index (exclusive); None for end of file
    
    Yields:
        Tuple of (nx.Graph, current_index)
    """

    if start_index < 0:
        raise ValueError("start_index must be >= 0")
    if end_index is not None and end_index < start_index:
        raise ValueError("end_index must be >= start_index")

    with open(path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
            n = len(m)
            line_count = 0
            byte_pos = 0

            # Skip to start_index
            while line_count < start_index:
                if byte_pos >= n:
                    return  # start_index beyond file
                nxt = m.find(b"\n", byte_pos)
                if nxt == -1:
                    return  # no more newlines => EOF before start_index
                byte_pos = nxt + 1
                line_count += 1

            # Stream until end_index or EOF
            while True:
                if end_index is not None and line_count >= end_index:
                    return
                if byte_pos >= n:
                    return  # EOF

                line_end = m.find(b"\n", byte_pos)
                eof = False
                if line_end == -1:
                    line_end = n
                    eof = True

                # Extract line bytes (graph6 is ASCII-ish)
                line = m[byte_pos:line_end].rstrip(b"\r\n")
                if not line:
                    raise ValueError(f"Graph at index {line_count} is empty")

                yield nx.from_graph6_bytes(line), line_count

                line_count += 1
                byte_pos = line_end + 1
                if eof:
                    return

def scout_face_set(G: nx.Graph) -> Dict[int, int]:
    """
    Given a dict mapping face types (e.g., "triangle", "square", "pentagon") to counts,
    return a concise string representation like "[8 triangles, 3 squares, 2 pentagons]".
    
    The face types are sorted by vertex count (e.g., triangle before square).
    """
    face_type_counts: Dict[int, int] = {}
    for node in G.nodes():
        degree = G.degree[node]
        if degree < 3:
            raise ValueError(f"Invalid face with degree {degree} at node {node}")
        face_type_counts[degree] = face_type_counts.get(degree, 0) + 1
    return face_type_counts

def create_polyhedron_from_graph(G: nx.Graph) -> RegularFacedPolyhedron:
    """
    G is a 3-connected simple planar graph where:

      - nodes = faces of the polyhedron
      - planar faces of G = vertices of the polyhedron
      - edges of G = edges of the polyhedron (adjacency of faces)

    This builds a RegularFacedPolyhedron where:
      polyhedron.faces    ↔ nodes of G
      polyhedron.vertices ↔ faces(embedding) of G
      polyhedron.edges    ↔ edges of G
    """
    is_planar, embedding = nx.check_planarity(G)
    if not is_planar:
        raise ValueError("Input graph is not planar")

    # --- step 1: graph nodes → polyhedron faces ---

    node_list = list(G.nodes())
    node_to_face_idx = {u: i for i, u in enumerate(node_list)}
    faces: List[Face] = [Face(index=i) for i in range(len(node_list))]

    # --- step 2: planar faces of G → polyhedron vertices ---

    face_cycles = enumerate_embedding_faces(embedding)  # list of node-cycles
    vertices: List[Vertex] = [Vertex(index=i) for i in range(len(face_cycles))]

    # Optional Euler sanity check: V - E + F = 2
    V_poly = len(face_cycles)          # vertices of polyhedron
    E_poly = G.number_of_edges()       # edges of polyhedron
    F_poly = len(node_list)            # faces of polyhedron
    if V_poly - E_poly + F_poly != 2:
        raise RuntimeError(
            f"Euler check failed: V={V_poly}, E={E_poly}, F={F_poly}"
        )

    # Fill face <-> vertex incidence using indices
    for v_idx, cycle in enumerate(face_cycles):
        V = vertices[v_idx]
        for node in cycle:
            f_idx = node_to_face_idx[node]
            F = faces[f_idx]
            if V not in F.vertices:
                F.vertices.append(V)
            if F not in V.faces:
                V.faces.append(F)

    # --- step 3: graph edges → polyhedron edges ---

    edges: List[Edge] = []
    edge_index = 0

    for u, v in G.edges():
        Fu = faces[node_to_face_idx[u]]
        Fv = faces[node_to_face_idx[v]]

        # Use vertex indices instead of Vertex objects for set operations
        indices_u = {vert.index for vert in Fu.vertices}
        indices_v = {vert.index for vert in Fv.vertices}
        common_indices = list(indices_u & indices_v)

        if len(common_indices) != 2:
            raise ValueError(
                f"Expected exactly 2 common vertices for face adjacency {u}-{v}, "
                f"got {len(common_indices)}"
            )

        v0 = vertices[common_indices[0]]
        v1 = vertices[common_indices[1]]

        e = Edge(index=edge_index, vertices=(v0, v1))
        edge_index += 1

        # register adjacency
        e.faces = [Fu, Fv]
        Fu.edges.append(e)
        Fv.edges.append(e)
        v0.edges.append(e)
        v1.edges.append(e)

        edges.append(e)

    # --- step 4: order vertices around each face (cyclic boundary) ---

    for F in faces:
        if not F.vertices:
            continue

        # Work in index space to avoid hashing Vertex
        idxs = [v.index for v in F.vertices]
        adj: Dict[int, List[int]] = {i: [] for i in idxs}

        for e in F.edges:
            i0 = e.vertices[0].index
            i1 = e.vertices[1].index
            if i0 in adj and i1 in adj:
                adj[i0].append(i1)
                adj[i1].append(i0)

        start_idx = idxs[0]
        order_idxs = [start_idx]
        prev_idx = None
        cur_idx = start_idx

        while True:
            neighbors = adj[cur_idx]
            if prev_idx is None:
                next_candidates = neighbors
            else:
                next_candidates = [nb for nb in neighbors if nb != prev_idx]

            if not next_candidates:
                break

            nxt_idx = next_candidates[0]

            if nxt_idx == start_idx:
                if len(order_idxs) == len(idxs):
                    break
                else:
                    # closed early; inconsistent boundary
                    break

            order_idxs.append(nxt_idx)
            prev_idx, cur_idx = cur_idx, nxt_idx

        if len(order_idxs) != len(idxs):
            raise ValueError("Couldn't reconstruct cyclic order of vertices around a face")

        # Map indices back to Vertex objects in the right cyclic order
        F.vertices = [vertices[i] for i in order_idxs]

        # Reorder the face's edges to match the cyclic vertex ordering.
        # For each consecutive vertex pair (v_i, v_{i+1}) there must be
        # an edge in the face connecting them; find that edge and attach
        # in the same cyclic order as vertices.
        new_face_edges: List[Edge] = []
        m = len(F.vertices)
        for i in range(m):
            v0 = F.vertices[i]
            v1 = F.vertices[(i + 1) % m]
            found_edge = None
            # Prefer searching v0.edges which should contain the boundary edges
            for e in v0.edges:
                if (e.vertices[0] is v0 and e.vertices[1] is v1) or (e.vertices[1] is v0 and e.vertices[0] is v1):
                    found_edge = e
                    break
            # Fallback: search the previously collected F.edges
            if found_edge is None:
                for e in F.edges:
                    if v0 in e.vertices and v1 in e.vertices:
                        found_edge = e
                        break
            if found_edge is None:
                raise ValueError("Couldn't find edge for consecutive vertices in face")
            new_face_edges.append(found_edge)
        F.edges = new_face_edges

    # --- step 5: reconstruct vertex.edge ordering to be cyclic around vertex ---

    for V in vertices:
        if not V.faces:
            continue
        new_vertex_edges: List[Edge] = []
        n = len(V.faces)
        # For each consecutive pair of faces around the vertex, the shared
        # edge between those faces is the incident polygon edge at that
        # position around the vertex.
        for i in range(n):
            f0 = V.faces[i]
            f1 = V.faces[(i + 1) % n]
            found_edge = None
            # Search edges of f0 for an edge that also has f1 and contains V
            for e in f0.edges:
                if f1 in e.faces and (e.vertices[0] is V or e.vertices[1] is V):
                    found_edge = e
                    break
            if found_edge is None:
                # Fallback: scan all incident edges attached earlier
                for e in V.edges:
                    if f0 in e.faces and f1 in e.faces and (e.vertices[0] is V or e.vertices[1] is V):
                        found_edge = e
                        break
            if found_edge is None:
                # As a last resort search global edge list
                for e in edges:
                    if f0 in e.faces and f1 in e.faces and (e.vertices[0] is V or e.vertices[1] is V):
                        found_edge = e
                        break
            if found_edge is None:
                raise ValueError("Couldn't reconstruct cyclic edge order for vertex")
            new_vertex_edges.append(found_edge)
        V.edges = new_vertex_edges

    # --- step 6: Ensure cyclic winding order is consistent in the Face.vertices arrays ---
    # For each face, check its winding order against its neighbors.
    # Consistent winding means adjacent faces traverse shared edges in opposite directions.
    # Use a BFS to propagate winding direction from face 0 to all other faces.
    
    if faces:
        # Start with face 0 having the "correct" winding (don't flip it)
        visited_faces = set()
        queue = [faces[0]]
        visited_faces.add(faces[0].index)
        
        while queue:
            current_face = queue.pop(0)
            
            # Check each adjacent face
            for edge in current_face.edges:
                other_face = edge.faces[0] if edge.faces[0] is not current_face else edge.faces[1]
                
                if other_face.index in visited_faces:
                    continue
                
                visited_faces.add(other_face.index)
                queue.append(other_face)
                
                # Determine if current_face traverses the edge as v0→v1 or v1→v0
                current_face_v0_idx = None
                current_face_v1_idx = None
                for i, v in enumerate(current_face.vertices):
                    if v.index == edge.vertices[0].index:
                        current_face_v0_idx = i
                    if v.index == edge.vertices[1].index:
                        current_face_v1_idx = i
                
                if current_face_v0_idx is None or current_face_v1_idx is None:
                    raise ValueError(f"Edge {edge.index} vertices not found in current_face {current_face.index}")
                
                # Check if traversing v0→v1 is in the cyclic order of current_face
                current_traverses_forward = (current_face_v1_idx == (current_face_v0_idx + 1) % len(current_face.vertices))
                
                # Determine if other_face traverses the edge as v0→v1 or v1→v0
                other_face_v0_idx = None
                other_face_v1_idx = None
                for i, v in enumerate(other_face.vertices):
                    if v.index == edge.vertices[0].index:
                        other_face_v0_idx = i
                    if v.index == edge.vertices[1].index:
                        other_face_v1_idx = i
                
                if other_face_v0_idx is None or other_face_v1_idx is None:
                    raise ValueError(f"Edge {edge.index} vertices not found in other_face {other_face.index}")
                
                # Check if traversing v0→v1 is in the cyclic order of other_face
                other_traverses_forward = (other_face_v1_idx == (other_face_v0_idx + 1) % len(other_face.vertices))
                
                # For consistent winding: if current_face traverses forward, other_face should traverse backward
                if current_traverses_forward == other_traverses_forward:
                    # Winding is inconsistent - need to flip other_face's vertex order
                    other_face.vertices.reverse()
                    other_face.edges.reverse()

    poly = RegularFacedPolyhedron(faces=faces, edges=edges, vertices=vertices)

    return poly