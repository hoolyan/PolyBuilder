"""
Symmetry detection for regular-faced polyhedra.

Detects nontrivial automorphisms using graph isomorphism and dihedral matching.
"""

import math
from typing import Optional, Dict, Tuple, Any

import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher

from data_structures import RegularFacedPolyhedron


TAU = 2.0 * math.pi

def _wrap_0_tau(a: float) -> float:
    """Normalize angle to [0, 2π)."""
    a_wrapped = a % TAU
    if a_wrapped < 0:
        a_wrapped += TAU
    return a_wrapped


def _ang_dist(a: float, b: float) -> float:
    """Compute circular distance between two angles."""
    a = _wrap_0_tau(a)
    b = _wrap_0_tau(b)
    diff = abs(a - b)
    return min(diff, TAU - diff)


def build_vertex_graph_with_dihedrals(solution: RegularFacedPolyhedron) -> nx.Graph:
    """
    Build a NetworkX graph where:
      - Nodes: vertices (with attributes valence and face_type_multiset)
      - Edges: edges with dihedral angle attributes (normalized to [0, 2π))
    """
    G = nx.Graph()
    
    for v_idx, v in enumerate(solution.vertices):
        valence = len(v.edges)
        face_types = sorted([len(f.vertices) for f in v.faces])
        face_type_str = str(face_types)
        
        G.add_node(v_idx, valence=valence, face_type_multiset=face_type_str)
    
    for e in solution.edges:
        v0_idx = None
        for vertex_index in range(len(solution.vertices)):
            if solution.vertices[vertex_index].index == e.vertices[0].index:
                v0_idx = vertex_index
                break
        v1_idx = None
        for vertex_index in range(len(solution.vertices)):
            if solution.vertices[vertex_index].index == e.vertices[1].index:
                v1_idx = vertex_index
                break
        if v0_idx is None or v1_idx is None:
            raise ValueError(f"Edge vertices not found in vertex list: {e.vertices[0].index}, {e.vertices[1].index}")
        
        dihedral_normalized = _wrap_0_tau(e.dihedral)
        G.add_edge(v0_idx, v1_idx, dihedral=dihedral_normalized)
    
    return G


def has_nontrivial_automorphism_with_dihedrals(
    poly: RegularFacedPolyhedron,
    *,
    dihedral_tol: float = 1e-12,  # <-- important to be tight
    return_mapping: bool = False,
) -> bool | Tuple[bool, Optional[Dict[int, int]]]:
    """
    Check if the polyhedron has a nontrivial automorphism that preserves:
      - Vertex valences
      - Face type multisets at vertices
      - Dihedral angles (within tolerance)
    
    Args:
      solution: RegularFacedPolyhedron to check
      dihedral_tol: Tolerance for dihedral angle matching
      return_mapping: If True, return the mapping dict; if False, return None for mapping
    
    Returns:
      (has_automorphism, mapping)
      where mapping is None or the vertex permutation dict depending on return_mapping
    """
    H = build_vertex_graph_with_dihedrals(poly)
    identity = {n: n for n in H.nodes()}
    
    def node_match(n1: Dict[str, Any], n2: Dict[str, Any]) -> bool:
        """Match nodes by valence and face type multiset."""
        return (n1['valence'] == n2['valence'] and 
                n1['face_type_multiset'] == n2['face_type_multiset'])
    
    def edge_match(e1: Dict[str, Any], e2: Dict[str, Any]) -> bool:
        """Match edges by dihedral angle (within tolerance)."""
        d1 = e1.get('dihedral', 0.0)
        d2 = e2.get('dihedral', 0.0)
        return _ang_dist(d1, d2) < dihedral_tol
    
    gm = GraphMatcher(H, H, node_match=node_match, edge_match=edge_match)

    for phi in gm.isomorphisms_iter():
        if phi != identity:
            return (True, {int(k): int(v) for k, v in phi.items()}) if return_mapping else True

    return (False, None) if return_mapping else False
