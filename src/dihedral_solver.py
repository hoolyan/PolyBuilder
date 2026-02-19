"""
Dihedral angle solving for regular-faced polyhedra.

Implements spherical triangulation and dihedral propagation:
- Spherical triangle solving (SAS and SSS methods)
- Spherical triangulation construction and solving
- Dihedral angle calculation and validation
- Vertex solution counting and dihedral branching
"""

from __future__ import annotations

import math
import random
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

from data_structures import Vertex, Edge, Face, RegularFacedPolyhedron


def num_solutions(vertex: Vertex) -> int:
    """
    Return:
      0: no possible completions for remaining dihedrals at this vertex
      1: unique completion
      2: exactly two completions (mirror pair)
      math.inf: not enough info yet to solve
    """
    if len(vertex.edges) < 3: # Something has gone horribly wrong
        raise ValueError("Vertices must be attached to at least 3 edges")
    elif len(vertex.edges) == 3:
        cornerAAngle = math.pi - 2.0 * math.pi/len(vertex.faces[0].vertices)
        cornerBAngle = math.pi - 2.0 * math.pi/len(vertex.faces[1].vertices)
        cornerCAngle = math.pi - 2.0 * math.pi/len(vertex.faces[2].vertices)
        if cornerAAngle + cornerBAngle <= cornerCAngle + 1e-12 or cornerAAngle + cornerCAngle <= cornerBAngle + 1e-12 or cornerBAngle + cornerCAngle <= cornerAAngle + 1e-12: # Simple check to determine if there are no solutions. It's impossible for two connected faces to span far enough to connect to the remaining edges of the third corner either at all, or without becoming coplanar.
            return 0
        if cornerAAngle + cornerBAngle + cornerCAngle >= 2.0 * math.pi - 1e-12:
            return 0
        # Now check for realized dihedrals
        num_assigned_dihedrals = sum(1 for edge in vertex.edges if edge.has_assigned_dihedral)
        if num_assigned_dihedrals > 0:
            return 1 # If at least one dihedral already calculated, no ambiguity. Could check if assigned dihedral is valid at this vertex, since if not, there are really no solutions. But calculating dihedrals aren't really the intended job of this function, just figuring out which solution route to go down.
        return 2
    else:
        num_assigned_dihedrals = sum(1 for edge in vertex.edges if edge.has_assigned_dihedral)
        num_missing_dihedrals = len(vertex.edges) - num_assigned_dihedrals
        if num_missing_dihedrals < 3:
            return 1 # Should always be enough info to unambiguously figure out the remaining dihedrals given the assigned dihedrals are valid.
        elif num_missing_dihedrals == 3:
            return 2 # Except for edge cases, like if any dihedrals are multiples of pi. But those should be caught before extend_solved_vertices() is called.
        else:
            return math.inf # Except for highly specific situations, vertices with >3 missing dihedrals will always have at least one degree of flex.

def clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))

def spherical_triangle_solvable(a, b, c):
    if a is None or b is None or c is None: # Not enough info to determine invalidity
        return True
    if ((abs(a - math.pi) < 1e-12 and abs(b + c - math.pi) < 1e-12) or
        (abs(b - math.pi) < 1e-12 and abs(a + c - math.pi) < 1e-12) or
        (abs(c - math.pi) < 1e-12 and abs(a + b - math.pi) < 1e-12)):
        return False
    return True


def spherical_triangle_arcs_valid(a, b, c):
    """
    Check if the given arc lengths can form a valid spherical triangle.
    Inputs: side lengths a,b,c in (0, pi).
    Returns: True if valid, False otherwise.
    """
    if a is None or b is None or c is None: # Not enough info to determine invalidity
        return True
    a_min_remaining_arc = min(a, 2.0 * math.pi - a)
    b_min_remaining_arc = min(b, 2.0 * math.pi - b)
    c_min_remaining_arc = min(c, 2.0 * math.pi - c)
    if a_min_remaining_arc - b - c >= 1e-12:
        # Not an error case, just means the arcs cannot be oriented to connect to each other.
        return False
    if b_min_remaining_arc - a - c >= 1e-12:
        # Not an error case, just means the arcs cannot be oriented to connect to each other.
        return False
    if c_min_remaining_arc - a - b >= 1e-12:
        # Not an error case, just means the arcs cannot be oriented to connect to each other.
        return False
    if a < 1e-12 or b < 1e-12 or c < 1e-12:
        # Not an error case, just means an arc of the triangulation is 0, therefore the spherical polygon must have 2 vertices in the same location.
        return False
    return True

def spherical_triangle_angles_from_sides(a, b, c):
    """
    Unit-sphere spherical triangle.
    Inputs: side lengths a,b,c in (0, pi).
    Returns: angles A,B,C in [0, pi].
    """
    
    # Denominators: sin(b)sin(c) etc. Must be nonzero.
    sb, sc, sa = math.sin(b), math.sin(c), math.sin(a)
    
    if abs(sb * sc) < 1e-12 or abs(sa * sc) < 1e-12 or abs(sa * sb) < 1e-12:
        print("Invalid spherical triangle SSS input:" )
        print("a =", a, "b =", b, "c =", c)
        raise ValueError("Degenerate spherical triangle (sin terms too small).")
    
    if abs(a + b + c - 2.0 * math.pi) < 1e-12: # coplanar arcs are valid input, but precision issues will cause A, B, and C to be imprecise approximations of pi.
        return math.pi, math.pi, math.pi

    cosA = (math.cos(a) - math.cos(b) * math.cos(c)) / (sb * sc)
    cosB = (math.cos(b) - math.cos(a) * math.cos(c)) / (sa * sc)
    cosC = (math.cos(c) - math.cos(a) * math.cos(b)) / (sa * sb)

    # Clamp for floating noise
    A = math.acos(clamp(cosA))
    B = math.acos(clamp(cosB))
    C = math.acos(clamp(cosC))
    return A, B, C

class SphericalTriangle:
    """
    Representation of a spherical triangle on the unit sphere.

    Attributes:
      a, b, c: side lengths (arc lengths in radians) opposite vertices A, B, C
      A, B, C: angles at vertices A, B, C (in radians)
    """

    vertex_index_A: int
    vertex_index_B: int
    vertex_index_C: int
    internal_angle_A: float = None
    internal_angle_B: float = None
    internal_angle_C: float = None
    arc_index_ab: int
    arc_index_bc: int
    arc_index_ca: int
    arc_ab: float = None
    arc_bc: float = None
    arc_ca: float = None
    solvable = True
    solution_valid = True

    def compute_SSS(self, convex: bool = True): # Solve for angles given arcs. If convex is False, return the concave solution instead of the convex solution.
        if not (self.arc_ab is not None and self.arc_bc is not None and self.arc_ca is not None):
            raise ValueError("Insufficient data for SSS computation in spherical triangle.")
        if not spherical_triangle_arcs_valid(self.arc_ab, self.arc_bc, self.arc_ca):
            self.solution_valid = False
        A, B, C = spherical_triangle_angles_from_sides(self.arc_bc, self.arc_ca, self.arc_ab)
        if self.internal_angle_A is not None and (abs(self.internal_angle_A - A) > 1e-12 and abs(self.internal_angle_A - (2.0 * math.pi - A)) > 1e-12):
            self.solution_valid = False # Inconsistent data; this triangle can't be realized with the given arcs and internal angle
        if self.internal_angle_B is not None and (abs(self.internal_angle_B - B) > 1e-12 and abs(self.internal_angle_B - (2.0 * math.pi - B)) > 1e-12):
            self.solution_valid = False # Inconsistent data; this triangle can't be realized with the given arcs and internal angle
        if self.internal_angle_C is not None and (abs(self.internal_angle_C - C) > 1e-12 and abs(self.internal_angle_C - (2.0 * math.pi - C)) > 1e-12):
            self.solution_valid = False # Inconsistent data; this triangle can't be realized with the given arcs and internal angle
        self.internal_angle_A = A
        self.internal_angle_B = B
        self.internal_angle_C = C
        if not convex:
            self.internal_angle_A = 2.0 * math.pi - self.internal_angle_A
            self.internal_angle_B = 2.0 * math.pi - self.internal_angle_B
            self.internal_angle_C = 2.0 * math.pi - self.internal_angle_C

    def compute_SAS(self): # Solve for angles given two arcs and the included angle. Solution will be unambiguous.
        internal_angle_A_defined = self.internal_angle_A is not None
        internal_angle_B_defined = self.internal_angle_B is not None
        internal_angle_C_defined = self.internal_angle_C is not None

        arc_ab_defined = self.arc_ab is not None
        arc_bc_defined = self.arc_bc is not None
        arc_ca_defined = self.arc_ca is not None

        used_angle: float
        used_side_1: float
        used_side_2: float
        use_case = 0
        
        if internal_angle_A_defined and arc_ab_defined and arc_ca_defined:
            used_angle = self.internal_angle_A
            used_side_1 = self.arc_ca
            used_side_2 = self.arc_ab
        elif internal_angle_B_defined and arc_bc_defined and arc_ab_defined:
            used_angle = self.internal_angle_B
            used_side_1 = self.arc_ab
            used_side_2 = self.arc_bc
            use_case = 1
        elif internal_angle_C_defined and arc_ca_defined and arc_bc_defined:
            used_angle = self.internal_angle_C
            used_side_1 = self.arc_bc
            used_side_2 = self.arc_ca
            use_case = 2
        else:
            raise ValueError("Insufficient data for SAS computation in spherical triangle.")
        
        # Use spherical law of cosines to compute missing arc
        angle_clamp = math.floor(used_angle / (2.0 * math.pi))
        used_angle = used_angle - angle_clamp * 2.0 * math.pi
        if math.pi * 2.0 - used_angle < 0: # This should probably never happen
            raise ValueError(f"Spherical triangle internal angle {used_angle} is out of bounds.")
        cos_angle = math.cos(used_side_1) * math.cos(used_side_2) + math.sin(used_side_1) * math.sin(used_side_2) * math.cos(used_angle)
        arc_side_3 = math.acos(clamp(cos_angle))
        concave = False
        if (used_angle > math.pi):
            concave = True
        if use_case == 0:
            if self.arc_bc is not None and (abs(self.arc_bc - arc_side_3) > 1e-12):
                self.solution_valid = False # Arc length mismatch
            self.arc_bc = arc_side_3
            if not spherical_triangle_solvable(self.arc_ab, self.arc_bc, self.arc_ca):
                self.solvable = False
            elif not spherical_triangle_arcs_valid(self.arc_ab, self.arc_bc, self.arc_ca):
                self.solution_valid = False
            else:
                A, B, C = spherical_triangle_angles_from_sides(arc_side_3, self.arc_ca, self.arc_ab)
                if concave:
                    A = 2.0 * math.pi - A
                    B = 2.0 * math.pi - B
                    C = 2.0 * math.pi - C
                if self.internal_angle_A is not None and (abs(self.internal_angle_A - A) > 1e-12):
                    self.solution_valid = False # Inconsistent data; this triangle can't be realized with the given arcs and internal angle
                if self.internal_angle_B is not None and (abs(self.internal_angle_B - B) > 1e-12):
                    self.solution_valid = False # Inconsistent data; this triangle can't be realized with the given arcs and internal angle
                if self.internal_angle_C is not None and (abs(self.internal_angle_C - C) > 1e-12):
                    self.solution_valid = False # Inconsistent data; this triangle can't be realized with the given arcs and internal angle
                self.internal_angle_A = A
                self.internal_angle_B = B
                self.internal_angle_C = C
        elif use_case == 1:
            if self.arc_ca is not None and (abs(self.arc_ca - arc_side_3) > 1e-12):
                self.solution_valid = False # Arc length mismatch
            self.arc_ca = arc_side_3
            if not spherical_triangle_solvable(self.arc_ab, self.arc_bc, self.arc_ca):
                self.solvable = False
            elif not spherical_triangle_arcs_valid(self.arc_ab, self.arc_bc, self.arc_ca):
                self.solution_valid = False
            else:
                A, B, C = spherical_triangle_angles_from_sides(self.arc_bc, arc_side_3, self.arc_ab)
                if concave:
                    A = 2.0 * math.pi - A
                    B = 2.0 * math.pi - B
                    C = 2.0 * math.pi - C
                if self.internal_angle_A is not None and (abs(self.internal_angle_A - A) > 1e-12):
                    self.solution_valid = False # Inconsistent data; this triangle can't be realized with the given arcs and internal angle
                if self.internal_angle_B is not None and (abs(self.internal_angle_B - B) > 1e-12):
                    self.solution_valid = False # Inconsistent data; this triangle can't be realized with the given arcs and internal angle
                if self.internal_angle_C is not None and (abs(self.internal_angle_C - C) > 1e-12):
                    self.solution_valid = False # Inconsistent data; this triangle can't be realized with the given arcs and internal angle
                self.internal_angle_A = A
                self.internal_angle_B = B
                self.internal_angle_C = C
        elif use_case == 2:
            if self.arc_ab is not None and (abs(self.arc_ab - arc_side_3) > 1e-12):
                self.solution_valid = False # Arc length mismatch
            self.arc_ab = arc_side_3
            if not spherical_triangle_solvable(self.arc_ab, self.arc_bc, self.arc_ca):
                self.solvable = False
            elif not spherical_triangle_arcs_valid(self.arc_ab, self.arc_bc, self.arc_ca):
                self.solution_valid = False
            else:
                A, B, C = spherical_triangle_angles_from_sides(self.arc_bc, self.arc_ca, arc_side_3)
                if concave:
                    A = 2.0 * math.pi - A
                    B = 2.0 * math.pi - B
                    C = 2.0 * math.pi - C
                if self.internal_angle_A is not None and (abs(self.internal_angle_A - A) > 1e-12):
                    self.solution_valid = False # Inconsistent data; this triangle can't be realized with the given arcs and internal angle
                if self.internal_angle_B is not None and (abs(self.internal_angle_B - B) > 1e-12):
                    self.solution_valid = False # Inconsistent data; this triangle can't be realized with the given arcs and internal angle
                if self.internal_angle_C is not None and (abs(self.internal_angle_C - C) > 1e-12):
                    self.solution_valid = False # Inconsistent data; this triangle can't be realized with the given arcs and internal angle
                self.internal_angle_A = A
                self.internal_angle_B = B
                self.internal_angle_C = C
    
    def __repr__(self):
        repr_str = (f"\nSphericalTriangle(vertex_index_A={self.vertex_index_A}, vertex_index_B={self.vertex_index_B}, vertex_index_C={self.vertex_index_C}, arc_index_ab={self.arc_index_ab}, arc_index_bc={self.arc_index_bc}, arc_index_ca={self.arc_index_ca}")
        if hasattr(self, 'arc_ab') and self.arc_ab is not None:
            repr_str += f", arc_ab: {self.arc_ab} radians"
        else:
            repr_str += f", arc_ab: None"
        if hasattr(self, 'arc_bc') and self.arc_bc is not None:
            repr_str += f", arc_bc: {self.arc_bc} radians"
        else:
            repr_str += f", arc_bc: None"
        if hasattr(self, 'arc_ca') and self.arc_ca is not None:
            repr_str += f", arc_ca: {self.arc_ca} radians"
        else:
            repr_str += f", arc_ca: None"
        if hasattr(self, 'internal_angle_A') and self.internal_angle_A is not None:
            repr_str += f", internal_angle_A: {self.internal_angle_A} radians"
        else:
            repr_str += f", internal_angle_A: None"
        if hasattr(self, 'internal_angle_B') and self.internal_angle_B is not None:
            repr_str += f", internal_angle_B: {self.internal_angle_B} radians"
        else:
            repr_str += f", internal_angle_B: None"
        if hasattr(self, 'internal_angle_C') and self.internal_angle_C is not None:
            repr_str += f", internal_angle_C: {self.internal_angle_C} radians)"
        else:
            repr_str += f", internal_angle_C: None)"
        
        return repr_str
    
    def __init__(self, vertex_index_A: int, vertex_index_B: int, vertex_index_C: int, arc_index_ab: int, arc_index_bc: int, arc_index_ca: int):
        self.vertex_index_A = vertex_index_A
        self.vertex_index_B = vertex_index_B
        self.vertex_index_C = vertex_index_C
        self.arc_index_ab = arc_index_ab
        self.arc_index_bc = arc_index_bc
        self.arc_index_ca = arc_index_ca


class SphericalTriangulation:
    """
    Abstract representation of a triangulated spherical polygon. Each triangle is represented as a SphericalTriangle object.
    Spherical triangles contain indices for their arcs and vertices to indicated which arcs/vertices of the spherical polygon they correspond to.
    Solving for a spherical triangulation involves filling in the missing arc lengths and internal angles of each spherical triangle.
    Solved arcs and angles are stored in the SphericalTriangle objects.
    Once solved, arcs with a certain index can be common between multiple triangles, and the internal angles at vertices can be summed to determine the dihedral angles at edges of the polyhedron.

    Attributes:
      triangles: list of SphericalTriangle objects forming the triangulation
      dihedrals: list of known dihedral angles (in radians) of the polyhedron corresponding to the spherical polygon's vertices. The sum of internal angles at each vertex must be equal to the dihedral angle at that edge.
    """

    triangles: List[SphericalTriangle]
    dihedrals: List[Optional[float]]
    solvable: bool = True
    solution_valid: bool = True

    def get_dihedral(self, index=0) -> Optional[float]: # Assumes triangulation has been solved. Missing edge dihedral assumed to be at index 0
        """
        After solving the triangulation, compute the dihedral angle at the specified missing edge by summing the internal angles at the corresponding vertex.
        """
        dihedral_angle = None
        for triangle in self.triangles:
            if triangle.vertex_index_A == index:
                if triangle.internal_angle_A is None:
                    raise ValueError("Triangulation not fully solved; missing internal angles.")
                if dihedral_angle is None:
                    dihedral_angle = 0.0
                dihedral_angle += triangle.internal_angle_A
            elif triangle.vertex_index_B == index:
                if triangle.internal_angle_B is None:
                    raise ValueError("Triangulation not fully solved; missing internal angles.")
                if dihedral_angle is None:
                    dihedral_angle = 0.0
                dihedral_angle += triangle.internal_angle_B
            elif triangle.vertex_index_C == index:
                if triangle.internal_angle_C is None:
                    raise ValueError("Triangulation not fully solved; missing internal angles.")
                if dihedral_angle is None:
                    dihedral_angle = 0.0
                dihedral_angle += triangle.internal_angle_C
        if dihedral_angle is not None:
            dihedral_clamp = math.floor(dihedral_angle / (2.0 * math.pi))
            dihedral_angle = dihedral_angle - dihedral_clamp * 2.0 * math.pi
            if math.pi * 2.0 - dihedral_angle < 1e-12:
                dihedral_angle -= math.pi * 2.0
        return dihedral_angle

    def solve_triangulation(self, convex: bool):
        """
        Solve the spherical triangulation by iteratively applying SAS and SSS methods on the triangles until all arcs and internal angles are determined.
        """
        resolved_triangles_indices: List[int] = []
        progress_made = True
        while len(self.triangles) > len(resolved_triangles_indices) and progress_made:
            progress_made = False
            solvable = True
            solution_valid = True
            for triangle_index, triangle in enumerate(self.triangles):
                if triangle_index in resolved_triangles_indices:
                    continue
                if ((triangle.arc_ca is not None and triangle.arc_ab is not None and triangle.internal_angle_A is not None) or
                    (triangle.arc_ab is not None and triangle.arc_bc is not None and triangle.internal_angle_B is not None) or
                    (triangle.arc_bc is not None and triangle.arc_ca is not None and triangle.internal_angle_C is not None)):
                    # Have enough info to solve for the remaining arc and internal angles using SAS, yielding an unambiguous solution for this triangle.
                    triangle.compute_SAS()
                    if not triangle.solution_valid:
                        solution_valid = False
                    progress_made = True
                elif (triangle.arc_ab is not None and triangle.arc_bc is not None and triangle.arc_ca is not None):
                    # Have enough info to solve for the internal angles using SSS, yields ambiguous solutions for this triangle.
                    triangle.compute_SSS(convex)
                    if not triangle.solution_valid:
                        solution_valid = False
                    progress_made = True
                else:
                    if triangle.internal_angle_A is None:
                        if self.dihedrals[triangle.vertex_index_A] is not None:
                            vertex_A_internal_angle_remainder = self.dihedrals[triangle.vertex_index_A]
                            vertex_A_internal_angle_solvable = True
                            for other_triangle in self.triangles:
                                if other_triangle == triangle:
                                    continue
                                if other_triangle.vertex_index_A == triangle.vertex_index_A:
                                    if other_triangle.internal_angle_A is not None:
                                        vertex_A_internal_angle_remainder -= other_triangle.internal_angle_A
                                    else:
                                        vertex_A_internal_angle_solvable = False
                                        break
                                if other_triangle.vertex_index_B == triangle.vertex_index_A:
                                    if other_triangle.internal_angle_B is not None:
                                        vertex_A_internal_angle_remainder -= other_triangle.internal_angle_B
                                    else:
                                        vertex_A_internal_angle_solvable = False
                                        break
                                if other_triangle.vertex_index_C == triangle.vertex_index_A:
                                    if other_triangle.internal_angle_C is not None:
                                        vertex_A_internal_angle_remainder -= other_triangle.internal_angle_C
                                    else:
                                        vertex_A_internal_angle_solvable = False
                                        break
                            if vertex_A_internal_angle_solvable:
                                angle_clamp = math.floor(vertex_A_internal_angle_remainder / (2.0 * math.pi))
                                vertex_A_internal_angle_remainder -= angle_clamp * 2.0 * math.pi
                                if math.pi * 2.0 - vertex_A_internal_angle_remainder < 1e-12:
                                    vertex_A_internal_angle_remainder -= math.pi * 2.0
                                triangle.internal_angle_A = vertex_A_internal_angle_remainder
                                if triangle.internal_angle_A < 1e-12:
                                    triangle.solution_valid = False
                                    solution_valid = False
                                progress_made = True
                    if triangle.internal_angle_B is None:
                        if self.dihedrals[triangle.vertex_index_B] is not None:
                            vertex_B_internal_angle_remainder = self.dihedrals[triangle.vertex_index_B]
                            vertex_B_internal_angle_solvable = True
                            for other_triangle in self.triangles:
                                if other_triangle == triangle:
                                    continue
                                if other_triangle.vertex_index_A == triangle.vertex_index_B:
                                    if other_triangle.internal_angle_A is not None:
                                        vertex_B_internal_angle_remainder -= other_triangle.internal_angle_A
                                    else:
                                        vertex_B_internal_angle_solvable = False
                                        break
                                if other_triangle.vertex_index_B == triangle.vertex_index_B:
                                    if other_triangle.internal_angle_B is not None:
                                        vertex_B_internal_angle_remainder -= other_triangle.internal_angle_B
                                    else:
                                        vertex_B_internal_angle_solvable = False
                                        break
                                if other_triangle.vertex_index_C == triangle.vertex_index_B:
                                    if other_triangle.internal_angle_C is not None:
                                        vertex_B_internal_angle_remainder -= other_triangle.internal_angle_C
                                    else:
                                        vertex_B_internal_angle_solvable = False
                                        break
                            if vertex_B_internal_angle_solvable:
                                angle_clamp = math.floor(vertex_B_internal_angle_remainder / (2.0 * math.pi))
                                vertex_B_internal_angle_remainder -= angle_clamp * 2.0 * math.pi
                                if math.pi * 2.0 - vertex_B_internal_angle_remainder < 1e-12:
                                    vertex_B_internal_angle_remainder -= math.pi * 2.0
                                triangle.internal_angle_B = vertex_B_internal_angle_remainder
                                if triangle.internal_angle_B < 1e-12:
                                    triangle.solution_valid = False
                                    solution_valid = False
                                progress_made = True
                    if triangle.internal_angle_C is None:
                        if self.dihedrals[triangle.vertex_index_C] is not None:
                            vertex_C_internal_angle_remainder = self.dihedrals[triangle.vertex_index_C]
                            vertex_C_internal_angle_solvable = True
                            for other_triangle in self.triangles:
                                if other_triangle == triangle:
                                    continue
                                if other_triangle.vertex_index_A == triangle.vertex_index_C:
                                    if other_triangle.internal_angle_A is not None:
                                        vertex_C_internal_angle_remainder -= other_triangle.internal_angle_A
                                    else:
                                        vertex_C_internal_angle_solvable = False
                                        break
                                if other_triangle.vertex_index_B == triangle.vertex_index_C:
                                    if other_triangle.internal_angle_B is not None:
                                        vertex_C_internal_angle_remainder -= other_triangle.internal_angle_B
                                    else:
                                        vertex_C_internal_angle_solvable = False
                                        break
                                if other_triangle.vertex_index_C == triangle.vertex_index_C:
                                    if other_triangle.internal_angle_C is not None:
                                        vertex_C_internal_angle_remainder -= other_triangle.internal_angle_C
                                    else:
                                        vertex_C_internal_angle_solvable = False
                                        break
                            if vertex_C_internal_angle_solvable:
                                angle_clamp = math.floor(vertex_C_internal_angle_remainder / (2.0 * math.pi))
                                vertex_C_internal_angle_remainder -= angle_clamp * 2.0 * math.pi
                                if math.pi * 2.0 - vertex_C_internal_angle_remainder < 1e-12:
                                    vertex_C_internal_angle_remainder -= math.pi * 2.0
                                triangle.internal_angle_C = vertex_C_internal_angle_remainder
                                if triangle.internal_angle_C < 1e-12:
                                    triangle.solution_valid = False
                                    solution_valid = False
                                progress_made = True
                    if triangle.arc_ab is None:
                        for other_triangle in self.triangles:
                            if other_triangle == triangle:
                                continue
                            if other_triangle.arc_index_ab == triangle.arc_index_ab:
                                if other_triangle.arc_ab is not None:
                                    triangle.arc_ab = other_triangle.arc_ab
                                    progress_made = True
                                    break
                            if other_triangle.arc_index_bc == triangle.arc_index_ab:
                                if other_triangle.arc_bc is not None:
                                    triangle.arc_ab = other_triangle.arc_bc
                                    progress_made = True
                                    break
                            if other_triangle.arc_index_ca == triangle.arc_index_ab:
                                if other_triangle.arc_ca is not None:
                                    triangle.arc_ab = other_triangle.arc_ca
                                    progress_made = True
                                    break
                    if triangle.arc_bc is None:
                        for other_triangle in self.triangles:
                            if other_triangle == triangle:
                                continue
                            if other_triangle.arc_index_ab == triangle.arc_index_bc:
                                if other_triangle.arc_ab is not None:
                                    triangle.arc_bc = other_triangle.arc_ab
                                    progress_made = True
                                    break
                            if other_triangle.arc_index_bc == triangle.arc_index_bc:
                                if other_triangle.arc_bc is not None:
                                    triangle.arc_bc = other_triangle.arc_bc
                                    progress_made = True
                                    break
                            if other_triangle.arc_index_ca == triangle.arc_index_bc:
                                if other_triangle.arc_ca is not None:
                                    triangle.arc_bc = other_triangle.arc_ca
                                    progress_made = True
                                    break
                    if triangle.arc_ca is None:
                        for other_triangle in self.triangles:
                            if other_triangle == triangle:
                                continue
                            if other_triangle.arc_index_ab == triangle.arc_index_ca:
                                if other_triangle.arc_ab is not None:
                                    triangle.arc_ca = other_triangle.arc_ab
                                    progress_made = True
                                    break
                            if other_triangle.arc_index_bc == triangle.arc_index_ca:
                                if other_triangle.arc_bc is not None:
                                    triangle.arc_ca = other_triangle.arc_bc
                                    progress_made = True
                                    break
                            if other_triangle.arc_index_ca == triangle.arc_index_ca:
                                if other_triangle.arc_ca is not None:
                                    triangle.arc_ca = other_triangle.arc_ca
                                    progress_made = True
                                    break
                if not triangle.solvable:
                    solvable = False
                    break
                elif ((triangle.arc_ca is not None and
                       triangle.arc_ab is not None and
                       triangle.arc_bc is not None and
                       triangle.internal_angle_A is not None and
                       triangle.internal_angle_B is not None and
                       triangle.internal_angle_C is not None)):
                    resolved_triangles_indices.append(triangle_index)
            if not solvable or not solution_valid:
                self.solvable = solvable
                self.solution_valid = solution_valid
                break
        if not progress_made:
            raise ValueError("Solve incomplete. Triangles after solve: " + str(self.triangles))
        
    def __repr__(self):
        return f"SphericalTriangulation(triangles={self.triangles})"

    def __init__(self, polyhedron_vertex: Vertex, missing_edge: Edge):
        # Initialize instance lists to avoid sharing mutable defaults across instances
        self.triangles = []
        self.dihedrals = []

        if missing_edge not in polyhedron_vertex.edges:
            raise ValueError("Missing edge is not incident to the given vertex")
        if missing_edge.has_assigned_dihedral:
            raise ValueError("Missing edge already has assigned dihedral; cannot build triangulation")
        num_missing_dihedrals = sum(1 for edge in polyhedron_vertex.edges if not edge.has_assigned_dihedral)
        if num_missing_dihedrals > 3:
            raise ValueError("Too many unknown dihedrals to build spherical triangulation")
        
        for edge in polyhedron_vertex.edges:
            if not edge.has_assigned_dihedral:
                self.dihedrals.append(None)
            else:
                self.dihedrals.append(edge.dihedral)
        
        missing_edge_index = polyhedron_vertex.edges.index(missing_edge)
        polyhedron_vertex_degree = len(polyhedron_vertex.edges)

        spherical_polygon_A_vertex_index = missing_edge_index
        spherical_polygon_B_vertex_index = (missing_edge_index + 1) % polyhedron_vertex_degree
        spherical_polygon_C_vertex_index = (missing_edge_index + 2) % polyhedron_vertex_degree
        current_arc_ab_index = None
        current_arc_bc_index = None
        current_arc_ca_index = None
        internal_arc_index = max(e.index for e in polyhedron_vertex.edges) + 1

        for face in polyhedron_vertex.edges[spherical_polygon_A_vertex_index].faces:
            if face in polyhedron_vertex.edges[spherical_polygon_B_vertex_index].faces:
                spherical_polygon_arc_ab = math.pi - 2.0 * math.pi / len(face.vertices)
                current_arc_ab_index = polyhedron_vertex.faces.index(face)
                break

        spherical_polygon_arc_bc = None
        for face in polyhedron_vertex.edges[spherical_polygon_B_vertex_index].faces:
            if face in polyhedron_vertex.edges[spherical_polygon_C_vertex_index].faces:
                spherical_polygon_arc_bc = math.pi - 2.0 * math.pi / len(face.vertices)
                current_arc_bc_index = polyhedron_vertex.faces.index(face)
                break

        spherical_polygon_arc_ca = None
        for face in polyhedron_vertex.edges[spherical_polygon_C_vertex_index].faces:
            if face in polyhedron_vertex.edges[spherical_polygon_A_vertex_index].faces:
                spherical_polygon_arc_ca = math.pi - 2.0 * math.pi / len(face.vertices)
                current_arc_ca_index = polyhedron_vertex.faces.index(face)
                break

        root_vertex_index_fan_1 = spherical_polygon_A_vertex_index
        root_vertex_index_fan_2 = spherical_polygon_B_vertex_index
        root_vertex_index_fan_3 = spherical_polygon_C_vertex_index

        central_triangle_arc_index_fan_1 = None
        central_triangle_arc_index_fan_2 = None
        central_triangle_arc_index_fan_3 = None

        triangulation_complete = False

        # First fan
        for face in polyhedron_vertex.edges[spherical_polygon_A_vertex_index].faces:
            if face in polyhedron_vertex.edges[spherical_polygon_B_vertex_index].faces:
                central_triangle_arc_index_fan_1 = polyhedron_vertex.faces.index(face)
                break

        while polyhedron_vertex.edges[spherical_polygon_B_vertex_index].has_assigned_dihedral and not polyhedron_vertex.edges[spherical_polygon_C_vertex_index] == missing_edge:
            
            spherical_polygon_arc_ab = None
            for face in polyhedron_vertex.edges[spherical_polygon_A_vertex_index].faces:
                if face in polyhedron_vertex.edges[spherical_polygon_B_vertex_index].faces:
                    spherical_polygon_arc_ab = math.pi - 2.0 * math.pi / len(face.vertices)
                    current_arc_ab_index = polyhedron_vertex.faces.index(face)
                    break

            spherical_polygon_arc_bc = None
            for face in polyhedron_vertex.edges[spherical_polygon_B_vertex_index].faces:
                if face in polyhedron_vertex.edges[spherical_polygon_C_vertex_index].faces:
                    spherical_polygon_arc_bc = math.pi - 2.0 * math.pi / len(face.vertices)
                    current_arc_bc_index = polyhedron_vertex.faces.index(face)
                    break

            spherical_polygon_arc_ca = None
            for face in polyhedron_vertex.edges[spherical_polygon_C_vertex_index].faces:
                if face in polyhedron_vertex.edges[spherical_polygon_A_vertex_index].faces:
                    spherical_polygon_arc_ca = math.pi - 2.0 * math.pi / len(face.vertices)
                    current_arc_ca_index = polyhedron_vertex.faces.index(face)
                    break
        
            if spherical_polygon_arc_ab is None:
                current_arc_ab_index = internal_arc_index
            if spherical_polygon_arc_bc is None:
                 # The BC arc is always an edge of the spherical polygon in the triple-fan triangulation algorithm. If it's missing, that means the triangulation construction has failed and we can't continue.
                draw_vertex_triangulation(self, out_path=f"tri_error.png")
                print("Error in computing spherical polygon with edges:")
                for face in polyhedron_vertex.faces:
                    print(polyhedron_vertex.faces.index(face), ":",len(face.vertices))
                print("Vertices", spherical_polygon_B_vertex_index, "and", spherical_polygon_C_vertex_index, "must share an edge")
                print("And vertex internal angles:")
                for edge in polyhedron_vertex.edges:
                    print(edge.has_assigned_dihedral)
                    print(edge.dihedral)
                raise ValueError("Spherical polygon BC arcs cannot be missing in spherical triangulation construction. Outputting debug image of current triangulation state to 'tri_error.png'.")
            if spherical_polygon_arc_ca is None:
                internal_arc_index += 1
                current_arc_ca_index = internal_arc_index

            new_triangle = SphericalTriangle(spherical_polygon_A_vertex_index, spherical_polygon_B_vertex_index, spherical_polygon_C_vertex_index, current_arc_ab_index, current_arc_bc_index, current_arc_ca_index)
            
            central_triangle_arc_index_fan_1 = current_arc_ca_index
        
            if spherical_polygon_arc_ca is not None and spherical_polygon_arc_ab is not None and polyhedron_vertex.edges[spherical_polygon_A_vertex_index].has_assigned_dihedral:
                new_triangle.internal_angle_A = polyhedron_vertex.edges[spherical_polygon_A_vertex_index].dihedral
            if spherical_polygon_arc_ab is not None and spherical_polygon_arc_bc is not None and polyhedron_vertex.edges[spherical_polygon_B_vertex_index].has_assigned_dihedral:
                new_triangle.internal_angle_B = polyhedron_vertex.edges[spherical_polygon_B_vertex_index].dihedral
            if spherical_polygon_arc_bc is not None and spherical_polygon_arc_ca is not None and polyhedron_vertex.edges[spherical_polygon_C_vertex_index].has_assigned_dihedral:
                new_triangle.internal_angle_C = polyhedron_vertex.edges[spherical_polygon_C_vertex_index].dihedral

            new_triangle.arc_ab = spherical_polygon_arc_ab
            new_triangle.arc_bc = spherical_polygon_arc_bc
            new_triangle.arc_ca = spherical_polygon_arc_ca
            
            self.triangles.append(new_triangle)
            spherical_polygon_B_vertex_index = spherical_polygon_C_vertex_index
            spherical_polygon_C_vertex_index = (spherical_polygon_C_vertex_index + 1) % polyhedron_vertex_degree

        if spherical_polygon_B_vertex_index == missing_edge_index: # Only 1 unknown dihedral, so first fan reaches missing edge
            triangulation_complete = True
        
        # Second fan
        spherical_polygon_A_vertex_index = spherical_polygon_B_vertex_index
        spherical_polygon_B_vertex_index = (spherical_polygon_A_vertex_index + 1) % polyhedron_vertex_degree
        spherical_polygon_C_vertex_index = (spherical_polygon_B_vertex_index + 1) % polyhedron_vertex_degree
        
        for face in polyhedron_vertex.edges[spherical_polygon_A_vertex_index].faces:
            if face in polyhedron_vertex.edges[spherical_polygon_B_vertex_index].faces:
                central_triangle_arc_index_fan_2 = polyhedron_vertex.faces.index(face)
                break
        
        root_vertex_index_fan_2 = spherical_polygon_A_vertex_index
        
        while polyhedron_vertex.edges[spherical_polygon_B_vertex_index].has_assigned_dihedral and not triangulation_complete:
            
            spherical_polygon_arc_ab = None
            for face in polyhedron_vertex.edges[spherical_polygon_A_vertex_index].faces:
                if face in polyhedron_vertex.edges[spherical_polygon_B_vertex_index].faces:
                    spherical_polygon_arc_ab = math.pi - 2.0 * math.pi / len(face.vertices)
                    current_arc_ab_index = polyhedron_vertex.faces.index(face)
                    break

            spherical_polygon_arc_bc = None
            for face in polyhedron_vertex.edges[spherical_polygon_B_vertex_index].faces:
                if face in polyhedron_vertex.edges[spherical_polygon_C_vertex_index].faces:
                    spherical_polygon_arc_bc = math.pi - 2.0 * math.pi / len(face.vertices)
                    current_arc_bc_index = polyhedron_vertex.faces.index(face)
                    break

            spherical_polygon_arc_ca = None
            for face in polyhedron_vertex.edges[spherical_polygon_C_vertex_index].faces:
                if face in polyhedron_vertex.edges[spherical_polygon_A_vertex_index].faces:
                    spherical_polygon_arc_ca = math.pi - 2.0 * math.pi / len(face.vertices)
                    current_arc_ca_index = polyhedron_vertex.faces.index(face)
                    break
        
            if spherical_polygon_arc_ab is None:
                current_arc_ab_index = internal_arc_index
            if spherical_polygon_arc_bc is None:
                 # The BC arc is always an edge of the spherical polygon in the triple-fan triangulation algorithm. If it's missing, that means the triangulation construction has failed and we can't continue.
                draw_vertex_triangulation(self, out_path=f"tri_error.png")
                print("Error in computing spherical polygon with edges:")
                for face in polyhedron_vertex.faces:
                    print(polyhedron_vertex.faces.index(face), ":",len(face.vertices))
                print("Vertices", spherical_polygon_B_vertex_index, "and", spherical_polygon_C_vertex_index, "must share an edge")
                print("And vertex internal angles:")
                for edge in polyhedron_vertex.edges:
                    print(edge.has_assigned_dihedral)
                    print(edge.dihedral)
                raise ValueError("Spherical polygon BC arcs cannot be missing in spherical triangulation construction. Outputting debug image of current triangulation state to 'tri_error.png'.")
            if spherical_polygon_arc_ca is None:
                internal_arc_index += 1
                current_arc_ca_index = internal_arc_index

            central_triangle_arc_index_fan_2 = current_arc_ca_index

            new_triangle = SphericalTriangle(spherical_polygon_A_vertex_index, spherical_polygon_B_vertex_index, spherical_polygon_C_vertex_index, current_arc_ab_index, current_arc_bc_index, current_arc_ca_index)
        
            if spherical_polygon_arc_ca is not None and spherical_polygon_arc_ab is not None and polyhedron_vertex.edges[spherical_polygon_A_vertex_index].has_assigned_dihedral:
                new_triangle.internal_angle_A = polyhedron_vertex.edges[spherical_polygon_A_vertex_index].dihedral
            if spherical_polygon_arc_ab is not None and spherical_polygon_arc_bc is not None and polyhedron_vertex.edges[spherical_polygon_B_vertex_index].has_assigned_dihedral:
                new_triangle.internal_angle_B = polyhedron_vertex.edges[spherical_polygon_B_vertex_index].dihedral
            if spherical_polygon_arc_bc is not None and spherical_polygon_arc_ca is not None and polyhedron_vertex.edges[spherical_polygon_C_vertex_index].has_assigned_dihedral:
                new_triangle.internal_angle_C = polyhedron_vertex.edges[spherical_polygon_C_vertex_index].dihedral
            
            new_triangle.arc_ab = spherical_polygon_arc_ab
            new_triangle.arc_bc = spherical_polygon_arc_bc
            new_triangle.arc_ca = spherical_polygon_arc_ca
            
            self.triangles.append(new_triangle)
            spherical_polygon_B_vertex_index = spherical_polygon_C_vertex_index
            spherical_polygon_C_vertex_index = (spherical_polygon_C_vertex_index + 1) % polyhedron_vertex_degree
            
        if spherical_polygon_B_vertex_index == missing_edge_index: # Only 2 unknown dihedrals, so second fan reaches missing edge
            triangulation_complete = True
        
        # Third fan
        spherical_polygon_A_vertex_index = spherical_polygon_B_vertex_index
        spherical_polygon_B_vertex_index = (spherical_polygon_A_vertex_index + 1) % polyhedron_vertex_degree
        spherical_polygon_C_vertex_index = (spherical_polygon_B_vertex_index + 1) % polyhedron_vertex_degree

        for face in polyhedron_vertex.edges[spherical_polygon_A_vertex_index].faces:
            if face in polyhedron_vertex.edges[spherical_polygon_B_vertex_index].faces:
                central_triangle_arc_index_fan_3 = polyhedron_vertex.faces.index(face)
                break

        root_vertex_index_fan_3 = spherical_polygon_A_vertex_index

        while polyhedron_vertex.edges[spherical_polygon_B_vertex_index].has_assigned_dihedral and not triangulation_complete:
            
            spherical_polygon_arc_ab = None
            for face in polyhedron_vertex.edges[spherical_polygon_A_vertex_index].faces:
                if face in polyhedron_vertex.edges[spherical_polygon_B_vertex_index].faces:
                    spherical_polygon_arc_ab = math.pi - 2.0 * math.pi / len(face.vertices)
                    current_arc_ab_index = polyhedron_vertex.faces.index(face)
                    break

            spherical_polygon_arc_bc = None
            for face in polyhedron_vertex.edges[spherical_polygon_B_vertex_index].faces:
                if face in polyhedron_vertex.edges[spherical_polygon_C_vertex_index].faces:
                    spherical_polygon_arc_bc = math.pi - 2.0 * math.pi / len(face.vertices)
                    current_arc_bc_index = polyhedron_vertex.faces.index(face)
                    break

            spherical_polygon_arc_ca = None
            for face in polyhedron_vertex.edges[spherical_polygon_C_vertex_index].faces:
                if face in polyhedron_vertex.edges[spherical_polygon_A_vertex_index].faces:
                    spherical_polygon_arc_ca = math.pi - 2.0 * math.pi / len(face.vertices)
                    current_arc_ca_index = polyhedron_vertex.faces.index(face)
                    break
        
            if spherical_polygon_arc_ab is None:
                current_arc_ab_index = internal_arc_index
            if spherical_polygon_arc_bc is None:
                 # The BC arc is always an edge of the spherical polygon in the triple-fan triangulation algorithm. If it's missing, that means the triangulation construction has failed and we can't continue.
                draw_vertex_triangulation(self, out_path=f"tri_error.png")
                print("Error in computing spherical polygon with edges:")
                for face in polyhedron_vertex.faces:
                    print(polyhedron_vertex.faces.index(face), ":",len(face.vertices))
                print("Vertices", spherical_polygon_B_vertex_index, "and", spherical_polygon_C_vertex_index, "must share an edge")
                print("And vertex internal angles:")
                for edge in polyhedron_vertex.edges:
                    print(edge.has_assigned_dihedral)
                    print(edge.dihedral)
                raise ValueError("Spherical polygon BC arcs cannot be missing in spherical triangulation construction. Outputting debug image of current triangulation state to 'tri_error.png'.")
            if spherical_polygon_arc_ca is None:
                internal_arc_index += 1
                current_arc_ca_index = internal_arc_index

            new_triangle = SphericalTriangle(spherical_polygon_A_vertex_index, spherical_polygon_B_vertex_index, spherical_polygon_C_vertex_index, current_arc_ab_index, current_arc_bc_index, current_arc_ca_index)
            
            central_triangle_arc_index_fan_3 = current_arc_ca_index
        
            if spherical_polygon_arc_ca is not None and spherical_polygon_arc_ab is not None and polyhedron_vertex.edges[spherical_polygon_A_vertex_index].has_assigned_dihedral:
                new_triangle.internal_angle_A = polyhedron_vertex.edges[spherical_polygon_A_vertex_index].dihedral
            if spherical_polygon_arc_ab is not None and spherical_polygon_arc_bc is not None and polyhedron_vertex.edges[spherical_polygon_B_vertex_index].has_assigned_dihedral:
                new_triangle.internal_angle_B = polyhedron_vertex.edges[spherical_polygon_B_vertex_index].dihedral
            if spherical_polygon_arc_bc is not None and spherical_polygon_arc_ca is not None and polyhedron_vertex.edges[spherical_polygon_C_vertex_index].has_assigned_dihedral:
                new_triangle.internal_angle_C = polyhedron_vertex.edges[spherical_polygon_C_vertex_index].dihedral
        
            new_triangle.arc_ab = spherical_polygon_arc_ab
            new_triangle.arc_bc = spherical_polygon_arc_bc
            new_triangle.arc_ca = spherical_polygon_arc_ca
            
            self.triangles.append(new_triangle)
            spherical_polygon_B_vertex_index = spherical_polygon_C_vertex_index
            spherical_polygon_C_vertex_index = (spherical_polygon_C_vertex_index + 1) % polyhedron_vertex_degree

        if not triangulation_complete: # 3 unknown dihedrals, so need to add central triangle between them which will need to be solved via SSS and then propagate to solve the rest of the fan, in most cases this yields ambiguous solutions.
            central_triangle_vertex_index_A = root_vertex_index_fan_1
            central_triangle_vertex_index_B = root_vertex_index_fan_2
            central_triangle_vertex_index_C = root_vertex_index_fan_3

            central_triangle_arc_ab = None
            central_triangle_arc_ab_index = central_triangle_arc_index_fan_1
            for face in polyhedron_vertex.edges[central_triangle_vertex_index_A].faces:
                if face in polyhedron_vertex.edges[central_triangle_vertex_index_B].faces:
                    central_triangle_arc_ab = math.pi - 2.0 * math.pi / len(face.vertices)
                    central_triangle_arc_ab_index = polyhedron_vertex.faces.index(face)
                    break

            central_triangle_arc_bc = None
            central_triangle_arc_bc_index = central_triangle_arc_index_fan_2
            for face in polyhedron_vertex.edges[central_triangle_vertex_index_B].faces:
                if face in polyhedron_vertex.edges[central_triangle_vertex_index_C].faces:
                    central_triangle_arc_bc = math.pi - 2.0 * math.pi / len(face.vertices)
                    central_triangle_arc_bc_index = polyhedron_vertex.faces.index(face)
                    break

            central_triangle_arc_ca = None
            central_triangle_arc_ca_index = central_triangle_arc_index_fan_3
            for face in polyhedron_vertex.edges[central_triangle_vertex_index_C].faces:
                if face in polyhedron_vertex.edges[central_triangle_vertex_index_A].faces:
                    central_triangle_arc_ca = math.pi - 2.0 * math.pi / len(face.vertices)
                    central_triangle_arc_ca_index = polyhedron_vertex.faces.index(face)
                    break

            central_triangle = SphericalTriangle(central_triangle_vertex_index_A, central_triangle_vertex_index_B, central_triangle_vertex_index_C, central_triangle_arc_ab_index, central_triangle_arc_bc_index, central_triangle_arc_ca_index)
            
            central_triangle.arc_ab = central_triangle_arc_ab
            central_triangle.arc_bc = central_triangle_arc_bc
            central_triangle.arc_ca = central_triangle_arc_ca

            self.triangles.append(central_triangle)
            triangulation_complete = True


def make_test_vertex(n: int = 20, known_count: int = 17, seed: int = 0) -> Vertex:
    """Create a synthetic Vertex with n incident faces and known_count
    assigned dihedrals at random positions. Faces get random polygon
    sizes between 3 and 8. This is only for debugging triangulation.
    """
    random.seed(seed)
    v = Vertex(index=0)
    faces: List[Face] = []
    for i in range(n):
        sides = random.randint(3, 8)
        f = Face(index=i)
        # place dummy vertices list used only for computing face size
        f.vertices = [None] * sides
        faces.append(f)

    # create circular edges connecting faces[i] and faces[i+1]
    edges: List[Edge] = []
    for i in range(n):
        # dummy geometric vertices for Edge constructor
        va = Vertex(index=1000 + 2 * i)
        vb = Vertex(index=1000 + 2 * i + 1)
        e = Edge(index=i, vertices=(va, vb))
        e.faces = [faces[i], faces[(i + 1) % n]]
        faces[i].edges.append(e)
        faces[(i + 1) % n].edges.append(e)
        edges.append(e)

    # attach faces and edges to central vertex
    v.faces = faces
    v.edges = edges

    # choose known dihedral indices
    known_idxs = set(random.sample(range(n), known_count))
    for i, e in enumerate(edges):
        print(f"Edge {i} assigned known dihedral: {i in known_idxs}")
        if i in known_idxs:
            e.has_assigned_dihedral = True
            e.dihedral = math.radians(random.uniform(10.0, 170.0))
        else:
            e.has_assigned_dihedral = False

    return v


def draw_vertex_triangulation(triangulation: SphericalTriangulation, out_path: str = "tri_debug.png", missing_edge: Optional[Edge] = None) -> None:
    """Draw a flat debug visualization of the spherical triangulation for
    `vertex` and save to `out_path`.

    Visualization notes (updated):
      - Polygon edges (arc side lengths) are known and drawn uniformly.
      - Dihedral (angle) unknown/known status is shown at each polygon
        vertex as a colored marker: green = known, red = unknown.
      - If `missing_edge` is provided (or auto-selected), attempt to use
        the ABC fan triangulation built from that missing dihedral; if
        unavailable, fall back to the simple fan triangulation.
    """
    # `triangulation` now stores SphericalTriangle objects and a dihedrals list.
    # Use the dihedrals list to determine known/unknown dihedrals.
    if not hasattr(triangulation, 'dihedrals'):
        raise ValueError("Triangulation object missing 'dihedrals' attribute")

    n = len(triangulation.dihedrals)
    alpha_known: List[bool] = [d is not None for d in triangulation.dihedrals]

    # positions on unit circle for flat layout
    angles = [2.0 * math.pi * i / n for i in range(n)]
    pts = [(math.cos(a), math.sin(a)) for a in angles]

    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.axis('off')

    # draw polygon edges uniformly (side lengths are known)
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        ax.plot([x0, x1], [y0, y1], color='black', linewidth=2)

    # draw triangulation diagonals (use SphericalTriangle objects)
    for sph_tri in triangulation.triangles:
        a = sph_tri.vertex_index_A
        b = sph_tri.vertex_index_B
        c = sph_tri.vertex_index_C
        for u, v in ((a, b), (b, c), (c, a)):
            # skip polygon boundary edges
            if (v == (u + 1) % n) or (u == (v + 1) % n):
                continue
            x0, y0 = pts[u]
            x1, y1 = pts[v]
            ax.plot([x0, x1], [y0, y1], color='gray', linewidth=1, alpha=0.6)

    # annotate vertices: color marker by dihedral-known status
    for i, (x, y) in enumerate(pts):
        color = 'green' if alpha_known[i] else 'red'
        ax.plot(x, y, 'o', color=color, markersize=6)
        ax.text(x * 1.08, y * 1.08, str(i), fontsize=8, ha='center', va='center')

    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()

def calculate_dihedral(vertex: Vertex, missing_edge: Edge) -> Tuple[bool, bool, float]:
    """
    Compute the unique dihedral for `missing_edge` when the vertex is
    in a '1 solution' state using SAS fan triangulation propagation.
    """
    convex_solvable, concave_solvable, convex_valid, concave_valid, convex_solution, concave_solution = calculate_possible_dihedrals(vertex, missing_edge)

    if not convex_solvable and not concave_solvable:
        return False, False, None
    if convex_solvable and not concave_solvable:
        return convex_valid, True, convex_solution
    if not convex_solvable and concave_solvable:
        return concave_valid, True, concave_solution
    if abs(convex_solution - concave_solution) < 1e-12:
        # In the '1 solution' state, SAS propagation should yield a unique value, if it doesn't, that's a problem.
        return convex_valid and concave_valid, True, convex_solution
    else:
        raise ValueError(f"Vertex {vertex.index} is not in '1 solution' state; dihedral is ambiguous")

def calculate_possible_dihedrals(vertex: Vertex, missing_edge: Edge) -> Tuple[bool, bool, bool, bool, float, float]:
    """
    Compute the two mirror dihedrals (convex, concave) for the
    `missing_edge` by performing an SAS fan-triangulation and summing the
    triangle-angle contributions at the target vertex.
    """
    # Use the 3-fan triangulation and perform SAS propagation

    missing_edge_index = vertex.edges.index(missing_edge)
    first_missing_edge_index = vertex.edges.index(next(e for e in vertex.edges if not e.has_assigned_dihedral))

    triangulation_convex = SphericalTriangulation(vertex, vertex.edges[first_missing_edge_index])
    triangulation_convex.solve_triangulation(convex=True)
    convex_solvable = True
    convex_valid = True
    for triangle in triangulation_convex.triangles:
        if not triangle.solvable:
            convex_solvable = False
        if not triangle.solution_valid:
            convex_valid = False
    triangulation_concave = SphericalTriangulation(vertex, vertex.edges[first_missing_edge_index])
    triangulation_concave.solve_triangulation(convex=False)
    concave_solvable = True
    concave_valid = True
    for triangle in triangulation_concave.triangles:
        if not triangle.solvable:
            concave_solvable = False
        if not triangle.solution_valid:
            concave_valid = False
    
    dihedral_convex = 0.0
    if convex_solvable and convex_valid:
        dihedral_convex = triangulation_convex.get_dihedral(missing_edge_index)
    dihedral_concave = 0.0
    if concave_solvable and concave_valid:
        dihedral_concave = triangulation_concave.get_dihedral(missing_edge_index)

    return convex_solvable, concave_solvable, convex_valid, concave_valid, dihedral_convex, dihedral_concave


def extend_solved_vertices(input_solutions: List[RegularFacedPolyhedron]) -> List[RegularFacedPolyhedron]:
    """
    For each partial solution, perform one of the following:
     - find vertices with num_solutions == 0 and discard solution
     - find one not fully realized vertex with num_solutions == 1 and fill in dihedrals
     - find one not fully realized vertex with num_solutions == 2 and create branches
       for each possible dihedral at that vertex
     - return unchanged if all vertices are fully realized
    """
    aggregated: List[RegularFacedPolyhedron] = []
    for in_sol_index, in_sol in enumerate(input_solutions):
        base = in_sol.deep_copy()
        valid = True
        selected_vertex_idx: Optional[int] = None

        for vidx, in_v in enumerate(in_sol.vertices):
            n_solutions = num_solutions(in_v)
            if in_v.vertex_dihedrals_valid:
                continue  # fully solved
            if n_solutions == 0:
                valid = False
                break
            if n_solutions == 1:
                selected_vertex_idx = vidx
            if n_solutions == 2 and selected_vertex_idx is None:
                selected_vertex_idx = vidx
            if n_solutions > 2:
                continue
        if not valid:
            continue
        
        if selected_vertex_idx is None:
            # all vertices fully realized
            aggregated.append(base)
            continue
        
        if selected_vertex_idx is not None:
            v = base.vertices[selected_vertex_idx]
            if sum(1 for e in v.edges if not e.has_assigned_dihedral) == 0:
                # all dihedrals already assigned at this vertex, but vertex_dihedrals_valid is False, so check if those dihedrals are valid at this vertex; if not, discard solution
                v.vertex_dihedrals_valid = True # assume valid until proven otherwise
                for e in v.edges:
                    check_dihedral = e.dihedral
                    e.has_assigned_dihedral = False # temporarily unassign to recalculate
                    e.dihedral = None
                    recalculated_solvable, recalculated_valid, recalculated_dihedral = calculate_dihedral(v, e)
                    e.dihedral = check_dihedral
                    e.has_assigned_dihedral = True # restore assigned status
                    if recalculated_solvable and (not recalculated_valid or abs(recalculated_dihedral - check_dihedral) > 1e-6):
                        v.vertex_dihedrals_valid = False
                        break
                if v.vertex_dihedrals_valid:
                    aggregated.append(base)
            elif num_solutions(base.vertices[selected_vertex_idx]) == 1:
                # fill in dihedrals for selected vertex
                dihedrals_validated = 0
                for e_out in v.edges:
                    if not e_out.has_assigned_dihedral:
                        d_solvable, d_valid, d = calculate_dihedral(v, e_out)
                        e_out.dihedral = d
                        if d_valid:
                            dihedrals_validated += 1
                    else:
                        dihedrals_validated += 1
                if dihedrals_validated == len(v.edges):
                    for e_out in v.edges:
                        e_out.has_assigned_dihedral = True
                    v.vertex_dihedrals_valid = True
                    aggregated.append(base)
            elif num_solutions(base.vertices[selected_vertex_idx]) == 2:
                # create two branches
                conv_branch = base.deep_copy()
                conc_branch = base.deep_copy()
                v_conv = conv_branch.vertices[selected_vertex_idx]
                v_conc = conc_branch.vertices[selected_vertex_idx]
                conv_dihedrals_validated = 0
                conc_dihedrals_validated = 0
                for e_out_idx in range(len(v_conv.edges)):
                    if not v.edges[e_out_idx].has_assigned_dihedral:
                        e_out_conv = v_conv.edges[e_out_idx]
                        e_out_conc = v_conc.edges[e_out_idx]
                        d_conv_solvable, d_conc_solvable, d_conv_valid, d_conc_valid, d_conv, d_conc = calculate_possible_dihedrals(v_conv, e_out_conv)
                        if d_conv is not None:
                            e_out_conv.dihedral = d_conv
                            if d_conv_valid:
                                conv_dihedrals_validated += 1
                        if d_conc is not None:
                            e_out_conc.dihedral = d_conc
                            if d_conc_valid:
                                conc_dihedrals_validated += 1
                    else:
                        conv_dihedrals_validated += 1
                        conc_dihedrals_validated += 1

                if conv_dihedrals_validated == len(v_conv.edges):
                    for e_out in v_conv.edges:
                        e_out.has_assigned_dihedral = True
                    v_conv.vertex_dihedrals_valid = True
                    aggregated.append(conv_branch)
                if conc_dihedrals_validated == len(v_conc.edges):
                    for e_out in v_conc.edges:
                        e_out.has_assigned_dihedral = True
                    v_conc.vertex_dihedrals_valid = True
                    aggregated.append(conc_branch)
    if len(aggregated) < len(input_solutions):
        return aggregated
    for out in aggregated:
        matched = False
        for inp in input_solutions:
            if out.solutions_match(inp):
                matched = True
                break
        if not matched:
            return aggregated
    return input_solutions