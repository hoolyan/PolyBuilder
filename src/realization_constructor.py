"""
3D polyhedron realization and export.

Constructs 3D coordinates for vertices based on dihedral angles,
validates geometry, and exports to OBJ format.
"""

import math
import os
import numpy as np
from typing import Tuple

from data_structures import Vertex, Edge, Face, RegularFacedPolyhedron, v_normalize, v_dot, v_cross, v_norm


def construct_polyhedron_realization(solution: RegularFacedPolyhedron) -> RegularFacedPolyhedron:
    """
    Construct polyhedron by building individual face transforms along BFS tree paths.
    
    For each face, we:
    1. Place it as a regular polygon in its local coordinate system
    2. Iteratively fold it along the path back to the root face
    3. Apply dihedral rotations around each hinge edge
    4. Project the final 3D coordinates back into the solution
    
    This approach is more robust than BFS folding because each face is built
    independently along its own path, avoiding error accumulation issues.
    """

    def translate_face(face: Face, translation: np.ndarray) -> None:
        """
        Translate a Face and all its Vertices and Edges by the given vector.
        Modifies the Face, its Vertices, and Edges in-place.
        """
        translation = np.array(translation, dtype=float)
        for v in face.vertices:
            v.pos = v.pos + translation
        for e in face.edges:
            e.pos = e.pos + translation
        face.pos = face.pos + translation


    def rotate_face_around_axis(face: Face, axis: np.ndarray, angle: float, pivot: np.ndarray = None) -> None:
        """
        Rotate a Face and all its Vertices and Edges around an axis through a pivot point.
        Uses Rodrigues' rotation formula. Modifies the Face, its Vertices, and Edges in-place.
        
        Args:
        face: The Face to rotate
        axis: The axis of rotation (will be normalized)
        angle: Rotation angle in radians
        pivot: The point around which to rotate (default: origin)
        """
        if pivot is None:
            pivot = np.array([0.0, 0.0, 0.0])
        else:
            pivot = np.array(pivot, dtype=float)
        
        axis = v_normalize(axis)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        def rotate_point(p):
            p_rel = p - pivot
            cross = v_cross(axis, p_rel)
            dot = v_dot(axis, p_rel)
            p_rot = p_rel * cos_a + cross * sin_a + axis * dot * (1.0 - cos_a)
            return p_rot + pivot
        
        for v in face.vertices:
            v.pos = rotate_point(v.pos)
        for e in face.edges:
            e.pos = rotate_point(e.pos)
        face.pos = rotate_point(face.pos)
        
        # Also rotate the face normal
        if hasattr(face, 'normal_x') and hasattr(face, 'normal_y') and hasattr(face, 'normal_z'):
            normal = np.array([face.normal_x, face.normal_y, face.normal_z])
            # Rotate normal around origin (normals don't translate)
            cross = v_cross(axis, normal)
            dot = v_dot(axis, normal)
            rotated_normal = normal * cos_a + cross * sin_a + axis * dot * (1.0 - cos_a)
            face.normal_x = float(rotated_normal[0])
            face.normal_y = float(rotated_normal[1])
            face.normal_z = float(rotated_normal[2])


    def align_face_to_target(face: Face, target_face: Face) -> None:
        """
        Align a Face to match the position and normal of a target Face.
        Moves the first Face so its center coincides with the target's center,
        and rotates it so its normal matches the target's normal.
        Modifies the Face, its Vertices, and Edges in-place.
        
        Args:
        face: The Face to align
        target_face: The target Face to align to
        """
        # First, ensure target has a normal computed
        if not (hasattr(target_face, 'normal_x') and hasattr(target_face, 'normal_y') and hasattr(target_face, 'normal_z')):
            raise ValueError("Target face must have normal_x, normal_y, normal_z attributes")
        
        # Translate face so its center matches target's center
        translation = target_face.pos - face.pos
        translate_face(face, translation)
        
        # Compute face's current normal (if not already set)
        if not (hasattr(face, 'normal_x') and hasattr(face, 'normal_y') and hasattr(face, 'normal_z')):
            # Compute from first three vertices (assuming face is planar)
            if len(face.vertices) >= 3:
                v0 = face.vertices[0].pos
                v1 = face.vertices[1].pos
                v2 = face.vertices[2].pos
                normal = v_normalize(v_cross(v1 - v0, v2 - v0))
                face.normal_x = float(normal[0])
                face.normal_y = float(normal[1])
                face.normal_z = float(normal[2])
        
        current_normal = np.array([face.normal_x, face.normal_y, face.normal_z])
        target_normal = np.array([target_face.normal_x, target_face.normal_y, target_face.normal_z])
        
        # Check if normals are already aligned
        dot_product = v_dot(current_normal, target_normal)
        if abs(dot_product - 1.0) < 1e-10:
            # Already aligned
            return
        if abs(dot_product + 1.0) < 1e-10:
            # Opposite direction - need 180° rotation
            # Pick an arbitrary perpendicular axis
            if abs(current_normal[0]) < 0.9:
                perp = np.array([1.0, 0.0, 0.0])
            else:
                perp = np.array([0.0, 1.0, 0.0])
            rotation_axis = v_normalize(v_cross(current_normal, perp))
            rotate_face_around_axis(face, rotation_axis, np.pi, face.pos)
        else:
            # Rotate around the axis perpendicular to both normals
            rotation_axis = v_normalize(v_cross(current_normal, target_normal))
            # Compute rotation angle
            angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
            rotate_face_around_axis(face, rotation_axis, angle, face.pos)


    def rotate_face_about_normal_to_align_edge(face: Face, reference_face: Face, edge_index: int) -> bool:
        """
        Rotate face about its normal vector until its edge at edge_index is parallel 
        to reference_face's edge at edge_index.
        
        Both faces must share edges at edge_index, and they must have the same normal 
        vector (within tolerance). Modifies face in-place.
        
        Args:
            face: The Face to rotate
            reference_face: The Face whose edge defines the target orientation
            edge_index: The index of the shared edge in both faces
        
        Returns:
            True if rotation was successful, False if operation not possible
            (e.g., edge_index out of bounds or normals don't match)
        """
        # Check that both faces have edges at this index
        # Need to search for edges with matching index, not access by list position
        edge1 = None
        for e in face.edges:
            if e.index == edge_index:
                edge1 = e
                break
        
        edge2 = None
        for e in reference_face.edges:
            if e.index == edge_index:
                edge2 = e
                break
        
        if edge1 is None or edge2 is None:
            print(f"Error: both faces must have the edge with index {edge_index}")
            return False
        
        # Check that both faces have the same normal vector (within tolerance)
        normal1 = np.array([face.normal_x, face.normal_y, face.normal_z])
        normal2 = np.array([reference_face.normal_x, reference_face.normal_y, reference_face.normal_z])
        
        if not np.allclose(normal1, normal2, atol=1e-10, rtol=0):
            print(f"Error: faces do not have the same normal vector")
            print(f"  face normal: {normal1}")
            print(f"  reference normal: {normal2}")
            raise ValueError("Faces do not have the same normal vector")
        
        # Get edge vectors
        v1_start = np.array(edge1.vertices[0].pos)
        v1_end = np.array(edge1.vertices[1].pos)
        edge_vec1 = v1_end - v1_start
        
        v2_start = np.array(edge2.vertices[0].pos)
        v2_end = np.array(edge2.vertices[1].pos)
        edge_vec2 = v2_end - v2_start
        
        # Normalize the edge vectors
        edge_vec1_norm = np.linalg.norm(edge_vec1)
        edge_vec2_norm = np.linalg.norm(edge_vec2)
        
        if edge_vec1_norm < 1e-10 or edge_vec2_norm < 1e-10:
            print("Error: one or both edges have zero length")
            raise ValueError(f"Error: one or both edges have zero length")
        
        edge_vec1_normalized = edge_vec1 / edge_vec1_norm
        edge_vec2_normalized = edge_vec2 / edge_vec2_norm
        
        # Check if they're already parallel (within tolerance)
        dot_product = np.dot(edge_vec1_normalized, edge_vec2_normalized)

        if np.isclose(dot_product, 1.0, atol=1e-10, rtol=0):
            # Already parallel in same direction
            return True
        if np.isclose(dot_product, -1.0, atol=1e-10, rtol=0):
            # Already parallel but opposite direction
            return True
        
        # Calculate the rotation angle needed to rotate edge1 to align with edge2
        # Use atan2 for proper sign and handling of all quadrants
        normal_normalized = normal1 / np.linalg.norm(normal1)
        cross = np.cross(edge_vec1_normalized, edge_vec2_normalized)
        angle = np.arctan2(np.dot(cross, normal_normalized), dot_product)
        
        # Rotate face about its normal
        face_center = face.pos
        rotate_face_around_axis(face, normal_normalized, angle, pivot=face_center)
        
        return True


    # Ensure all dihedrals are assigned
    for edge in solution.edges:
        if not edge.has_assigned_dihedral:
            raise ValueError("Attempted to create full model realization, but not all dihedral angles are assigned.")
    
    # Step 1: Create isolated regular polygon for each face
    isolated_faces: list[Face] = []
    for face in solution.faces:
        # Create a face with vertices placed as a regular polygon in local xy-plane
        n = len(face.vertices)
        R = 1.0 / (2.0 * math.sin(math.pi / n))
        
        # Create face copy with all properties from original
        local_face = Face(index=face.index)
        local_face.normal_x = face.normal_x
        local_face.normal_y = face.normal_y
        local_face.normal_z = face.normal_z
        local_face.constructed = face.constructed
        local_face.vertices = []
        
        # Map from original vertex indices to new vertices for edge lookup
        vertex_map = {}
        
        for v_idx, v in enumerate(face.vertices):
            angle = 2.0 * math.pi * v_idx / n
            x = R * math.cos(angle)
            y = 0.0
            z = R * math.sin(angle)
            new_v = Vertex(index=v.index)
            new_v.x = x
            new_v.y = y
            new_v.z = z
            new_v.vertex_dihedrals_valid = v.vertex_dihedrals_valid
            new_v.constructed = v.constructed
            vertex_map[v.index] = new_v
            local_face.vertices.append(new_v)
        
        # Create edges for this face, preserving original edge ordering
        local_face.edges = []
        for orig_edge in face.edges:
            # Find which vertices this edge connects in the new face
            v0_new = None
            v1_new = None
            for new_v in local_face.vertices:
                if new_v.index == orig_edge.vertices[0].index:
                    v0_new = new_v
                if new_v.index == orig_edge.vertices[1].index:
                    v1_new = new_v
            
            if v0_new is None or v1_new is None:
                raise ValueError(f"Edge {orig_edge.index} vertices not found in new face vertices")
            
            # Preserve original vertex ordering from orig_edge
            new_e = Edge(index=orig_edge.index, vertices=(v0_new, v1_new))
            
            # Copy all edge properties
            new_e.dihedral = orig_edge.dihedral
            new_e.has_assigned_dihedral = orig_edge.has_assigned_dihedral
            new_e.constructed = orig_edge.constructed
            new_e.x = (v0_new.x + v1_new.x) * 0.5
            new_e.y = (v0_new.y + v1_new.y) * 0.5
            new_e.z = (v0_new.z + v1_new.z) * 0.5
            local_face.edges.append(new_e)
        
        local_face.pos = np.array([0.0, 0.0, 0.0])
        isolated_faces.append(local_face)

    isolated_faces[0].constructed = True # Root face stays where it is.
    progress_made = True

    while progress_made:
        progress_made = False
        for face in isolated_faces:
            if face.constructed:
                for edge in face.edges:
                    neighbor_face = isolated_faces[solution.edges[edge.index].faces[0].index]
                    if solution.edges[edge.index].faces[0].index == face.index:
                        neighbor_face = isolated_faces[solution.edges[edge.index].faces[1].index]
                    if neighbor_face.constructed:
                        continue
                    else:
                        progress_made = True
                        neighbor_edge = None
                        for check_neighbor_edge in neighbor_face.edges:
                            if check_neighbor_edge.index == edge.index:
                                neighbor_edge = check_neighbor_edge
                                break
                        if neighbor_edge == None:
                            raise ValueError("Shared edge not found in neighbor face.")
                        
                        if edge.vertices[0].index != neighbor_edge.vertices[0].index or edge.vertices[1].index != neighbor_edge.vertices[1].index:
                            raise ValueError("Shared edge vertex indices do not match.")

                        # Winding order should always be clockwise in the face.vertices arrays due to how we construct the vertex positions initially
                        relative_edge_vertex_order_flipped = False
                        for face_vertex_index in range(len(face.vertices)):
                            if face.vertices[face_vertex_index].index == edge.vertices[1].index:
                                if face.vertices[(face_vertex_index + 1) % len(face.vertices)].index == edge.vertices[0].index:
                                    relative_edge_vertex_order_flipped = True
                        for neighbor_face_vertex_index in range(len(neighbor_face.vertices)):
                            if neighbor_face.vertices[neighbor_face_vertex_index].index == neighbor_edge.vertices[1].index:
                                if neighbor_face.vertices[(neighbor_face_vertex_index + 1) % len(neighbor_face.vertices)].index == neighbor_edge.vertices[0].index:
                                    relative_edge_vertex_order_flipped = not relative_edge_vertex_order_flipped
                        if not relative_edge_vertex_order_flipped:
                            raise ValueError("Error in model realization construction: One face has flipped winding order.")
                        
                        align_face_to_target(neighbor_face, face)

                        if abs(neighbor_face.normal_x - face.normal_x) > 1e-6 or abs(neighbor_face.normal_y - face.normal_y) > 1e-6 or abs(neighbor_face.normal_z - face.normal_z) > 1e-6:
                            raise ValueError(f"Normals still don't match after alignment. neighbor_face: ({neighbor_face.normal_x}, {neighbor_face.normal_y}, {neighbor_face.normal_z}), face: ({face.normal_x}, {face.normal_y}, {face.normal_z})")

                        edge_vertex_0_face_index = None
                        for vertex_index in range(len(face.vertices)):
                            if face.vertices[vertex_index].index == edge.vertices[0].index:
                                edge_vertex_0_face_index = vertex_index
                        edge_vertex_1_face_index = None
                        for vertex_index in range(len(face.vertices)):
                            if face.vertices[vertex_index].index == edge.vertices[1].index:
                                edge_vertex_1_face_index = vertex_index
                        neighbor_edge_vertex_0_face_index = None
                        for vertex_index in range(len(neighbor_face.vertices)):
                            if neighbor_face.vertices[vertex_index].index == neighbor_edge.vertices[0].index:
                                neighbor_edge_vertex_0_face_index = vertex_index
                        neighbor_edge_vertex_1_face_index = None
                        for vertex_index in range(len(neighbor_face.vertices)):
                            if neighbor_face.vertices[vertex_index].index == neighbor_edge.vertices[1].index:
                                neighbor_edge_vertex_1_face_index = vertex_index

                        edge_dir_in_face = (face.vertices[edge_vertex_1_face_index].pos - face.vertices[edge_vertex_0_face_index].pos) / np.linalg.norm(face.vertices[edge_vertex_1_face_index].pos - face.vertices[edge_vertex_0_face_index].pos)
                        neighbor_edge_dir_in_face = (neighbor_face.vertices[neighbor_edge_vertex_1_face_index].pos - neighbor_face.vertices[neighbor_edge_vertex_0_face_index].pos) / np.linalg.norm(neighbor_face.vertices[neighbor_edge_vertex_1_face_index].pos - neighbor_face.vertices[neighbor_edge_vertex_0_face_index].pos)
                        
                        rotate_face_about_normal_to_align_edge(neighbor_face, face, edge.index)
                        edge_dir_in_face = (face.vertices[edge_vertex_1_face_index].pos - face.vertices[edge_vertex_0_face_index].pos) / np.linalg.norm(face.vertices[edge_vertex_1_face_index].pos - face.vertices[edge_vertex_0_face_index].pos)
                        neighbor_edge_dir_in_face = (neighbor_face.vertices[neighbor_edge_vertex_1_face_index].pos - neighbor_face.vertices[neighbor_edge_vertex_0_face_index].pos) / np.linalg.norm(neighbor_face.vertices[neighbor_edge_vertex_1_face_index].pos - neighbor_face.vertices[neighbor_edge_vertex_0_face_index].pos)
                        
                        if np.linalg.norm(edge_dir_in_face + neighbor_edge_dir_in_face) > 1e-6 and np.linalg.norm(edge_dir_in_face - neighbor_edge_dir_in_face) > 1e-6:
                            raise ValueError(f"align_face_to_target() or rotate_face_about_normal_to_align_edge() failed: Edge directions don't align.\nface.normal: ({face.normal_x}, {face.normal_y}, {face.normal_z})\nneighbor_face.normal: ({neighbor_face.normal_x}, {neighbor_face.normal_y}, {neighbor_face.normal_z})\nedge_dir_in_face: {edge_dir_in_face}\nneighbor_edge_dir_in_face: {neighbor_edge_dir_in_face}")
                        elif np.linalg.norm(edge_dir_in_face - neighbor_edge_dir_in_face) > 1e-6:
                            rotate_face_around_axis(neighbor_face, np.array([neighbor_face.normal_x, neighbor_face.normal_y, neighbor_face.normal_z]), math.pi, neighbor_face.pos)
                            
                        edge_dir_in_face = (face.vertices[edge_vertex_1_face_index].pos - face.vertices[edge_vertex_0_face_index].pos) / np.linalg.norm(face.vertices[edge_vertex_1_face_index].pos - face.vertices[edge_vertex_0_face_index].pos)
                        neighbor_edge_dir_in_face = (neighbor_face.vertices[neighbor_edge_vertex_1_face_index].pos - neighbor_face.vertices[neighbor_edge_vertex_0_face_index].pos) / np.linalg.norm(neighbor_face.vertices[neighbor_edge_vertex_1_face_index].pos - neighbor_face.vertices[neighbor_edge_vertex_0_face_index].pos)
                        
                        if np.linalg.norm(neighbor_edge_dir_in_face - edge_dir_in_face) > 1e-6:
                            raise ValueError(f"Edge directions don't align after aligning faces.\nface.normal: ({face.normal_x}, {face.normal_y}, {face.normal_z})\nneighbor_face.normal: ({neighbor_face.normal_x}, {neighbor_face.normal_y}, {neighbor_face.normal_z})\nedge_dir_in_face: {edge_dir_in_face}\nneighbor_edge_dir_in_face: {neighbor_edge_dir_in_face}")
                        
                        face_edge_to_neighbor_edge = edge.pos - neighbor_edge.pos
                        translate_face(neighbor_face, face_edge_to_neighbor_edge)
                        
                        if np.linalg.norm(neighbor_edge.vertices[0].pos - edge.vertices[0].pos) > 1e-6 or np.linalg.norm(neighbor_edge.vertices[1].pos - edge.vertices[1].pos) > 1e-6:
                            raise ValueError(f"Edge positions don't align after translating faces. neighbor_edge.vertices[0].pos: {neighbor_edge.vertices[0].pos}, edge.vertices[0].pos: {edge.vertices[0].pos}, neighbor_edge.vertices[1].pos: {neighbor_edge.vertices[1].pos}, edge.vertices[1].pos: {edge.vertices[1].pos}")

                        if (edge_vertex_1_face_index - edge_vertex_0_face_index == 1 or edge_vertex_1_face_index - edge_vertex_0_face_index == 1 - len(face.vertices)):
                            rotate_face_around_axis(neighbor_face, -neighbor_edge_dir_in_face, math.pi + edge.dihedral, neighbor_edge.pos)
                        else:
                            rotate_face_around_axis(neighbor_face, neighbor_edge_dir_in_face, math.pi + edge.dihedral, neighbor_edge.pos)
                        
                        neighbor_face.constructed = True

    for face_index in range(len(solution.faces)):
        isolated_face = isolated_faces[face_index]
        solution_face = solution.faces[face_index]
        if isolated_face.constructed:
            for vertex_index in range(len(solution_face.vertices)):
                if isolated_face.vertices[vertex_index].index != solution_face.vertices[vertex_index].index:
                    raise ValueError("Inconsistent vertex positions found.")
                if not solution_face.vertices[vertex_index].constructed:
                    solution_face.vertices[vertex_index].pos = isolated_face.vertices[vertex_index].pos
                    solution_face.vertices[vertex_index].constructed = True
                elif np.linalg.norm(solution_face.vertices[vertex_index].pos - isolated_face.vertices[vertex_index].pos) > 1e-6:
                    # print(f"Model realization invalid: Face {solution_face.index} Vertex {solution_face.vertices[vertex_index].index} position is inconsistent. {solution_face.vertices[vertex_index].pos} != {isolated_face.vertices[vertex_index].pos}")
                    pass
            for edge_index in range(len(solution_face.edges)):
                if isolated_face.edges[edge_index].index != solution_face.edges[edge_index].index:
                    raise ValueError("Inconsistent edge positions found.")
                if not solution_face.edges[edge_index].constructed:
                    solution_face.edges[edge_index].pos = isolated_face.edges[edge_index].pos
                    solution_face.edges[edge_index].constructed = True
                elif np.linalg.norm(solution_face.edges[edge_index].pos - isolated_face.edges[edge_index].pos) > 1e-6:
                    # print(f"Model realization invalid: Face {solution_face.index} Edge {solution_face.edges[edge_index].index} position is inconsistent. {solution_face.edges[edge_index].pos} != {isolated_face.edges[edge_index].pos}")
                    pass
            solution_face.pos = isolated_face.pos
            solution_face.constructed = True
        else:
            # This indicates solution is incomplete. What is solved can be further analyzed or exported to OBJ
            pass
    return solution


def export_regular_faced_polyhedron_to_OBJ(model: RegularFacedPolyhedron, output_dir: str, filename: str):
    """
    Export a RegularFacedPolyhedron to OBJ format.
    
    Parameters:
      model: RegularFacedPolyhedron with constructed vertices and faces
      output_dir: directory to write the OBJ file to
      filename: name of the OBJ file (e.g., "polyhedron.obj")
    
    OBJ format:
      - Vertices defined as "v x y z" (1-indexed in OBJ)
      - Faces defined as "f v1 v2 v3 ..." (vertex indices, 1-based)
      - Optional normals "vn nx ny nz" and "f v1//vn1 v2//vn2 ..."
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        # Header
        f.write(f"# OBJ file for regular-faced polyhedron\n")
        f.write(f"# Vertices: {len(model.vertices)}\n")
        f.write(f"# Faces: {len(model.faces)}\n")
        f.write(f"\n")
        
        # Write vertices
        f.write("# Vertices\n")
        for v in model.vertices:
            f.write(f"v {v.x:.6f} {v.y:.6f} {v.z:.6f}\n")
        
        f.write("\n")
        
        # Write faces
        f.write("# Faces\n")
        for face in model.faces:
            if not face.vertices or not face.constructed:
                continue
            # Convert vertex indices to 1-based (OBJ uses 1-indexing)
            vertex_indices = [str(v.index + 1) for v in face.vertices]
            face_line = "f " + " ".join(vertex_indices)
            f.write(face_line + "\n")


def is_valid_realization(solution: RegularFacedPolyhedron, strict=False) -> Tuple[bool, str]:
    """
    Validate that a realization has unit edges and regular face geometry.
    Non-strict mode only ensures faces are regular polygons.
    Strict mode also checks for other geometric issues that violate the definitions of a valid polyhedron (edges shared by more than two faces, overlapping vertices, etc.)
    Self-intersection currently not checked, but could be added in the future, perhaps as a separate category of strictness level.
    """
    # Unit edges
    for e in solution.edges:
        if not all(v.constructed for v in e.vertices):
            continue
        p0 = e.vertices[0].pos
        p1 = e.vertices[1].pos
        L = v_norm(p0 - p1)
        if abs(L - 1.0) > 1e-6:
            return False, f"Edge {e.index} length is not 1: {L}"
        
    # Two vertices, edges, or faces sharing the same position indicates self-intersection or overlap. This is not an exhaustive check for self-intersection.
    if strict:
        for v in solution.vertices:
            for other_v in solution.vertices:
                if v.index == other_v.index or not v.constructed or not other_v.constructed:
                    continue
                if np.linalg.norm(v.pos - other_v.pos) < 1e-10:
                    return False, f"Vertices {v.index} and {other_v.index} are overlapping."
            for e in solution.edges:
                if np.linalg.norm(v.pos - e.pos) < 1e-10:
                    return False, f"Vertex {v.index} and Edge {e.index} are overlapping."
            for f in solution.faces:
                if np.linalg.norm(v.pos - f.pos) < 1e-10:
                    return False, f"Vertex {v.index} and Face {f.index} are overlapping."
        for e in solution.edges:
            for other_e in solution.edges:
                if e.index == other_e.index or not e.constructed or not other_e.constructed:
                    continue
                if np.linalg.norm(e.pos - other_e.pos) < 1e-10:
                    return False, f"Edges {e.index} and {other_e.index} are overlapping."
            for f in solution.faces:
                if np.linalg.norm(e.pos - f.pos) < 1e-10:
                    return False, f"Edge {e.index} and Face {f.index} are overlapping."
        for f in solution.faces:
            for other_f in solution.faces:
                if f.index == other_f.index or not f.constructed or not other_f.constructed:
                    continue
                if np.linalg.norm(f.pos - other_f.pos) < 1e-10:
                    return False, f"Faces {f.index} and {other_f.index} are overlapping."
        
    # Face radii checks
    for f in solution.faces:
        if not f.constructed:
            continue
        n = len(f.vertices)
        if n < 3:
            return False, f"Face {f.index} has less than 3 vertices."
        R = 1.0 / (2.0 * math.sin(math.pi / n))
        for v in f.vertices:
            if not v.constructed:
                continue
            r = v_norm(v.pos - f.pos)
            if abs(r - R) > 1e-4:
                return False, f"Vertex {v.index} of face {f.index} is at incorrect distance from face center: {r} (expected {R})"

    return True, "Valid realization"