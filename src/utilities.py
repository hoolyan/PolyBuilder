"""
Utility functions for formatting and display.
"""

from collections import Counter
from typing import List, Optional
import numpy as np


def format_face_types(face_str_list: List[str]) -> str:
    """
    Convert a list of face type strings into a human-readable summary.
    
    Example:
        ['triangle', 'triangle', 'square', 'pentagon'] 
        -> "3 triangles, 2 squares, 1 pentagon"
    
    Args:
        face_str_list: List of face type strings (e.g., 'triangle', 'square', etc.)
    
    Returns:
        Formatted string with counts of each face type, sorted by vertex count.
    """
    face_counts = Counter(face_str_list)
    parts = []
    
    # Map face types to vertex counts for sorting
    vertex_count_map = {
        'triangle': 3, 'square': 4, 'pentagon': 5, 'hexagon': 6,
        'heptagon': 7, 'octagon': 8
    }
    
    # Sort by vertex count (ascending), with fallback for n-gons
    def get_vertex_count(face_type):
        if face_type in vertex_count_map:
            return vertex_count_map[face_type]
        # Handle cases like "9-gon", "10-gon", etc.
        if '-gon' in face_type:
            try:
                return int(face_type.split('-')[0])
            except:
                raise ValueError(f"Unrecognized face type: {face_type}")
        raise ValueError(f"Unrecognized face type: {face_type}")
    
    for face_type in sorted(face_counts.keys(), key=get_vertex_count):
        count = face_counts[face_type]
        if count == 1:
            parts.append(f"1 {face_type}")
        else:
            # Pluralize: triangle->triangles, pentagon->pentagons, etc.
            plural = face_type + "s" if not face_type.endswith("s") else face_type
            parts.append(f"{count} {plural}")
    
    return "[" + ", ".join(parts) + "]"


def format_dihedral_degrees(dihedral_set: List[Optional[float]], precision: int = 3) -> List:
    """
    Convert a list of dihedral angles (in radians) to degrees with reduced precision.
    
    Args:
        dihedral_set: List of dihedral angles in radians, or None for unassigned angles.
        precision: Number of decimal places to round to (default: 4).
    
    Returns:
        List of dihedral angles in degrees, rounded to specified precision, or None for unassigned.
    """
    return [round(d * 180.0 / np.pi, precision) if d is not None else None for d in dihedral_set]
