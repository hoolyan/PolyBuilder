"""
Checkpoint saving and loading for long-running polyhedron searches.

Allows resuming interrupted runs from a saved checkpoint.
"""

import json
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any


@dataclass
class CheckpointData:
    """Container for all checkpoint state."""
    run_settings: Dict[str, Any]
    current_graph_index: int
    graphs_unsolved: List[Tuple[int, List[str], List[List[float]]]]
    graphs_with_dihedral_solutions: List[Tuple[int, List[str], List[List[float]]]]
    graphs_with_realizations: List[Tuple[int, List[str], List[List[float]]]]
    graphs_with_asymmetric_realizations: List[Tuple[int, List[str], List[List[float]]]]


def _serialize_for_json(data):
    """Convert data structures to JSON-serializable format."""
    if isinstance(data, dict):
        return {k: _serialize_for_json(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [_serialize_for_json(item) for item in data]
    elif isinstance(data, float):
        # Handle special float values
        if data != data:  # NaN check
            return None
        elif data == float('inf'):
            return "Infinity"
        elif data == float('-inf'):
            return "-Infinity"
        return data
    else:
        return data


def _deserialize_from_json(data):
    """Convert JSON data back to Python structures."""
    if isinstance(data, dict):
        return {k: _deserialize_from_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_deserialize_from_json(item) for item in data]
    elif data == "Infinity":
        return float('inf')
    elif data == "-Infinity":
        return float('-inf')
    else:
        return data


def save_checkpoint(
    checkpoint_data: CheckpointData,
    output_path: str
) -> None:
    """
    Save checkpoint data to a JSON file.
    
    Args:
        checkpoint_data: CheckpointData instance with all run state
        output_path: Path to save checkpoint file (e.g., "checkpoint.json")
    """
    # Ensure directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    checkpoint_dict = {
        "run_settings": checkpoint_data.run_settings,
        "current_graph_index": checkpoint_data.current_graph_index,
        "graphs_unsolved": checkpoint_data.graphs_unsolved,
        "graphs_with_dihedral_solutions": checkpoint_data.graphs_with_dihedral_solutions,
        "graphs_with_realizations": checkpoint_data.graphs_with_realizations,
        "graphs_with_asymmetric_realizations": checkpoint_data.graphs_with_asymmetric_realizations,
    }
    
    # Serialize to JSON-compatible format
    serialized = _serialize_for_json(checkpoint_dict)
    
    with open(output_path, 'w') as f:
        json.dump(serialized, f, indent=2)


def load_checkpoint(checkpoint_path: str) -> CheckpointData:
    """
    Load checkpoint data from a JSON file.
    
    Args:
        checkpoint_path: Path to checkpoint file
    
    Returns:
        CheckpointData instance with all saved run state
    
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        json.JSONDecodeError: If checkpoint file is corrupted
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    with open(checkpoint_path, 'r') as f:
        data = json.load(f)
    
    # Deserialize from JSON format
    deserialized = _deserialize_from_json(data)
    
    # Convert lists back to tuples where needed
    def convert_to_tuples(results_list):
        return [
            (item[0], item[1], item[2])
            for item in results_list
        ]
    
    return CheckpointData(
        run_settings=deserialized["run_settings"],
        current_graph_index=deserialized["current_graph_index"],
        graphs_unsolved=convert_to_tuples(deserialized["graphs_unsolved"]),
        graphs_with_dihedral_solutions=convert_to_tuples(deserialized["graphs_with_dihedral_solutions"]),
        graphs_with_realizations=convert_to_tuples(deserialized["graphs_with_realizations"]),
        graphs_with_asymmetric_realizations=convert_to_tuples(deserialized["graphs_with_asymmetric_realizations"]),
    )
