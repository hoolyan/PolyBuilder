"""
Search over realizations of regular-faced polyhedra for a given set of planar graphs.

- Input: .g6 file from plantri (3-connected planar graphs)
- Args:  F (face count), g6_path, output_path (for logging / OBJ dumps later)

This file sets up:
- Polyhedron / Face / Edge / Vertex data structures
- Conversion from NetworkX planar graph to RegularFacedPolyhedron
- Finding possible dihedral solutions
- Folding the faces according the assigned dihedral angles
- Checking for symmetries
- Exporting solutions to OBJ model
"""

from __future__ import annotations

import gc

import argparse
import math
from typing import List, Tuple

import networkx as nx

from tqdm import tqdm

from utilities import *
from data_structures import *
from checkpoint import *
from dihedral_solver import *
from realization_constructor import *
from symmetry_checker import *


# ============================================================
# Main orchestration and results reporting
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--F", type=int, help="Face count")
    parser.add_argument("--g6_path", type=str, help="Input .g6 file from plantri")
    parser.add_argument("--graph_subset_range", type=int, nargs=2, default=[None, None], help="Range of graph indices to process (0-based, inclusive start, exclusive end)")
    parser.add_argument("--graph_memory", type=str, default="4GB", help="Maximum memory to use for loading graphs (e.g., 16GB, 1024MB). Graphs are loaded in chunks respecting this limit.")
    parser.add_argument("--combination_limit", type=int, default=None, help="Specify a limit on the number of combinations that can be explored during dihedral solution. Graph is rejected if this is exceeded.")
    parser.add_argument("--specify_face_set", type=str, default=None, help="Only process graphs with a specific set of face types. Format: 'sides:count,sides:count,...' (e.g., '3:8,4:3,5:2' for 8 triangles, 3 squares, 2 pentagons)")
    parser.add_argument("--allow_coplanar_dihedrals", action="store_true", help="Allow dihedrals of 180 degrees, which correspond to coplanar faces.")
    parser.add_argument("--disable_overlap_check", action="store_true", help="Disable the check for overlapping features during construction.")
    parser.add_argument("--perform_asymmetry_check", action="store_true", help="Perform the check for symmetries in the realized polyhedra.")
    parser.add_argument("--show_progress_details", action="store_true", help="Print progress info")
    parser.add_argument("--display_dihedral_solutions", action="store_true", help="Display dihedral solutions for each graph in each category after processing is complete.")
    parser.add_argument("--export_objs", action="store_true", help="Export OBJs of valid realizations")
    parser.add_argument("--export_invalid_objs", action="store_true", help="Export OBJs of invalid realizations")
    parser.add_argument("--output_path", type=str, help="Output path (currently just used for OBJ exports)")
    parser.add_argument("--save_progress", type=str, default=None, help="Path to save progress after every batch (e.g., checkpoint.json)")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from a previously saved progress file (e.g., checkpoint.json)")
    args = parser.parse_args()

    F = args.F
    g6_path = args.g6_path
    graph_subset_start = args.graph_subset_range[0]
    graph_subset_end = args.graph_subset_range[1]
    max_memory_str = args.graph_memory
    specified_face_set_str = args.specify_face_set
    show_progress_details = args.show_progress_details
    display_dihedral_solutions = args.display_dihedral_solutions
    export_objs = args.export_objs
    export_invalid_objs = args.export_invalid_objs
    output_path = args.output_path
    combination_limit = args.combination_limit
    allow_coplanar_dihedrals = args.allow_coplanar_dihedrals
    disable_overlap_check = args.disable_overlap_check
    perform_asymmetry_check = args.perform_asymmetry_check
    resume_from_path = args.resume_from
    save_progress_path = args.save_progress
    if (export_objs or export_invalid_objs) and output_path is None:
        raise ValueError("Output path is required when exporting OBJs")
    # Assume they want to just pick up where they left off, that way they don't have to specify both args and the same path twice. If they really want to specify a different save path, they can still do that by providing both args.
    if (resume_from_path and not save_progress_path):
        save_progress_path = resume_from_path

    if graph_subset_start is None:
        graph_subset_start = 0

    max_memory_bytes = 4 * 1024 * 1024 * 1024 # Default to 4 GB
    if max_memory_str.lower().endswith("gb"):
        max_memory_bytes = int(max_memory_str[:-2]) * 1024 * 1024 * 1024
    elif max_memory_str.lower().endswith("mb"):
        max_memory_bytes = int(max_memory_str[:-2]) * 1024 * 1024
    elif max_memory_str.lower().endswith("kb"):
        max_memory_bytes = int(max_memory_str[:-2]) * 1024
    elif max_memory_str.lower().endswith("b"):
        max_memory_bytes = int(max_memory_str[:-1])
    else:
        raise ValueError("Invalid graph_memory format. Use 'GB', 'MB', 'KB', or 'B', e.g., '16GB', '1024MB'.")
    
    target_face_set = parse_face_set(specified_face_set_str)
    
    print()
    
    # Handle checkpoint resume
    if resume_from_path:
        print(f"Loading checkpoint from {resume_from_path}")
        checkpoint = load_checkpoint(resume_from_path)
        
        # Verify checkpoint settings match current run
        if checkpoint.run_settings.get("F") != F:
            raise ValueError(f"Checkpoint F={checkpoint.run_settings.get('F')} does not match current F={F}")
        if checkpoint.run_settings.get("g6_path") != g6_path:
            raise ValueError(f"Checkpoint g6_path does not match current g6_path")
        
        # Just override graph_subset_start with checkpoint index if checkpoint index is higher to avoid complications.
        if graph_subset_start is None or graph_subset_start == checkpoint.current_graph_index:
            print(f"Resuming from graph index {checkpoint.current_graph_index}")
            graph_subset_start = checkpoint.current_graph_index
        elif graph_subset_start < checkpoint.current_graph_index:
            # Don't tell them we're ignoring their provided graph_subset_start, presumably they just want to resume from the checkpoint index and don't care about the provided graph_subset_start. This is to avoid unnecessary confirmation prompts when resuming.
            graph_subset_start = checkpoint.current_graph_index
        elif graph_subset_start > checkpoint.current_graph_index:
            ok = input(f"Warning: Provided graph subset start index {graph_subset_start} is after checkpoint index {checkpoint.current_graph_index}. Proceed? (y/n) ")
            if ok.lower() != 'y':
                exit(1)

        if show_progress_details: 
            print(f"Loaded checkpoint with:")
            print(f"  - {len(checkpoint.graphs_unsolved)} unsolved graphs")
            print(f"  - {len(checkpoint.graphs_with_dihedral_solutions)} graphs with dihedral solutions")
            print(f"  - {len(checkpoint.graphs_with_realizations)} graphs with realizations")
            print(f"  - {len(checkpoint.graphs_with_asymmetric_realizations)} graphs with asymmetric realizations")
    
    print("Reading graphs from", g6_path)

    # Graphs are loaded in chunks respecting max_memory limit
    # This outer loop handles multiple chunks if needed
    graphs: List[nx.Graph] = []
    graphs_unsolved: List[nx.Graph] = []
    graphs_with_dihedral_solutions: List[nx.Graph] = []
    graphs_with_realizations: List[nx.Graph] = []
    graphs_with_asymmetric_realizations: List[nx.Graph] = []

    # Additional info associated with each graph of interest: (index, face types, dihedral solutions)
    graphs_unsolved_indices_faces_and_solutions: List[Tuple[int, List[str], List[List[float]]]] = []
    graphs_with_dihedral_solutions_indices_faces_and_solutions: List[Tuple[int, List[str], List[List[float]]]] = []
    graphs_with_realizations_indices_faces_and_solutions: List[Tuple[int, List[str], List[List[float]]]] = []
    graphs_with_asymmetric_realizations_indices_faces_and_solutions: List[Tuple[int, List[str], List[List[float]]]] = []
    
    # Restore from checkpoint if resuming
    if resume_from_path:
        graphs_unsolved_indices_faces_and_solutions = list(checkpoint.graphs_unsolved)
        graphs_with_dihedral_solutions_indices_faces_and_solutions = list(checkpoint.graphs_with_dihedral_solutions)
        graphs_with_realizations_indices_faces_and_solutions = list(checkpoint.graphs_with_realizations)
        graphs_with_asymmetric_realizations_indices_faces_and_solutions = list(checkpoint.graphs_with_asymmetric_realizations)

    # Load and process graphs in chunks
    current_index = graph_subset_start
    filtered_graphs = 0
    chunk_index = 0
    pbar = None
    total_graph_count = None
    
    while True:
        if chunk_index > 0:
            # Save progress each chunk, useful in case of dry spells.
            if save_progress_path:
                checkpoint_data = CheckpointData(
                    run_settings={
                        "F": F,
                        "g6_path": g6_path,
                        "combination_limit": combination_limit,
                        "export_objs": export_objs,
                        "show_progress_details": show_progress_details,
                    },
                    current_graph_index=next_index,
                    graphs_unsolved=graphs_unsolved_indices_faces_and_solutions,
                    graphs_with_dihedral_solutions=graphs_with_dihedral_solutions_indices_faces_and_solutions,
                    graphs_with_realizations=graphs_with_realizations_indices_faces_and_solutions,
                    graphs_with_asymmetric_realizations=graphs_with_asymmetric_realizations_indices_faces_and_solutions,
                )
                save_checkpoint(checkpoint_data, save_progress_path)
        
        if graphs:
            del graphs
            gc.collect()
        
        # Suspend progress bar during chunk loading to avoid conflicts with loading progress display
        if pbar:
            pbar.disable = True
        
        graphs, next_index, hit_memory_limit = read_g6_graphs(g6_path, current_index, graph_subset_end, max_memory_bytes, chunk_index)
        
        # Resume progress bar after loading
        if pbar:
            pbar.disable = False
        
        if not graphs:
            if pbar:
                pbar.close()
            if hit_memory_limit:
                print(f"Graph processing failed: Graph memory allocation of {max_memory_bytes} bytes is too low to load next graph. Consider increasing the graph_memory setting.")
                exit(1)
            if show_progress_details:
                print("No more graphs to process")
            break
        
        # On first chunk, determine if we need scouting or can use this chunk's size
        if chunk_index == 0:
            if not hit_memory_limit:
                # Entire range fits in first chunk, no need to scout
                print(f"Loaded {len(graphs)} graphs.")
                total_graph_count = len(graphs)
            else:
                # Hit memory limit, need to scout the full range
                total_graph_count = len(graphs) + scout_g6_graph_count(g6_path, next_index, graph_subset_end)
                print(f"Total graphs to process: {total_graph_count}")
                print(f"Loaded graph chunk {chunk_index + 1}: {len(graphs)} graphs")
            
            # Create single progress bar with determined total
            pbar = tqdm(total=total_graph_count, desc="Graphs processed", unit=" graphs")
            pbar.set_postfix(passing_graphs=0, total_realizations=0)
        else:
            tqdm.write(f"Loaded graph chunk {chunk_index + 1}: {len(graphs)} graphs")
        
        for graph_array_index, G in enumerate(graphs):
            gi = graph_array_index + current_index

            V = G.number_of_nodes()

            if V != F:
                # This should never happen if g6 files are correctly generated.
                pbar.close()
                print(f"Failed to process graph. Graph {gi} contains {V} nodes, expected {F}. Ensure input graphs have the correct number of nodes (must equal specified face count).")
                exit(1)

            if target_face_set is not None:
                if target_face_set != scout_face_set(G):
                    pbar.update(1)
                    filtered_graphs += 1
                    continue

            try:
                poly = create_polyhedron_from_graph(G)
            except Exception as e:
                pbar.close()
                print(f"Failed to process graph. Cannot embed graph {gi} as polyhedron: {e}. Ensure input graphs are 3-connected and planar.")
                exit(1)
            
            pbar.update(1)

            # Part 1: Solving for dihedrals

            face_str_list = []
            for face in poly.faces:
                face_type = "invalid"
                if len(face.vertices) == 3:
                    face_type = "triangle"
                elif len(face.vertices) == 4:
                    face_type = "square"
                elif len(face.vertices) == 5:
                    face_type = "pentagon"
                elif len(face.vertices) == 6:
                    face_type = "hexagon"
                elif len(face.vertices) == 7:
                    face_type = "heptagon"
                elif len(face.vertices) == 8:
                    face_type = "octagon"
                else:
                    face_type = str(len(face.vertices)) + "-gon"
                face_str_list.append(face_type)
                
            if show_progress_details: print(f"Graph {gi}: Faces: {format_face_types(face_str_list)}")
            solution_list: List[RegularFacedPolyhedron] = [poly]
            extended = extend_solved_vertices(solution_list)

            if (all(num_solutions(v) == math.inf for v in solution_list[0].vertices)):
                if show_progress_details: print(f"Graph {gi}: unable to solve using vertex-dihedral rigidity propagation")
                graphs_unsolved.append(G)
                graphs_unsolved_indices_faces_and_solutions.append((gi, face_str_list, [[e.dihedral if e.has_assigned_dihedral else None for e in poly.edges] for poly in solution_list]))
                continue

            # iterate until no changes
            extend_step = 0
            while extended is not solution_list:
                if show_progress_details: print(f"Graph {gi}: solution step {extend_step}, {len(extended)} solutions")
                solution_list = extended
                r_indices_to_remove = []
                for r_index in range(len(solution_list)):
                    r = solution_list[r_index]
                    if not has_valid_dihedrals(r, not allow_coplanar_dihedrals):
                        if show_progress_details: print(f"Graph {gi}: solution failed dihedral solution validity check at step {extend_step}")
                        r_indices_to_remove.append(r_index)
                if not solution_list:
                    break
                for r_index in reversed(r_indices_to_remove):
                    solution_list.pop(r_index)
                extended = extend_solved_vertices(solution_list)

                solution_list_complete = True
                for sol in extended:
                    for v in sol.vertices:
                        if not v.vertex_dihedrals_valid:
                            solution_list_complete = False
                            break
                if not solution_list_complete:
                    if show_progress_details: print(f"Graph {gi}: {len(solution_list)} incomplete solutions after dihedral solution step {extend_step}")
                    if combination_limit is not None and len(solution_list) > combination_limit:
                        if show_progress_details: print(f"Graph {gi} reached combination limit. ({len(solution_list)}) combinations after dihedral solution step {extend_step}. Stopping further exploration of this graph.")
                        r_indices_to_remove = []
                        for sol_index in range(len(extended)):
                            sol = extended[sol_index]
                            for v in sol.vertices:
                                if not v.vertex_dihedrals_valid:
                                    r_indices_to_remove.append(sol_index)
                                    break
                        for r_index in reversed(r_indices_to_remove):
                            extended.pop(r_index)
                        solution_list = extended
                        break
                        
                    for sol in extended:
                        dihedrals = format_dihedral_degrees([e.dihedral if e.has_assigned_dihedral else None for e in sol.edges])
                        dihedrals_assigned = sum(1 for e in sol.edges if e.has_assigned_dihedral)
                        if show_progress_details: print(f"Graph {gi}: {dihedrals_assigned}/{len(sol.edges)} dihedrals assigned in incomplete solution:\n{dihedrals}")
                for s_index, s in enumerate(solution_list):
                    dihedrals = format_dihedral_degrees([e.dihedral if e.has_assigned_dihedral else None for e in s.edges])
                    if show_progress_details: print(f"Graph {gi}, Realization {s_index}: Base solution dihedrals:\n{dihedrals}")
                for s_index, s in enumerate(extended):
                    dihedrals = format_dihedral_degrees([e.dihedral if e.has_assigned_dihedral else None for e in s.edges])
                    if show_progress_details: print(f"Graph {gi}, Realization {s_index}: Extended solution dihedrals:\n{dihedrals}")
                extend_step += 1

            if not solution_list:
                # No possible dihedral assignments -> graph impossible
                if show_progress_details: print(f"Graph {gi}: no realizations with valid dihedral solutions")
                continue

            if show_progress_details: print(f"Graph {gi}: {len(solution_list)} complete solutions with valid dihedral solutions")

            for sol in solution_list:
                dihedrals = format_dihedral_degrees([e.dihedral if e.has_assigned_dihedral else None for e in sol.edges])
                if show_progress_details: print(f"Solution dihedrals:\n{dihedrals}")
            
            # Removing inside-out/mirror duplicates
            duplicate_indices = []
            for r_index in range(len(solution_list)):
                r = solution_list[r_index]
                if any(e.has_assigned_dihedral is False for e in r.edges):
                    continue
                check_is_inside_out_copy = False
                for r_check_index in range(len(solution_list)):
                    if r_check_index == r_index:
                        continue
                    r_check = solution_list[r_check_index]
                    check_is_inside_out_copy = True
                    for d_index in range(len(r.edges)):
                        if not abs(2.0 * math.pi - r.edges[d_index].dihedral - r_check.edges[d_index].dihedral) < 1e-12:
                            check_is_inside_out_copy = False
                            break
                    if check_is_inside_out_copy:
                        break
                r_avg_dihedral = sum(e.dihedral for e in r.edges) / len(r.edges)
                if abs(r_avg_dihedral - math.pi) < 1e-12:
                    for r_check_index in range(len(solution_list)):
                        if r_check_index >= r_index:
                            continue
                        r_check = solution_list[r_check_index]
                        if abs(r_avg_dihedral - sum(e.dihedral for e in r_check.edges) / len(r_check.edges)) < 1e-12:
                            duplicate_indices.append(r_index)
                            if show_progress_details: print(f"Removing duplicate solution {r_index} with average dihedral {r_avg_dihedral}")
                            break
                elif check_is_inside_out_copy and r_avg_dihedral > math.pi:
                    if show_progress_details: print(f"Removing duplicate solution with average dihedral > pi: {r_avg_dihedral}")
                    dihedrals = format_dihedral_degrees([e.dihedral if e.has_assigned_dihedral else None for e in r.edges])
                    if show_progress_details: print(f"Duplicate solution dihedrals:\n{dihedrals}")
                    duplicate_indices.append(r_index)

            if (any(num_solutions(v) == math.inf for v in solution_list[0].vertices)):
                if show_progress_details: print(f"Graph {gi}: unable to solve using vertex-dihedral rigidity propagation")
                graphs_unsolved.append(G)
                graphs_unsolved_indices_faces_and_solutions.append((gi, face_str_list, [[e.dihedral for e in poly.edges] for poly in solution_list]))
                continue

            for r_index in reversed(duplicate_indices):
                solution_list.pop(r_index)

            graphs_with_dihedral_solutions.append(G)
            graphs_with_dihedral_solutions_indices_faces_and_solutions.append((gi, face_str_list, [[e.dihedral if e.has_assigned_dihedral else None for e in poly.edges] for poly in solution_list]))
            
            tqdm.write(f"Graph {gi} produced {len(solution_list)} dihedral solution" + ("s" if len(solution_list) != 1 else "") + ".")
                
            # At this point, all realizations in the solution_list have solved dihedrals. This does not necessarily mean the polyhedron is constructible. Once folded, there may be edge misalignments or self-intersection.

            # Part 2: Constructing models of realizations

            realized_solutions: List[RegularFacedPolyhedron] = []
            for s_index in range(len(solution_list)):
                s = solution_list[s_index]
                model = construct_polyhedron_realization(s)
                valid, validation_message = is_valid_realization(model, not disable_overlap_check)
                if valid:
                    realized_solutions.append(model)
                else:
                    if show_progress_details: print(f"Realization of graph {gi} solution {s_index} is invalid: {validation_message}")
                    if export_invalid_objs: # Just to see what's wrong with the invalid ones.
                        export_regular_faced_polyhedron_to_OBJ(model, output_path, f"graph_{gi}_invalid_realization_{s_index - len(realized_solutions)}.obj")

            # Part 3: Checking for symmetry and exporting results

            if realized_solutions:
                asymmetric_realizations: List[RegularFacedPolyhedron] = []
                for i, m in enumerate(realized_solutions):
                    symmetric, mapping = has_nontrivial_automorphism_with_dihedrals(m, return_mapping=True)
                    if not symmetric:
                        if show_progress_details: print(f"Graph {gi} realization {i} has no non-trivial symmetries.")
                        asymmetric_realizations.append(m)
                    if export_objs:
                        export_regular_faced_polyhedron_to_OBJ(m, output_path, f"graph_{gi}_realization_{i}.obj")
                graphs_with_realizations.append(G)
                graphs_with_realizations_indices_faces_and_solutions.append((gi, face_str_list, [[e.dihedral if e.has_assigned_dihedral else None for e in poly.edges] for poly in realized_solutions]))
                if len(asymmetric_realizations) > 0:
                    graphs_with_asymmetric_realizations.append(G)
                    graphs_with_asymmetric_realizations_indices_faces_and_solutions.append((gi, face_str_list, [[e.dihedral if e.has_assigned_dihedral else None for e in poly.edges] for poly in asymmetric_realizations]))
                    
            pbar.set_postfix(passing_graphs=len(graphs_with_realizations_indices_faces_and_solutions), total_realizations=sum(len(r_list) for _, _, r_list in graphs_with_realizations_indices_faces_and_solutions))
            
            # Save progress after finding something interesting
            if save_progress_path:
                checkpoint_data = CheckpointData(
                    run_settings={
                        "F": F,
                        "g6_path": g6_path,
                        "combination_limit": combination_limit,
                        "export_objs": export_objs,
                        "show_progress_details": show_progress_details,
                    },
                    current_graph_index=gi + 1,
                    graphs_unsolved=graphs_unsolved_indices_faces_and_solutions,
                    graphs_with_dihedral_solutions=graphs_with_dihedral_solutions_indices_faces_and_solutions,
                    graphs_with_realizations=graphs_with_realizations_indices_faces_and_solutions,
                    graphs_with_asymmetric_realizations=graphs_with_asymmetric_realizations_indices_faces_and_solutions,
                )
                save_checkpoint(checkpoint_data, save_progress_path)
        
        # End of per-graph loop
        current_index = next_index
        chunk_index += 1
        
        # If we've reached the end_index or read fewer graphs than requested, we're done
        if next_index >= (graph_subset_end if graph_subset_end is not None else float('inf')):
            break

    # Close progress bar after all chunks
    if pbar:
        pbar.close()

    # Save progress at the end of the full run
    if save_progress_path:
        checkpoint_data = CheckpointData(
            run_settings={
                "F": F,
                "g6_path": g6_path,
                "combination_limit": combination_limit,
                "export_objs": export_objs,
                "show_progress_details": show_progress_details,
            },
            current_graph_index=next_index,
            graphs_unsolved=graphs_unsolved_indices_faces_and_solutions,
            graphs_with_dihedral_solutions=graphs_with_dihedral_solutions_indices_faces_and_solutions,
            graphs_with_realizations=graphs_with_realizations_indices_faces_and_solutions,
            graphs_with_asymmetric_realizations=graphs_with_asymmetric_realizations_indices_faces_and_solutions,
        )
        save_checkpoint(checkpoint_data, save_progress_path)

    print(f"\nResults:")
    if target_face_set:
        print(f"\nGraphs filtered due to face set mismatch: {filtered_graphs}")
    if (resume_from_path):
        print(f"\nNew unsolved graphs: {len(graphs_unsolved)}")
        print(f"Total unsolved graphs: {len(graphs_unsolved_indices_faces_and_solutions)}")
    else:
        print(f"\nUnsolved graphs: {len(graphs_unsolved_indices_faces_and_solutions)}")
    for gi, face_str_list, dihedrals in graphs_unsolved_indices_faces_and_solutions:
        print(f"Graph {gi} with faces {format_face_types(face_str_list)} could not be solved.")
        if display_dihedral_solutions:
            for r_index, dihedral_set in enumerate(dihedrals):
                dihedral_degrees = format_dihedral_degrees(dihedral_set)
                print(f"Solution {r_index} has dihedrals:\n{dihedral_degrees}")

    if (resume_from_path):
        print(f"\nNew graphs with dihedral solutions: {len(graphs_with_dihedral_solutions)}")
        print(f"Total graphs with dihedral solutions: {len(graphs_with_dihedral_solutions_indices_faces_and_solutions)}")
    else:
        print(f"\nGraphs with dihedral solutions: {len(graphs_with_dihedral_solutions_indices_faces_and_solutions)}")
    for gi, face_str_list, dihedrals in graphs_with_dihedral_solutions_indices_faces_and_solutions:
        print(f"Graph {gi} with faces {format_face_types(face_str_list)} produced {len(dihedrals)} valid dihedral solution" + ("s" if len(dihedrals) != 1 else "") + ".")
        # Debug: display dihedral solutions
        if display_dihedral_solutions:
            for r_index, dihedral_set in enumerate(dihedrals):
                dihedral_degrees = format_dihedral_degrees(dihedral_set)
                print(f"Solution {r_index} has dihedrals:\n{dihedral_degrees}")

    if (resume_from_path):
        print(f"\nNew graphs with realizable solutions: {len(graphs_with_realizations)}")
        print(f"Total graphs with realizable solutions: {len(graphs_with_realizations_indices_faces_and_solutions)}")
    else:
        print(f"\nGraphs with realizable solutions: {len(graphs_with_realizations_indices_faces_and_solutions)}")
    for gi, face_str_list, dihedrals in graphs_with_realizations_indices_faces_and_solutions:
        print(f"Graph {gi} with faces {format_face_types(face_str_list)} produced {len(dihedrals)} realizable solution" + ("s" if len(dihedrals) != 1 else "") + ".")
        if display_dihedral_solutions:
            for r_index, dihedral_set in enumerate(dihedrals):
                dihedral_degrees = format_dihedral_degrees(dihedral_set)
                print(f"Solution {r_index} has dihedrals:\n{dihedral_degrees}")
        
    if perform_asymmetry_check:
        if (resume_from_path):
            print(f"\nNew graphs with asymmetric realizations: {len(graphs_with_asymmetric_realizations)}")
            print(f"Total graphs with asymmetric realizations: {len(graphs_with_asymmetric_realizations_indices_faces_and_solutions)}")
        else:
            print(f"\nGraphs with asymmetric realizations: {len(graphs_with_asymmetric_realizations_indices_faces_and_solutions)}")
        for gi, face_str_list, dihedrals in graphs_with_asymmetric_realizations_indices_faces_and_solutions:
            print(f"Graph {gi} with faces {format_face_types(face_str_list)} produced {len(dihedrals)} asymmetric realization" + ("s" if len(dihedrals) != 1 else "") + ".")
            if display_dihedral_solutions:
                for r_index, dihedral_set in enumerate(dihedrals):
                    dihedral_degrees = format_dihedral_degrees(dihedral_set)
                    print(f"Solution {r_index} has dihedrals:\n{dihedral_degrees}")

    print()

if __name__ == "__main__":
    main()