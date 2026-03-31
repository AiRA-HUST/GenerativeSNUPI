import torch
import random
import warnings
import numpy as np
from typing import Dict, Set, List
from collections import defaultdict
from automated_design.ults import *
from automated_design.visual import plot_dna
from networkx.algorithms import tree
from torch_geometric.utils import degree, coalesce, remove_self_loops, to_undirected

# ============================================================================
# SCAFFOLD ROUTING FUNCTIONS
# ============================================================================

def scaffold_routing(CO_index, RBedge_index, SSedge_index, bp_Hindex, bp_Lindex, coords):
    """
    Generate scaffold routing through DNA origami structure.
    
    Args:
        CO_index: Trial Cross-Over edge index [2xm]
        RBedge_index: Regular base-pairs edge index [2xn]
        SSedge_index: Single-Strand edge index [2xk]
        bp_Hindex: Base-pair helix indices
        bp_Lindex: Base-pair loop indices
        coords: Base-pairs coordinates [Nx3] for visualization (optional)
    
    Returns:
        scaf_route: Scaffold routes [num_bps]
        scaf_edge_index: Edge index over scaffold routes [2xn]
        staples_edge_index: Edge index over staples routes [2xm]
    """
    device = RBedge_index.device
    num_bps = int(RBedge_index.amax() + 1)
    
    # ========================================
    # Create loops graph from cross-overs
    # ========================================
    loop_edge_index, _ = remove_self_loops(coalesce(bp_Lindex[CO_index]))
    num_loops = loop_edge_index.amax() + 1
    loop_edge_index = to_undirected(loop_edge_index, num_nodes=num_loops)
    
    # Convert to NetworkX for spanning tree algorithm
    loop_graph = Data(edge_index=loop_edge_index, num_nodes=num_loops)
    loop_graph_nx = to_networkx(loop_graph, to_undirected=True, remove_self_loops=True)
    
    # ========================================
    # Create minimum spanning tree
    # ========================================
    mst = tree.minimum_spanning_edges(loop_graph_nx, algorithm="kruskal", data=False)
    edgelist = list(mst)
    loop_edge_index = torch.tensor(sorted(sorted(e) for e in edgelist), 
                                   dtype=torch.long, device=device).t()

    # ========================================
    # Create scaffold cross-overs from spanning tree
    # ========================================
    scaffold_crossovers = []
    for i in range(loop_edge_index.size(1)):
        # Select cross-over steps between loops
        src_loop, dst_loop = loop_edge_index[:, i]
        
        # Find cross-overs connecting these loops
        mask_src = torch.logical_or(bp_Lindex[CO_index[0]] == src_loop, 
                                   bp_Lindex[CO_index[1]] == src_loop)
        mask_dst = torch.logical_or(bp_Lindex[CO_index[0]] == dst_loop, 
                                   bp_Lindex[CO_index[1]] == dst_loop)
        mask = torch.logical_and(mask_src, mask_dst)
        
        # Keep only first 4 cross-overs (1 scaffold cross-over step)
        scaffold_crossovers.append(CO_index[:, mask][:, :4])
    
    COedge_index_scaf = torch.hstack(scaffold_crossovers)
    
    # ========================================
    # Identify non-scaffold cross-overs
    # ========================================
    bp_mask = torch.ones(num_bps, dtype=torch.bool, device=device)
    bp_mask[COedge_index_scaf.flatten()] = False
    alignCO_index_noscaf = CO_index[:, torch.logical_and(bp_mask[CO_index[0]], 
                                                        bp_mask[CO_index[1]])]
    
    # ========================================
    # Create cycle loop graph for routing
    # ========================================
    # Combine all edge types
    edge_index = torch.hstack((RBedge_index, COedge_index_scaf, SSedge_index))
    deg = degree(edge_index[0])
    
    # Remove regular base-pair edges with degree 3 (internal to helices)
    RBedge_mask = torch.logical_and(deg[RBedge_index[0]] == 3, 
                                   deg[RBedge_index[1]] == 3)
    edge_index_cycle = torch.hstack((RBedge_index[:, ~RBedge_mask], 
                                   COedge_index_scaf, SSedge_index))
    
    # Verify it's a proper cycle (all nodes should have degree 2)
    deg_cycle = degree(edge_index_cycle[0], num_bps)
    if len((deg_cycle != 2).nonzero()) > 0:
        warnings.warn("Warning: error in scaffold routing, it's not one-loop graph")
    
    # ========================================
    # Select nick position
    # ========================================
    print("Selecting nick position for scaffold...")
    manual_select = 0 # int(input("Manual select Nick position [1/0]: "))
    
    if manual_select == 1:
        # Manual selection with visualization
        fig = plot_dna(coords, edge_index=COedge_index_scaf)
        fig.show()
        nick_idx = int(input("Enter Nick position: "))
    else:
        # Automatic selection at furthest helix
        furthest_helix_id, with_cos = get_furthest_helixID(coords, bp_Hindex, alignCO_index_noscaf)
        nick_candidates = (bp_Hindex == furthest_helix_id).nonzero().to(torch.float32)
        nick_idx = int(nick_candidates.amin() + 5) if with_cos else int(nick_candidates.mean())
        
    print(f"Scaffold nick index: {nick_idx}")
    
    # ========================================
    # Generate scaffold route
    # ========================================
    scaf_route = cycle_graph_to_route(edge_index_cycle, nick_idx, num_bps)
    
    return scaf_route, COedge_index_scaf, alignCO_index_noscaf


def cycle_graph_to_route(loop_edge_index, nick_idx, num_bps, second_idx=None):
    """
    Convert a cycle graph to a linear route starting from nick position.
    
    Args:
        loop_edge_index: Edge index of the cycle graph [2xN]
        nick_idx: Starting position (nick point)
        num_bps: Total number of base pairs
        second_idx: Optional second starting position
    
    Returns:
        route: Linear route through the cycle [num_bps]
    """
    src, _ = loop_edge_index
    route = [nick_idx]
    num_iterations = num_bps - 1
    
    # Add second starting point if provided
    if second_idx is not None:
        route.append(second_idx)
        num_iterations = num_bps - 2
    
    # Traverse the cycle graph
    for i in range(num_iterations):
        # Find neighbors of current position
        mask = src == route[-1]
        neighbors = loop_edge_index[-1, mask]
        
        if len(route) > 1:
            # Choose neighbor that's not the previous node
            if neighbors[0] != route[-2]:
                route.append(neighbors[0].item())
            else:
                route.append(neighbors[1].item())
        else:
            # First step: choose any neighbor
            route.append(neighbors[0].item())
    
    return torch.as_tensor(route, dtype=torch.long, device=loop_edge_index.device)


def assign_helix_direction(scaf_route, helix_index):
    """
    Determine helix direction based on scaffold route.
    
    Args:
        scaf_route: Scaffold route [num_bps]
        helix_index: Helix index for each base pair [num_bps]
    
    Returns:
        helix_direction: Direction flag for each helix [num_helices]
                        True = low to high index, False = high to low index
    """
    num_helices = helix_index.amax() + 1
    low_to_high = torch.zeros(num_helices, dtype=torch.bool, device=scaf_route.device)
    
    for helix_id in range(num_helices):
        # Get scaffold route through this helix
        helix_route = scaf_route[helix_index[scaf_route] == helix_id]
        
        # Check if route goes from low to high index
        if helix_route[0] < helix_route[1]:
            low_to_high[helix_id] = True
    
    return low_to_high

# ============================================================================
# STAPLE ROUTING FUNCTIONS
# ============================================================================

def staples_routing(RBedge_index, scCO_index, stCO_index, SS_index, scaf_route, 
                   min_pairing_len=4, coords=None):
    """
    Generate staple routing for DNA origami structure.
    
    Args:
        RBedge_index: Regular base-pair edges [2xn]
        scCO_index: Scaffold cross-over edges [2xm]
        stCO_index: Staple cross-over edges [2xk]
        SS_index: Single-strand edges [2xj]
        scaf_route: Scaffold route [num_bps]
        min_pairing_len: Minimum pairing length for staples
        coords: Base-pair coordinates [Nx3] for visualization (optional)
    
    Returns:
        routes: List of staple routes
        stap_len: Length of each staple route
        bp_stap_index: Base-pair to staple index mapping [N]
    """
    device = RBedge_index.device
    num_bps = scaf_route.size(0)
    
    # ========================================
    # Create staple edge graph
    # ========================================
    staple_edge_index = torch.cat((RBedge_index, stCO_index, SS_index), dim=1)
    deg = degree(staple_edge_index[0, :], num_bps)
    src, dst = staple_edge_index
    
    # Remove internal helix edges (degree 3, adjacent base pairs)
    mask = torch.logical_and(deg[src] == 3, deg[dst] == 3)
    mask = torch.logical_and(mask, torch.abs(dst - src) < 2)
    staple_edge_index = staple_edge_index[:, ~mask]
    
    # ========================================
    # Segment staples into separate routes
    # ========================================
    bp_stap_index = segmentation(staple_edge_index)
    
    # ========================================
    # Identify safe nick positions
    # ========================================
    nicks_mask = Nicks(scaf_route, scCO_index, stCO_index, RBedge_index, SS_index, 
                      min_pairing_len=min_pairing_len)
    
    # ========================================
    # Generate individual staple routes
    # ========================================
    num_staples = bp_stap_index.max() + 1
    routes = []
    
    for stap_id in range(num_staples):
        print(f"Processing staple {stap_id + 1}/{num_staples}")
        
        # Get base pairs belonging to this staple
        bp_mask = bp_stap_index == stap_id
        num_nucleotides = bp_mask.sum().item()
        
        # Get edges for this staple
        edge_mask = torch.logical_and(bp_mask[staple_edge_index[0]], 
                                     bp_mask[staple_edge_index[1]])
        stap_edge_index = staple_edge_index[:, edge_mask]
        stap_bp_list = torch.unique(stap_edge_index.flatten())
        
        # Find valid nick positions for this staple
        nick_candidates = stap_bp_list[nicks_mask[stap_bp_list]]
        
        if nick_candidates.size(0) < 1:
            warnings.warn(f"No valid nick positions found for staple {stap_id}")
            if coords is not None:
                fig = plot_dna(coords, edge_index=stap_edge_index)
                fig.show()
        
        # Try different nick positions and minimum lengths
        min_len_options = torch.tensor([30, 35, 40, 25], dtype=torch.long, device=device)
        
        for attempt in range(100):
            # Select nick position strategy
            if attempt < 25:
                nick_pos = min(nick_candidates) if len(nick_candidates) > 0 else stap_bp_list[0]
            elif attempt < 50:
                nick_pos = max(nick_candidates) if len(nick_candidates) > 0 else stap_bp_list[-1]
            else:
                nick_pos = random.choice(nick_candidates) if len(nick_candidates) > 0 else random.choice(stap_bp_list)
            
            # Generate route from nick position
            route = _generate_staple_route(stap_edge_index, nick_pos, num_nucleotides)
            
            # Design staple segments with length constraints
            min_len = min_len_options[attempt % 4]
            segment_routes, max_segment_len = design_staples(route, nicks_mask, min_len=min_len)
            
            # Check if segments meet length constraints (< 51 nucleotides)
            if max_segment_len < 51:
                break
        
        routes.extend(segment_routes)
    
    # ========================================
    # Reconfigure staple routes
    # ========================================
    routes = reconfig_STroutes(routes, scaf_route)
    
    # ========================================
    # Calculate staple lengths and update indices
    # ========================================
    stap_len = torch.zeros(len(routes), device=device)
    for i, route in enumerate(routes):
        stap_len[i] = len(route)
        bp_stap_index[route] = i
    
    return routes, stap_len, bp_stap_index


def _generate_staple_route(stap_edge_index, start_pos, num_nucleotides):
    """
    Generate a linear route through staple edges starting from a given position.
    
    Args:
        stap_edge_index: Edge index for this staple [2xN]
        start_pos: Starting position
        num_nucleotides: Total number of nucleotides to visit
    
    Returns:
        route: Linear route through the staple
    """
    device = stap_edge_index.device
    route = [start_pos]
    src, dst = stap_edge_index
    
    # Traverse the staple graph
    for _ in range(num_nucleotides - 1):
        # Find neighbors of current position
        mask = src == route[-1]
        neighbors = stap_edge_index[-1, mask]
        
        if len(route) > 1:
            # Choose neighbor that's not the previous node
            if neighbors[0] != route[-2]:
                route.append(neighbors[0].item())
            else:
                route.append(neighbors[1].item())
        else:
            # First step: choose any neighbor
            route.append(neighbors[0].item())
    
    return torch.as_tensor(route, dtype=torch.long, device=device)


def reconfig_STroutes(staple_routes, scaffold_route):
    """
    Reconfigure staple routes to follow 5'->3' direction rule relative to scaffold.
    
    Args:
        staple_routes: List of staple routes
        scaffold_route: Scaffold route for reference
    
    Returns:
        staple_routes: Reconfigured staple routes
    """
    num_staples = len(staple_routes)
    
    for i in range(num_staples):
        route = staple_routes[i]
        bp_first = route[0]
        bp_second = route[1]
        
        # Find positions in scaffold route
        pos_first = (scaffold_route == bp_first).nonzero().item()
        pos_second = (scaffold_route == bp_second).nonzero().item()
        
        # Reverse route if it goes in same direction as scaffold
        if pos_first < pos_second:
            staple_routes[i] = torch.flip(route, dims=[0])
    
    return staple_routes


def design_staples(route, nicks_mask, min_len=20):
    """
    Design staple segments with length constraints.
    
    Args:
        route: Full staple route
        nicks_mask: Mask indicating valid nick positions
        min_len: Minimum segment length
    
    Returns:
        segment_list: List of staple segments
        max_seg_len: Maximum segment length
    """
    route_len = route.size(0)
    nicks = nicks_mask[route]
    segment_list = []
    
    # Only segment if route is longer than 50 nucleotides
    if route_len > 50:
        current_segment = []
        
        # Build segments based on nick positions
        for i in range(route_len):
            current_segment.append(route[i])
            
            # End segment if we've reached minimum length and found a nick
            if len(current_segment) > min_len and nicks[i]:
                segment_list.append(torch.as_tensor(current_segment, dtype=torch.long, device=route.device))
                current_segment = []
            # End segment if we've reached the end of the route
            elif i == (route_len - 1):
                segment_list.append(torch.as_tensor(current_segment, dtype=torch.long, device=route.device))
                current_segment = []
        
        # ========================================
        # Handle final segment if it's too short
        # ========================================
        if len(segment_list) > 1 and segment_list[-1].size(0) < 25:
            final_segment = segment_list[-1]
            
            # Check if final segment has no valid nicks or is very short
            if nicks_mask[final_segment].sum().item() == 0 or final_segment.size(0) < 10:
                # Merge with smaller adjacent segment
                if segment_list[0].size(0) < segment_list[-2].size(0):
                    segment_list[0] = torch.hstack((final_segment, segment_list[0]))
                else:
                    segment_list[-2] = torch.hstack((segment_list[-2], final_segment))
            else:
                # Split final segment at optimal nick position
                nick_indices = nicks_mask[final_segment].nonzero().reshape(-1)
                nick_distances = (nick_indices - final_segment.size(0) / 2).abs()
                optimal_nick_idx = nick_indices[nick_distances.argmin()]
                
                # Split and merge with adjacent segments
                segment_list[0] = torch.hstack((final_segment[optimal_nick_idx:], segment_list[0]))
                segment_list[-2] = torch.hstack((segment_list[-2], final_segment[:optimal_nick_idx]))
            
            # Remove the final segment
            segment_list = segment_list[:-1]
        
        # Calculate maximum segment length
        max_seg_len = max(len(segment) for segment in segment_list)
    else:
        # Keep route as single segment if it's short enough
        segment_list.append(route)
        max_seg_len = route_len
    
    return segment_list, max_seg_len

# ============================================================================
# SEQUENCE ASSIGNMENT FUNCTIONS
# ============================================================================

def SS_to_SeqLen(SSedge_index, scaf_coords, stap_coords, add_seqlen: int = 0):
    """
    Convert single-strand edges to sequence lengths based on geometric distance.
    
    Args:
        SSedge_index: Single-strand edges [2xN]
        scaf_coords: Scaffold base coordinates [nx3]
        stap_coords: Staple base coordinates [nx3]
        add_seqlen: Additional sequence length to add
    
    Returns:
        scaf_SSseq: Scaffold single-strand sequence lengths
        stap_SSseq: Staple single-strand sequence lengths
        SSedge_index_undirected: Undirected single-strand edges
        scaf_SS_coords: Scaffold single-strand coordinates
        stap_SS_coords: Staple single-strand coordinates
    """
    # Get undirected edges (every other edge)
    src, dst = SSedge_index[:, ::2]
    
    # Calculate geometric lengths
    scaf_distance_vectors = scaf_coords[dst, :] - scaf_coords[src, :]
    scaf_lengths = scaf_distance_vectors.norm(dim=-1, keepdim=False)
    
    stap_distance_vectors = stap_coords[dst, :] - stap_coords[src, :]
    stap_lengths = stap_distance_vectors.norm(dim=-1, keepdim=False)
    
    # Convert to sequence lengths (assuming 0.42 nm per nucleotide)
    scaf_SSseq = torch.ceil(scaf_lengths / 0.42) - 1 + add_seqlen
    stap_SSseq = torch.ceil(stap_lengths / 0.42) - 1 + add_seqlen
    
    # Ensure minimum length of 1
    scaf_SSseq[scaf_SSseq < 1] = 1
    stap_SSseq[stap_SSseq < 1] = 1
    
    # Generate coordinates for single-strand nucleotides
    scaf_SS_coords = []
    stap_SS_coords = []
    
    for i in range(scaf_SSseq.size(0)):
        src_ID, dst_ID = SSedge_index[:, 2 * i]
        scaf_num_insertions = scaf_SSseq[i]
        stap_num_insertions = stap_SSseq[i]
        
        # Insert intermediate coordinates
        scaf_insertions = ins_inner_nodes(scaf_coords, src_ID, dst_ID, scaf_num_insertions)
        stap_insertions = ins_inner_nodes(stap_coords, src_ID, dst_ID, stap_num_insertions)
        
        scaf_SS_coords.append(scaf_insertions)
        stap_SS_coords.append(stap_insertions)
    
    return (scaf_SSseq, stap_SSseq, SSedge_index[:, ::2], 
            torch.vstack(scaf_SS_coords), torch.vstack(stap_SS_coords))


from automated_design.m13 import m13_7249, m13_7560

def assign_nucleidx_to_route(SCroute, STroutes, SSedge_index, scaf_coords, stap_coords, 
                           add_seqlen=0, scaf_unbound=True):
    """
    Assign nucleotide indices and sequences to scaffold and staple routes.
    
    Args:
        SCroute: Scaffold route
        STroutes: List of staple routes
        SSedge_index: Single-strand edge index
        scaf_coords: Scaffold coordinates
        stap_coords: Staple coordinates
        add_seqlen: Additional sequence length
        scaf_unbound: Whether scaffold is unbound
    
    Returns:
        Multiple outputs including nucleotide indices, sequences, and coordinates
    """
    # ========================================
    # Generate single-strand sequences
    # ========================================
    scaf_SSseq, stap_SSseq, SSedge_index_undirected, scaf_SS_coords, stap_SS_coords = SS_to_SeqLen(
        SSedge_index, scaf_coords, stap_coords, add_seqlen)
    
    num_bps = scaf_coords.size(0)
    
    # ========================================
    # Assign single-strand nucleotides to routes
    # ========================================
    SCroute_bpidx = assign_SStrand_to_route(SCroute, SSedge_index_undirected, 
                                           scaf_SSseq, num_bps)
    
    STroutes_bpidx = []
    for route in STroutes:
        STroutes_bpidx.append(assign_SStrand_to_route(route, SSedge_index_undirected, 
                                                     stap_SSseq, num_bps))
    
    # ========================================
    # Convert base-pair indices to nucleotide indices
    # ========================================
    current_SS_nucleidx = 2 * num_bps + 1
    
    # Process scaffold route
    SCroute_Nidx = SCroute_bpidx.clone().detach()
    for i in range(len(SCroute_Nidx)):
        if SCroute_Nidx[i] > -1:
            SCroute_Nidx[i] = 1 + 2 * SCroute_Nidx[i]
        else:
            SCroute_Nidx[i] = current_SS_nucleidx
            current_SS_nucleidx += 1
    
    # Process staple routes
    STroutes_Nidx = []
    for route in STroutes_bpidx:
        route_copy = route.clone().detach()
        for j in range(len(route_copy)):
            if route_copy[j] > -1:
                route_copy[j] = 2 * (route_copy[j] + 1)
            else:
                route_copy[j] = current_SS_nucleidx
                current_SS_nucleidx += 1
        STroutes_Nidx.append(route_copy)
    
    # ========================================
    # Assign nucleotide sequences
    # ========================================
    m13_7249_list = list(m13_7249)
    m13_7560_list = list(m13_7560)
    
    # Select appropriate scaffold sequence
    scaffold_length = SCroute_Nidx.size(0)
    
    if scaffold_length > len(m13_7249_list):
        if scaffold_length < len(m13_7560_list):
            scaffold_seq_list = m13_7560_list
            print(f"Scaffold length {scaffold_length} (scaffold type M13 7560)")
        else:
            scaffold_seq_list = m13_7249_list * int((scaffold_length / len(m13_7249_list)) + 1)
            print(f"Scaffold length {scaffold_length} exceeds M13 length!")
    else:
        scaffold_seq_list = m13_7249_list
        print(f"Scaffold length {scaffold_length} (scaffold type M13 7249)")
    
    scaffold_seq = scaffold_seq_list[:scaffold_length]
    
    # Get scaffold nucleotides for each base pair
    BP_scaf_nucleotides = []
    for i in range(num_bps):
        nucleotide_idx = (SCroute_bpidx == i).nonzero().squeeze()
        BP_scaf_nucleotides.append(scaffold_seq[nucleotide_idx])
    
    # ========================================
    # Generate complementary staple sequences
    # ========================================
    staples_seq = []
    complement_map = {'G': 'C', 'C': 'G', 'A': 'T', 'T': 'A'}
    
    for stap_route in STroutes_bpidx:
        stap_seq = []
        for bp_idx in stap_route:
            if bp_idx > -1:
                # Find corresponding scaffold nucleotide
                scaffold_idx = (SCroute_bpidx == bp_idx).nonzero().item()
                scaffold_nucleotide = scaffold_seq[scaffold_idx]
                # Add complement
                stap_seq.append(complement_map.get(scaffold_nucleotide, 'T'))
            else:
                # Single-strand region: use default nucleotide
                stap_seq.append('T')
        staples_seq.append(stap_seq)
    
    return (SCroute_Nidx, STroutes_Nidx, SCroute_bpidx, STroutes_bpidx, 
            scaffold_seq, staples_seq, BP_scaf_nucleotides, 
            scaf_SS_coords, stap_SS_coords)


def assign_SStrand_to_route(route, SSedge_index_undirected, SSedges_len, num_bps):
    """
    Assign single-strand nucleotides to route with '-1' labels.
    
    Args:
        route: Base-pair route
        SSedge_index_undirected: Undirected single-strand edges
        SSedges_len: Length of each single-strand edge
        num_bps: Number of base pairs
    
    Returns:
        route_with_SS: Route with single-strand insertions marked as -1
    """
    # Identify positions where single-strand insertions are needed
    src, dst = SSedge_index_undirected
    edit_indices = SSedge_index_undirected.flatten()
    edit_mask = torch.zeros(num_bps, dtype=torch.bool, device=route.device)
    edit_mask[edit_indices] = True
    
    # Find positions in route that need single-strand insertions
    edit_positions = edit_mask[route].nonzero().squeeze()
    split_positions = edit_positions[1::2].to('cpu')
    
    # Split route at single-strand positions
    route_segments = list(torch.tensor_split(route, split_positions))
    
    # Insert single-strand sequences between segments
    segment_count = 0
    while segment_count < 2 * len(split_positions) - 1:
        # Find single-strand length for this position
        last_bp = route_segments[segment_count][-1]
        ss_mask = torch.logical_or(src == last_bp, dst == last_bp)
        ss_insertion_length = int(SSedges_len[ss_mask])
        
        # Create single-strand sequence (marked with -1)
        ss_insertion = (-1) * torch.ones((ss_insertion_length,), 
                                        dtype=torch.int32, device=route.device)
        
        # Insert single-strand sequence
        route_segments.insert(segment_count + 1, ss_insertion)
        segment_count += 2
    
    return torch.hstack(route_segments)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_furthest_helixID(coords, bp_Hindex, alignCO_index_noscaf):
    """
    Determine the furthest helix from center for nick placement.
    
    Args:
        coords: Base-pair coordinates [Nx3]
        bp_Hindex: Helix index for each base pair
        alignCO_index_noscaf: Non-scaffold cross-over indices
    
    Returns:
        furthest_helix_id: ID of the furthest helix
        with_COs: Whether the selected helix has cross-overs
    """
    num_helices = bp_Hindex.amax() + 1
    
    # Get helices with non-scaffold cross-overs
    no_scaffold_CO_helices = torch.unique(bp_Hindex[alignCO_index_noscaf])
    
    # Get helix lengths
    _, helix_lengths = torch.unique(bp_Hindex, return_counts=True)
    average_length = helix_lengths.sum() / num_helices
    
    # Create masks for selection criteria
    has_crossovers_mask = torch.zeros(num_helices, dtype=torch.bool, device=coords.device)
    has_crossovers_mask[no_scaffold_CO_helices] = True
    above_average_length_mask = helix_lengths > average_length
    
    # Combined mask for preferred helices
    preferred_helices_mask = torch.logical_and(has_crossovers_mask, above_average_length_mask)
    
    # Calculate distances from center
    dna_center = coords.mean(dim=0, keepdim=True)
    relative_vectors = coords - dna_center
    distances = relative_vectors.norm(dim=-1, keepdim=False)
    
    # Average distance per helix
    helix_distances = scatter(distances, bp_Hindex, dim=0, reduce='mean')
    
    # Select furthest helix
    if helix_distances[preferred_helices_mask].size(0) > 0:
        # Use preferred helices if available
        furthest_helix_id = (helix_distances == helix_distances[preferred_helices_mask].amax()).nonzero()
        with_COs = False
    else:
        # Use any helix
        furthest_helix_id = (helix_distances == helix_distances.amax()).nonzero()
        with_COs = True
    
    return furthest_helix_id.squeeze(), with_COs
