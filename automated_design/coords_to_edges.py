import torch
import warnings
import numpy as np
from torch_geometric.nn import radius_graph
from torch_geometric.utils import scatter, degree, subgraph, to_undirected, k_hop_subgraph
from automated_design.ults import *
from automated_design.visual import *


# =============================================================================
# REGULAR BASE-PAIR EDGES ASSIGNMENT
# =============================================================================

def assign_RegularBp_edges(coords, equi_len=0.4):
    """
    Assign regular base-pair edges to given coordinates.
    
    Input: 
        coords: base-pairs coordinate [Nx3]
        equi_len: equilibrium length of regular base-pairs edges [nm]

    Output:
        coords: base-pairs coordinate [Nx3]
        helix_index: base-pairs helix index [N]
        edge_index: regular base-pair edges in each helix [2xN]
        num_bps: number of remaining base-pairs
    """
    dev = coords.device
    
    # Apply point cloud filter
    coords, edge_index = pcl_filter(coords, r=equi_len)
    
    # Perform helix indexing and remove helices that are too short
    bp_Hindex = segmentation(edge_index, min_nodes=10, num_nodes=coords.size(0))
    node_mask = bp_Hindex > -1
    subset = node_mask.nonzero()
    
    # Update edge index and coordinates based on valid nodes
    edge_index, _ = subgraph(subset.squeeze(), edge_index, relabel_nodes=True)
    bp_Hindex = bp_Hindex[node_mask]
    coords = coords[node_mask, :]
    edge_index = radius_graph(coords, r=equi_len, loop=False)
    
    # Re-index base-pairs for routing model
    coords, bp_Hindex = reindexing_base_pair(coords, edge_index, bp_Hindex)
    num_bps = coords.size(0)
    edge_index = radius_graph(coords, r=equi_len, loop=False)
    deg = degree(edge_index[0], num_bps)
    
    # Analyze unsafe regions
    helix_ID, helix_len = torch.unique(bp_Hindex, return_counts=True)
    num_helices = helix_ID.amax() + 1

    # Find helix end base-pair IDs
    helix_ends_bpID = torch.zeros(num_helices, 2, dtype=torch.long, device=dev)
    for i in range(num_helices):
        helix_bpID_lst = (bp_Hindex == i).nonzero()
        helix_ends_bpID[i, 0] = helix_bpID_lst.amin().to(torch.long)
        helix_ends_bpID[i, 1] = helix_bpID_lst.amax().to(torch.long)
        
    # Identify end helices and circular helices
    endhelix_bpID = helix_ends_bpID[deg[helix_ends_bpID[:, 0]] < 2, :]
    helix_cycle = torch.logical_and(
        deg[helix_ends_bpID[:, 0]] > 1, 
        deg[helix_ends_bpID[:, 1]] > 1
    )
    
    # Check for circular helices
    if helix_cycle.sum().item() > 0:
        print("There are some helices that form a circle. Please remove some base-pairs!")

    # Collect base-pairs to remove from circular helices
    remove_bpID_lst = []
    for i in range(num_helices):
        if helix_cycle[i] == True:
            remove_bpID_lst.append(int(helix_ends_bpID[i, -1].item()))
    
    return coords, edge_index, bp_Hindex, endhelix_bpID, remove_bpID_lst


# =============================================================================
# SINGLE-STRAND EDGES ASSIGNMENT
# =============================================================================

def assign_SingleStrand_edges(coords, RBedge_index, alignCOs_index, bp_Hindex):
    """
    Assign single-strand edges to DNA structure.
    
    Input: 
        coords: base-pairs coordinate [Nx3]
        RBedge_index: regular base-pair edges in each helices [2xm]
        alignCOs_index: alignment cross-over edges
        bp_Hindex: base-pair helix index
        
    Output:
        SSedge_index: single-strand edges [2xn]
        bp_Lindex: base-pair loop index
    """
    # Assign single-strand edges using substructure method
    SSedge_index = assign_SSedges_wsubstr(coords, RBedge_index, alignCOs_index, bp_Hindex)
    
    # Segment the combined edge structure
    bp_Lindex = segmentation(torch.hstack((RBedge_index, SSedge_index)))
    
    return SSedge_index, bp_Lindex


def assign_SSedges_wsubstr(coords, RBedge_index, alignCO_index, bp_Hindex):
    """
    Assign single-strand edges using substructure information.
    
    Method 2: assign single-strand edges from coordinate and substructure index
    """
    num_bps = coords.size(0)
    deg = degree(RBedge_index[0], num_nodes=num_bps)
    
    # Create combined edge index and calculate degrees
    edge_index = torch.hstack((RBedge_index, alignCO_index))
    deg = degree(edge_index[0])
    
    # Remove some staple cross-overs for better staple routing
    RBedge_mask = torch.logical_and(
        deg[RBedge_index[0]] == 3, 
        deg[RBedge_index[1]] == 3
    )
    edge_index = torch.hstack((RBedge_index[:, ~RBedge_mask], alignCO_index))
    
    # Perform segment indexing
    bp_segID, _, _ = segmentation(edge_index, min_nodes=0, return_extra=True)
    
    # Determine end base-pairs of helices
    Hend_mask = deg == 1
    Hend_coords = coords[Hend_mask, :]
    Hend_index = Hend_mask.nonzero().reshape((-1,))
    
    if Hend_coords.size(0) > 1:
        # Find nearest neighbors for helix ends
        pairwise_dist = torch.cdist(Hend_coords, Hend_coords, p=2)
        pairwise_dist.fill_diagonal_(10.0)  # Remove self-distance        
        
        # Use 1st nearest neighbor
        _, shortest_indices = torch.topk(
            pairwise_dist, k=1, dim=1, largest=False, sorted=True
        )
        
        src = Hend_index
        dst = Hend_index[shortest_indices.squeeze()]
        SSedge_index_guess = torch.vstack((src, dst))
        
        # Re-check connections between different segments
        SSedge_guess_mask = bp_segID[SSedge_index_guess[0]] != bp_segID[SSedge_index_guess[1]]
        SSedge_index = SSedge_index_guess[:, SSedge_guess_mask]   
        
        # Fix single-strand edges from directed to undirected graph
        deg = degree(SSedge_index.flatten(), num_nodes=num_bps)
        SSedge_mask = torch.logical_or(
            deg[SSedge_index[0]] < 2, 
            deg[SSedge_index[1]] < 2
        )
        SSedge_index_directed = SSedge_index[:, SSedge_mask]
        
        if SSedge_index_directed.size(1) > 0:
            SSedge_index_fix = torch.zeros(
                2, 2 * SSedge_index_directed.size(1), 
                dtype=torch.long, device=SSedge_index.device
            )
            SSedge_index_fix[:, ::2] = SSedge_index_directed
            SSedge_index_fix[:, 1::2] = SSedge_index_directed.flip([0])
            SSedge_index = torch.hstack((SSedge_index, SSedge_index_fix))
        
        # Find remaining unconnected helix ends
        combined_edges = torch.hstack((RBedge_index, SSedge_index))
        Hend_remain_mask = degree(combined_edges[0], num_nodes=num_bps) < 2
        remain_Hend_bpID = Hend_remain_mask.nonzero().reshape((-1,))
        
        # Connect remaining points (first attempt with segment constraints)
        if remain_Hend_bpID.size(0) > 0:
            remain_Hend_coords = coords[remain_Hend_bpID, :]
            
            # Find nearest neighbors for remaining ends
            pairwise_dist = torch.cdist(remain_Hend_coords, remain_Hend_coords, p=2)
            pairwise_dist.fill_diagonal_(10.0)
            
            _, shortest_indices = torch.topk(
                pairwise_dist, k=1, dim=1, largest=False, sorted=True
            )
            
            src = remain_Hend_bpID.repeat_interleave(1)
            dst = remain_Hend_bpID[shortest_indices.squeeze()].reshape(-1)
            mask = bp_segID[src] != bp_segID[dst]
            SSedge_index = torch.hstack((SSedge_index, torch.vstack((src[mask], dst[mask]))))
        
        # Second attempt without segment constraints
        combined_edges = torch.hstack((RBedge_index, SSedge_index))
        Hend_remain_mask = degree(combined_edges[0], num_nodes=num_bps) < 2
        remain_Hend_bpID = Hend_remain_mask.nonzero().reshape((-1,))
        
        if remain_Hend_bpID.size(0) > 0:
            remain_Hend_coords = coords[remain_Hend_bpID, :]
            
            # Find nearest neighbors without segment constraints
            pairwise_dist = torch.cdist(remain_Hend_coords, remain_Hend_coords, p=2)
            pairwise_dist.fill_diagonal_(10.0)
            
            _, shortest_indices = torch.topk(
                pairwise_dist, k=1, dim=1, largest=False, sorted=True
            )
            
            src = remain_Hend_bpID.repeat_interleave(1)
            dst = remain_Hend_bpID[shortest_indices.squeeze()].reshape(-1)
            SSedge_index = torch.hstack((SSedge_index, torch.vstack((src, dst))))
            
        # Final validation check
        final_combined_edges = torch.hstack((RBedge_index, SSedge_index))
        remaining_incomplete = (degree(final_combined_edges[0], num_nodes=num_bps) < 2).sum().item()
        
        if remaining_incomplete > 0:
            warnings.warn("Warning: FAILED to define single-strands")
        
        # Convert to undirected edge index format
        SSedge_index_Undirected = Undirected_edge_index(coords, SSedge_index)
        
    else:
        # No helix ends found, return empty edge index
        SSedge_index_Undirected = torch.empty((2, 0), dtype=torch.long, device=coords.device)
    
    return SSedge_index_Undirected


# =============================================================================
# STAPLE CROSS-OVER GENERATION
# =============================================================================

def gen_Staples_CrossOvers(coords, bp_Hindex, scaf_route, scafCO_index, 
                          nonscafCO_index, RBedge_index, SSedge_index,
                          effective_len=10, min_pairing_len=4, adaptive=False):
    """
    Generate staple cross-over steps.
    
    Input: 
        coords: base-pairs coordinate [Nx3]
        bp_Hindex: base-pair helix index
        scaf_route: scaffold routing
        scafCO_index: scaffold cross-over index
        nonscafCO_index: non-scaffold cross-over index
        RBedge_index: regular base-pairs edge_index [2xM]
        SSedge_index: single-strand edge index
        effective_len: effective length parameter
        min_pairing_len: minimum pairing length
        adaptive: whether to use adaptive selection

    Output:
        stapCO_index: staple cross-over edge_index [2xL]
    """
    dev = scafCO_index.device
    num_bps = coords.size(0)
    alignCO_index = torch.hstack((scafCO_index, nonscafCO_index))
    num_aligns = alignCO_index[:, ::4].size(1)
    num_scafCOs = scafCO_index[:, ::4].size(1)
    
    # Generate trial cross-over step candidates
    CO_guess = radius_graph(coords, r=3.0, loop=False)   
    CO_mask = bp_Hindex[CO_guess[0]] != bp_Hindex[CO_guess[1]]
    CO_guess = CO_guess[:, CO_mask]
    
    # Generate staple cross-over candidates
    staple_COs = guess_staple_COs(
        coords, CO_guess, alignCO_index, RBedge_index, SSedge_index, 
        bp_Hindex, num_bps, num_aligns, num_scafCOs, min_pairing_len
    )
    
    # Create selection mask for suitable staple cross-overs
    bp_mask = torch.ones(num_bps, dtype=torch.bool, device=dev)
    
    # Mask scaffold nick and end helices
    seeds = torch.cat((scaf_route[:1], scaf_route[-1:], SSedge_index.flatten()), dim=0)   
    bp_mask = update_bpmask(bp_mask, RBedge_index, seeds, k_hop=min_pairing_len)
    
    # Mask scaffold cross-overs
    seeds = scafCO_index.flatten()
    bp_mask = update_bpmask(bp_mask, RBedge_index, seeds, k_hop=12)
    
    # Select optimal staple cross-overs
    stapCO_index = select_staple_COs(
        staple_COs, bp_mask, RBedge_index, SSedge_index, 
        effective_len, adaptive=adaptive
    )
    
    # Check and optimize staple nick configuration
    opt_vert = check_staple_Nick(
        RBedge_index, scafCO_index, stapCO_index, SSedge_index, 
        scaf_route, num_bps, min_pairing_len
    )
    
    if opt_vert:
        stapCO_index = select_staple_COs(
            staple_COs, bp_mask, RBedge_index, SSedge_index, 
            effective_len, adaptive=adaptive, opt_vert=opt_vert
        )
    
    return stapCO_index.long()


def gen_newCOs(CO_guess, RBedge_index, coords, bp_mask, neighbors_side, effective_len=10):
    """
    Generate new cross-over indices in selected areas.
    
    Parameters:
        CO_guess: candidate cross-over edges
        RBedge_index: regular base-pair edges
        coords: base-pair coordinates
        bp_mask: base-pair selection mask
        neighbors_side: neighbor selection parameter
        effective_len: effective length for masking
    
    Returns:
        List of cross-over indices or None if no valid cross-overs found
    """
    # Select shortest cross-over steps
    sorted_len, sorted_idx = edges_len(coords, CO_guess, sort="increasing")
    COs_lst = []
    
    # Process cross-overs while shortest length is acceptable
    while sorted_len[0] < 2.5 and CO_guess.size(1) > 11:
        
        # Select the shortest cross-over edges
        CO_select = CO_guess[:, sorted_idx[:1]] 
        CO_select = torch.hstack((CO_select, CO_select.flip([0])))
        
        # Generate cross-over neighbors based on side preference
        if neighbors_side < 1:
            CO_neighbors = torch.tensor([
                [CO_select[0, 0] + 1, CO_select[0, 0] - 1, CO_select[1, 0] + 1, CO_select[1, 0] - 1],
                [CO_select[1, 0] + 1, CO_select[1, 0] - 1, CO_select[0, 0] + 1, CO_select[0, 0] - 1]
            ], dtype=torch.long, device=CO_guess.device)
        else:
            CO_neighbors = torch.tensor([
                [CO_select[0, 0] - 1, CO_select[0, 0] + 1, CO_select[1, 0] + 1, CO_select[1, 0] - 1],
                [CO_select[1, 0] + 1, CO_select[1, 0] - 1, CO_select[0, 0] - 1, CO_select[0, 0] + 1]
            ], dtype=torch.long, device=CO_guess.device)
        
        # Mask invalid neighbors (out of bounds)
        mask = torch.logical_and(
            CO_neighbors > 0, 
            CO_neighbors < coords.size(0)
        ).all(axis=0)
        CO_neighbors = CO_neighbors[:, mask]

        # Select best neighbors based on distance
        neigh_sorted_len, sorted_idx = edges_len(coords, CO_neighbors, sort="increasing")
        
        if neigh_sorted_len[0] < 2.5:
            CO_index = torch.hstack((CO_select, CO_neighbors[:, sorted_idx][:, :2]))
            
            if CO_index.size(1) == 4:
                COs_lst.append(CO_index)
            
        # Update base-pair mask to exclude processed regions
        bp_mask = update_bpmask(bp_mask, RBedge_index, CO_select.flatten(), k_hop=effective_len + 1)
        
        # Filter remaining cross-over candidates
        CO_guess = CO_guess[:, torch.logical_and(bp_mask[CO_guess[0]], bp_mask[CO_guess[1]])]
        
        if CO_guess.size(1) > 0:
            sorted_len, sorted_idx = edges_len(coords, CO_guess, sort="increasing")
        else:
            break
    
    return COs_lst if len(COs_lst) > 0 else None


def guess_staple_COs(coords, CO_guess, alignCO_index, RBedge_index, SSedge_index, 
                    bp_Hindex, num_bps, num_aligns, num_scafCOs, min_pairing_len):
    """
    Generate all possible staple cross-over candidates.
    
    Parameters:
        coords: base-pair coordinates
        CO_guess: initial cross-over guess
        alignCO_index: alignment cross-over index
        RBedge_index: regular base-pair edges
        SSedge_index: single-strand edges
        bp_Hindex: base-pair helix index
        num_bps: number of base pairs
        num_aligns: number of alignments
        num_scafCOs: number of scaffold cross-overs
        min_pairing_len: minimum pairing length
    
    Returns:
        List of staple cross-over candidates
    """
    effective_len = 7
    staple_COs = []
    
    # Process each alignment
    for i in range(num_aligns):
        
        # Get cross-overs with respect to alignment cross-over
        align_COs = alignCO_index[:, 4*i: 4*(i+1)]
        
        # Get helix IDs for source and destination
        helixID_src, helixID_dst = bp_Hindex[align_COs[:, 0]]   
        
        # Filter cross-over candidates for this substructure
        CO_mask = torch.logical_or(
            bp_Hindex[CO_guess] == helixID_src, 
            bp_Hindex[CO_guess] == helixID_dst
        ).all(axis=0)
        CO_substr = CO_guess[:, CO_mask].clone().detach()
        
        # Calculate neighbor side preference
        neighbors_side = (
            (align_COs[:, 2][0] - align_COs[:, 2][1]).abs() - 
            (align_COs[:, 0][0] - align_COs[:, 0][1]).abs()
        ).abs()
        
        # Initialize substructure cross-overs
        CO_radius = CO_substr.clone().detach()
        bp_mask = torch.ones(num_bps, dtype=torch.bool, device=CO_radius.device)
        
        sub_COs = []
        
        # Add scaffold cross-overs if applicable
        if i < num_scafCOs:
            sub_COs.append(align_COs)
            bp_mask = update_bpmask(bp_mask, RBedge_index, align_COs.flatten(), min_pairing_len)
          
        # Generate new cross-overs for this substructure
        if CO_radius.size(1) > 0:
            COs = gen_newCOs(
                CO_radius, RBedge_index, coords, bp_mask, 
                neighbors_side, effective_len=effective_len
            )
            
            if COs is not None: 
                sub_COs = sub_COs + COs
        
        # Collect cross-overs (excluding scaffold cross-overs for staples)
        if len(sub_COs) > 0:  
            if i < num_scafCOs and len(sub_COs) > 1:      
                staple_COs.append(sub_COs[1:]) 
            elif i >= num_scafCOs:
                staple_COs.append(sub_COs)
    
    return staple_COs


def select_staple_COs(staple_COs, bp_mask, RBedge_index, SSedge_index, 
                     effective_len, adaptive=True, opt_vert=False):
    """
    Select optimal staple cross-overs from candidates.
    
    Parameters:
        staple_COs: list of staple cross-over candidates
        bp_mask: base-pair selection mask
        RBedge_index: regular base-pair edges
        SSedge_index: single-strand edges
        effective_len: effective length for masking
        adaptive: whether to use adaptive selection
        opt_vert: whether to optimize vertices
    
    Returns:
        Selected staple cross-over indices
    """
    num_subs = len(staple_COs)
    num_innerCOs = torch.zeros(num_subs, dtype=torch.long)

    staple_COs_select = []
    outerCOs_lst = []
    
    # Process each substructure
    for i, sub_COs in enumerate(staple_COs):
        
        # Filter and sort cross-overs by position
        sub_COs_pos = []
        sub_COs_select = []
        
        for _, COs in enumerate(sub_COs):
            if bp_mask[COs].sum().item() == 8:
                sub_COs_select.append(COs)
                sub_COs_pos.append(COs.min())
        
        # Sort by position
        _, indices = torch.sort(torch.as_tensor(sub_COs_pos))
        sub_COs_select = [sub_COs_select[k] for k in indices]
        
        # Adjust effective length adaptively
        adaptive_effective_len = effective_len
        if adaptive and len(sub_COs_select) > 12:
            adaptive_effective_len = int(1.85 * effective_len)
        
        # Select cross-overs with spacing constraints
        subCOs = [] 
        mask = bp_mask.detach().clone()
        
        for j, COs in enumerate(sub_COs_select):
            if mask[COs].sum().item() == 8:  
                subCOs.append(COs)
                mask = update_bpmask(mask, RBedge_index, COs.flatten(), k_hop=adaptive_effective_len)
                
        num_innerCOs[i] = int(len(subCOs) - 2)
        
        # Handle vertex optimization
        if opt_vert:
            if len(subCOs) > 2:
                staple_COs_select.append(torch.hstack((subCOs[1:-1])))
                outerCOs_lst.append([subCOs[0], subCOs[-1]])
            else:
                outerCOs_lst.append([])
                if len(subCOs) > 0:
                    staple_COs_select.append(torch.hstack((subCOs)))
        elif len(subCOs) > 0:
            staple_COs_select.append(torch.hstack((subCOs)))
    
    # Handle outer cross-overs optimization
    if len(outerCOs_lst) > 0:
        # Select optimal outer cross-overs
        mask_vert = bp_mask.detach().clone()
        mask_vert = update_bpmask(
            mask_vert, RBedge_index, torch.hstack(staple_COs_select).flatten(), 
            k_hop=effective_len, additional_edge_index=SSedge_index
        )
        
        # Sort substructures by number of inner cross-overs
        _, indices = torch.sort(num_innerCOs, descending=False)
        
        outerCOs_select = []
        for k in range(num_subs):
            
            sub_idx = indices[k]
            outerCOs = outerCOs_lst[sub_idx]
            
            # Add first outer cross-over if valid
            if len(outerCOs) > 0:
                if mask_vert[outerCOs[0]].sum().item() == 8:
                    outerCOs_select.append(outerCOs[0])
            
            # Add second outer cross-over if valid        
            if len(outerCOs) > 1:
                if mask_vert[outerCOs[1]].sum().item() == 8:
                    outerCOs_select.append(outerCOs[1])
            
            # Update mask with selected outer cross-overs
            if len(outerCOs_select) > 0:
                mask_vert = update_bpmask(
                    mask_vert, RBedge_index, torch.hstack(outerCOs_select).flatten(), 
                    k_hop=20, additional_edge_index=SSedge_index
                )
        
        # Add outer cross-overs to selection
        if len(outerCOs_select) > 0:
            staple_COs_select.append(torch.hstack((outerCOs_select)))
        
    return torch.hstack((staple_COs_select))


def opt_endhelix_CO(COextra_index, RBedge_index, SSedge_index, 
                   stapCO_index, scafCO_index):
    """
    Remove some staple cross-overs near vertices to optimize staple sequence length.
    
    Parameters:
        COextra_index: extra cross-over indices
        RBedge_index: regular base-pair edges
        SSedge_index: single-strand edges
        stapCO_index: staple cross-over indices
        scafCO_index: scaffold cross-over indices
    
    Returns:
        Optimized cross-over indices
    """
    dev = stapCO_index.device
    COedge_index = torch.hstack((stapCO_index, scafCO_index))
    
    # Remove some staple cross-overs for better staple routing
    edge_index = torch.hstack((RBedge_index, SSedge_index, COedge_index))
    deg = degree(edge_index[0])
    
    # Create mask for regular base-pair edges
    RBedge_mask = torch.logical_and(
        deg[RBedge_index[0]] == 3, 
        deg[RBedge_index[1]] == 3
    )
    
    # Create staple edge index
    stap_edge_index = torch.hstack((
        RBedge_index[:, ~RBedge_mask], SSedge_index, COedge_index
    ))
    
    # Segment staples
    bp_stapID, _, _ = segmentation(stap_edge_index, min_nodes=0, return_extra=True)
    num_staples = bp_stapID.amax() + 1   
    
    # Count cross-overs on each staple
    stap_ID, counts = torch.unique(bp_stapID[COedge_index[0]], sorted=True, return_counts=True)
    num_CO_onStaple = torch.zeros(num_staples, device=dev)
    num_CO_onStaple[stap_ID] = counts / 2
    
    # Count extra cross-overs on each staple
    stap_ID, counts = torch.unique(bp_stapID[COextra_index[0]], sorted=True, return_counts=True)
    num_COextra_onStaple = torch.zeros(num_staples, device=dev)
    num_COextra_onStaple[stap_ID] = counts / 4
    
    # Select staples with appropriate number of cross-overs
    COextra_index_new = []
    
    for i in range(num_staples):
        
        if num_COextra_onStaple[i] > 0:
            # Get cross-overs for this staple
            mask = torch.logical_and(
                bp_stapID[COextra_index[0]] == i, 
                bp_stapID[COextra_index[1]] == i
            )
            COs = COextra_index[:, mask]
            num_CO_wNick = num_CO_onStaple[i] - num_COextra_onStaple[i]
            
            # Determine minimum required cross-overs
            if num_CO_onStaple[i] < 5:
                min_CO_wNick = 1
            elif num_CO_onStaple[i] < 6:
                min_CO_wNick = 2
            elif num_CO_onStaple[i] < 8:
                min_CO_wNick = 3
            elif num_CO_onStaple[i] < 10:
                min_CO_wNick = 4
            else:
                min_CO_wNick = 5
            
            # Keep all cross-overs if sufficient
            if num_CO_wNick > min_CO_wNick:
                COextra_index_new.append(COs)
            else:
                # Remove some extra cross-overs
                num_remove_COextra = int(min_CO_wNick - num_CO_wNick) 
                idx_np = np.random.choice(
                    int(num_COextra_onStaple[i]), num_remove_COextra, replace=False
                ) 
                idx = torch.from_numpy(idx_np).to(device=COs.device)
                idx = 4 * idx.repeat_interleave(4) + torch.arange(4, device=COs.device).repeat(num_remove_COextra)
                
                COs_mask = torch.zeros(COs.size(1), dtype=torch.bool, device=COs.device)
                COs_mask[idx] = True
                COextra_index_new.append(COs[:, ~COs_mask])
                
    return torch.hstack((COextra_index_new))


# =============================================================================
# ALIGNMENT CROSS-OVER GENERATION
# =============================================================================

def gen_align_CrossOvers(coords, bp_Hindex, RBedge_index, endhelix_bpID,
                        effective_len=10, adjust_short_edges=4, adjust_long_edges=4):
    """
    Generate alignment cross-over steps.
    
    Input: 
        coords: base-pairs coordinate [Nx3]
        bp_Hindex: base-pair helix index
        RBedge_index: regular base-pairs edge_index [2xM]
        endhelix_bpID: end helix base-pair IDs
        effective_len: effective length for masking
        adjust_short_edges: adjustment for short edges
        adjust_long_edges: adjustment for long edges

    Output:
        align_COs: alignment cross-over edge_index [2xL]
    """
    dev = coords.device
    num_bps = coords.size(0)
    num_helices = bp_Hindex.amax() + 1
      
    # Generate radius graph for cross-helix connections
    edge_index = radius_graph(coords, r=2.75, loop=False)  # Default = 2.75
    edge_mask = bp_Hindex[edge_index[0]] != bp_Hindex[edge_index[1]]
    COs_index = edge_index[:, edge_mask]
    
    # Apply end helix masking if needed
    if endhelix_bpID.size(0) > 0:
        bp_mask = EndHelix_bpmask(
            num_bps, RBedge_index, 
            mask_seed_idx=endhelix_bpID.flatten(), 
            k_hop=effective_len
        )
        edge_mask = torch.logical_and(bp_mask[COs_index[0]], bp_mask[COs_index[1]])
        COs_index = COs_index[:, edge_mask]
    
    # Create helix assignment index
    helix_asign_index = coalesce(bp_Hindex[COs_index], num_nodes=num_helices)
    helix_cen_coords = scatter(coords, bp_Hindex, dim=0, reduce='mean')
    helix_asign_index = To_undirected_edge_index(helix_cen_coords, helix_asign_index)[:, ::2]
    
    # Sort helix assignments by average length
    _, helix_len = torch.unique(bp_Hindex, return_counts=True)
    helix_asign_len = helix_len[helix_asign_index]
    helix_asign_minlen, _ = torch.min(helix_asign_len, 0)
    helix_asign_maxlen, _ = torch.max(helix_asign_len, 0)
    helix_asign_avglen = (helix_asign_minlen + helix_asign_maxlen) / 2
    
    _, indices = torch.sort(helix_asign_avglen, descending=True)
    helix_asign_index_sorted = helix_asign_index[:, indices]
    
    # Assign cross-overs to substructures
    num_sub = helix_asign_index_sorted.size(1)
    align_COs = []
    
    for i in range(num_sub):
        helixID_src, helixID_dst = helix_asign_index_sorted[:, i]
        
        # Filter cross-overs for this helix pair
        mask = torch.logical_or(
            bp_Hindex[COs_index] == helixID_src, 
            bp_Hindex[COs_index] == helixID_dst
        ).all(axis=0)
        COs_sub = COs_index[:, mask]
        
        # Get base-pair lists for source and destination helices
        sub_bpID_list = COs_sub.flatten()
        src_bpID_list = sub_bpID_list[bp_Hindex[sub_bpID_list] == helixID_src] 
        dst_bpID_list = sub_bpID_list[bp_Hindex[sub_bpID_list] == helixID_dst] 
        
        # Calculate central positions
        src_cen_bpID = int((src_bpID_list.amin() + src_bpID_list.amax()) / 2) 
        dst_cen_bpID = int((dst_bpID_list.amin() + dst_bpID_list.amax()) / 2) 
        
        # Calculate helix lengths
        src_len = src_bpID_list.max() - src_bpID_list.min()
        dst_len = dst_bpID_list.max() - dst_bpID_list.min()
        
        # Determine center position based on helix lengths
        if min(src_len, dst_len) < 25:  # Default threshold = 25
            if src_len < dst_len:
                # Use shorter helix with adjustment
                cen_bpID = int(src_bpID_list.min() + adjust_short_edges)
                if cen_bpID not in src_bpID_list:
                    idx = ((src_bpID_list - cen_bpID).abs() == 
                          (src_bpID_list - cen_bpID).abs().min()).nonzero()[0]
                    cen_bpID = src_bpID_list[idx]
            else:
                # Use shorter helix with adjustment
                cen_bpID = int(dst_bpID_list.min() + adjust_short_edges)
                if cen_bpID not in dst_bpID_list:
                    idx = ((dst_bpID_list - cen_bpID).abs() == 
                          (dst_bpID_list - cen_bpID).abs().min()).nonzero()[0]
                    cen_bpID = dst_bpID_list[idx]
        else:     
            # Use source helix center with adjustment
            cen_bpID = int(src_cen_bpID + adjust_long_edges)
        
        # Create cross-overs around the center position
        mask = torch.logical_or(COs_sub[0] == cen_bpID, COs_sub[1] == cen_bpID)
        COs_sub_select = to_undirected(COs_sub[:, mask], num_bps)
        
        if COs_sub_select.size(1) > 1:
            # Select shortest cross-over
            _, sorted_edge_idx = edges_len(coords, COs_sub_select, sort="increasing")
            Co = COs_sub_select[:, sorted_edge_idx[0]].reshape(2, -1)
            Co = torch.hstack((Co, Co.flip(0)))
            
            # Create neighbor cross-overs
            Co_neighbors = torch.tensor([
                [Co[0, 0] - 1, Co[0, 0] - 1, Co[0, 0] + 1, Co[0, 0] + 1, 
                 Co[1, 0] + 1, Co[1, 0] - 1, Co[1, 0] - 1, Co[1, 0] + 1],
                [Co[1, 0] + 1, Co[1, 0] - 1, Co[1, 0] - 1, Co[1, 0] + 1, 
                 Co[0, 0] - 1, Co[0, 0] - 1, Co[0, 0] + 1, Co[0, 0] + 1]
            ], dtype=torch.long, device=dev)
            
            # Select best neighbors
            _, sorted_idx = edges_len(coords, Co_neighbors, sort="increasing")
            COs = torch.hstack((Co, Co_neighbors[:, sorted_idx][:, :2]))
            align_COs.append(COs)
      
    # Combine all alignment cross-overs
    align_COs = torch.hstack(align_COs) 
    
    # Validation: check if all helices are connected
    align_helices, _ = remove_self_loops(coalesce(bp_Hindex[COs_index], num_nodes=num_helices))
    deg_helix = degree(align_helices[0])
    
    if deg_helix.sum().item() < num_helices:
        print("Some helices are not aligned by cross-over steps")
     
    return align_COs