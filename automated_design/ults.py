import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils import scatter, degree, coalesce, remove_self_loops, k_hop_subgraph
from typing import Tuple, List, Optional, Union


# =======================================================================================
# NICK CREATION AND VALIDATION FUNCTIONS
# =======================================================================================

def Nicks(scaf_route: torch.Tensor, 
          scCO_index: torch.Tensor, 
          stCO_index: torch.Tensor, 
          RB_index: torch.Tensor, 
          SS_index: torch.Tensor, 
          min_pairing_len: int = 4) -> torch.Tensor:
    """
    Create nicks mask for base-pairs to determine safe cutting positions.
    
    Args:
        scaf_route: Tensor of scaffold route indices
        scCO_index: Scaffold crossover indices
        stCO_index: Staple crossover indices  
        RB_index: Rigid body (base pair) edge indices
        SS_index: Single strand edge indices
        min_pairing_len: Minimum pairing length for safe nicking
        
    Returns:
        torch.Tensor: Boolean mask indicating safe positions for nicks
    """
    num_bps = scaf_route.size(0)
    
    # Create initial mask for unsafe base-pairs around helix ends
    bp_mask = EndHelix_bpmask(
        num_bps, 
        RB_index, 
        mask_seed_idx=SS_index.flatten(), 
        k_hop=min_pairing_len
    ) 
    
    # Combine all critical indices that need protection from nicking
    seed_idx = torch.cat((
        scaf_route[:1],           # Start of scaffold
        scaf_route[-1:],          # End of scaffold
        scCO_index.flatten(),     # Scaffold crossovers
        stCO_index.flatten()      # Staple crossovers
    ), dim=0)
    
    # Update mask to protect areas around critical indices
    bp_mask = update_bpmask(bp_mask, RB_index, seed_idx, k_hop=min_pairing_len)
    
    return bp_mask


def check_staple_Nick(RBedge_index: torch.Tensor, 
                      scCO_index: torch.Tensor, 
                      stCO_index: torch.Tensor, 
                      SS_index: torch.Tensor, 
                      scaf_route: torch.Tensor, 
                      num_bps: int, 
                      min_pairing_len: int) -> bool:
    """
    Check if staple nicking is valid by analyzing connectivity and safety constraints.
    
    Args:
        RBedge_index: Rigid body edge indices
        scCO_index: Scaffold crossover indices
        stCO_index: Staple crossover indices
        SS_index: Single strand edge indices
        scaf_route: Scaffold route tensor
        num_bps: Total number of base pairs
        min_pairing_len: Minimum pairing length for safe nicking
        
    Returns:
        bool: True if staple nicking is valid, False otherwise
    """
    # Generate staple edge indices by combining different edge types
    STedge_index = torch.cat((RBedge_index, stCO_index, SS_index), dim=1)
    
    # Calculate node degrees to identify problematic connections
    deg = degree(STedge_index[0, :], num_bps)
    src, dst = STedge_index
    
    # Filter out edges between nodes with degree 3 that are too close
    # (these represent problematic tight connections)
    mask = torch.logical_and(deg[src] == 3, deg[dst] == 3)
    mask = torch.logical_and(mask, torch.abs(dst - src) < 2) 
    STedge_index = STedge_index[:, ~mask]
    
    # Segment the staple graph to identify connected components
    bp_stap_index = segmentation(STedge_index)
    
    # Generate nicks mask for safety validation
    nicks_mask = Nicks(
        scaf_route, 
        scCO_index, stCO_index, 
        RBedge_index, SS_index, 
        min_pairing_len=min_pairing_len
    )
    
    # Check if any staple segment has no safe nicking positions
    status = scatter(nicks_mask.long(), bp_stap_index, dim=0, reduce='sum')
    status = (status == 0).sum().item() > 0
    
    return status


# =======================================================================================
# GRAPH ANALYSIS AND INDEXING FUNCTIONS
# =======================================================================================

def loop_indexing(RBedge_index: torch.Tensor, 
                  SSedge_index: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """
    Get face or loop indices from coordinates and helix indices.
    
    Args:
        RBedge_index: Rigid body edge indices
        SSedge_index: Single strand edge indices
        
    Returns:
        Tuple[torch.Tensor, int]: Loop indices and number of loops
    """
    # Combine all edge types to form complete graph
    edge_index = torch.hstack((RBedge_index, SSedge_index))
    
    # Segment the graph to identify connected components (loops)
    loop_index = segmentation(edge_index)
    num_loops = loop_index.max() + 1
    
    return loop_index, num_loops


def substr_indexing(alignCOs_index: torch.Tensor, 
                    bp_Hindex: torch.Tensor, 
                    coords: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create sub-structure indexing from aligned crossover indices.
    
    Args:
        alignCOs_index: Aligned crossover indices
        bp_Hindex: Base pair helix indices
        coords: Optional coordinate tensor (unused in current implementation)
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Base pair sub-structure indices and helix sub-structure indices
    """
    align_index = alignCOs_index
    
    # Remove self-loops and coalesce edges for clean helix connections
    align_Hindex, _ = remove_self_loops(
        coalesce(bp_Hindex[align_index], num_nodes=bp_Hindex.amax() + 1)
    )
    
    # Segment helices into sub-structures
    H_Sindex = segmentation(align_Hindex, min_nodes=0)
    
    # Map base pairs to their corresponding sub-structures
    bps_Sindex = H_Sindex[bp_Hindex]
    
    return bps_Sindex, H_Sindex


def segmentation(edge_index: torch.Tensor, 
                 min_nodes: int = 0, 
                 return_extra: bool = False, 
                 num_nodes: Optional[int] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]]:
    """
    Divide graph into connected components (groups) like helices or sub-structures.
    
    Args:
        edge_index: Edge connectivity tensor
        min_nodes: Minimum number of nodes required for a valid group
        return_extra: Whether to return additional information (group lengths and lists)
        num_nodes: Total number of nodes (inferred if None)
        
    Returns:
        Union[torch.Tensor, Tuple]: Node group indices, optionally with group lengths and lists
    """
    if num_nodes is None:
        num_nodes = edge_index.amax() + 1
    
    # Convert to NetworkX graph for connected components analysis
    groups_graph = to_networkx(
        Data(edge_index=edge_index, num_nodes=num_nodes), 
        to_undirected=True
    )
    groups = list(nx.connected_components(groups_graph))
    num_groups = len(groups)
    
    # Initialize group assignment tensor
    node_Gindex = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
    group_len = []
    group_lst = []
    idx = 0
    
    # Process each connected component
    for i in range(num_groups):
        bp_lst = []
        
        if len(groups[i]) > min_nodes:  # Only keep groups with sufficient nodes
            group_len.append(len(groups[i]))
            
            # Assign group index to all nodes in this component
            for node_ID in groups[i]:
                node_Gindex[node_ID] = idx
                bp_lst.append(node_ID)
                
            group_lst.append(torch.tensor(bp_lst, dtype=torch.long, device=edge_index.device))
            idx += 1
        else:
            # Mark nodes in small groups as invalid (-1)
            for node_ID in groups[i]:
                node_Gindex[node_ID] = -1
                
    group_len = torch.tensor(group_len, dtype=torch.long, device=edge_index.device)
    
    if return_extra:
        return node_Gindex, group_len, group_lst
    else:
        return node_Gindex


# =======================================================================================
# COORDINATE AND ROUTING FUNCTIONS
# =======================================================================================

def reindexing_base_pair(coords: torch.Tensor, 
                         edge_index: torch.Tensor, 
                         bp_Hindex: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Re-index base pairs following the route order within each helix.
    
    Args:
        coords: Base pair coordinates
        edge_index: Edge connectivity
        bp_Hindex: Base pair helix indices
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Reordered coordinates and helix indices
    """
    num_bps = coords.size(0)
    deg = degree(edge_index[0], num_nodes=num_bps)
    dev = coords.device
    
    # Process each helix separately
    num_helices = bp_Hindex.amax() + 1
    new_coords = torch.zeros_like(coords, device=dev)
    new_bpsdeg = torch.zeros_like(deg, device=dev)
    new_helix_index = torch.zeros_like(bp_Hindex, device=dev)
    src, dst = edge_index
    offset_bpsidx = 0
    
    for i in range(num_helices):
        # Get base pairs belonging to current helix
        bp_mask = bp_Hindex == i
        num_bps = bp_mask.long().sum()
        
        # Find helix end points (nodes with degree 1)
        end_bpidx = bp_mask.nonzero()[(deg[bp_mask] == 1).nonzero()]
        
        # Get edges within this helix
        helix_edge_mask = torch.logical_and(bp_mask[src], bp_mask[dst])
        helix_edge_index = edge_index[:, helix_edge_mask]
        
        # Choose starting point for route traversal
        start_idx = end_bpidx[0].squeeze() if (len(end_bpidx) != 0) else helix_edge_index[0, 0]
        
        # Generate Hamiltonian route through the helix
        old_bpsidx = edge2route(helix_edge_index, start_idx, num_bps)
        
        # Re-index base pairs in sequential order
        new_bpsidx = torch.arange(num_bps, dtype=torch.long, device=dev) + offset_bpsidx
        offset_bpsidx = offset_bpsidx + num_bps
        
        # Update coordinate and index arrays
        new_coords[new_bpsidx, :] = coords[old_bpsidx, :]
        new_bpsdeg[new_bpsidx] = deg[old_bpsidx]
        new_helix_index[new_bpsidx] = i
        
    return new_coords, new_helix_index


def edge2route(edge_index: torch.Tensor, 
               start_idx: torch.Tensor, 
               num_nodes: int) -> torch.Tensor:
    """
    Generate route from edge indices using depth-first traversal.
    
    Args:
        edge_index: Edge connectivity tensor
        start_idx: Starting node index
        num_nodes: Total number of nodes to traverse
        
    Returns:
        torch.Tensor: Route as sequence of node indices
    """
    route = []
    route.append(start_idx)
    src, dst = edge_index
    
    # Traverse graph by following edges, avoiding backtracking
    for i in range(num_nodes - 1):
        # Find neighbors of current node
        mask = src == route[-1]
        neighbors = edge_index[-1, mask]
        
        if len(route) > 1:
            # Choose neighbor that's not the previous node (avoid backtracking)
            if neighbors[0] != route[-2]:
                route.append(neighbors[0])
            else:
                route.append(neighbors[1])
        else:
            # First step: choose first available neighbor
            route.append(neighbors[0])
            
    return torch.hstack(route).long()


# =======================================================================================
# GEOMETRIC CALCULATION FUNCTIONS
# =======================================================================================

def vectors_to_angles(src_vec: torch.Tensor, dst_vec: torch.Tensor) -> torch.Tensor:
    """
    Calculate angles between pairs of vectors.
    
    Args:
        src_vec: Source vectors
        dst_vec: Destination vectors
        
    Returns:
        torch.Tensor: Angles between vector pairs
    """
    # Calculate dot product
    inner_product = (src_vec * dst_vec).sum(dim=1)
    
    # Calculate vector magnitudes
    src_norm = src_vec.pow(2).sum(dim=1).pow(0.5)
    dst_norm = dst_vec.pow(2).sum(dim=1).pow(0.5)
    
    # Calculate angles using arccos formula
    angles = torch.acos(inner_product / (2 * src_norm * dst_norm))
    
    return angles


def edges_len(coords: torch.Tensor, 
              edge_index: torch.Tensor, 
              sort: Optional[str] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Calculate edge lengths from coordinates and edge indices.
    
    Args:
        coords: Node coordinates
        edge_index: Edge connectivity
        sort: Optional sorting mode ("descending" or "increasing")
        
    Returns:
        Union[torch.Tensor, Tuple]: Edge lengths, optionally with sorting indices
    """
    src, dst = edge_index
    
    # Calculate relative vectors and their magnitudes
    edge_relvec = coords[dst] - coords[src]
    edge_len = edge_relvec.norm(dim=-1, keepdim=False)

    # Apply optional sorting
    if sort == "descending":
        sorted_edge_len, sorted_edge_idx = torch.sort(edge_len, descending=True)
    elif sort == "increasing":
        sorted_edge_len, sorted_edge_idx = torch.sort(edge_len, descending=False)
    else:
        return edge_len
        
    return sorted_edge_len, sorted_edge_idx


def To_undirected_edge_index(coords: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """
    Generate undirected edge index from directed edges.
    
    Args:
        coords: Node coordinates
        edge_index: Directed edge indices
        
    Returns:
        torch.Tensor: Undirected edge indices
    """
    # Calculate edge centers to identify unique edges
    cens = (coords[edge_index[0]] + coords[edge_index[1]]) / 2
    cens = cens.cpu().detach().numpy()
    
    # Find unique edges based on their centers
    _, idx = np.unique(cens, axis=0, return_index=True)
    
    # Create undirected edge index (both directions for each edge)
    edge_index_undirected = torch.zeros_like(edge_index)
    edge_index_undirected[:, ::2] = edge_index[:, idx]
    edge_index_undirected[:, 1::2] = torch.vstack((edge_index[1, idx], edge_index[0, idx]))
    
    return edge_index_undirected


# =======================================================================================
# DNA STRUCTURE POSITIONING FUNCTIONS
# =======================================================================================

def set_position(cm_bp: torch.Tensor, 
                 e1_bp: torch.Tensor, 
                 e2_bp: torch.Tensor, 
                 e3_bp: torch.Tensor, 
                 scale_factor: float = 1.0, 
                 preferred: bool = True) -> torch.Tensor:
    """
    Set DNA base position relative to base pair center with directional offset.
    
    Args:
        cm_bp: Base pair center of mass
        e1_bp: First basis vector
        e2_bp: Second basis vector  
        e3_bp: Third basis vector
        scale_factor: Scaling factor for positioning
        preferred: Whether to use preferred (True) or complementary (False) strand direction
        
    Returns:
        torch.Tensor: Positioned coordinates
    """
    COM_SHIFT = 0.54  # Standard DNA base pair center offset
    
    if not preferred:
        # Position on complementary strand (negative direction)
        cm_pos = cm_bp - scale_factor * COM_SHIFT * e2_bp
    else:
        # Position on preferred strand (positive direction)
        cm_pos = cm_bp + scale_factor * COM_SHIFT * e2_bp
        
    return cm_pos


def bps_to_bases(coords: torch.Tensor, 
                 triads: torch.Tensor, 
                 scale_factor: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert base pair coordinates to individual base coordinates.
    
    Args:
        coords: Base pair center coordinates
        triads: Orientation triads for each base pair
        scale_factor: Scaling factor for base positioning
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Scaffold and staple base coordinates
    """
    num_bps = coords.size(0)
    coords_sca = torch.zeros_like(coords)  # Scaffold coordinates
    coords_sta = torch.zeros_like(coords)  # Staple coordinates
    
    # Process each base pair
    for pid in range(num_bps):
        cm_bp = coords[pid, :]
        e1_bp = triads[pid, 1, :]  # First basis vector
        e2_bp = triads[pid, 2, :]  # Second basis vector  
        e3_bp = triads[pid, 0, :]  # Third basis vector
        
        # Position scaffold and staple bases on opposite sides
        coords_sca[pid, :] = set_position(cm_bp, e1_bp, e2_bp, e3_bp, scale_factor, preferred=True)
        coords_sta[pid, :] = set_position(cm_bp, e1_bp, e2_bp, e3_bp, scale_factor, preferred=False)
        
    return coords_sca, coords_sta


# =======================================================================================
# UTILITY FUNCTIONS FOR TENSOR OPERATIONS
# =======================================================================================

def delete_row_tensor(a: torch.Tensor, del_row: int, device: torch.device) -> torch.Tensor:
    """
    Delete a row from a tensor.
    
    Args:
        a: Input tensor
        del_row: Row index to delete
        device: Target device
        
    Returns:
        torch.Tensor: Tensor with specified row removed
    """
    n = a.cpu().detach().numpy()
    n = np.delete(n, del_row, 0)
    n = torch.from_numpy(n).to(device)
    return n


def pcl_filter(coords: torch.Tensor, r: float = 0.4) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Filter point cloud by removing high-density regions.
    
    Args:
        coords: Point cloud coordinates
        r: Radius for neighborhood analysis
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Filtered coordinates and edge indices
    """
    num_nodes = coords.size(0)
    
    # Build radius graph to analyze local density
    edge_index = radius_graph(coords, r=r, loop=False)
    
    # Calculate node degrees (local density measure)
    deg = degree(edge_index[0], num_nodes=coords.size(0))
    
    # Identify high-density edges for removal
    src, dst = edge_index
    edge_mask = deg[src] + deg[dst] > 4
    remove_index = torch.unique(edge_index[:, edge_mask].flatten())
    
    # Create node mask to keep low-density nodes
    node_mask = torch.ones(num_nodes, dtype=torch.bool, device=coords.device)
    node_mask[remove_index] = False
    
    # Filter coordinates and rebuild edge index
    coords = coords[node_mask, :]
    edge_index = radius_graph(coords, r=r, loop=False)
    
    return coords, edge_index


def unique_WOsort(array: np.ndarray) -> np.ndarray:
    """
    Get unique elements without sorting (preserves original order).
    
    Args:
        array: Input numpy array
        
    Returns:
        np.ndarray: Unique elements in original order
    """
    uniq, index = np.unique(array, return_index=True)
    return uniq[index.argsort()]


# =======================================================================================
# MASKING AND FILTERING FUNCTIONS
# =======================================================================================

def EndHelix_bpmask(num_bps: int, 
                    edge_index: torch.Tensor, 
                    mask_seed_idx: torch.Tensor, 
                    k_hop: int) -> torch.Tensor:
    """
    Initialize base-pair mask around helix ends.
    
    Args:
        num_bps: Total number of base pairs
        edge_index: Edge connectivity
        mask_seed_idx: Seed indices for masking
        k_hop: Number of hops for mask expansion
        
    Returns:
        torch.Tensor: Boolean mask for base pairs
    """
    bp_mask = torch.ones(num_bps, dtype=torch.bool, device=edge_index.device)
    node_idx = mask_seed_idx.flatten().tolist()
    
    if len(node_idx) > 0:
        # Find k-hop neighborhood around seed nodes
        subset, _, _, _ = k_hop_subgraph(node_idx, k_hop, edge_index, relabel_nodes=False)
        bp_mask[subset] = False
        
    return bp_mask


def Radius_bpmask(num_bps: int, 
                  edge_index: torch.Tensor, 
                  mask_seed_idx: torch.Tensor, 
                  k_hop: int) -> torch.Tensor:
    """
    Initialize base-pair mask using radius-based selection.
    
    Args:
        num_bps: Total number of base pairs
        edge_index: Edge connectivity
        mask_seed_idx: Seed indices for masking
        k_hop: Number of hops for mask expansion
        
    Returns:
        torch.Tensor: Boolean mask for base pairs
    """
    bp_mask = torch.zeros(num_bps, dtype=torch.bool, device=edge_index.device)
    node_idx = mask_seed_idx.to(torch.long).flatten().tolist()
    
    # Find k-hop neighborhood and set mask to True
    subset, _, _, _ = k_hop_subgraph(node_idx, k_hop, edge_index, relabel_nodes=False)
    bp_mask[subset] = True
    
    return bp_mask


def update_bpmask(bp_mask: torch.Tensor, 
                  edge_index: torch.Tensor, 
                  mask_seed_idx: torch.Tensor, 
                  k_hop: int = 10, 
                  additional_edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Update base-pair mask by removing areas around specified indices.
    
    Args:
        bp_mask: Existing base pair mask
        edge_index: Primary edge connectivity
        mask_seed_idx: Seed indices for mask updates
        k_hop: Number of hops for mask expansion
        additional_edge_index: Optional additional edges to consider
        
    Returns:
        torch.Tensor: Updated base pair mask
    """
    # Combine edge indices if additional edges provided
    if additional_edge_index is not None:
        edge_index = torch.hstack((edge_index, additional_edge_index))
    
    # Find k-hop neighborhood around seed nodes and mask them
    node_idx = mask_seed_idx.to(torch.long).flatten().tolist()
    subset, _, _, _ = k_hop_subgraph(node_idx, k_hop, edge_index, relabel_nodes=False)
    bp_mask[subset] = False
    
    return bp_mask


def Undirected_edge_index(coords: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """
    Convert directed edge index to undirected by removing duplicate edges.
    
    Args:
        coords: Node coordinates
        edge_index: Directed edge indices
        
    Returns:
        torch.Tensor: Undirected edge indices
    """
    # Calculate edge centers to identify unique edges
    cens = (coords[edge_index[0], :] + coords[edge_index[1], :]) / 2
    cens = cens.cpu().detach().numpy()
    
    # Find unique edges and create bidirectional representation
    _, idx = np.unique(cens, axis=0, return_index=True)
    edge_index_Undirected = torch.zeros(2, 2 * idx.shape[0], dtype=torch.long, device=edge_index.device)
    edge_index_Undirected[:, ::2] = edge_index[:, idx]
    edge_index_Undirected[:, 1::2] = edge_index[:, idx].flip([0])
    
    return edge_index_Undirected


# =======================================================================================
# EDGE AND NODE MANIPULATION FUNCTIONS
# =======================================================================================

def del_edges(edge_index: torch.Tensor, remove_ID: torch.Tensor, num_bps: int) -> torch.Tensor:
    """
    Delete edges connected to specified nodes.
    
    Args:
        edge_index: Edge connectivity
        remove_ID: Node indices to remove
        num_bps: Total number of base pairs
        
    Returns:
        torch.Tensor: Filtered edge index
    """
    # Create mask for nodes to keep
    node_mask = torch.ones(num_bps, dtype=torch.bool, device=edge_index.device)
    node_mask[remove_ID] = False
    
    # Filter edges to keep only those between valid nodes
    row, col = edge_index
    edge_mask = torch.logical_and(node_mask[row], node_mask[col])
    edge_index = edge_index[:, edge_mask]
    
    return edge_index


def replace_edges(edge_index: torch.Tensor, 
                  old_edge_index: torch.Tensor, 
                  new_edge_index: torch.Tensor, 
                  num_bps: int) -> torch.Tensor:
    """
    Replace edges between specified nodes.
    
    Args:
        edge_index: Original edge connectivity
        old_edge_index: Nodes whose edges to replace
        new_edge_index: New edge connectivity
        num_bps: Total number of base pairs
        
    Returns:
        torch.Tensor: Updated edge index
    """
    # Create mask for nodes whose edges need replacement
    node_mask = torch.zeros(num_bps, dtype=torch.bool, device=edge_index.device)
    node_mask[old_edge_index] = True
    
    # Find edges between the specified nodes
    row, col = edge_index
    edge_mask = torch.logical_and(node_mask[row], node_mask[col])
    
    # Replace the identified edges
    edge_index[:, edge_mask] = new_edge_index
    
    return edge_index


def ins_inner_nodes(coords: torch.Tensor, 
                    src_ID: int, 
                    dst_ID: int, 
                    num_ins: Optional[int] = None) -> torch.Tensor:
    """
    Insert intermediate nodes between two points.
    
    Args:
        coords: Node coordinates
        src_ID: Source node ID
        dst_ID: Destination node ID
        num_ins: Number of nodes to insert (calculated if None)
        
    Returns:
        torch.Tensor: Coordinates of inserted nodes
    """
    src_coords = coords[src_ID: src_ID + 1, :]
    dst_coords = coords[dst_ID: dst_ID + 1, :]
    
    # Calculate relative vector and distance
    relvec = (dst_coords - src_coords)
    length = relvec.norm(dim=-1, keepdim=False)
    
    # Determine number of insertions based on DNA base spacing (0.34 nm)
    if num_ins is None:
        num_ins = int(torch.round(length / 0.34) + 1)
    
    # Generate evenly spaced intermediate points
    pts_idx = torch.linspace(0., 1.0, steps=int(num_ins + 2), device=coords.device)
    inner_pts = pts_idx[1:-1, None] * relvec + src_coords
    
    return inner_pts


def del_nodes(coords: torch.Tensor, remove_ID: torch.Tensor) -> torch.Tensor:
    """
    Delete specified nodes from coordinate tensor.
    
    Args:
        coords: Node coordinates
        remove_ID: Node indices to remove
        
    Returns:
        torch.Tensor: Filtered coordinates
    """
    node_mask = torch.ones(coords.size(0), dtype=torch.bool, device=coords.device)
    node_mask[remove_ID] = False
    coords = coords[node_mask, :]
    
    return coords


def perpendicular_vector(v: torch.Tensor) -> torch.Tensor:
    """
    Calculate a perpendicular vector to the input vector.
    
    Args:
        v: Input vector
        
    Returns:
        torch.Tensor: Perpendicular vector
        
    Raises:
        ValueError: If input is zero vector
    """
    if v[1] == 0 and v[2] == 0:
        if v[0] == 0:
            raise ValueError('zero vector')
        else:
            arbitrary_vec = torch.tensor([0, 1, 0], dtype=v.dtype, device=v.device)
            return torch.cross(v, arbitrary_vec)
    else:
        arbitrary_vec = torch.tensor([1, 0, 0], dtype=v.dtype, device=v.device)
        
    return torch.cross(v, arbitrary_vec)