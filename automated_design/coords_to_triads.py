import roma
import math
import torch
import warnings
import numpy as np
from scipy import interpolate
from numpy import linalg as LA
from automated_design.visual import plot_dna
from torch_geometric.utils import scatter, to_undirected
from automated_design.routing import cycle_graph_to_route
from automated_design.ults import *

# Constants
DEFAULT_ONE_TURN_ANGLE = (-2) * math.pi / 10.5
HELIX_TURNS_PER_REVOLUTION = 10.5
SPLINE_SMOOTHING_FACTOR = 200
SPLINE_DEGREE = 3

#%% triads alignment
def assign_triads(alignCOs_index,
                  RB_index, 
                  SS_index,
                  scafCO_index, 
                  coords, 
                  bp_Hindex, 
                  bp_Lindex, 
                  scaf_route,
                  one_turn_angle=DEFAULT_ONE_TURN_ANGLE):
    """
    Assign triads to base-pairs based on helix alignment and scaffold routing.
    
    Args:
        alignCOs_index (2D Tensor): Alignment cross-over indices
        RB_index (2D Tensor): Regular bond indices
        SS_index (2D Tensor): Single-strand indices
        scafCO_index (2D Tensor): Scaffold cross-over indices
        coords (2D Tensor): Base-pair coordinates [num_bps x 3]
        bp_Hindex (1D Tensor): Base-pair helix index [num_bps]
        bp_Lindex (1D Tensor): Base-pair loop (or face) index [num_bps]
        scaf_route (1D Tensor): List of base-pair indices for scaffold route
        one_turn_angle (float): Angle for one helical turn (default: -2π/10.5)
        
    Returns:
        tuple: (triads, dna_axis, scaf_route, scafCO_index)
            - triads (3D Tensor): Triads for aligned helices [[Tx];[-Tz];[Ty]] [3x3]
            - dna_axis (3D Tensor): DNA axis vectors
            - scaf_route (1D Tensor): Updated scaffold route
            - scafCO_index (2D Tensor): Updated scaffold cross-over indices
    """
    # Extract alignment positions (every 4th element)
    align_index = alignCOs_index[:, ::4]
    
    # Re-order aligned positions from high degree to low degree
    # This prioritizes helices with more connections for better alignment
    align_helices = bp_Hindex[align_index]
    helix_degrees = degree(align_helices.flatten())
    alignment_degrees = helix_degrees[align_helices]
    total_degrees = alignment_degrees[0] + alignment_degrees[1]
    
    # Sort by degree (descending) to process high-degree helices first
    _, indices = torch.sort(total_degrees, descending=True, stable=True)
    align_index = align_index[:, indices]
    
    # Generate local coordinate system for each base-pair
    dna_axis, rotation_axis, helix_bp_list = SCroute_to_rotaxis(
        coords, bp_Hindex, scaf_route, align_index
    )
    
    # Assign helix turns and triads
    turns, _ = assign_helix_triads(align_index, bp_Hindex, helix_bp_list, dna_axis)
    turn_angles = one_turn_angle * turns
    triads = angles_to_triads(turn_angles, dna_axis, rotation_axis)
    
    # Convert base-pairs to bases and adjust cross-overs
    scaffold_coords, _ = bps_to_bases(coords, triads)
    scafCO_index = adjust_scafCOs(scafCO_index, scaffold_coords)
          
    # Update scaffold route and cross-over edge indices
    num_bps = coords.size(0)
    edge_index = torch.cat((RB_index, scafCO_index, SS_index), dim=1)    
    
    # Filter regular bonds to maintain proper connectivity
    node_degrees = degree(edge_index[0, :], num_bps)
    regular_bond_mask = torch.logical_and(
        node_degrees[RB_index[0]] == 3, 
        node_degrees[RB_index[1]] == 3
    )
    
    # Construct scaffold edge index
    scaffold_edge_index = torch.hstack((
        RB_index[:, ~regular_bond_mask], 
        scafCO_index, 
        SS_index
    ))
    
    # Validate scaffold routing (should form a single loop)
    final_degrees = degree(scaffold_edge_index[0], num_bps)
    if len((final_degrees != 2).nonzero()) > 0:
        warnings.warn("Warning: error in scaffold routing, it's not one-loop graph")
        
    # Update scaffold route based on new edge connectivity
    scaf_route = cycle_graph_to_route(
        scaffold_edge_index, 
        scaf_route[0], 
        num_bps, 
        second_idx=scaf_route[1]
    )
    
    return triads, dna_axis, scaf_route, scafCO_index

#%% align helix with align_index
def assign_helix_triads(align_index, bp_Hindex, helix_bp_lst, dna_axis):
    """
    Assign triads to helices based on alignment constraints.
    
    This function processes helix alignments sequentially, ensuring that
    connected helices have compatible orientations by adjusting their
    rotational angles.

    Args:
        align_index (2D Tensor): Alignment edges [2 x N]
        bp_Hindex (1D Tensor): Helix index for each base-pair [num_bps]
        helix_bp_lst (list): List of base-pair indices for each helix
        dna_axis (3D Tensor): DNA axis vectors for each base-pair

    Returns:
        tuple: (turns, aligned_helix)
            - turns (1D Tensor): Turn angles for each base-pair
            - aligned_helix (1D Tensor): Boolean mask of aligned helices
    """
    device = align_index.device
    num_bps = bp_Hindex.size(0)
    num_helices = bp_Hindex.amax() + 1
    num_alignments = align_index.size(1)
    
    # Initialize turn angles and alignment status
    turns = torch.zeros(num_bps, dtype=torch.float32, device=device)
    aligned_helix = torch.zeros(num_helices, dtype=torch.bool, device=device)
    
    # Process each alignment constraint
    for i in range(num_alignments):
        alignment_indices = align_index[:, i]
        helix_indices = bp_Hindex[alignment_indices]
        alignment_status = aligned_helix[helix_indices]
        
        # Check if both helices are already aligned (invalid configuration)
        if alignment_status.sum().item() > 1:
            print(f"These 2 helices cannot be aligned: [{helix_indices[0]}, {helix_indices[1]}]")
            continue
            
        # Perform helix alignment
        helix_src = helix_bp_lst[helix_indices[0]].squeeze()
        helix_dst = helix_bp_lst[helix_indices[1]].squeeze()
        
        # Calculate turn angles for source and destination helices
        if alignment_status[0]:
            # Source helix already aligned, use existing turns
            helix_src_turns = turns[helix_src].clone().detach()
        else:
            # Source helix not aligned, calculate relative turns
            helix_src_turns = (helix_src - alignment_indices[0]).to(torch.float32)
            
        if alignment_status[1]:
            # Destination helix already aligned, use existing turns
            helix_dst_turns = turns[helix_dst].clone().detach()
        else:
            # Destination helix not aligned, calculate relative turns
            helix_dst_turns = (helix_dst - alignment_indices[1]).to(torch.float32)
        
        # Ensure turns start from negative values (consistent direction)
        helix_src_turns = helix_src_turns if helix_src_turns[0] < 0 else -helix_src_turns
        helix_dst_turns = helix_dst_turns if helix_dst_turns[0] < 0 else -helix_dst_turns
        
        # Extract axis vectors at alignment points
        src_e1 = dna_axis[alignment_indices[0], 0, :][None, :]  # Z-axis
        src_e2 = dna_axis[alignment_indices[0], 2, :][None, :]  # Y-axis
        dst_e1 = dna_axis[alignment_indices[1], 0, :][None, :]  # Z-axis
        dst_e2 = dna_axis[alignment_indices[1], 2, :][None, :]  # Y-axis
        
        # Calculate angular offset between helices and adjust accordingly
        if alignment_status[1]:
            # Destination helix is fixed, adjust source helix
            offset_turns = HELIX_TURNS_PER_REVOLUTION * (
                vectors_to_angles(src_e2, -dst_e2) / (2 * math.pi)
            )
            offset_axis = torch.cross(src_e2, -dst_e2)
            offset_sign = torch.sign(torch.dot(offset_axis.squeeze(), src_e1.squeeze()))
            helix_src_turns = helix_src_turns - offset_sign * offset_turns
        else:
            # Source helix is fixed, adjust destination helix
            offset_turns = HELIX_TURNS_PER_REVOLUTION * (
                vectors_to_angles(dst_e2, -src_e2) / (2 * math.pi)
            )
            offset_axis = torch.cross(dst_e2, -src_e2)
            offset_sign = torch.sign(torch.dot(offset_axis.squeeze(), dst_e1.squeeze()))
            helix_dst_turns = helix_dst_turns - offset_sign * offset_turns
                
        # Update turn angles and alignment status
        turns[helix_src] = helix_src_turns 
        turns[helix_dst] = helix_dst_turns  
        aligned_helix[helix_indices] = True
    
    # Handle unaligned helices (if any remain)
    if aligned_helix.sum().item() < num_helices:
        print("Some helices are not long enough to form DX edges!")
        for helix_id in range(num_helices):
            if not aligned_helix[helix_id]:
                bp_list = helix_bp_lst[helix_id].squeeze()
                # Set turns relative to the first base-pair in the helix
                turns[bp_list] = (bp_list - bp_list.min()).to(torch.float32)
    
    return turns, aligned_helix

#%% adjust aligned_positions for better configuration of scaffold route
def adjust_scafCOs(crossover_indices, coords):
    """
    Adjust scaffold cross-over positions for optimal routing configuration.
    
    This function finds the closest base-pair positions between aligned helices
    and creates optimized cross-over connections with neighboring bonds.
    
    Args:
        crossover_indices (2D Tensor): Cross-over indices [2 x N]
        coords (2D Tensor): Base-pair coordinates [num_bps x 3]
        
    Returns:
        2D Tensor: Adjusted cross-over indices with neighbors
    """
    # Extract alignment positions (every 4th element)
    align_index = crossover_indices[:, ::4]
    num_alignments = align_index.size(1)
    
    adjusted_crossovers = []
    
    for i in range(num_alignments):
        src_bp_id, dst_bp_id = align_index[:, i]
        
        # Get coordinate windows around alignment points
        src_coords = coords[src_bp_id - 6 : src_bp_id + 7, :]
        dst_coords = coords[dst_bp_id - 6 : dst_bp_id + 7, :]
        
        # Find closest pair of base-pairs
        pairwise_distances = torch.cdist(src_coords, dst_coords, p=2.0)
        min_distance_mask = pairwise_distances == torch.amin(pairwise_distances)
        closest_indices = (min_distance_mask.nonzero() - 6).reshape(-1, 2)
        
        # Create new cross-over at closest positions
        new_crossover = torch.tensor([
            [src_bp_id + closest_indices[0, 0], dst_bp_id + closest_indices[0, 1]], 
            [dst_bp_id + closest_indices[0, 1], src_bp_id + closest_indices[0, 0]]
        ], dtype=torch.long, device=coords.device)
        
        # Add neighboring cross-over bonds for better connectivity
        original_neighbor_distance = abs(
            (crossover_indices[:, 4*i+2][0] - crossover_indices[:, 4*i+2][1]).abs() - 
            (crossover_indices[:, 4*i][0] - crossover_indices[:, 4*i][1]).abs()
        ).abs()
        
        if original_neighbor_distance > 1:
            # Different direction neighbors
            crossover_neighbors = torch.tensor([
                [new_crossover[0, 0] - 1, new_crossover[1, 0] + 1, 
                 new_crossover[0, 0] + 1, new_crossover[1, 0] - 1],
                [new_crossover[1, 0] + 1, new_crossover[0, 0] - 1, 
                 new_crossover[1, 0] - 1, new_crossover[0, 0] + 1]
            ], dtype=torch.long, device=coords.device)
        else:
            # Same direction neighbors
            crossover_neighbors = torch.tensor([
                [new_crossover[0, 0] - 1, new_crossover[1, 0] - 1, 
                 new_crossover[0, 0] + 1, new_crossover[1, 0] + 1],
                [new_crossover[1, 0] - 1, new_crossover[0, 0] - 1, 
                 new_crossover[1, 0] + 1, new_crossover[0, 0] + 1]
            ], dtype=torch.long, device=coords.device)
        
        # Select the two shortest neighbor bonds
        _, sorted_indices = edges_len(coords, crossover_neighbors, sort="increasing")
        new_crossover = torch.hstack((new_crossover, crossover_neighbors[:, sorted_indices][:, :2]))
        
        adjusted_crossovers.append(new_crossover)
    
    return torch.hstack(adjusted_crossovers)

#%% convert turn angles to triads
def angles_to_triads(angles, dna_axis, rotation_matrices):
    """
    Convert turn angles to triads (local coordinate systems).
    
    Args:
        angles (1D Tensor): Turn angles for each base-pair
        dna_axis (3D Tensor): DNA axis vectors [num_bps x 3 x 3]
        rotation_matrices (3D Tensor): Rotation matrices for alignment
        
    Returns:
        3D Tensor: Triads for each base-pair [num_bps x 3 x 3]
    """
    # Create rotation vectors around x-axis
    rotation_vectors = angles[:, None] * torch.tensor(
        [[1., 0., 0.]], dtype=torch.float32, device=angles.device
    )
    
    # Convert rotation vectors to rotation matrices
    reference_triads = roma.rotvec_to_rotmat(rotation_vectors, epsilon=1e-12)
    
    # Apply rotation alignment
    triads = torch.bmm(reference_triads, rotation_matrices.transpose(-1, -2))
    
    return triads

#%% generate base-pair z-axis follow scaffold route
def SCroute_to_rotaxis(coords, bp_Hindex, scaf_route, align_index):
    """
    Generate rotation axes for base-pairs following the scaffold route.
    
    This function creates local coordinate systems for each base-pair,
    with the z-axis following the helix direction and x,y-axes aligned
    based on cross-over connections.
    
    Args:
        coords (2D Tensor): Base-pair coordinates [num_bps x 3]
        bp_Hindex (1D Tensor): Helix index for each base-pair
        scaf_route (1D Tensor): Scaffold route indices
        align_index (2D Tensor): Alignment indices
        
    Returns:
        tuple: (dna_axis, rotation_matrices, helices)
            - dna_axis (3D Tensor): DNA axis vectors [num_bps x 3 x 3]
            - rotation_matrices (3D Tensor): Rotation matrices for alignment
            - helices (list): List of base-pair indices for each helix
    """
    num_bps = coords.size(0)
    num_helices = bp_Hindex.amax() + 1
    align_index = to_undirected(align_index)
    
    # Initialize axis vectors
    dna_x_axis = torch.zeros_like(coords)  # Cross-over direction
    dna_y_axis = torch.zeros_like(coords)  # Perpendicular to x and z
    dna_z_axis = torch.zeros_like(coords)  # Helix direction
    dna_axis = torch.zeros(num_bps, 3, 3, dtype=torch.float32, device=coords.device)
    
    helices = []
    
    # Process each helix
    for helix_id in range(num_helices):
        # Get base-pairs belonging to this helix
        helix_bp_mask = (bp_Hindex == helix_id)
        helix = helix_bp_mask.nonzero().squeeze()
        
        # Determine helix direction based on scaffold route
        helix_route = scaf_route[bp_Hindex[scaf_route] == helix_id]
        if len(helix_route) > 1 and helix_route[0] > helix_route[1]:
            helix = helix.flip([0])
        
        helices.append(helix)
        
        # Generate smooth z-axis (helix direction) using spline interpolation
        dna_z_axis[helix, :] = slerp_z_axis(coords[helix, :])
        
        # Find cross-over connections for this helix
        alignment_mask = bp_Hindex[align_index][0] == helix_id
        helix_alignments = align_index[:, alignment_mask].reshape(2, -1)
        
        if helix_alignments.nelement() == 0:
            # No cross-overs: use perpendicular vector to z-axis
            perpendicular_vec = perpendicular_vector(dna_z_axis[helix[0], :]).reshape(-1, 3)
            perpendicular_vec = perpendicular_vec / perpendicular_vec.norm(dim=-1, keepdim=True)
            dna_x_axis[helix, :] = torch.cross(
                perpendicular_vec, 
                dna_z_axis[helix[0], :][None, :], 
                dim=1
            )
        else:
            # Use cross-over direction for x-axis
            crossover_vector = (
                coords[helix_alignments[1, 0], :] - coords[helix_alignments[0, 0], :]
            ).reshape(-1, 3)
            crossover_vector = crossover_vector / crossover_vector.norm(dim=-1, keepdim=True)
            dna_x_axis[helix, :] = torch.cross(
                crossover_vector, 
                dna_z_axis[helix_alignments[0, 0], :][None, :], 
                dim=1
            )
        
        # Normalize x-axis and compute y-axis
        dna_x_axis[helix, :] = dna_x_axis[helix, :] / dna_x_axis[helix, :].norm(dim=-1, keepdim=True)
        dna_y_axis[helix, :] = torch.cross(dna_z_axis[helix, :], dna_x_axis[helix, :], dim=1)
    
    # Assemble final axis system
    dna_axis[:, 0, :] = dna_z_axis / dna_z_axis.norm(dim=-1, keepdim=True)  # Z-axis
    dna_axis[:, 1, :] = dna_x_axis / dna_x_axis.norm(dim=-1, keepdim=True)  # X-axis
    dna_axis[:, 2, :] = dna_y_axis / dna_y_axis.norm(dim=-1, keepdim=True)  # Y-axis
    
    # Calculate rotation matrices for alignment
    reference_axis = torch.eye(3, device=coords.device)
    reference_axis = reference_axis.unsqueeze(0).expand(num_bps, 3, 3)
    reference_x_axis = torch.tensor(
        [[1., 0., 0.]], dtype=torch.float32, device=coords.device
    )
    
    if num_helices < 5:
        # For simple structures, align only z-axis
        rotation_matrices = roma.rigid_vectors_registration(
            reference_x_axis[None, :, :], 
            dna_z_axis[:, None, :]
        )
    else:
        # For complex structures, align both x and z axes
        rotation_matrices = roma.rigid_vectors_registration(
            reference_axis[:, :2, :], 
            dna_axis[:, :2, :]
        )
    
    return dna_axis, rotation_matrices, helices

def slerp_z_axis(coords):
    """
    Generate smooth z-axis vectors using spherical linear interpolation.
    
    This function fits a spline to the coordinate trajectory and computes
    tangent vectors, which represent the helix direction at each point.
    
    Args:
        coords (2D Tensor): Coordinates along the helix [num_points x 3]
        
    Returns:
        1D Tensor: Normalized tangent vectors [num_points x 3]
    """
    # Convert to numpy for spline fitting
    coords_np = coords.cpu().detach().numpy()
    x_coords, y_coords, z_coords = coords_np[:, 0], coords_np[:, 1], coords_np[:, 2]
    
    # Fit parametric spline to the trajectory
    spline_params, _ = interpolate.splprep(
        [x_coords, y_coords, z_coords], 
        k=SPLINE_DEGREE, 
        s=SPLINE_SMOOTHING_FACTOR
    )
    
    # Evaluate spline derivatives (tangent vectors)
    parameter_values = np.linspace(0, 1, coords_np.shape[0])
    tangent_x, tangent_y, tangent_z = interpolate.splev(
        [parameter_values], spline_params, der=1
    )
    
    # Assemble and normalize tangent vectors
    tangent_vectors = np.transpose(np.vstack((tangent_x, tangent_y, tangent_z)))
    tangent_vectors = tangent_vectors / LA.norm(tangent_vectors, axis=1)[:, None]
    
    # Convert back to tensor
    z_axis = torch.from_numpy(tangent_vectors).to(
        device=coords.device, dtype=torch.float32
    )
    
    return z_axis

    