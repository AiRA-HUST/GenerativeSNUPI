"""
Structural Nucleic Acid Potential Energy Calculations

This module provides functions for calculating potential energy components
in DNA nanostructures, including mechanical and electrostatic energies.

@author: Truong Quoc Chien (chientrq92@gmail.com)
"""

import roma
import torch
import scipy.io as scio
from typing import Tuple, Optional
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph

# Physical constants
DEBYE_LENGTH = 1.2576  # Debye screening length
ELECTROSTATIC_COEFFICIENT = 1.4131  # Electrostatic interaction coefficient
RADIUS_CUTOFF = 2.5  # Cutoff radius for electrostatic interactions
EPSILON = 1e-8  # Small value to avoid division by zero


def total_PE(
    coords: torch.Tensor,
    graph: Data,
    coeff_ele: float = 1.0,
    substr_edges: bool = True,
    interfere_area: bool = True
) -> Tuple[torch.Tensor, float, float]:
    """
    Calculate total potential energy of DNA nanostructure.
    
    Combines mechanical (structural) and electrostatic potential energies
    to compute the total system energy.
    
    Args:
        coords: Tensor of shape (N, 6) containing positions and orientations
        graph: PyTorch Geometric Data object containing graph structure
        coeff_ele: Coefficient for electrostatic energy weighting
        substr_edges: Whether to subtract edges in mechanical energy calculation
        interfere_area: Whether to consider interference area in electrostatic calculation
    
    Returns:
        Tuple containing:
        - total_loss: Combined potential energy
        - mechanical_energy: Mechanical component (as float)
        - electrostatic_energy: Electrostatic component (as float)
    """
    coords = coords.to(device=graph.x.device)
    helix_indices = graph.x[:, 6].view(-1)
    
    # Extract substrate indices if available
    try:
        substrate_indices = graph.x[:, 7].view(-1)
    except IndexError:
        substrate_indices = None
    
    # Calculate mechanical potential energy
    if substr_edges:
        mechanical_energy = _calculate_mechanical_PE(
            coords, graph.edge_index[:, ::2], graph.edge_attr[::2, :], substrate_indices
        )
    else:
        mechanical_energy = _calculate_mechanical_PE(
            coords, graph.edge_index[:, ::2], graph.edge_attr[::2, :], None
        )
    
    # Calculate electrostatic potential energy
    if interfere_area:
        electrostatic_energy = _calculate_electrostatic_PE(coords, helix_indices, substrate_indices)
    else:
        electrostatic_energy = _calculate_electrostatic_PE(coords, helix_indices, None)
    
    total_loss = mechanical_energy + coeff_ele * electrostatic_energy
    
    return total_loss, mechanical_energy.item(), electrostatic_energy.item()


def _calculate_electrostatic_PE(
    coords: torch.Tensor,
    helix_indices: torch.Tensor,
    substrate_indices: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Calculate electrostatic potential energy using Debye-Hückel screening.
    
    Computes repulsive electrostatic interactions between charged nucleotides
    on different helices, considering ionic screening effects.
    
    Args:
        coords: Tensor of shape (N, 6) containing positions and orientations
        helix_indices: Tensor indicating which helix each nucleotide belongs to
        substrate_indices: Optional tensor for substrate grouping
    
    Returns:
        Total electrostatic potential energy
    """
    positions = coords[:, 0:3]
    debye_length = torch.tensor(DEBYE_LENGTH, device=coords.device)
    coefficient = torch.tensor(ELECTROSTATIC_COEFFICIENT, device=coords.device)
    
    # Find neighboring pairs within cutoff radius
    row_indices, col_indices = radius_graph(positions, r=RADIUS_CUTOFF)
    
    # Filter for interactions between different helices
    different_helix_mask = helix_indices[row_indices] != helix_indices[col_indices]
    
    # Apply substrate interference filter if specified
    if substrate_indices is not None:
        same_substrate_mask = substrate_indices[row_indices] == substrate_indices[col_indices]
        different_helix_mask = torch.logical_and(different_helix_mask, same_substrate_mask)
    
    # Apply filters
    row_indices = row_indices[different_helix_mask]
    col_indices = col_indices[different_helix_mask]
    
    # Calculate pairwise distances
    distances = torch.norm(
        positions[col_indices] - positions[row_indices], 
        p=2, 
        dim=-1
    ).view(-1, 1)
    distances = distances + EPSILON  # Avoid division by zero
    
    # Calculate screened Coulomb potential
    electrostatic_energy = coefficient * torch.exp(-distances / debye_length) / distances
    
    return electrostatic_energy.sum()


def _calculate_mechanical_PE(
    coords: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    edge_type: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Calculate mechanical potential energy using beam theory.
    
    Computes structural deformation energy based on co-rotational beam
    formulation, considering bending, twisting, and stretching deformations.
    
    Args:
        coords: Tensor of shape (N, 6) containing positions and orientations
        edge_index: Tensor of shape (2, E) containing edge connectivity
        edge_attr: Tensor of shape (E, 30) containing edge properties
        edge_type: Optional tensor indicating edge types
    
    Returns:
        Total mechanical potential energy
    """
    n_edges = edge_index.size(1)
    row_indices = edge_index[0, :]
    col_indices = edge_index[1, :]
    
    # Extract material properties (first 30 components)
    material_properties = edge_attr[:, :30]
    
    # Calculate current edge lengths
    coord_n1 = coords[row_indices, :3]
    coord_n2 = coords[col_indices, :3]
    current_lengths = torch.sqrt(
        torch.sum((coord_n1 - coord_n2) ** 2, dim=1)
    ).unsqueeze(1)
    
    # Convert rotation vectors to rotation matrices
    rotation_matrices = roma.rotvec_to_rotmat(coords[:, 3:6])
    triad_n1 = rotation_matrices[row_indices, :, :]
    triad_n2 = rotation_matrices[col_indices, :, :]
    
    # Handle crossover (CO) configurations
    if edge_type is not None:
        co_mask = torch.logical_or(edge_type == 6, edge_type == 7)
        if co_mask.any():
            triad_n1[co_mask], triad_n2[co_mask] = _modify_crossover_triads(
                triad_n1[co_mask], triad_n2[co_mask],
                coord_n1[co_mask], coord_n2[co_mask]
            )
    
    # Calculate local coordinate system
    local_coords = _calculate_local_coordinates(
        triad_n1, triad_n2, coord_n1, coord_n2, current_lengths
    )
    
    # Calculate beam element strain energy
    strain_energy = _calculate_element_strain_energy(
        material_properties, current_lengths, local_coords['phi_1'], local_coords['phi_2']
    )
    
    # Zero out certain components for specific edge types
    if edge_type is not None:
        # Single-strand edges (type 10) - zero out bending/twisting components
        ss_mask = edge_type == 10
        strain_energy[ss_mask, 3:] = 0.0
    
    return strain_energy.sum()


def _calculate_local_coordinates(
    triad_n1: torch.Tensor,
    triad_n2: torch.Tensor,
    coord_n1: torch.Tensor,
    coord_n2: torch.Tensor,
    current_lengths: torch.Tensor
) -> dict:
    """
    Calculate local coordinate system for beam elements.
    
    Establishes local reference frames for each beam element to compute
    relative rotations and deformations.
    
    Args:
        triad_n1: Rotation matrices for first nodes
        triad_n2: Rotation matrices for second nodes
        coord_n1: Positions of first nodes
        coord_n2: Positions of second nodes
        current_lengths: Current lengths of edges
    
    Returns:
        Dictionary containing local coordinate transformations
    """
    n_edges = triad_n1.size(0)
    
    # Calculate average director vectors
    q1 = triad_n1[:, :, 1].squeeze()
    q2 = triad_n2[:, :, 1].squeeze()
    average_q = (q1 + q2) / 2
    
    # Calculate unit vector along element
    r1 = (coord_n2 - coord_n1) / current_lengths
    
    # Calculate perpendicular vectors
    r3_unnormalized = torch.cross(r1, average_q)
    r3 = r3_unnormalized / torch.norm(r3_unnormalized, dim=1).view(n_edges, 1).clamp(min=EPSILON)
    r2 = torch.cross(r3, r1)
    
    # Construct rotation matrix from global to local coordinates
    rotation_global_to_local = torch.cat((r1, r2, r3), dim=1).view(n_edges, 3, 3)
    
    # Transform triads to local coordinate system
    local_triad_1 = torch.bmm(rotation_global_to_local, triad_n1)
    local_triad_2 = torch.bmm(rotation_global_to_local, triad_n2)
    
    # Convert to rotation vectors
    phi_1 = roma.rotmat_to_rotvec(local_triad_1)
    phi_2 = roma.rotmat_to_rotvec(local_triad_2)
    
    return {
        'phi_1': phi_1,
        'phi_2': phi_2,
        'rotation_matrix': rotation_global_to_local
    }


def _calculate_element_strain_energy(
    material_props: torch.Tensor,
    current_lengths: torch.Tensor,
    phi_1: torch.Tensor,
    phi_2: torch.Tensor
) -> torch.Tensor:
    """
    Calculate strain energy for beam elements using co-rotational formulation.
    
    Computes energy contributions from axial deformation, bending, twisting,
    and shear deformations based on material properties and current configuration.
    
    Args:
        material_props: Material property tensor (N, 30)
        current_lengths: Current element lengths
        phi_1: Rotation vectors at first nodes
        phi_2: Rotation vectors at second nodes
    
    Returns:
        Strain energy tensor (N, 21) with different energy components
    """
    n_elements = material_props.size(0)
    
    # Extract rotation components
    rotations = {
        't1x': phi_1[:, 0].unsqueeze(1), 't1y': phi_1[:, 1].unsqueeze(1), 't1z': phi_1[:, 2].unsqueeze(1),
        't2x': phi_2[:, 0].unsqueeze(1), 't2y': phi_2[:, 1].unsqueeze(1), 't2z': phi_2[:, 2].unsqueeze(1)
    }
    
    # Extract material properties
    props = _extract_material_properties(material_props)
    
    # Calculate reference length
    reference_length = torch.sqrt(torch.sum(material_props[:, 0:3] ** 2, dim=1)).unsqueeze(1)
    
    # Initialize strain energy tensor
    strain_energy = torch.zeros((n_elements, 21), device=material_props.device)
    
    # Calculate individual energy components
    strain_energy = _calculate_energy_components(
        strain_energy, props, rotations, reference_length, current_lengths
    )
    
    return strain_energy


def _extract_material_properties(material_props: torch.Tensor) -> dict:
    """Extract and organize material properties from tensor."""
    return {
        'ea': material_props[:, 12].unsqueeze(1),      # Axial stiffness
        'ei22': material_props[:, 10].unsqueeze(1),    # Bending stiffness (y-axis)
        'ei33': material_props[:, 11].unsqueeze(1),    # Bending stiffness (z-axis)
        'gj': material_props[:, 9].unsqueeze(1),       # Torsional stiffness
        'ga2': material_props[:, 13].unsqueeze(1),     # Shear stiffness (y-axis)
        'ga3': material_props[:, 14].unsqueeze(1),     # Shear stiffness (z-axis)
        # Internal rotations
        't1x_int': material_props[:, 3].unsqueeze(1),
        't2x_int': material_props[:, 4].unsqueeze(1),
        't1y_int': material_props[:, 5].unsqueeze(1),
        't2y_int': material_props[:, 6].unsqueeze(1),
        't1z_int': material_props[:, 7].unsqueeze(1),
        't2z_int': material_props[:, 8].unsqueeze(1),
        # Coupling terms
        'g_Tx_Ty': material_props[:, 15].unsqueeze(1),
        'g_Tx_Tz': material_props[:, 16].unsqueeze(1),
        'g_Ty_Tz': material_props[:, 17].unsqueeze(1),
        'g_Dx_Dy': material_props[:, 18].unsqueeze(1),
        'g_Dx_Dz': material_props[:, 19].unsqueeze(1),
        'g_Dy_Dz': material_props[:, 20].unsqueeze(1),
        'g_Tx_Dx': material_props[:, 21].unsqueeze(1),
        'g_Ty_Dx': material_props[:, 22].unsqueeze(1),
        'g_Tz_Dx': material_props[:, 23].unsqueeze(1),
        'g_Tx_Dy': material_props[:, 24].unsqueeze(1),
        'g_Ty_Dy': material_props[:, 25].unsqueeze(1),
        'g_Tz_Dy': material_props[:, 26].unsqueeze(1),
        'g_Tx_Dz': material_props[:, 27].unsqueeze(1),
        'g_Ty_Dz': material_props[:, 28].unsqueeze(1),
        'g_Tz_Dz': material_props[:, 29].unsqueeze(1),
    }


def _calculate_energy_components(
    strain_energy: torch.Tensor,
    props: dict,
    rotations: dict,
    reference_length: torch.Tensor,
    current_length: torch.Tensor
) -> torch.Tensor:
    """Calculate all strain energy components using co-rotational beam theory."""
    # Axial displacement
    axial_displacement = torch.zeros_like(reference_length)
    
    # Axial strain energy
    strain_energy[:, 0] = (
        (props['ea'] * current_length ** 2) / (2 * reference_length) +
        (props['ea'] * current_length * axial_displacement) / reference_length -
        props['ea'] * current_length +
        (props['ea'] * axial_displacement ** 2) / (2 * reference_length) -
        props['ea'] * axial_displacement +
        (props['ea'] * reference_length) / 2
    ).squeeze()
    
    # Bending strain energy (z-rotation)
    strain_energy[:, 1] = (
        (props['ei22'] * rotations['t1z'] ** 2) / (2 * reference_length) -
        (props['ei22'] * rotations['t1z'] * rotations['t2z']) / reference_length -
        (props['ei22'] * rotations['t1z'] * props['t1z_int']) / reference_length +
        (props['ei22'] * rotations['t1z'] * props['t2z_int']) / reference_length +
        (props['ei22'] * rotations['t2z'] ** 2) / (2 * reference_length) +
        (props['ei22'] * rotations['t2z'] * props['t1z_int']) / reference_length -
        (props['ei22'] * rotations['t2z'] * props['t2z_int']) / reference_length +
        (props['ei22'] * props['t1z_int'] ** 2) / (2 * reference_length) -
        (props['ei22'] * props['t1z_int'] * props['t2z_int']) / reference_length +
        (props['ei22'] * props['t2z_int'] ** 2) / (2 * reference_length)
    ).squeeze()
    
    # Bending strain energy (y-rotation)
    strain_energy[:, 2] = (
        (props['ei33'] * rotations['t1y'] ** 2) / (2 * reference_length) -
        (props['ei33'] * rotations['t1y'] * rotations['t2y']) / reference_length -
        (props['ei33'] * rotations['t1y'] * props['t1y_int']) / reference_length +
        (props['ei33'] * rotations['t1y'] * props['t2y_int']) / reference_length +
        (props['ei33'] * rotations['t2y'] ** 2) / (2 * reference_length) +
        (props['ei33'] * rotations['t2y'] * props['t1y_int']) / reference_length -
        (props['ei33'] * rotations['t2y'] * props['t2y_int']) / reference_length +
        (props['ei33'] * props['t1y_int'] ** 2) / (2 * reference_length) -
        (props['ei33'] * props['t1y_int'] * props['t2y_int']) / reference_length +
        (props['ei33'] * props['t2y_int'] ** 2) / (2 * reference_length)
    ).squeeze()
    
    # Torsional strain energy
    strain_energy[:, 3] = (
        (props['gj'] * rotations['t1x'] ** 2) / (2 * reference_length) -
        (props['gj'] * rotations['t1x'] * rotations['t2x']) / reference_length -
        (props['gj'] * rotations['t1x'] * props['t1x_int']) / reference_length +
        (props['gj'] * rotations['t1x'] * props['t2x_int']) / reference_length +
        (props['gj'] * rotations['t2x'] ** 2) / (2 * reference_length) +
        (props['gj'] * rotations['t2x'] * props['t1x_int']) / reference_length -
        (props['gj'] * rotations['t2x'] * props['t2x_int']) / reference_length +
        (props['gj'] * props['t1x_int'] ** 2) / (2 * reference_length) -
        (props['gj'] * props['t1x_int'] * props['t2x_int']) / reference_length +
        (props['gj'] * props['t2x_int'] ** 2) / (2 * reference_length)
    ).squeeze()
    
    # Shear strain energy components (simplified representation)
    # Components 4-5: Shear deformation energy
    strain_energy[:, 4] = (
        (23 * props['ga2'] * reference_length * rotations['t1z'] ** 2) / 120 +
        (13 * props['ga2'] * reference_length * rotations['t1z'] * rotations['t2z']) / 60 -
        (23 * props['ga2'] * reference_length * rotations['t1z'] * props['t1z_int']) / 60 -
        (13 * props['ga2'] * reference_length * rotations['t1z'] * props['t2z_int']) / 60 +
        (23 * props['ga2'] * reference_length * rotations['t2z'] ** 2) / 120 -
        (13 * props['ga2'] * reference_length * rotations['t2z'] * props['t1z_int']) / 60 -
        (23 * props['ga2'] * reference_length * rotations['t2z'] * props['t2z_int']) / 60 +
        (23 * props['ga2'] * reference_length * props['t1z_int'] ** 2) / 120 +
        (13 * props['ga2'] * reference_length * props['t1z_int'] * props['t2z_int']) / 60 +
        (23 * props['ga2'] * reference_length * props['t2z_int'] ** 2) / 120
    ).squeeze()
    
    strain_energy[:, 5] = (
        (23 * props['ga3'] * reference_length * rotations['t1y'] ** 2) / 120 +
        (13 * props['ga3'] * reference_length * rotations['t1y'] * rotations['t2y']) / 60 -
        (23 * props['ga3'] * reference_length * rotations['t1y'] * props['t1y_int']) / 60 -
        (13 * props['ga3'] * reference_length * rotations['t1y'] * props['t2y_int']) / 60 +
        (23 * props['ga3'] * reference_length * rotations['t2y'] ** 2) / 120 -
        (13 * props['ga3'] * reference_length * rotations['t2y'] * props['t1y_int']) / 60 -
        (23 * props['ga3'] * reference_length * rotations['t2y'] * props['t2y_int']) / 60 +
        (23 * props['ga3'] * reference_length * props['t1y_int'] ** 2) / 120 +
        (13 * props['ga3'] * reference_length * props['t1y_int'] * props['t2y_int']) / 60 +
        (23 * props['ga3'] * reference_length * props['t2y_int'] ** 2) / 120
    ).squeeze()
    
    # Coupling terms (components 6-20)
    strain_energy[:, 6] = (
        (props['g_Tx_Ty'] * rotations['t1x'] * rotations['t1y']) / reference_length -
        (props['g_Tx_Ty'] * rotations['t1x'] * rotations['t2y']) / reference_length -
        (props['g_Tx_Ty'] * rotations['t2x'] * rotations['t1y']) / reference_length +
        (props['g_Tx_Ty'] * rotations['t2x'] * rotations['t2y']) / reference_length -
        (props['g_Tx_Ty'] * rotations['t1x'] * props['t1y_int']) / reference_length -
        (props['g_Tx_Ty'] * rotations['t1y'] * props['t1x_int']) / reference_length +
        (props['g_Tx_Ty'] * rotations['t1x'] * props['t2y_int']) / reference_length +
        (props['g_Tx_Ty'] * rotations['t2x'] * props['t1y_int']) / reference_length +
        (props['g_Tx_Ty'] * rotations['t1y'] * props['t2x_int']) / reference_length +
        (props['g_Tx_Ty'] * rotations['t2y'] * props['t1x_int']) / reference_length -
        (props['g_Tx_Ty'] * rotations['t2x'] * props['t2y_int']) / reference_length -
        (props['g_Tx_Ty'] * rotations['t2y'] * props['t2x_int']) / reference_length +
        (props['g_Tx_Ty'] * props['t1x_int'] * props['t1y_int']) / reference_length -
        (props['g_Tx_Ty'] * props['t1x_int'] * props['t2y_int']) / reference_length -
        (props['g_Tx_Ty'] * props['t2x_int'] * props['t1y_int']) / reference_length +
        (props['g_Tx_Ty'] * props['t2x_int'] * props['t2y_int']) / reference_length
    ).squeeze()
    
    # Additional coupling terms (7-20) follow similar patterns
    # For brevity, implementing the remaining terms with similar structure
    _calculate_remaining_coupling_terms(strain_energy, props, rotations, reference_length, current_length)
    
    return strain_energy


def _calculate_remaining_coupling_terms(
    strain_energy: torch.Tensor,
    props: dict,
    rotations: dict,
    reference_length: torch.Tensor,
    current_length: torch.Tensor
) -> None:
    """Calculate remaining coupling terms for strain energy (components 7-20)."""
    # Implementation of remaining 14 coupling terms
    # Each follows the same pattern as component 6 but with different variable combinations
    
    # Tx-Tz coupling (component 7)
    strain_energy[:, 7] = (
        (props['g_Tx_Tz'] * rotations['t1x'] * rotations['t1z']) / reference_length -
        (props['g_Tx_Tz'] * rotations['t1x'] * rotations['t2z']) / reference_length -
        (props['g_Tx_Tz'] * rotations['t2x'] * rotations['t1z']) / reference_length +
        (props['g_Tx_Tz'] * rotations['t2x'] * rotations['t2z']) / reference_length -
        (props['g_Tx_Tz'] * rotations['t1x'] * props['t1z_int']) / reference_length -
        (props['g_Tx_Tz'] * props['t1x_int'] * rotations['t1z']) / reference_length +
        (props['g_Tx_Tz'] * rotations['t1x'] * props['t2z_int']) / reference_length +
        (props['g_Tx_Tz'] * rotations['t2x'] * props['t1z_int']) / reference_length +
        (props['g_Tx_Tz'] * props['t1x_int'] * rotations['t2z']) / reference_length +
        (props['g_Tx_Tz'] * props['t2x_int'] * rotations['t1z']) / reference_length -
        (props['g_Tx_Tz'] * rotations['t2x'] * props['t2z_int']) / reference_length -
        (props['g_Tx_Tz'] * props['t2x_int'] * rotations['t2z']) / reference_length +
        (props['g_Tx_Tz'] * props['t1x_int'] * props['t1z_int']) / reference_length -
        (props['g_Tx_Tz'] * props['t1x_int'] * props['t2z_int']) / reference_length -
        (props['g_Tx_Tz'] * props['t2x_int'] * props['t1z_int']) / reference_length +
        (props['g_Tx_Tz'] * props['t2x_int'] * props['t2z_int']) / reference_length
    ).squeeze()
    
    # Continue with similar patterns for components 8-20
    # For brevity, showing the pattern for a few more components
    
    # Ty-Tz coupling (component 8)
    strain_energy[:, 8] = (
        (props['g_Ty_Tz'] * rotations['t1y'] * rotations['t1z']) / reference_length -
        (props['g_Ty_Tz'] * rotations['t1y'] * rotations['t2z']) / reference_length -
        (props['g_Ty_Tz'] * rotations['t2y'] * rotations['t1z']) / reference_length +
        (props['g_Ty_Tz'] * rotations['t2y'] * rotations['t2z']) / reference_length -
        (props['g_Ty_Tz'] * rotations['t1y'] * props['t1z_int']) / reference_length -
        (props['g_Ty_Tz'] * rotations['t1z'] * props['t1y_int']) / reference_length +
        (props['g_Ty_Tz'] * rotations['t1y'] * props['t2z_int']) / reference_length +
        (props['g_Ty_Tz'] * rotations['t2y'] * props['t1z_int']) / reference_length +
        (props['g_Ty_Tz'] * rotations['t1z'] * props['t2y_int']) / reference_length +
        (props['g_Ty_Tz'] * rotations['t2z'] * props['t1y_int']) / reference_length -
        (props['g_Ty_Tz'] * rotations['t2y'] * props['t2z_int']) / reference_length -
        (props['g_Ty_Tz'] * rotations['t2z'] * props['t2y_int']) / reference_length +
        (props['g_Ty_Tz'] * props['t1y_int'] * props['t1z_int']) / reference_length -
        (props['g_Ty_Tz'] * props['t1y_int'] * props['t2z_int']) / reference_length -
        (props['g_Ty_Tz'] * props['t2y_int'] * props['t1z_int']) / reference_length +
        (props['g_Ty_Tz'] * props['t2y_int'] * props['t2z_int']) / reference_length
    ).squeeze()
    
    # Components 9-20: Additional coupling terms with full mathematical expressions
    
    # Component 9: Dx-Dy coupling
    strain_energy[:, 9] = (
        (reference_length * props['g_Dx_Dy'] * rotations['t1y'] * rotations['t2z']) / 144 -
        (reference_length * props['g_Dx_Dy'] * rotations['t1y'] * rotations['t1z']) / 144 +
        (reference_length * props['g_Dx_Dy'] * rotations['t2y'] * rotations['t1z']) / 144 -
        (reference_length * props['g_Dx_Dy'] * rotations['t2y'] * rotations['t2z']) / 144 +
        (reference_length * props['g_Dx_Dy'] * rotations['t1y'] * props['t1z_int']) / 144 +
        (reference_length * props['g_Dx_Dy'] * rotations['t1z'] * props['t1y_int']) / 144 -
        (reference_length * props['g_Dx_Dy'] * rotations['t1y'] * props['t2z_int']) / 144 -
        (reference_length * props['g_Dx_Dy'] * rotations['t2y'] * props['t1z_int']) / 144 -
        (reference_length * props['g_Dx_Dy'] * rotations['t1z'] * props['t2y_int']) / 144 -
        (reference_length * props['g_Dx_Dy'] * rotations['t2z'] * props['t1y_int']) / 144 +
        (reference_length * props['g_Dx_Dy'] * rotations['t2y'] * props['t2z_int']) / 144 +
        (reference_length * props['g_Dx_Dy'] * rotations['t2z'] * props['t2y_int']) / 144 -
        (reference_length * props['g_Dx_Dy'] * props['t1y_int'] * props['t1z_int']) / 144 +
        (reference_length * props['g_Dx_Dy'] * props['t1y_int'] * props['t2z_int']) / 144 +
        (reference_length * props['g_Dx_Dy'] * props['t2y_int'] * props['t1z_int']) / 144 -
        (reference_length * props['g_Dx_Dy'] * props['t2y_int'] * props['t2z_int']) / 144
    ).squeeze()
    
    # Component 10: Dx-Dz coupling
    axial_displacement = torch.zeros_like(reference_length)
    strain_energy[:, 10] = (
        (reference_length * props['g_Dx_Dz'] * rotations['t1y']) / 24 -
        (reference_length * props['g_Dx_Dz'] * rotations['t2y']) / 24 -
        (reference_length * props['g_Dx_Dz'] * props['t1y_int']) / 24 +
        (reference_length * props['g_Dx_Dz'] * props['t2y_int']) / 24 -
        (current_length * props['g_Dx_Dz'] * rotations['t1y']) / 24 +
        (current_length * props['g_Dx_Dz'] * rotations['t2y']) / 24 +
        (current_length * props['g_Dx_Dz'] * props['t1y_int']) / 24 -
        (current_length * props['g_Dx_Dz'] * props['t2y_int']) / 24 -
        (props['g_Dx_Dz'] * rotations['t1y'] * axial_displacement) / 24 +
        (props['g_Dx_Dz'] * rotations['t2y'] * axial_displacement) / 24 +
        (props['g_Dx_Dz'] * props['t1y_int'] * axial_displacement) / 24 -
        (props['g_Dx_Dz'] * props['t2y_int'] * axial_displacement) / 24
    ).squeeze()
    
    # Component 11: Dy-Dz coupling
    strain_energy[:, 11] = (
        (reference_length * props['g_Dy_Dz'] * rotations['t2z']) / 24 -
        (reference_length * props['g_Dy_Dz'] * rotations['t1z']) / 24 +
        (reference_length * props['g_Dy_Dz'] * props['t1z_int']) / 24 -
        (reference_length * props['g_Dy_Dz'] * props['t2z_int']) / 24 +
        (current_length * props['g_Dy_Dz'] * rotations['t1z']) / 24 -
        (current_length * props['g_Dy_Dz'] * rotations['t2z']) / 24 -
        (current_length * props['g_Dy_Dz'] * props['t1z_int']) / 24 +
        (current_length * props['g_Dy_Dz'] * props['t2z_int']) / 24 +
        (props['g_Dy_Dz'] * rotations['t1z'] * axial_displacement) / 24 -
        (props['g_Dy_Dz'] * rotations['t2z'] * axial_displacement) / 24 -
        (props['g_Dy_Dz'] * props['t1z_int'] * axial_displacement) / 24 +
        (props['g_Dy_Dz'] * props['t2z_int'] * axial_displacement) / 24
    ).squeeze()
    
    # Component 12: Tx-Dx coupling
    strain_energy[:, 12] = (
        (props['g_Tx_Dx'] * rotations['t1x'] * rotations['t1y']) / 12 -
        (props['g_Tx_Dx'] * rotations['t1x'] * rotations['t2y']) / 12 -
        (props['g_Tx_Dx'] * rotations['t2x'] * rotations['t1y']) / 12 +
        (props['g_Tx_Dx'] * rotations['t2x'] * rotations['t2y']) / 12 -
        (props['g_Tx_Dx'] * rotations['t1x'] * props['t1y_int']) / 12 -
        (props['g_Tx_Dx'] * rotations['t1y'] * props['t1x_int']) / 12 +
        (props['g_Tx_Dx'] * rotations['t1x'] * props['t2y_int']) / 12 +
        (props['g_Tx_Dx'] * rotations['t2x'] * props['t1y_int']) / 12 +
        (props['g_Tx_Dx'] * rotations['t1y'] * props['t2x_int']) / 12 +
        (props['g_Tx_Dx'] * rotations['t2y'] * props['t1x_int']) / 12 -
        (props['g_Tx_Dx'] * rotations['t2x'] * props['t2y_int']) / 12 -
        (props['g_Tx_Dx'] * rotations['t2y'] * props['t2x_int']) / 12 +
        (props['g_Tx_Dx'] * props['t1x_int'] * props['t1y_int']) / 12 -
        (props['g_Tx_Dx'] * props['t1x_int'] * props['t2y_int']) / 12 -
        (props['g_Tx_Dx'] * props['t2x_int'] * props['t1y_int']) / 12 +
        (props['g_Tx_Dx'] * props['t2x_int'] * props['t2y_int']) / 12
    ).squeeze()
    
    # Component 13: Ty-Dx coupling
    strain_energy[:, 13] = (
        (props['g_Ty_Dx'] * rotations['t1y'] ** 2) / 12 -
        (props['g_Ty_Dx'] * rotations['t1y'] * rotations['t2y']) / 6 -
        (props['g_Ty_Dx'] * rotations['t1y'] * props['t1y_int']) / 6 +
        (props['g_Ty_Dx'] * rotations['t1y'] * props['t2y_int']) / 6 +
        (props['g_Ty_Dx'] * rotations['t2y'] ** 2) / 12 +
        (props['g_Ty_Dx'] * rotations['t2y'] * props['t1y_int']) / 6 -
        (props['g_Ty_Dx'] * rotations['t2y'] * props['t2y_int']) / 6 +
        (props['g_Ty_Dx'] * props['t1y_int'] ** 2) / 12 -
        (props['g_Ty_Dx'] * props['t1y_int'] * props['t2y_int']) / 6 +
        (props['g_Ty_Dx'] * props['t2y_int'] ** 2) / 12
    ).squeeze()
    
    # Component 14: Tz-Dx coupling
    strain_energy[:, 14] = (
        (props['g_Tz_Dx'] * rotations['t1y'] * rotations['t1z']) / 12 -
        (props['g_Tz_Dx'] * rotations['t1y'] * rotations['t2z']) / 12 -
        (props['g_Tz_Dx'] * rotations['t2y'] * rotations['t1z']) / 12 +
        (props['g_Tz_Dx'] * rotations['t2y'] * rotations['t2z']) / 12 -
        (props['g_Tz_Dx'] * rotations['t1y'] * props['t1z_int']) / 12 -
        (props['g_Tz_Dx'] * rotations['t1z'] * props['t1y_int']) / 12 +
        (props['g_Tz_Dx'] * rotations['t1y'] * props['t2z_int']) / 12 +
        (props['g_Tz_Dx'] * rotations['t2y'] * props['t1z_int']) / 12 +
        (props['g_Tz_Dx'] * rotations['t1z'] * props['t2y_int']) / 12 +
        (props['g_Tz_Dx'] * rotations['t2z'] * props['t1y_int']) / 12 -
        (props['g_Tz_Dx'] * rotations['t2y'] * props['t2z_int']) / 12 -
        (props['g_Tz_Dx'] * rotations['t2z'] * props['t2y_int']) / 12 +
        (props['g_Tz_Dx'] * props['t1y_int'] * props['t1z_int']) / 12 -
        (props['g_Tz_Dx'] * props['t1y_int'] * props['t2z_int']) / 12 -
        (props['g_Tz_Dx'] * props['t2y_int'] * props['t1z_int']) / 12 +
        (props['g_Tz_Dx'] * props['t2y_int'] * props['t2z_int']) / 12
    ).squeeze()
    
    # Component 15: Tx-Dy coupling
    strain_energy[:, 15] = (
        (props['g_Tx_Dy'] * rotations['t1x'] * rotations['t2z']) / 12 -
        (props['g_Tx_Dy'] * rotations['t1x'] * rotations['t1z']) / 12 +
        (props['g_Tx_Dy'] * rotations['t2x'] * rotations['t1z']) / 12 -
        (props['g_Tx_Dy'] * rotations['t2x'] * rotations['t2z']) / 12 +
        (props['g_Tx_Dy'] * rotations['t1x'] * props['t1z_int']) / 12 +
        (props['g_Tx_Dy'] * props['t1x_int'] * rotations['t1z']) / 12 -
        (props['g_Tx_Dy'] * rotations['t1x'] * props['t2z_int']) / 12 -
        (props['g_Tx_Dy'] * rotations['t2x'] * props['t1z_int']) / 12 -
        (props['g_Tx_Dy'] * props['t1x_int'] * rotations['t2z']) / 12 -
        (props['g_Tx_Dy'] * props['t2x_int'] * rotations['t1z']) / 12 +
        (props['g_Tx_Dy'] * rotations['t2x'] * props['t2z_int']) / 12 +
        (props['g_Tx_Dy'] * props['t2x_int'] * rotations['t2z']) / 12 -
        (props['g_Tx_Dy'] * props['t1x_int'] * props['t1z_int']) / 12 +
        (props['g_Tx_Dy'] * props['t1x_int'] * props['t2z_int']) / 12 +
        (props['g_Tx_Dy'] * props['t2x_int'] * props['t1z_int']) / 12 -
        (props['g_Tx_Dy'] * props['t2x_int'] * props['t2z_int']) / 12
    ).squeeze()
    
    # Component 16: Ty-Dy coupling
    strain_energy[:, 16] = (
        (props['g_Ty_Dy'] * rotations['t1y'] * rotations['t2z']) / 12 -
        (props['g_Ty_Dy'] * rotations['t1y'] * rotations['t1z']) / 12 +
        (props['g_Ty_Dy'] * rotations['t2y'] * rotations['t1z']) / 12 -
        (props['g_Ty_Dy'] * rotations['t2y'] * rotations['t2z']) / 12 +
        (props['g_Ty_Dy'] * rotations['t1y'] * props['t1z_int']) / 12 +
        (props['g_Ty_Dy'] * rotations['t1z'] * props['t1y_int']) / 12 -
        (props['g_Ty_Dy'] * rotations['t1y'] * props['t2z_int']) / 12 -
        (props['g_Ty_Dy'] * rotations['t2y'] * props['t1z_int']) / 12 -
        (props['g_Ty_Dy'] * rotations['t1z'] * props['t2y_int']) / 12 -
        (props['g_Ty_Dy'] * rotations['t2z'] * props['t1y_int']) / 12 +
        (props['g_Ty_Dy'] * rotations['t2y'] * props['t2z_int']) / 12 +
        (props['g_Ty_Dy'] * rotations['t2z'] * props['t2y_int']) / 12 -
        (props['g_Ty_Dy'] * props['t1y_int'] * props['t1z_int']) / 12 +
        (props['g_Ty_Dy'] * props['t1y_int'] * props['t2z_int']) / 12 +
        (props['g_Ty_Dy'] * props['t2y_int'] * props['t1z_int']) / 12 -
        (props['g_Ty_Dy'] * props['t2y_int'] * props['t2z_int']) / 12
    ).squeeze()
    
    # Component 17: Tz-Dy coupling
    strain_energy[:, 17] = (
        (props['g_Tz_Dy'] * rotations['t1z'] * rotations['t2z']) / 6 -
        (props['g_Tz_Dy'] * rotations['t2z'] ** 2) / 12 -
        (props['g_Tz_Dy'] * props['t1z_int'] ** 2) / 12 -
        (props['g_Tz_Dy'] * props['t2z_int'] ** 2) / 12 -
        (props['g_Tz_Dy'] * rotations['t1z'] ** 2) / 12 +
        (props['g_Tz_Dy'] * rotations['t1z'] * props['t1z_int']) / 6 -
        (props['g_Tz_Dy'] * rotations['t1z'] * props['t2z_int']) / 6 -
        (props['g_Tz_Dy'] * rotations['t2z'] * props['t1z_int']) / 6 +
        (props['g_Tz_Dy'] * rotations['t2z'] * props['t2z_int']) / 6 +
        (props['g_Tz_Dy'] * props['t1z_int'] * props['t2z_int']) / 6
    ).squeeze()
    
    # Component 18: Tx-Dz coupling
    strain_energy[:, 18] = (
        (props['g_Tx_Dz'] * rotations['t1x']) / 2 -
        (props['g_Tx_Dz'] * rotations['t2x']) / 2 -
        (props['g_Tx_Dz'] * props['t1x_int']) / 2 +
        (props['g_Tx_Dz'] * props['t2x_int']) / 2 -
        (current_length * props['g_Tx_Dz'] * rotations['t1x']) / (2 * reference_length) +
        (current_length * props['g_Tx_Dz'] * rotations['t2x']) / (2 * reference_length) +
        (current_length * props['g_Tx_Dz'] * props['t1x_int']) / (2 * reference_length) -
        (current_length * props['g_Tx_Dz'] * props['t2x_int']) / (2 * reference_length) -
        (props['g_Tx_Dz'] * rotations['t1x'] * axial_displacement) / (2 * reference_length) +
        (props['g_Tx_Dz'] * rotations['t2x'] * axial_displacement) / (2 * reference_length) +
        (props['g_Tx_Dz'] * props['t1x_int'] * axial_displacement) / (2 * reference_length) -
        (props['g_Tx_Dz'] * props['t2x_int'] * axial_displacement) / (2 * reference_length)
    ).squeeze()
    
    # Component 19: Ty-Dz coupling
    strain_energy[:, 19] = (
        (props['g_Ty_Dz'] * rotations['t1y']) / 2 -
        (props['g_Ty_Dz'] * rotations['t2y']) / 2 -
        (props['g_Ty_Dz'] * props['t1y_int']) / 2 +
        (props['g_Ty_Dz'] * props['t2y_int']) / 2 -
        (current_length * props['g_Ty_Dz'] * rotations['t1y']) / (2 * reference_length) +
        (current_length * props['g_Ty_Dz'] * rotations['t2y']) / (2 * reference_length) +
        (current_length * props['g_Ty_Dz'] * props['t1y_int']) / (2 * reference_length) -
        (current_length * props['g_Ty_Dz'] * props['t2y_int']) / (2 * reference_length) -
        (props['g_Ty_Dz'] * rotations['t1y'] * axial_displacement) / (2 * reference_length) +
        (props['g_Ty_Dz'] * rotations['t2y'] * axial_displacement) / (2 * reference_length) +
        (props['g_Ty_Dz'] * props['t1y_int'] * axial_displacement) / (2 * reference_length) -
        (props['g_Ty_Dz'] * props['t2y_int'] * axial_displacement) / (2 * reference_length)
    ).squeeze()
    
    # Component 20: Tz-Dz coupling
    strain_energy[:, 20] = (
        (props['g_Tz_Dz'] * rotations['t1z']) / 2 -
        (props['g_Tz_Dz'] * rotations['t2z']) / 2 -
        (props['g_Tz_Dz'] * props['t1z_int']) / 2 +
        (props['g_Tz_Dz'] * props['t2z_int']) / 2 -
        (current_length * props['g_Tz_Dz'] * rotations['t1z']) / (2 * reference_length) +
        (current_length * props['g_Tz_Dz'] * rotations['t2z']) / (2 * reference_length) +
        (current_length * props['g_Tz_Dz'] * props['t1z_int']) / (2 * reference_length) -
        (current_length * props['g_Tz_Dz'] * props['t2z_int']) / (2 * reference_length) -
        (props['g_Tz_Dz'] * rotations['t1z'] * axial_displacement) / (2 * reference_length) +
        (props['g_Tz_Dz'] * rotations['t2z'] * axial_displacement) / (2 * reference_length) +
        (props['g_Tz_Dz'] * props['t1z_int'] * axial_displacement) / (2 * reference_length) -
        (props['g_Tz_Dz'] * props['t2z_int'] * axial_displacement) / (2 * reference_length)
    ).squeeze()


def _modify_crossover_triads(
    triad_n1: torch.Tensor,
    triad_n2: torch.Tensor,
    coord_n1: torch.Tensor,
    coord_n2: torch.Tensor,
    scafCO: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Modify triad orientations for crossover connections.
    
    Adjusts the local coordinate systems at crossover points to ensure
    proper geometric continuity in DNA crossover structures.
    
    Args:
        triad_n1: Rotation matrices for first crossover nodes
        triad_n2: Rotation matrices for second crossover nodes
        coord_n1: Positions of first crossover nodes
        coord_n2: Positions of second crossover nodes
        scafCO: Whether this is a scaffold crossover
    
    Returns:
        Tuple of modified rotation matrices (triad_n1_new, triad_n2_new)
    """
    n_crossovers = triad_n1.size(0)
    crossover_vector = coord_n2 - coord_n1
    
    # Process first node triads
    y_n1 = triad_n1[:, :, 1].squeeze().view(n_crossovers, 3)
    z_n1_trial = triad_n1[:, :, 0].squeeze().view(n_crossovers, 3)
    
    # Determine correct orientation based on crossover direction
    sign_n1 = torch.sign(
        (crossover_vector * torch.cross(y_n1, z_n1_trial)).sum(dim=1)
    ).view(n_crossovers, 1)
    
    z_n1 = sign_n1.expand_as(z_n1_trial) * z_n1_trial
    x_n1 = torch.cross(y_n1, z_n1)
    
    triad_n1_new = torch.cat([
        x_n1.unsqueeze(2),
        y_n1.unsqueeze(2),
        z_n1.unsqueeze(2)
    ], dim=2)
    
    # Process second node triads
    y_n2 = triad_n2[:, :, 1].squeeze().view(n_crossovers, 3)
    z_n2_trial = triad_n2[:, :, 0].squeeze().view(n_crossovers, 3)
    
    sign_n2 = torch.sign(
        (crossover_vector * torch.cross(y_n2, z_n2_trial)).sum(dim=1)
    ).view(n_crossovers, 1)
    
    z_n2 = sign_n2.expand_as(z_n2_trial) * z_n2_trial
    x_n2 = torch.cross(y_n2, z_n2)
    
    triad_n2_new = torch.cat([
        x_n2.unsqueeze(2),
        y_n2.unsqueeze(2),
        z_n2.unsqueeze(2)
    ], dim=2)
    
    return triad_n1_new, triad_n2_new


# Compatibility aliases for backward compatibility
mech_PE = _calculate_mechanical_PE
elec_PE = _calculate_electrostatic_PE
CO_triad_modify = _modify_crossover_triads