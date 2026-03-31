import torch
import math
from pathlib import Path
from typing import Tuple, List, Optional

from automated_design.ults import *
from automated_design.visual import *
from automated_design.routing import *
from automated_design.gen_bild import *
from automated_design.gen_cando import *
from automated_design.coords_to_edges import *
from automated_design.coords_to_triads import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def automated_DnaOrigami(
    coords: torch.Tensor, 
    min_COs_distance: int = 10, 
    min_pairing_len: int = 4,
    additional_SSseq: int = 0,
    scaf_unbound: bool = True, 
    adaptive_staples: bool = False,
    adjust_short_edges: int = 4,
    adjust_long_edges: int = 4,
    show: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, 
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List, torch.Tensor]:
    """
    Automated DNA origami design from 3D coordinates.
    
    This function takes 3D coordinates and generates a complete DNA origami design
    including scaffold routing, staple generation, and structural information.
    
    Args:
        coords: 3D coordinates of DNA structure points
        min_COs_distance: Minimum distance between crossovers (base-pairs)
        min_pairing_len: Minimum pairing length (base-pairs)
        additional_SSseq: Additional single-strand sequence length
        scaf_unbound: Whether scaffold has unbound regions
        adaptive_staples: Whether to use adaptive staple generation
        adjust_short_edges: Adjustment factor for short edges
        adjust_long_edges: Adjustment factor for long edges
        show: Whether to display visualization plots
        
    Returns:
        Tuple containing:
        - dnaGeom: DNA geometry information
        - dnaTop: DNA topology information
        - coords: Processed coordinates
        - triads: Base-pair triads
        - bp_Hindex: Base-pair helix indices
        - coords_scaf: Scaffold coordinates
        - scaf_route: Scaffold routing path
        - coords_stap: Staple coordinates
        - node_scaf_seq: Node scaffold sequence
        - stap_routes: Staple routing paths
        - unbound_nucle_coords: Unbound nucleotide coordinates
    """
    
    # ==================== DESIGN PARAMETERS ====================
    regular_BP_len = 0.4                    # Regular base-pair length [nm]
    one_turn_angle = (-2) * math.pi / 10.5  # One turn angle [radians]
    min_pairing_len = min_pairing_len       # Minimum pairing length [base-pairs]
    min_COs_distance = min_COs_distance     # Minimum crossover distance [base-pairs]
    
    # ==================== GENERATE HELIX CONFIGURATION ====================
    print("Generating helix configuration from coordinates...")
    coords, RBedge_index, bp_Hindex, endhelix_bpID, remove_bpID_lst = assign_RegularBp_edges(
        coords, equi_len=regular_BP_len
    )
    
    # Handle manual editing of problematic base-pairs if needed
    if len(remove_bpID_lst) > 0:
        print(f"Found {len(remove_bpID_lst)} problematic base-pairs that may need removal")
        
        # Show current structure with potential issues
        fig = plot_dna(coords, edge_index=RBedge_index)
        fig.show()
        
        # Allow manual intervention for base-pair removal
        manual_edit = 0 # int(input("Manual edit [1/0]: "))
        if manual_edit == 1:
            num_remove_bps = len(remove_bpID_lst)
            remove_bpID_lst = list(map(int, 
                input("\nEnter remove base-pair IDs : ").strip().split()))[:num_remove_bps]
            
        # Remove problematic base-pairs and regenerate structure
        coords = del_nodes(coords, remove_bpID_lst)
        coords, RBedge_index, bp_Hindex, endhelix_bpID, remove_bpID_lst = assign_RegularBp_edges(
            coords, equi_len=regular_BP_len
        )
    
    # Display the processed structure
    if show:
        print("Displaying processed DNA structure...")
        fig = plot_dna(coords, edge_index=RBedge_index)
        fig.show()
    
    # ==================== GENERATE CROSSOVER ALIGNMENT ====================
    print("Generating aligned crossovers between helices...")
    alignCO_index = gen_align_CrossOvers(
        coords, bp_Hindex, RBedge_index, endhelix_bpID, 
        effective_len=10,  # Default effective length for crossovers
        adjust_short_edges=adjust_short_edges,
        adjust_long_edges=adjust_long_edges
    )
    
    # ==================== GENERATE LOOP STRUCTURES ====================
    print("Generating single-strand loops and indexing...")
    SSedge_index, bp_Lindex = assign_SingleStrand_edges(
        coords, RBedge_index, alignCO_index, bp_Hindex
    )
    bp_Lindex, _ = loop_indexing(RBedge_index, SSedge_index)
    
    # ==================== SCAFFOLD ROUTING ====================
    print("Generating scaffold routing path...")
    scaf_route, scafCO_index, alignCO_index_noscaf = scaffold_routing(
        alignCO_index, RBedge_index, SSedge_index, 
        bp_Hindex, bp_Lindex, coords
    )
    
    # ==================== ASSIGN BASE-PAIR TRIADS ====================
    print("Assigning base-pair triads for 3D structure...")
    triads, _, scaf_route, scafCO_index = assign_triads(
        alignCO_index, RBedge_index, SSedge_index, scafCO_index,
        coords, bp_Hindex, bp_Lindex, scaf_route,
        one_turn_angle=one_turn_angle
    )
    
    # ==================== GENERATE BASE COORDINATES ====================
    print("Converting base-pairs to individual base coordinates...")
    coords_scaf, coords_stap = bps_to_bases(coords, triads)
    
    # ==================== GENERATE STAPLE CROSSOVERS ====================
    print("Generating staple crossovers...")
    stapCO_index = gen_Staples_CrossOvers(
        coords_stap, bp_Hindex, scaf_route, scafCO_index,
        alignCO_index_noscaf, RBedge_index, SSedge_index,
        effective_len=min_COs_distance,
        min_pairing_len=min_pairing_len,
        adaptive=adaptive_staples
    )
        
    # ==================== STAPLE ROUTING ====================
    print("Routing staples through the structure...")
    stap_routes, stap_len, _ = staples_routing(
        RBedge_index, scafCO_index, stapCO_index, SSedge_index, scaf_route,
        min_pairing_len=min_pairing_len, coords=coords
    )
    
    print(f"Generated {len(stap_routes)} staples with lengths ranging from {stap_len.amin()} to {stap_len.amax()} base-pairs")
    
    # ==================== FINAL CONFIGURATION ====================
    print("Assembling final DNA structure...")
    edge_index = torch.hstack((RBedge_index, scafCO_index, stapCO_index, SSedge_index))
    
    # Check for overly long staples and visualize if necessary
    if stap_len.amax() > 50:
        print(f"Warning: Found staple with length {stap_len.amax()} base-pairs (>50)")
        maxstap_ID = (stap_len == stap_len.amax()).nonzero()[0]
        fig = plot_dna(coords, edge_index=torch.vstack((stap_routes[maxstap_ID][:-1], stap_routes[maxstap_ID][1:])))
        fig.show()
    
    # Display final structure components if requested
    if show:
        print("Displaying final structure components...")
        
        # Show scaffold crossovers
        fig = plot_dna(coords_scaf, edge_index=scafCO_index)
        fig.show()
        
        # Show staple crossovers
        fig = plot_dna(coords_stap, edge_index=stapCO_index)
        fig.show()
        
        # Show complete structure
        fig = plot_dna(coords, edge_index=edge_index)
        fig.show()
    
    # ==================== GENERATE OUTPUT FILES ====================
    print("Generating CanDo design files...")
    
    # Generate geometry file (coordinates scaled by 10x for CanDo)
    dnaGeom = gen_dnageom(10 * coords, triads)
    
    # Generate topology file with routing information
    dnaTop, scaf_SS_coords, stap_SS_coords, node_scaf_seq = gen_dnatop(
        scaf_route, stap_routes, SSedge_index, 
        coords_scaf, coords_stap, coords=coords,
        add_seqlen=additional_SSseq, scaf_unbound=scaf_unbound
    )
    
    # Combine unbound nucleotide coordinates
    unbound_nucle_coords = torch.vstack((scaf_SS_coords, stap_SS_coords))
    
    # Generate edge indices for visualization
    scaf_edge_index = torch.vstack((scaf_route[:-1], scaf_route[1:]))
    
    stap_edge_index = []
    for i in range(len(stap_routes)):
        stap_edge_index.append(torch.vstack((stap_routes[i][:-1], stap_routes[i][1:])))
    stap_edge_index = torch.hstack(stap_edge_index)  
    
    print("DNA origami design completed successfully!")
    
    return (dnaGeom, dnaTop, coords, triads, bp_Hindex,
            coords_scaf, scaf_route, coords_stap, node_scaf_seq,
            stap_routes, unbound_nucle_coords)
