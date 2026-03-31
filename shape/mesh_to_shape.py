import torch
import meshio
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visual import *
from shape.shape_ults import *

# Constants
DEFAULT_MAX_SCAF_LEN = 7429
DEFAULT_RADIUS_DIFF = 1.0
DEFAULT_RADIUS_SHAPE = 1.0
MIN_EDGE_LENGTH = 10
SCAFFOLD_CONVERSION_FACTOR = 0.34  # nm per base pair
EDGE_LENGTH_THRESHOLD_MULTIPLIER = 1.1


def mesh_to_shape(file_name, 
                  design_scaffold_length=DEFAULT_MAX_SCAF_LEN - 300, 
                  max_scaffold_length=DEFAULT_MAX_SCAF_LEN, 
                  radius_diff=DEFAULT_RADIUS_DIFF, 
                  radius_shape=DEFAULT_RADIUS_SHAPE,
                  add_verts=False,
                  zero_gravity=True):
    """
    Convert a mesh file to a target shape point cloud for DNA origami design.
    
    Args:
        file_name (str): Path to mesh file (.ply, .stl)
        design_scaffold_length (int): Length of the DNA scaffold used in the design
        max_scaffold_length (int): Maximum length of scaffold
        radius_diff (float): Radius for BP estimation
        radius_shape (float): Radius for intersection removal
        add_verts (bool): Whether to add vertices
        zero_gravity (bool): Whether to center the shape at origin
        
    Returns:
        tuple: (shape, num_bps) where:
            - shape: Target shape point cloud coordinates [Nx3]
            - num_bps: Number of base-pairs for diffusion model
    """
    
    # Read mesh file
    msh = meshio.read(file_name)
    coords = msh.points
    
    # Process edges from mesh faces
    edge_index = _extract_edges_from_mesh(msh)
    edge_length = length_edge_np(coords, edge_index)
    total_path_length = np.sum(edge_length)
    min_path_length = np.min(edge_length)
    
    # Calculate maximum and design shape lengths in nm
    max_shape_length = ((max_scaffold_length - 300) / 2) * SCAFFOLD_CONVERSION_FACTOR
    design_shape_length = (design_scaffold_length / 2) * SCAFFOLD_CONVERSION_FACTOR
    
    # Validate and provide feedback on scaling
    scale_min = MIN_EDGE_LENGTH / min_path_length
    scale_max = max_shape_length / total_path_length

    if design_shape_length > max_shape_length:
        print(f"Warning: Design scaffold length is greater than max scaffold length!")
        design_shape_length = max_shape_length

    scale = design_shape_length / total_path_length

    if scale_min > scale_max:
        print(f"Warning: Target pattern is too large and detailed for scaffold M13!")
        print(f"Warning: Some edges will be too short to ensure the shape is stable!")
        scale = scale_max

    elif scale >= scale_max:
        scale = scale_max

    elif scale <= scale_min:
        scale = scale_min
    
    # Apply calculated scale
    coords = 0.95 * scale * coords
    
    # Generate shape point cloud
    edge_length = length_edge_np(coords, edge_index)
    # edge_mask = edge_length < edge_length.mean() * EDGE_LENGTH_THRESHOLD_MULTIPLIER
    # edge_index = edge_index[:, edge_mask]
    
    shape, pathID = get_shape_from_edges(coords, edge_index, Le=1.0)
    
    if zero_gravity:
        shape = shape - np.mean(shape, axis=0)
    
    # Convert to torch tensors for diffusion model
    shape = torch.from_numpy(shape)
    pathID = torch.from_numpy(pathID)
    num_bps = estimate_diff_BPs(shape, pathID, radius_diff)
    shape = remove_intersection(shape, pathID, radius_shape, add_verts)
    
    print(f"Fitting target shape with {num_bps} base pairs")
    return shape.to(torch.float32), num_bps


def _extract_edges_from_mesh(msh):
    """Extract unique edges from mesh faces."""
    edge_index_lst = []
    
    for cell in msh.cells:
        cells = cell.data
        
        # Generate edges from faces
        for i in range(cells.shape[1] - 1):
            edge_index_lst.append(cells[:, i:i+2])
        edge_index_lst.append(cells[:, -2:])
        edge_index_lst.append(np.hstack((cells[:, 0:1], cells[:, -1:])))
    
    # Process edges
    edge_index = np.vstack(edge_index_lst)
    sorted_edge_index = np.sort(edge_index, axis=1)
    edge_index = np.unique(sorted_edge_index, axis=0)
    
    return edge_index.T



def main():
    """Test the mesh_to_shape function with STL file."""
    # Test with the specific STL file requested
    #test_file = "target_shapes/stl/miura_ori_3x3_40PercentFolded.stl"
    test_file = "target_shapes/ply/3D/01_tetrahedron.ply"
    
    if os.path.exists(test_file):
        print(f"\n{'='*50}")
        print(f"Testing with: {test_file}")
        print(f"{'='*50}")

        # Test with default scaffold length
        try:
            print("\n--- Testing with default scaffold length (7429) ---")
            shape, num_bps = mesh_to_shape(
                file_name=test_file,
                zero_gravity=True
            )
            
            print(f"Successfully processed {test_file}")
            print(f"Shape dimensions: {shape.shape}")
            print(f"Number of base pairs: {num_bps}")
            print(f"Shape bounds: min={shape.min(dim=0)[0]}, max={shape.max(dim=0)[0]}")
            
        except Exception as e:
            print(f"Error processing {test_file}: {e}")
        
        # Test with smaller scaffold length
        try:
            print("\n--- Testing with smaller scaffold length (3000) ---")
            shape_scaled, num_bps_scaled = mesh_to_shape(
                file_name=test_file,
                design_scaffold_length=3000,
                zero_gravity=True
            )
            
            print(f"Successfully processed {test_file} with design_scaffold_length=3000")
            print(f"Shape dimensions: {shape_scaled.shape}")
            print(f"Number of base pairs: {num_bps_scaled}")
            print(f"Shape bounds: min={shape_scaled.min(dim=0)[0]}, max={shape_scaled.max(dim=0)[0]}")

            # Show visualization for smaller scaffold version
            fig = plot_dna(shape_scaled)
            fig.show()
            
        except Exception as e:
            print(f"Error processing {test_file} with design_scaffold_length=3000: {e}")
    else:
        print(f"File not found: {test_file}")


if __name__ == "__main__":
    main()
