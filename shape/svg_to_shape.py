"""
SVG to Shape Converter Module

This module provides functionality to convert SVG files to 3D shape representations
suitable for DNA origami design. It processes SVG paths, scales them appropriately,
and generates point clouds with associated metadata.
"""

import os
from pathlib import Path
from typing import Tuple, List, Optional, Union
import numpy as np
import torch
from xml.dom import minidom
from svg.path import parse_path
from torch_geometric.nn import knn_graph

try:
    # Try relative import (when imported as a module)
    from .shape_ults import length_edge_pyt, estimate_diff_BPs, remove_intersection
    from .shape_ults import *
except ImportError:
    # Fall back to absolute import (when run directly)
    from shape_ults import length_edge_pyt, estimate_diff_BPs, remove_intersection
    from shape_ults import *


# Constants
DEFAULT_SCALE = 1.0
DEFAULT_MAX_SCAFFOLD_LENGTH = 7249
DEFAULT_RADIUS_DIFF = 1.0
DEFAULT_RADIUS_SHAPE = 1.0
MIN_PATH_LENGTH = 10
SCAFFOLD_CONVERSION_FACTOR = 0.34
EDGE_LENGTH_TARGET = 1.0
EDGE_LENGTH_TOLERANCE = 0.1


def get_point_at(path, distance: float, scale: float, offset: complex) -> np.ndarray:
    """
    Get a point at a specific distance along a path.
    
    Args:
        path: SVG path object
        distance: Normalized distance along the path (0 to 1)
        scale: Scale factor for the coordinates
        offset: Complex number representing the offset
        
    Returns:
        numpy array containing the 2D coordinates
    """
    pos = path.point(distance)
    pos += offset
    pos *= scale
    return np.array([[pos.real, pos.imag]])


def points_from_path(path, scale: float, offset: complex):
    """
    Extract points from an SVG path.
    
    Args:
        path: SVG path object
        scale: Scale factor for the coordinates
        offset: Complex number representing the offset
        
    Yields:
        numpy arrays containing 2D coordinates along the path
    """
    step = int(scale * path.length()) 
    last_step = step - 1
    
    if last_step == 0:
        yield get_point_at(path, 0, scale, offset)
        return
        
    for distance in range(step):
        yield get_point_at(path, distance / last_step, scale, offset)


def points_from_svg(svg, design_scaffold_length: int = DEFAULT_MAX_SCAFFOLD_LENGTH - 300, 
                    max_scaffold_length: int = DEFAULT_MAX_SCAFFOLD_LENGTH, 
                    offset: Tuple[float, float] = (0.0, 0.0)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract points from all paths in an SVG file.
    
    Args:
        svg: Parsed SVG document
        design_scaffold_length: length of the DNA scaffold used in the design
        max_scaffold_length: Maximum length of the DNA scaffold
        offset: Offset to apply to all coordinates
        
    Returns:
        Tuple of (point_cloud, path_ids) where:
        - point_cloud: numpy array of shape (N, 2) with coordinates
        - path_ids: numpy array of shape (N,) with path IDs for each point
    """
    offset_complex = offset[0] + offset[1] * 1j
    
    # Calculate total path length
    path_lengths = []
    for element in svg.getElementsByTagName("path"):
        for path in parse_path(element.getAttribute("d")):
            if path.length() > 0:
                path_lengths.append(path.length())
    
    if not path_lengths:
        raise ValueError("No valid paths found in SVG")
    
    total_path_length = sum(path_lengths)
    min_path_length = min(path_lengths)
            
    max_shape_length = ((max_scaffold_length - 300) / 2) * SCAFFOLD_CONVERSION_FACTOR
    design_shape_length = (design_scaffold_length / 2) * SCAFFOLD_CONVERSION_FACTOR
    
    # Validate and provide feedback on scaling
    scale_min = MIN_PATH_LENGTH / min_path_length
    scale_max = max_shape_length / total_path_length

    if design_shape_length > max_shape_length:
        print(f"Warning: Design scaffold length is greater than max scaffold length!")
        design_shape_length = max_shape_length

    scale = design_shape_length / total_path_length

    if scale_min > scale_max:
        print(f"Warning: Target pattern is too large and detailed for scaffold M13!")
        print(f"Warning: Some edges will be too short to ensure the shape is stable!")
        scale = scale_max

    elif scale > scale_max:
        scale = scale_max
        print(f"Warning: Design scaffold length is over the max scaffold length!")

    elif scale < scale_min:
        scale = scale_min
        print(f"Warning: Design scaffold length is too short to show the shape in detail!")


    # Extract points from all paths
    paths = []
    for element in svg.getElementsByTagName("path"):
        for path in parse_path(element.getAttribute("d")):
            points = list(points_from_path(path, scale, offset_complex))
            if points: 
                paths.append(np.vstack(points))
                
    if not paths:
        raise ValueError("No points extracted from SVG paths")
        
    point_cloud = np.vstack(paths)    
        
    # Assign path IDs for each point
    num_paths = len(paths)
    num_points_per_path = np.array([len(path) for path in paths])
    path_ids = np.repeat(np.arange(num_paths), num_points_per_path)
    
    return point_cloud, path_ids


def svg_to_shape(svg_file: Union[str, Path], 
                design_scaffold_length: int = DEFAULT_MAX_SCAFFOLD_LENGTH - 200, 
                max_scaffold_length: int = DEFAULT_MAX_SCAFFOLD_LENGTH, 
                radius_diff: float = DEFAULT_RADIUS_DIFF, 
                radius_shape: float = DEFAULT_RADIUS_SHAPE,
                add_vertices: bool = False, 
                zero_gravity: bool = True) -> Tuple[torch.Tensor, int]:
    """
    Convert an SVG file to a 3D shape representation suitable for DNA origami.
    
    Args:
        svg_file: Path to the SVG file
        scale: Scale factor for the coordinates
        max_scaffold_length: Maximum length of the DNA scaffold
        radius_diff: Radius for differential base pair estimation
        radius_shape: Radius for shape processing
        add_vertices: Whether to add additional vertices
        zero_gravity: Whether to center the shape at origin
        
    Returns:
        Tuple of (shape, num_base_pairs) where:
        - shape: torch.Tensor of shape (N, 3) with 3D coordinates
        - num_base_pairs: estimated number of base pairs needed
        
    Raises:
        FileNotFoundError: If the SVG file doesn't exist
        ValueError: If the SVG file is invalid or contains no paths
    """
    svg_path = Path(svg_file)
    if not svg_path.exists():
        raise FileNotFoundError(f"SVG file not found: {svg_file}")
    
    try:
        # Parse the SVG file
        svg = minidom.parse(str(svg_path))
        path_strings = [path.getAttribute('d') for path in svg.getElementsByTagName('path')]
        
        if not path_strings:
            raise ValueError("No paths found in SVG file")
            
        # Process only the first path for now
        svg_string = f"<svg> <path d=\"{path_strings[0]}\" /> </svg>"
        svg_parsed = minidom.parseString(svg_string)
        
        # Extract 2D shape points
        shape_2d, path_ids = points_from_svg(
            svg_parsed, 
            design_scaffold_length=design_scaffold_length, 
            max_scaffold_length=max_scaffold_length, 
            offset=(0.0, 0.0)
        )
        
        svg.unlink()
        svg_parsed.unlink()
        
    except Exception as e:
        raise ValueError(f"Failed to parse SVG file: {e}")
    
    # Remove duplicate points
    shape_2d = np.round(shape_2d, decimals=2)
    shape_2d, unique_indices = np.unique(shape_2d, axis=0, return_index=True)
    path_ids = path_ids[unique_indices]
    
    # Convert to tensors and add z-dimension
    shape_2d_tensor = torch.from_numpy(shape_2d).float()
    path_ids_tensor = torch.from_numpy(path_ids).long()
    shape_3d = torch.hstack((
        shape_2d_tensor, 
        torch.zeros((shape_2d_tensor.shape[0], 1), device=shape_2d_tensor.device)
    ))
    
    # Create edges between consecutive points in the same path
    edge_index = knn_graph(shape_3d, k=2, loop=False)
    edge_index = edge_index[:, path_ids_tensor[edge_index[0]] == path_ids_tensor[edge_index[1]]]
    
    # Normalize edge lengths to target length
    edge_lengths = length_edge_pyt(coords=shape_3d, edge_index=edge_index)
    mean_edge_length = edge_lengths.mean()
    
    if mean_edge_length > 0:
        shape_3d /= (mean_edge_length / EDGE_LENGTH_TARGET)
    
    if zero_gravity:
        shape_3d -= shape_3d.mean(dim=0)
    
    # Validate edge lengths after normalization
    edge_lengths_normalized = length_edge_pyt(coords=shape_3d, edge_index=edge_index)
    mean_normalized = edge_lengths_normalized.mean().item()
    
    if abs(mean_normalized - EDGE_LENGTH_TARGET) > EDGE_LENGTH_TOLERANCE:
        print(f"Warning: Edge length normalization may be inaccurate. "
              f"Mean edge length: {mean_normalized:.3f}")
    
    # Estimate number of base pairs and remove intersections
    num_base_pairs = estimate_diff_BPs(shape_3d, path_ids_tensor, radius_diff)
    shape_3d = remove_intersection(shape_3d, path_ids_tensor, radius_shape, add_vertices)
    
    print(f"Target shape fitted with {num_base_pairs} base pairs")
    
    return shape_3d.to(torch.float32), num_base_pairs


def main():
    """Main function for testing the SVG to shape conversion."""
    # Get the directory of this script and go to parent directory
    script_dir = Path(__file__).parent
    parent_dir = script_dir.parent
    os.chdir(parent_dir)
    
    try:
        from visual import plot_dna
        
        # Test with a sample SVG file
        svg_file = parent_dir / 'target_shapes' / 'svg' / 'Flower_Logo.svg'
        
        if not svg_file.exists():
            print(f"Test file not found: {svg_file}")
            return
            
        shape, num_bps = svg_to_shape(
            svg_file, 
            design_scaffold_length=1000, 
            max_scaffold_length=7249, 
            radius_diff=1.0, 
            radius_shape=1.0,
            add_vertices=True,
            zero_gravity=True
        )
        
        # Visualize the result
        fig = plot_dna(shape)
        fig.show()
        
        print(f"Successfully converted SVG to shape with {num_bps} base pairs")
        print(f"Shape dimensions: {shape.shape}")
        
    except ImportError:
        print("Visualization module not available")
    except Exception as e:
        print(f"Error during testing: {e}")


if __name__ == "__main__":
    main()

