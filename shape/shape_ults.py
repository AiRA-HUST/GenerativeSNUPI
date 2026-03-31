import torch
import numpy as np
import networkx as nx
from itertools import repeat
from shapely import LineString
from shapely.ops import substring
from torch_geometric.data import Data
from torch_geometric.utils import degree, scatter
from torch_geometric.nn import radius_graph
from torch_geometric.utils.convert import to_networkx
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visual import *

device = torch.device("cpu")

# ================================================================================================
# SHAPE GENERATION AND PROCESSING FUNCTIONS
# ================================================================================================

def get_shape_from_edges(verts, edge_index, Le=1.0):
    """
    Generate shape coordinates from vertices and edge indices by interpolating points along edges.
    
    Args:
        verts: Vertex coordinates array
        edge_index: Edge connectivity information 
        Le: Length spacing between interpolated points (default: 1.0)
    
    Returns:
        shape: Interpolated coordinates along edges
        pathID: Path ID for each generated point
    """
    shape, pathID = interpolate_edge(verts, edge_index, Le=Le)
    return shape, pathID


def get_shape_from_contours(cnt_coords, pts_cntID):
    """
    Generate shape coordinates from contour data by cutting contours into equal-length segments.
    
    Args:
        cnt_coords: Contour coordinates array
        pts_cntID: Contour ID for each point
    
    Returns:
        cnt_coords: Processed contour coordinates
        pts_cntID: Updated contour IDs
    """
    num_cnts = np.max(pts_cntID) + 1
    cnt_lst = []
    cnt_len = np.zeros(num_cnts, dtype=int)
    
    for cntID in range(num_cnts):
        # Extract contour points for current contour ID
        cnt = cnt_coords[pts_cntID == cntID, :2]
        cnt = np.vstack((cnt, cnt[:1, :]))  # Close the contour by adding first point at end
        
        # Create LineString and cut into equal segments
        line = LineString(cnt)
        lines = cut(line, distance=1.0)
        
        # Extract coordinates from segmented lines
        cnt = np.zeros((len(lines), 3))
        for i in range(len(lines)):
            x, y = lines[i].coords.xy
            cnt[i, 0] = x[0]
            cnt[i, 1] = y[0]
            cnt[i, 2] = 0.0
        
        cnt_lst.append(cnt)    
        cnt_len[cntID] = cnt.shape[0]
    
    # Combine all contours and create updated contour IDs
    cnt_coords = np.vstack(cnt_lst)   
    pts_cntID = np.arange(num_cnts)
    pts_cntID = np.repeat(pts_cntID, cnt_len)
    
    return cnt_coords, pts_cntID


def interpolate_edge(coords, edge_index, Le=1.0):
    """
    Interpolate points along edges with approximately equal spacing.
    
    Args:
        coords: Vertex coordinates
        edge_index: Edge connectivity [2, num_edges]
        Le: Desired length spacing between points (default: 1.0)
    
    Returns:
        interpolated_coords: Stacked coordinates of all interpolated points
        pts_edgeID: Edge ID for each interpolated point
    """
    num_edges = edge_index.shape[1]
    
    # Calculate relative vectors and lengths for all edges
    edges_relvec = coords[edge_index[1]] - coords[edge_index[0]]
    edges_length = np.linalg.norm(edges_relvec, axis=-1)
    
    edges = []
    for i in range(num_edges):
        # Calculate number of points needed for this edge
        num_pts = round(edges_length[i].item() / Le + 1)
        
        # Generate evenly spaced parameter values from 0 to 1
        pts_idx = np.linspace(0., 1.0, num=num_pts, endpoint=True)
        
        # Interpolate points along the edge
        inner_pts = pts_idx[:, None] * edges_relvec[i, :] + coords[edge_index[0, i]]
        edges.append(inner_pts)
    
    # Assign edge ID to each interpolated point
    num_pts_per_edge = [len(edges[i]) for i in range(num_edges)]
    num_pts_per_edge = np.array(num_pts_per_edge)
    pts_edgeID = np.arange(num_edges)
    pts_edgeID = np.repeat(pts_edgeID, num_pts_per_edge)
        
    return np.vstack(edges), pts_edgeID


# ================================================================================================
# EDGE LENGTH CALCULATION FUNCTIONS
# ================================================================================================

def length_edge_np(coords, edge_index):
    """
    Calculate edge lengths using NumPy operations.
    
    Args:
        coords: Vertex coordinates (numpy array)
        edge_index: Edge connectivity [2, num_edges]
    
    Returns:
        edge_length: Length of each edge
    """
    src, dst = edge_index
    edge_relvec = coords[dst] - coords[src]
    edge_length = np.linalg.norm(edge_relvec, axis=-1)
    return edge_length


def length_edge_pyt(coords, edge_index):
    """
    Calculate edge lengths using PyTorch operations.
    
    Args:
        coords: Vertex coordinates (PyTorch tensor)
        edge_index: Edge connectivity [2, num_edges]
    
    Returns:
        edge_length: Length of each edge
    """
    src, dst = edge_index
    edge_relvec = coords[dst] - coords[src]
    edge_length = edge_relvec.norm(dim=-1, keepdim=False)
    return edge_length


# ================================================================================================
# BASE-PAIR ESTIMATION FUNCTIONS
# ================================================================================================

def estimate_diff_BPs(coords, pathID, offset_radius=2.0):
    """
    Estimate the number of base pairs needed for a DNA structure from pattern coordinates.
    
    Args:
        coords: Coordinate tensor of shape points
        pathID: Path ID for each point
        offset_radius: Radius for finding nearby points (default: 2.0)
    
    Returns:
        num_bps: Estimated number of base pairs
    """
    num_paths = pathID.amax() + 1
    
    # Find edges between nearby points from different paths
    edge_index = radius_graph(coords, r=offset_radius, loop=False)
    edge_index = edge_index[:, pathID[edge_index[0]] != pathID[edge_index[1]]]
    
    # Get points involved in intersections
    pointID_offset = torch.unique(edge_index.flatten())
    num_nodes = coords.size(0)
    
    # Segment intersection points into groups
    node_gID = segmentation(edge_index, num_nodes, min_nodes=1)
    
    # Estimate group sizes and calculate nucleotides to remove
    gsize = estimate_group_size(node_gID, edge_index, pathID)
    num_remove_nts = gsize[gsize > 2].sum() * 5
    
    # Calculate final base pair count
    num_bps = (num_nodes - num_paths) * 6 - num_remove_nts
    return num_bps


def estimate_group_size(node_gID, edge_index, pathID):
    """
    Estimate the size of each intersection group based on the number of unique paths.
    
    Args:
        node_gID: Group ID for each node
        edge_index: Edge connectivity information
        pathID: Path ID for each point
    
    Returns:
        group_size: Size of each group (number of unique paths)
    """
    num_groups = int(node_gID.max() + 1)
    group_size = torch.zeros(num_groups, dtype=torch.long, device=node_gID.device)
    
    for i in range(num_groups):
        # Find edges within this group
        mask = torch.logical_and(node_gID[edge_index[0]] == i, node_gID[edge_index[1]] == i)
        
        # Get unique paths in this group
        group_path_index = pathID[edge_index[:, mask].flatten()]
        group_path = torch.unique(group_path_index)
        group_size[i] = group_path.size(0)
    
    return group_size


# ================================================================================================
# INTERSECTION REMOVAL FUNCTIONS
# ================================================================================================

def remove_intersection(coords, pathID, offset_radius=1.0, add_verts=True):
    """
    Remove intersection points from coordinates to clean up the structure.
    
    Args:
        coords: Coordinate tensor
        pathID: Path ID for each point
        offset_radius: Radius for finding intersections (default: 1.0)
        add_verts: Whether to add vertices (currently unused)
    
    Returns:
        coords: Cleaned coordinates with intersections removed
    """
    # Find intersections between different paths
    edge_index = radius_graph(coords, r=offset_radius, loop=False)
    edge_index = edge_index[:, pathID[edge_index[0]] != pathID[edge_index[1]]]
    num_nodes = coords.size(0)
    
    # Segment intersection points
    node_gID = segmentation(edge_index, num_nodes, min_nodes=1)
    
    # Remove intersections with more than 2 paths
    src, dst = edge_index
    edge_index = edge_index[:, pathID[src] != pathID[dst]]
    Gsize = estimate_group_size(node_gID, edge_index, pathID)
    node_Gsize = Gsize[node_gID]
    
    # Filter out nodes in large intersection groups
    mask = torch.logical_and(node_gID > -1, node_Gsize > 2)
    coords = coords[~mask]
    pathID = pathID[~mask]
    
    # Second pass: remove remaining close intersections
    edge_index = radius_graph(coords, r=0.5, loop=False)
    edge_index = edge_index[:, pathID[edge_index[0]] != pathID[edge_index[1]]]
    num_nodes = coords.size(0)
    node_gID = segmentation(edge_index, num_nodes, min_nodes=1)
    
    # Keep only isolated points
    coords = coords[node_gID < 0, :]
    return coords


def segmentation(edge_index, num_nodes, min_nodes=1):
    """
    Segment nodes into connected components and filter by minimum size.
    
    Args:
        edge_index: Edge connectivity information
        num_nodes: Total number of nodes
        min_nodes: Minimum nodes required for a valid group (default: 1)
    
    Returns:
        node_Gindex: Group index for each node (-1 for filtered out nodes)
    """
    # Create NetworkX graph and find connected components
    groups_graph = to_networkx(Data(edge_index=edge_index, num_nodes=num_nodes), to_undirected=True)
    groups = list(nx.connected_components(groups_graph))
    num_groups = len(groups)
    
    # Initialize group indices
    node_Gindex = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
    idx = 0
    
    # Assign group indices, filtering out small groups
    for i in range(num_groups):
        if len(groups[i]) > min_nodes:  # Keep groups with sufficient nodes
            for node_ID in groups[i]:
                node_Gindex[node_ID] = idx
            idx += 1
        else:  # Mark small groups as invalid
            for node_ID in groups[i]:
                node_Gindex[node_ID] = -1
    
    return node_Gindex


# ================================================================================================
# UTILITY FUNCTIONS
# ================================================================================================

def cut(line, distance):
    """
    Cut a Shapely LineString into segments of specified distance.
    
    Args:
        line: Shapely LineString object
        distance: Distance between cut points
    
    Returns:
        List of LineString segments
    """
    distances = np.arange(0, line.length + distance, distance)
    return list(map(substring, repeat(line), distances[:-1], distances[1:]))


import roma
def random_rotate_3dpcl(point_cloud):
    """
    Rotate a 3D point cloud using a random rotation matrix.
    
    Args:
        point_cloud: 3D PyTorch tensor of shape (n, 3) where n is the number of points
    
    Returns:
        rotated_point_cloud: Rotated point cloud with same shape as input
    """
    rotation_matrix = roma.random_rotmat()
    return torch.mm(point_cloud, rotation_matrix.t())


# ================================================================================================
# GEOMETRIC SHAPE GENERATION FUNCTIONS
# ================================================================================================

def circle2D(radius, x0=0., y0=0., z0=0., return_num_bps=False):
    """
    Generate a 2D circle with evenly spaced points.
    
    Args:
        radius: Radius of the circle
        x0, y0, z0: Center coordinates (default: origin)
        return_num_bps: Whether to return base pair count (default: False)
    
    Returns:
        coords: Circle coordinates as PyTorch tensor
        num_bps: Number of base pairs (only if return_num_bps=True)
    """
    # Calculate number of angular steps based on circumference
    num_angsteps = round(2 * np.pi * radius)
    theta = np.linspace(0, 2 * np.pi, num_angsteps)
    
    # Generate circle coordinates
    x = x0 + radius * np.cos(theta)
    y = y0 + radius * np.sin(theta)
    coords = np.vstack((x, y))
    coords = coords.astype(np.float32)
    coords = np.transpose(coords)
    
    # Divide curve into equal-length segments
    line = LineString(coords)
    lines = cut(line, distance=1.0)
    
    coords = np.zeros((len(lines), 3))
    for i in range(len(lines)):
        x, y = lines[i].coords.xy
        coords[i, 0] = x[0]
        coords[i, 1] = y[0]
        coords[i, 2] = z0
    
    # Convert to PyTorch tensor
    coords = torch.from_numpy(coords)
    coords = coords.to(torch.float32)
    
    if return_num_bps:
        num_bps = int(coords.size(0) * 6)
        print("Fitting target shape with %d bps" % (num_bps))
        return coords, num_bps
    else:
        return coords


def arc2D(radius, x0=0., y0=0., z0=0., arc=2*np.pi):
    """
    Generate a 2D arc with evenly spaced points.
    
    Args:
        radius: Radius of the arc
        x0, y0, z0: Center coordinates (default: origin)
        arc: Arc length in radians (default: 2π for full circle)
    
    Returns:
        coords: Arc coordinates as NumPy array
    """
    # Calculate number of angular steps
    num_angsteps = round(2 * np.pi * radius)
    theta = np.linspace(0, arc, num_angsteps)
    
    # Generate arc coordinates
    x = x0 + radius * np.cos(theta)
    y = y0 + radius * np.sin(theta)
    coords = np.vstack((x, y))
    coords = coords.astype(np.float32)
    coords = np.transpose(coords)
    
    # Divide curve into equal-length segments
    line = LineString(coords)
    lines = cut(line, distance=1.0)
    
    coords = np.zeros((len(lines), 3))
    for i in range(len(lines)):
        x, y = lines[i].coords.xy
        coords[i, 0] = x[0]
        coords[i, 1] = y[0]
        coords[i, 2] = z0
    
    return coords


def generate_helical_point_cloud(radius, pitch, height, init_angle, num_points=None):
    """
    Generate a helical point cloud with approximately equal spacing between points.
    
    Args:
        radius: Radius of the helix
        pitch: Pitch of the helix (distance between consecutive turns)
        height: Total height of the helix
        init_angle: Initial angle offset
        num_points: Number of points (auto-calculated if None)
    
    Returns:
        point_cloud: NumPy array of (x, y, z) coordinates
    """
    if num_points is None:
        # Calculate points needed based on helix geometry
        num_points = round(2 * np.pi * radius * (height / pitch))
    
    # Generate parametric values along helix height
    t = np.linspace(0, height, num_points)
    
    # Calculate helical coordinates
    x = radius * np.cos((2 * np.pi * t / pitch) + init_angle)
    y = radius * np.sin((2 * np.pi * t / pitch) + init_angle)
    z = t
    
    point_cloud = np.column_stack((x, y, z))
    return point_cloud


def rotate_point_cloud(point_cloud, axis, angle_degrees):
    """
    Rotate a 3D point cloud around a specified axis.
    
    Args:
        point_cloud: Nx3 numpy array of 3D points
        axis: Rotation axis ('x', 'y', or 'z')
        angle_degrees: Rotation angle in degrees
    
    Returns:
        rotated_point_cloud: Rotated 3D point cloud
    """
    angle_radians = np.radians(angle_degrees)
    
    # Define rotation matrices for each axis
    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle_radians), -np.sin(angle_radians)],
            [0, np.sin(angle_radians), np.cos(angle_radians)]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [np.cos(angle_radians), 0, np.sin(angle_radians)],
            [0, 1, 0],
            [-np.sin(angle_radians), 0, np.cos(angle_radians)]
        ])
    elif axis == 'z':
        rotation_matrix = np.array([
            [np.cos(angle_radians), -np.sin(angle_radians), 0],
            [np.sin(angle_radians), np.cos(angle_radians), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Invalid axis. Choose 'x', 'y', or 'z'.")
    
    rotated_point_cloud = np.dot(point_cloud, rotation_matrix.T)
    return rotated_point_cloud


# ================================================================================================
# COORDINATE MAPPING CLASSES
# ================================================================================================

import torch.nn as nn

class RectangularToSphereMapping(nn.Module):
    """
    Neural network module for mapping rectangular coordinates to spherical coordinates.
    """
    
    def __init__(self, sphere_radius):
        """
        Initialize the mapping with specified sphere radius.
        
        Args:
            sphere_radius: Radius of the target sphere
        """
        super(RectangularToSphereMapping, self).__init__()
        self.sphere_radius = sphere_radius

    def forward(self, x, y):
        """
        Map rectangular (x, y) coordinates to 3D spherical coordinates.
        
        Args:
            x, y: Input rectangular coordinates
        
        Returns:
            coords: 3D Cartesian coordinates on sphere surface
        """
        # Convert rectangular to spherical coordinates
        r = torch.sqrt(x**2 + y**2)
        theta = torch.atan2(y, x)  # Azimuthal angle
        phi = (torch.pi / 2) - torch.acos(r / self.sphere_radius)  # Polar angle
        
        # Convert spherical to 3D Cartesian coordinates
        x_sphere = self.sphere_radius * torch.sin(phi) * torch.cos(theta)
        y_sphere = self.sphere_radius * torch.sin(phi) * torch.sin(theta)
        z_sphere = self.sphere_radius * torch.cos(phi)
        
        coords = torch.hstack((x_sphere[:, None], y_sphere[:, None], z_sphere[:, None]))
        return coords


class RectangularToCylinderMapping(torch.nn.Module):
    """
    Neural network module for mapping rectangular coordinates to cylindrical coordinates.
    """
    
    def __init__(self, cylinder_radius, cylinder_height):
        """
        Initialize the mapping with specified cylinder parameters.
        
        Args:
            cylinder_radius: Radius of the target cylinder
            cylinder_height: Height of the target cylinder
        """
        super(RectangularToCylinderMapping, self).__init__()
        self.cylinder_radius = cylinder_radius
        self.cylinder_height = cylinder_height

    def forward(self, x, y):
        """
        Map rectangular (x, y) coordinates to 3D cylindrical coordinates.
        
        Args:
            x, y: Input rectangular coordinates
        
        Returns:
            coords: 3D Cartesian coordinates on cylinder surface
        """
        # Map y directly to cylinder height
        y_cylinder = y
        
        # Convert x to angular coordinate and map to cylinder surface
        r = torch.tensor(self.cylinder_radius)
        theta = x / r
        z_cylinder = r * torch.cos(theta)
        x_cylinder = r * torch.sin(theta)
        
        coords = torch.hstack((x_cylinder[:, None], y_cylinder[:, None], z_cylinder[:, None]))
        return coords