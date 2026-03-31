# Import libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import imageio
import roma
import plotly.graph_objects as go


# Constants
DEFAULT_NODE_SIZE = 5
DEFAULT_EDGE_WIDTH = 1.5
DEFAULT_SINGLE_EDGE_WIDTH = 1.0
DEFAULT_AUG_EDGE_WIDTH = 5.0
DEFAULT_MARKER_SIZE = 20
DEFAULT_FRAMES_PER_SECOND = 10
DEFAULT_STEP_SIZE = 5


def plot_samples(x, path, edge_index=None, edge_attr=None, show=True, target_shape=None, **kwargs):
    """
    Plot DNA samples using the plot_dna function.
    
    Args:
        x: Sample coordinates tensor
        path: Path to save the plot
        edge_index: Edge connectivity information
        edge_attr: Edge attributes
        show: Whether to display the plot
        target_shape: Target shape coordinates for comparison
        **kwargs: Additional arguments
    """
    fig = plot_dna(x[:, 0:3], edge_index=edge_index, edge_attr=edge_attr, target_shape=target_shape)
    if show:
        fig.show()


def create_gif(samples, edge_index, edge_attr, path, index):
    """
    Create a GIF animation from DNA diffusion samples.
    
    Args:
        samples: Tensor of sample coordinates over time
        edge_index: Edge connectivity information
        edge_attr: Edge attributes
        path: Directory path to save images and GIF
        index: Index for naming the output GIF
        
    Returns:
        Generated GIF object
    """
    num_time_steps = samples.size(1)
    
    # Generate individual frames
    for t in range(0, num_time_steps, DEFAULT_STEP_SIZE):
        save_path = os.path.join(path, f"img_{t}.png")
        fig = plot_dna(samples[:, t, :], edge_index=edge_index, edge_attr=edge_attr)
        fig.write_image(save_path)
    
    # Collect frames for GIF
    frames = []
    for t in range(0, num_time_steps, DEFAULT_STEP_SIZE):
        img_path = os.path.join(path, f"img_{t}.png")
        image = imageio.v2.imread(img_path)
        frames.append(image)
    
    # Create and save GIF
    gif_path = os.path.join(path, f"pos_diffusion_{index}.gif")
    gif = imageio.mimsave(gif_path, frames, fps=DEFAULT_FRAMES_PER_SECOND, loop=1)
    return gif


def _create_edge_trace(coord, edge_index, color, width):
    """
    Helper function to create edge traces for plotly visualization.
    
    Args:
        coord: Node coordinates
        edge_index: Edge connectivity
        color: Edge color
        width: Edge width
        
    Returns:
        Plotly Scatter3d trace for edges
    """
    x_edges, y_edges, z_edges = [], [], []
    row, col = edge_index
    
    for i in range(edge_index.size(1)):
        x_coords = [coord[row[i], 0], coord[col[i], 0], None]
        x_edges.extend(x_coords)
        y_coords = [coord[row[i], 1], coord[col[i], 1], None]
        y_edges.extend(y_coords)
        z_coords = [coord[row[i], 2], coord[col[i], 2], None]
        z_edges.extend(z_coords)
    
    return go.Scatter3d(
        x=x_edges, y=y_edges, z=z_edges,
        mode='lines',
        line=dict(color=color, width=width),
        hoverinfo='none'
    )


def _create_node_trace(coord, node_labels, color='blue', size=DEFAULT_NODE_SIZE):
    """
    Helper function to create node traces for plotly visualization.
    
    Args:
        coord: Node coordinates
        node_labels: Node labels for hover text
        color: Node color
        size: Node size
        
    Returns:
        Plotly Scatter3d trace for nodes
    """
    return go.Scatter3d(
        x=coord[:, 0], y=coord[:, 1], z=coord[:, 2],
        mode='markers',
        marker=dict(
            symbol='circle',
            size=size,
            color=color,
            colorscale=['lightgreen', 'magenta'],
            line=dict(color='black', width=0.5)
        ),
        text=node_labels,
        hoverinfo='text'
    )


def _get_axis_config():
    """Get standard axis configuration for plotly 3D plots."""
    return dict(
        showbackground=False,
        showline=False,
        zeroline=False,
        showgrid=False,
        showticklabels=False,
        title=''
    )


def _get_layout_config():
    """Get standard layout configuration for plotly 3D plots."""
    axis = _get_axis_config()
    return go.Layout(
        showlegend=False,
        scene=dict(
            xaxis=dict(axis),
            yaxis=dict(axis),
            zaxis=dict(axis)
        ),
        hovermode='closest'
    )


def plot_dna(coord, triads=None, edge_index=None, edge_attr=None, 
             show_edge_type='phys', node_labels=None, target_shape=None):
    """
    Plot DNA structure with various visualization options.
    
    Args:
        coord: Node coordinates tensor
        triads: Orientation triads (optional)
        edge_index: Edge connectivity information
        edge_attr: Edge attributes
        show_edge_type: Type of edges to show ('full', '0', '1', 'aug', 'phys')
        node_labels: Labels for nodes
        target_shape: Target shape coordinates for comparison
        
    Returns:
        Plotly Figure object
    """
    coord = coord.to("cpu")
    n_nodes = coord.size(0)
    
    # Prepare node labels
    if node_labels is not None:
        try:
            node_labels = node_labels.to("cpu")
        except:
            pass
    else:
        node_labels = torch.arange(0, n_nodes)
    
    data = []
    
    # Add target shape if provided
    if target_shape is not None:
        coords_shape = target_shape.to("cpu")
        trace_shape = _create_node_trace(coords_shape, node_labels, color='green', size=DEFAULT_NODE_SIZE)
        data.append(trace_shape)
    
    # Add edges if provided
    if edge_index is not None:
        edge_index = edge_index.to("cpu")
        
        if edge_attr is not None:
            edge_attr = edge_attr.to("cpu")
            
            # Double edges (type 0)
            if show_edge_type in ['full', 'phys', '0']:
                double_edge_index = edge_index[:, edge_attr[:, 0] == 1]
                if double_edge_index.size(1) > 0:
                    trace_double = _create_edge_trace(coord, double_edge_index, 'blue', DEFAULT_EDGE_WIDTH)
                    data.append(trace_double)
            
            # Single edges (type 1)
            if show_edge_type in ['full', 'phys', '1']:
                single_edge_index = edge_index[:, edge_attr[:, 1] == 1]
                if single_edge_index.size(1) > 0:
                    trace_single = _create_edge_trace(coord, single_edge_index, 'blue', DEFAULT_SINGLE_EDGE_WIDTH)
                    data.append(trace_single)
            
            # Augmented edges
            if edge_attr.size(1) > 2 and show_edge_type in ['full', 'aug']:
                # First augmented edge type
                aug_edge_index_1 = edge_index[:, edge_attr[:, 2] == 1]
                if aug_edge_index_1.size(1) > 0:
                    trace_aug_1 = _create_edge_trace(coord, aug_edge_index_1, 'red', DEFAULT_AUG_EDGE_WIDTH)
                    data.append(trace_aug_1)
                
                # Second augmented edge type
                if edge_attr.size(1) > 3:
                    aug_edge_index_2 = edge_index[:, edge_attr[:, 3] == 1]
                    if aug_edge_index_2.size(1) > 0:
                        trace_aug_2 = _create_edge_trace(coord, aug_edge_index_2, 'green', 0.5)
                        data.append(trace_aug_2)
        else:
            # Simple edge plotting without attributes
            trace_edges = _create_edge_trace(coord, edge_index, 'red', DEFAULT_AUG_EDGE_WIDTH)
            data.append(trace_edges)
    
    # Add orientation triads if provided
    if triads is not None:
        if len(triads.size()) > 2:
            triads = roma.rotmat_to_rotvec(triads)
        triads = triads.to("cpu")
        
        trace_triads = go.Cone(
            x=coord[:, 0], y=coord[:, 1], z=coord[:, 2],
            u=triads[:, 0], v=triads[:, 1], w=triads[:, 2],
            sizemode="scaled",
            sizeref=2
        )
        data.append(trace_triads)
    else:
        # Add main node trace
        trace_nodes = _create_node_trace(coord, node_labels)
        data.append(trace_nodes)
    
    # Create figure with layout
    layout = _get_layout_config()
    fig = go.Figure(data=data, layout=layout)
    
    # Set aspect ratio based on coordinate ranges
    x_range = torch.abs(coord[:, 0].max() - coord[:, 0].min()).item() / 100
    y_range = torch.abs(coord[:, 1].max() - coord[:, 1].min()).item() / 100
    z_range = torch.abs(coord[:, 2].max() - coord[:, 2].min()).item() / 100
    
    fig.update_layout(
        scene_aspectmode='manual',
        scene_aspectratio=dict(x=x_range, y=y_range, z=z_range)
    )
    
    return fig


def plot_system(coords, bonds=None, marker="o", ms=DEFAULT_MARKER_SIZE, azim=45, elev=60):
    """
    Plot DNA system using matplotlib 3D visualization.
    
    Args:
        coords: Node coordinates tensor
        bonds: Bond connectivity information
        marker: Marker style for nodes
        ms: Marker size
        azim: Azimuth angle for 3D view
        elev: Elevation angle for 3D view
        
    Returns:
        Matplotlib Figure object
    """
    coords = coords.cpu().detach().numpy()
    
    # Create 3D figure
    fig = plt.figure()
    fig.tight_layout()
    ax = fig.add_subplot(111, projection="3d")
    ax.grid(False)
    
    # Plot nodes
    ax.scatter(*coords.T, s=ms, marker=marker, c='blue', alpha=0.25, edgecolors='w')
    
    # Plot bonds if provided
    if bonds is not None:
        bonds = bonds.cpu().detach().numpy()
        edge_xyz = np.array([
            (coords[bonds[0, i], 0:3], coords[bonds[1, i], 0:3])
            for i in range(bonds.shape[1])
        ])
        
        for edge in edge_xyz:
            ax.plot(*edge.T, color="tab:gray", linewidth=5)
    
    # Set axis limits and properties
    padding = 2
    x_min, x_max = np.amin(coords[:, 0]) - padding, np.amax(coords[:, 0]) + padding
    y_min, y_max = np.amin(coords[:, 1]) - padding, np.amax(coords[:, 1]) + padding
    z_min, z_max = np.amin(coords[:, 2]) - padding, np.amax(coords[:, 2]) + padding
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    
    # Set aspect ratio
    x_range = np.abs(coords[:, 0].max() - coords[:, 0].min())
    y_range = np.abs(coords[:, 1].max() - coords[:, 1].min())
    z_range = np.abs(coords[:, 2].max() - coords[:, 2].min())
    ax.set_box_aspect(aspect=(x_range, y_range, z_range))
    
    ax.set_axis_off()
    ax.view_init(azim, elev)
    
    return fig


def _faces_to_edges(faces):
    """
    Convert face connectivity to edge connectivity.
    
    Args:
        faces: Face connectivity array
        
    Returns:
        Edge connectivity array
    """
    edge_index = np.concatenate((faces[:, :1], faces[:, 2:]), axis=1)
    edge_index = np.concatenate((edge_index, faces[:, :2], faces[:, 1:]), axis=0)
    return np.transpose(edge_index)


def plot_mesh(verts, faces, boundary=None):
    """
    Plot mesh using plotly visualization.
    
    Args:
        verts: Vertex coordinates
        faces: Face connectivity
        boundary: Boundary vertex coordinates (optional)
        
    Returns:
        Plotly Figure object
    """
    data = []
    
    # Extract coordinates
    x_nodes = verts[:, 0]
    y_nodes = verts[:, 1]
    z_nodes = verts[:, 2] if verts.shape[1] > 2 else np.zeros_like(verts[:, 0])
    
    # Create main node trace
    trace_nodes = go.Scatter3d(
        x=x_nodes, y=y_nodes, z=z_nodes,
        mode='markers',
        marker=dict(
            symbol='circle',
            size=4,
            color='blue',
            colorscale=['lightgreen', 'magenta'],
            line=dict(color='black', width=0.5)
        ),
        hoverinfo='text'
    )
    data.append(trace_nodes)
    
    # Add boundary nodes if provided
    if boundary is not None:
        x_bnodes = boundary[:, 0]
        y_bnodes = boundary[:, 1]
        z_bnodes = boundary[:, 2] if boundary.shape[1] > 2 else np.zeros_like(boundary[:, 0])
        
        trace_boundary = go.Scatter3d(
            x=x_bnodes, y=y_bnodes, z=z_bnodes,
            mode='markers',
            marker=dict(
                symbol='circle',
                size=8,
                color='red',
                colorscale=['lightgreen', 'magenta'],
                line=dict(color='black', width=0.5)
            ),
            hoverinfo='text'
        )
        data.append(trace_boundary)
    
    # Convert faces to edges and plot
    edge_index = _faces_to_edges(faces)
    
    x_edges, y_edges, z_edges = [], [], []
    row, col = edge_index
    
    for i in range(edge_index.shape[1]):
        x_coords = [x_nodes[row[i]], x_nodes[col[i]], None]
        x_edges.extend(x_coords)
        y_coords = [y_nodes[row[i]], y_nodes[col[i]], None]
        y_edges.extend(y_coords)
        z_coords = [z_nodes[row[i]], z_nodes[col[i]], None]
        z_edges.extend(z_coords)
    
    trace_edges = go.Scatter3d(
        x=x_edges, y=y_edges, z=z_edges,
        mode='lines',
        line=dict(color='red', width=0.75),
        hoverinfo='none'
    )
    data.append(trace_edges)
    
    # Create figure with layout
    layout = _get_layout_config()
    fig = go.Figure(data=data, layout=layout)
    
    # Set aspect ratio
    x_range = np.abs(x_nodes.max() - x_nodes.min()) / 100
    y_range = np.abs(y_nodes.max() - y_nodes.min()) / 100
    z_range = np.abs(z_nodes.max() - z_nodes.min()) / 100
    
    fig.update_layout(
        scene_aspectmode='manual',
        scene_aspectratio=dict(x=x_range, y=y_range, z=z_range)
    )
    
    return fig


def dna_matplot(coords, edge_index=None, color="cadetblue", size=30, 
                azim=45, elev=60, scale=True):
    """
    Plot DNA structure using matplotlib with customizable styling.
    
    Args:
        coords: Node coordinates
        edge_index: Edge connectivity information
        color: Color for nodes and edges
        size: Node size
        azim: Azimuth angle for 3D view
        elev: Elevation angle for 3D view
        scale: Whether to scale the plot to fit data
        
    Returns:
        Matplotlib Figure object
    """
    # Setup 3D figure
    fig = plt.figure()
    fig.tight_layout()
    fig.suptitle('Sample plot', fontsize=10)
    ax = fig.add_subplot(111, projection="3d")
    ax.grid(False)
    
    # Plot edges if provided
    if edge_index is not None:
        row, col = edge_index[:, ::2]
        
        for src, dst in zip(row, col):
            ax.plot3D(
                [coords[src, 0], coords[dst, 0]],
                [coords[src, 1], coords[dst, 1]],
                [coords[src, 2], coords[dst, 2]],
                c=color, alpha=0.5, linewidth=5
            )
    
    # Plot nodes
    ax.scatter(
        *coords.T, s=size, marker="o", c=color,
        alpha=1.0, edgecolors='gray', linewidth=0.25
    )
    
    # Scale plot if requested
    if scale:
        padding = 2
        x_min, x_max = np.amin(coords[:, 0]) - padding, np.amax(coords[:, 0]) + padding
        y_min, y_max = np.amin(coords[:, 1]) - padding, np.amax(coords[:, 1]) + padding
        z_min, z_max = np.amin(coords[:, 2]) - padding, np.amax(coords[:, 2]) + padding
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        
        # Set aspect ratio
        x_range = np.abs(coords[:, 0].max() - coords[:, 0].min())
        y_range = np.abs(coords[:, 1].max() - coords[:, 1].min())
        z_range = np.abs(coords[:, 2].max() - coords[:, 2].min())
        ax.set_box_aspect(aspect=(x_range, y_range, z_range))
    
    ax.set_axis_off()
    ax.view_init(azim, elev)
    
    return fig