# import library
import os
import roma
import torch
import imageio
import plotly.graph_objects as go

def plot_dna(coord, triads=None, edge_index=None, edge_attr=None, 
             show_edge_type='phys', node_labels=None, target_shape=None):
    # show_edge_type = 'full', '0', '1', 'aug', 'phys'
    coord = coord.to("cpu")
    # number of nodes
    n_nodes = coord.size(0)   
    if node_labels is not None:
        try:
            node_labels = node_labels.to("cpu")
        except:
            node_labels = node_labels
    else:
        node_labels = torch.arange(0, n_nodes)

    data = []    
    # we need to seperate the X,Y,Z coordinates for Plotly
    x_nodes = coord[:,0]      # x-coordinates of nodes
    y_nodes = coord[:,1]      # y-coordinates
    z_nodes = coord[:,2]      # z-coordinates
    
    #create a trace for the nodes
    if node_labels is None:
        node_labels = torch.arange(0, n_nodes)
    
    trace_nodes = go.Scatter3d( x = x_nodes,
                                y = y_nodes,
                                z = z_nodes,
                                mode = 'markers',
                                marker = dict(  symbol='circle',
                                                size = 5,
                                                color = 'blue', 
                                                colorscale = ['lightgreen','magenta'], # either green or mageneta
                                                line = dict(color='black', width=0.5)),
                                text = node_labels,
                                hoverinfo='text')
    
    if target_shape is not None:
        coords_shape = target_shape.to("cpu")
        x_shapenodes = coords_shape[:,0]      # x-coordinates of nodes
        y_shapenodes = coords_shape[:,1]      # y-coordinates
        z_shapenodes = coords_shape[:,2]      # z-coordinates
        trace_shapenodes = go.Scatter3d(x = x_shapenodes,
                                        y = y_shapenodes,
                                        z = z_shapenodes,
                                        mode = 'markers',
                                        marker = dict(  symbol='circle',
                                                        size = 5,
                                                        color = 'green', 
                                                        colorscale = ['lightgreen','magenta'], 
                                                        line = dict(color='green', width=0.25)),
                                        text = node_labels,
                                        hoverinfo='text')
        data.append(trace_shapenodes)
    # we  need to create lists that contain the starting and ending coordinates of each edge.
    if edge_index is not None:
        edge_index = edge_index.to("cpu")
        if edge_attr is not None:
            edge_attr = edge_attr.to("cpu")
            double_edge_index = edge_index[:,edge_attr[:,0]==1]
            if show_edge_type == 'full' or show_edge_type == 'phys' or show_edge_type == '0':
                # double edges plot
                x_edges =[]
                y_edges =[]
                z_edges =[]
                row, col = double_edge_index
                for i in range(double_edge_index.size(1)):
                    x_coords = [coord[row[i],0], coord[col[i],0],None]
                    x_edges += x_coords
                    y_coords = [coord[row[i],1], coord[col[i],1], None]
                    y_edges += y_coords
                    z_coords = [coord[row[i],2], coord[col[i],2],None]
                    z_edges += z_coords
                        
                trace_double_edges = go.Scatter3d(  x=x_edges,
                                                    y=y_edges,
                                                    z=z_edges,
                                                    mode = 'lines',
                                                    line = dict(color='blue', width=1.5),
                                                    hoverinfo = 'none')
                data.append(trace_double_edges)
            if show_edge_type == 'full' or show_edge_type == 'phys' or show_edge_type == '1': 
                single_edge_index = edge_index[:,edge_attr[:,1]==1]
                # single edges plot
                x_edges =[]
                y_edges =[]
                z_edges =[]
                row, col = single_edge_index
                for i in range(single_edge_index.size(1)):
                    x_coords = [coord[row[i],0], coord[col[i],0],None]
                    x_edges += x_coords
                    y_coords = [coord[row[i],1], coord[col[i],1], None]
                    y_edges += y_coords
                    z_coords = [coord[row[i],2], coord[col[i],2],None]
                    z_edges += z_coords
                        
                trace_single_edges = go.Scatter3d(  x=x_edges,
                                                    y=y_edges,
                                                    z=z_edges,
                                                    mode = 'lines',
                                                    line = dict(color='blue', width=1.0),
                                                    hoverinfo = 'none')
                
                # Include the trace edges  to plot and create a figure  
                data.append(trace_single_edges)
            if edge_attr.size(1) > 2:
                aug_edge_index_1 = edge_index[:, edge_attr[:,2]==1]
                if show_edge_type == 'full' or show_edge_type == 'aug':
                    # augmen edges plot
                    x_edges =[]
                    y_edges =[]
                    z_edges =[]
                    row, col = aug_edge_index_1
                    for i in range(aug_edge_index_1.size(1)):
                        x_coords = [coord[row[i],0], coord[col[i],0],None]
                        x_edges += x_coords
                        y_coords = [coord[row[i],1], coord[col[i],1], None]
                        y_edges += y_coords
                        z_coords = [coord[row[i],2], coord[col[i],2],None]
                        z_edges += z_coords
                            
                    trace_aug_edges_1 = go.Scatter3d(   x=x_edges,
                                                        y=y_edges,
                                                        z=z_edges,
                                                        mode = 'lines',
                                                        line = dict(color='red', width=5.),
                                                        hoverinfo = 'none')
                    # Include the trace edges  to plot and create a figure  
                    data.append(trace_aug_edges_1)
                    if edge_attr.size(1) > 3:
                        aug_edge_index_2 = edge_index[:, edge_attr[:,3]==1]
                        # augmen edges plot
                        x_edges =[]
                        y_edges =[]
                        z_edges =[]
                        row, col = aug_edge_index_2
                        for i in range(aug_edge_index_2.size(1)):
                            x_coords = [coord[row[i],0], coord[col[i],0],None]
                            x_edges += x_coords
                            y_coords = [coord[row[i],1], coord[col[i],1], None]
                            y_edges += y_coords
                            z_coords = [coord[row[i],2], coord[col[i],2],None]
                            z_edges += z_coords
                                
                        trace_aug_edges_2 = go.Scatter3d(   x=x_edges,
                                                            y=y_edges,
                                                            z=z_edges,
                                                            mode = 'lines',
                                                            line = dict(color='green', width=0.5),
                                                            hoverinfo = 'none')
                        # Include the trace edges  to plot and create a figure  
                        data.append(trace_aug_edges_2)
        else:
            x_edges = []
            y_edges = []
            z_edges = []
            row, col = edge_index
            for i in range(edge_index.size(1)):
                x_coords = [coord[row[i],0], coord[col[i],0], None]
                x_edges += x_coords
                y_coords = [coord[row[i],1], coord[col[i],1], None]
                y_edges += y_coords
                z_coords = [coord[row[i],2], coord[col[i],2], None]
                z_edges += z_coords
                    
            trace_edges = go.Scatter3d( x=x_edges,
                                        y=y_edges,
                                        z=z_edges,
                                        mode = 'lines',
                                        line = dict(color='red', width=5),
                                        hoverinfo = 'none')
            
            # Include the trace edges  to plot and create a figure  
            data.append(trace_edges)
    
    if triads is not None:
        if len(triads.size()) > 2:
            triads = roma.rotmat_to_rotvec(triads)
        triads = triads.to("cpu")
        u_nodes = triads[:,0]
        v_nodes = triads[:,1]
        w_nodes = triads[:,2]
        trace_triads = go.Cone( x=x_nodes,
                                y=y_nodes,
                                z=z_nodes,
                                u=u_nodes,
                                v=v_nodes,
                                w=w_nodes,
                                # colorscale='Blues',
                                sizemode="scaled",
                                # showscale=False,
                                sizeref=2)
        data.append(trace_triads)
        
    else:
        data.append(trace_nodes)
        
    # we need to set the axis for the plot 
    axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title='')
    # also need to create the layout for our plot
    layout = go.Layout( showlegend=False,
                        scene=dict(xaxis=dict(axis),
                                yaxis=dict(axis),
                                zaxis=dict(axis),),
                        hovermode='closest')
    fig = go.Figure(data=data, layout=layout)
    if coord.size(0) > 0:
        X_range = torch.abs(coord[:, 0].max() - coord[:, 0].min()).item()/100
        Y_range = torch.abs(coord[:, 1].max() - coord[:, 1].min()).item()/100
        Z_range = torch.abs(coord[:, 2].max() - coord[:, 2].min()).item()/100
        # Ensure non-zero ranges
        X_range = max(X_range, 1e-4)
        Y_range = max(Y_range, 1e-4)
        Z_range = max(Z_range, 1e-4)
    else:
        # Default ranges for empty data
        X_range = Y_range = Z_range = 1.0
    fig.update_layout(scene_aspectmode='manual',
                      scene_aspectratio=dict(x=X_range, y=Y_range, z=Z_range))
    return fig

def plot_nucleotides(scaf_coords, stap_coords, SS_coords,
                     scaf_edge_index, stap_edge_index):


    scaf_coords = scaf_coords.to("cpu")
    stap_coords = stap_coords.to("cpu")
    SS_coords = SS_coords.to("cpu")
    
    # number of nodes
    num_bps = scaf_coords.size(0)  
    data = []    
    
    # we need to seperate the X,Y,Z coordinates for Plotly
    x_scaf_nodes = scaf_coords[:,0]      # x-coordinates of nodes
    y_scaf_nodes = scaf_coords[:,1]      # y-coordinates
    z_scaf_nodes = scaf_coords[:,2]      # z-coordinates
    trace_scaf_nodes = go.Scatter3d(x = x_scaf_nodes,
                                    y = y_scaf_nodes,
                                    z = z_scaf_nodes,
                                    mode = 'markers',
                                    marker = dict(  symbol='circle',
                                                    size = 4,
                                                    color = 'red', 
                                                    colorscale = ['lightgreen','magenta'], # either green or mageneta
                                                    line = dict(color='black', width=0.5)),
                                                    hoverinfo='text')
    data.append(trace_scaf_nodes)
    
    # plot staple nodes
    x_stap_nodes = stap_coords[:,0]      # x-coordinates of nodes
    y_stap_nodes = stap_coords[:,1]      # y-coordinates
    z_stap_nodes = stap_coords[:,2]      # z-coordinates
    trace_stap_nodes = go.Scatter3d(x = x_stap_nodes,
                                    y = y_stap_nodes,
                                    z = z_stap_nodes,
                                    mode = 'markers',
                                    marker = dict(  symbol='circle',
                                                    size = 4,
                                                    color = 'blue', 
                                                    colorscale = ['lightgreen','magenta'], 
                                                    line = dict(color='green', width=0.25)),
                                    hoverinfo='text')
    data.append(trace_stap_nodes)
    
    # plot unbounded bases
    x_unbound_nodes = SS_coords[:,0]      # x-coordinates of nodes
    y_unbound_nodes = SS_coords[:,1]      # y-coordinates
    z_unbound_nodes = SS_coords[:,2]      # z-coordinates
    trace_SS_nodes = go.Scatter3d(  x = x_unbound_nodes,
                                    y = y_unbound_nodes,
                                    z = z_unbound_nodes,
                                    mode = 'markers',
                                    marker = dict(  symbol='circle',
                                                    size = 4,
                                                    color = 'green', 
                                                    colorscale = ['lightgreen','magenta'], 
                                                    line = dict(color='green', width=0.25)),
                                    hoverinfo='text')
    data.append(trace_SS_nodes)
    
    # scaffold edges plot
    scaf_edge_index = scaf_edge_index.to("cpu")
    x_scaf_edges =[]
    y_scaf_edges =[]
    z_scaf_edges =[]
    row, col = scaf_edge_index
    for i in range(scaf_edge_index.size(1)):
        x_scaf_coords = [scaf_coords[row[i],0], scaf_coords[col[i],0],None]
        x_scaf_edges += x_scaf_coords
        y_scaf_coords = [scaf_coords[row[i],1], scaf_coords[col[i],1], None]
        y_scaf_edges += y_scaf_coords
        z_scaf_coords = [scaf_coords[row[i],2], scaf_coords[col[i],2],None]
        z_scaf_edges += z_scaf_coords
                        
    trace_scaf_edges = go.Scatter3d(x=x_scaf_edges,
                                    y=y_scaf_edges,
                                    z=z_scaf_edges,
                                    mode = 'lines',
                                    line = dict(color='red', width=2.0),
                                    hoverinfo = 'none')
    data.append(trace_scaf_edges)
    
    # staple edges plot
    stap_edge_index = stap_edge_index.to("cpu")
    x_stap_edges =[]
    y_stap_edges =[]
    z_stap_edges =[]
    row, col = stap_edge_index
    for i in range(stap_edge_index.size(1)):
        x_stap_coords = [stap_coords[row[i],0], stap_coords[col[i],0], None]
        x_stap_edges += x_stap_coords
        y_stap_coords = [stap_coords[row[i],1], stap_coords[col[i],1], None]
        y_stap_edges += y_stap_coords
        z_stap_coords = [stap_coords[row[i],2], stap_coords[col[i],2], None]
        z_stap_edges += z_stap_coords
            
    trace_stap_edges = go.Scatter3d(x=x_stap_edges,
                                    y=y_stap_edges,
                                    z=z_stap_edges,
                                    mode = 'lines',
                                    line = dict(color='blue', width=2.0),
                                    hoverinfo = 'none')
    data.append(trace_stap_edges)
        
    # we need to set the axis for the plot 
    axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title='')
    
    # also need to create the layout for our plot
    layout = go.Layout( showlegend=False,
                        scene=dict( xaxis=dict(axis),
                                    yaxis=dict(axis),
                                    zaxis=dict(axis),),
                                    hovermode='closest')
    
    fig = go.Figure(data=data, layout=layout)
    
    if scaf_coords.size(0) > 0:
        X_range = torch.abs(scaf_coords[:, 0].max() - scaf_coords[:, 0].min()).item()/100
        Y_range = torch.abs(scaf_coords[:, 1].max() - scaf_coords[:, 1].min()).item()/100
        Z_range = torch.abs(scaf_coords[:, 2].max() - scaf_coords[:, 2].min()).item()/100
        # Ensure non-zero ranges
        X_range = max(X_range, 1e-4)
        Y_range = max(Y_range, 1e-4)
        Z_range = max(Z_range, 1e-4)
    else:
        # Default ranges for empty data
        X_range = Y_range = Z_range = 1.0
    fig.update_layout(scene_aspectmode='manual',
                      scene_aspectratio=dict(x=X_range, y=Y_range, z=Z_range))
    return fig


def plot_bases(coords, scaf_coords=None, stap_coords=None):
    coords = coords.to("cpu")
    coords_all = coords
    num_nodes = coords.size(0)   
    data = []    
    # we need to seperate the X,Y,Z coordinates for Plotly
    x_nodes = coords[:,0]      # x-coordinates of nodes
    y_nodes = coords[:,1]      # y-coordinates
    z_nodes = coords[:,2]      # z-coordinates
    node_labels = torch.arange(0, num_nodes)
    trace_bps = go.Scatter3d(   x = x_nodes,
                                y = y_nodes,
                                z = z_nodes,
                                mode = 'markers',
                                marker = dict(  symbol='circle',
                                                size = 5,
                                                color = 'red', 
                                                colorscale = ['lightgreen','magenta'], # either green or mageneta
                                                line = dict(color='black', width=0.5)),
                                text = node_labels,
                                hoverinfo='text')
    data.append(trace_bps)
    
    if scaf_coords is not None:
        scaf_coords = scaf_coords.to("cpu")
        x_nodes = scaf_coords[:,0]      # x-coordinates of nodes
        y_nodes = scaf_coords[:,1]      # y-coordinates
        z_nodes = scaf_coords[:,2]      # z-coordinates
        trace_scaf = go.Scatter3d(  x = x_nodes,
                                    y = y_nodes,
                                    z = z_nodes,
                                    mode = 'markers',
                                    marker = dict(  symbol='circle',
                                                    size = 5,
                                                    color = 'blue', 
                                                    colorscale = ['lightgreen','magenta'], # either green or mageneta
                                                    line = dict(color='black', width=0.5)),
                                    hoverinfo='text'    )
        data.append(trace_scaf)
        coords_all = torch.vstack((coords_all, scaf_coords))
        
    if stap_coords is not None:
        stap_coords = stap_coords.to("cpu")
        x_nodes = stap_coords[:,0]      # x-coordinates of nodes
        y_nodes = stap_coords[:,1]      # y-coordinates
        z_nodes = stap_coords[:,2]      # z-coordinates
        trace_stap = go.Scatter3d(  x = x_nodes,
                                    y = y_nodes,
                                    z = z_nodes,
                                    mode = 'markers',
                                    marker = dict(  symbol='circle',
                                                    size = 5,
                                                    color = 'green', 
                                                    colorscale = ['lightgreen','magenta'], # either green or mageneta
                                                    line = dict(color='black', width=0.5)),
                                    hoverinfo='text'    )
        data.append(trace_stap)  
        coords_all = torch.vstack((coords_all, stap_coords))  
     
    # we need to set the axis for the plot 
    axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title='')
    # also need to create the layout for our plot
    layout = go.Layout( showlegend=False,
                        scene=dict( xaxis=dict(axis),
                                    yaxis=dict(axis),
                                    zaxis=dict(axis),),
                                    hovermode='closest')
    fig = go.Figure(data=data, layout=layout)
    if coords_all.size(0) > 0:
        X_range = torch.abs(coords_all[:, 0].max() - coords_all[:, 0].min()).item()/100
        Y_range = torch.abs(coords_all[:, 1].max() - coords_all[:, 1].min()).item()/100
        Z_range = torch.abs(coords_all[:, 2].max() - coords_all[:, 2].min()).item()/100
        # Ensure non-zero ranges
        X_range = max(X_range, 1e-4)
        Y_range = max(Y_range, 1e-4)
        Z_range = max(Z_range, 1e-4)
    else:
        # Default ranges for empty data
        X_range = Y_range = Z_range = 1.0
    fig.update_layout(scene_aspectmode='manual',
                      scene_aspectratio=dict(x=X_range, y=Y_range, z=Z_range))
    return fig


def plot_dna_coloring_edges(coord, RBedge_index=None, SSedge_index=None, COedge_index=None, node_labels=None):
    # show_edge_type = 'full', '0', '1', 'aug', 'phys'
    coord = coord.to("cpu")
    # number of nodes
    n_nodes = coord.size(0)   
    if node_labels is not None:
        try:
            node_labels = node_labels.to("cpu")
        except:
            node_labels = node_labels
    else:
        node_labels = torch.arange(0, n_nodes)

    data = []    
    # we need to seperate the X,Y,Z coordinates for Plotly
    x_nodes = coord[:,0]      # x-coordinates of nodes
    y_nodes = coord[:,1]      # y-coordinates
    z_nodes = coord[:,2]      # z-coordinates
    
    #create a trace for the nodes
    if node_labels is None:
        node_labels = torch.arange(0, n_nodes)
    trace_nodes = go.Scatter3d( x = x_nodes,
                                y = y_nodes,
                                z = z_nodes,
                                mode = 'markers',
                                marker = dict(  symbol='circle',
                                                size = 5,
                                                color = 'blue', 
                                                colorscale = ['lightgreen','magenta'], # either green or mageneta
                                                line = dict(color='black', width=0.5)),
                                text = node_labels,
                                hoverinfo='text')
    data.append(trace_nodes)
    
    # we  need to create lists that contain the starting and ending coordinates of each edge.
    if RBedge_index is not None:
        x_edges =[]
        y_edges =[]
        z_edges =[]
        row, col = RBedge_index
        for i in range(RBedge_index.size(1)):
            x_coords = [coord[row[i],0], coord[col[i],0],None]
            x_edges += x_coords
            y_coords = [coord[row[i],1], coord[col[i],1], None]
            y_edges += y_coords
            z_coords = [coord[row[i],2], coord[col[i],2],None]
            z_edges += z_coords
                
        trace_RBedges = go.Scatter3d(x=x_edges,
                                    y=y_edges,
                                    z=z_edges,
                                    mode = 'lines',
                                    line = dict(color='blue', width=5),
                                    hoverinfo = 'none')
        # Include the trace edges  to plot and create a figure  
        data.append(trace_RBedges)
    
    if SSedge_index is not None:
        x_edges =[]
        y_edges =[]
        z_edges =[]
        row, col = SSedge_index
        for i in range(SSedge_index.size(1)):
            x_coords = [coord[row[i],0], coord[col[i],0],None]
            x_edges += x_coords
            y_coords = [coord[row[i],1], coord[col[i],1], None]
            y_edges += y_coords
            z_coords = [coord[row[i],2], coord[col[i],2],None]
            z_edges += z_coords
                
        trace_SSedges = go.Scatter3d(x=x_edges,
                                     y=y_edges,
                                     z=z_edges,
                                     mode = 'lines',
                                     line = dict(color='green', width=5),
                                     hoverinfo = 'none')
        # Include the trace edges  to plot and create a figure  
        data.append(trace_SSedges)
        
    if COedge_index is not None:
        x_edges =[]
        y_edges =[]
        z_edges =[]
        row, col = COedge_index
        for i in range(COedge_index.size(1)):
            x_coords = [coord[row[i],0], coord[col[i],0],None]
            x_edges += x_coords
            y_coords = [coord[row[i],1], coord[col[i],1], None]
            y_edges += y_coords
            z_coords = [coord[row[i],2], coord[col[i],2],None]
            z_edges += z_coords
                
        trace_COedges = go.Scatter3d(x=x_edges,
                                     y=y_edges,
                                     z=z_edges,
                                     mode = 'lines',
                                     line = dict(color='red', width=5),
                                     hoverinfo = 'none')
        # Include the trace edges  to plot and create a figure  
        data.append(trace_COedges)
    
    
        
    # we need to set the axis for the plot 
    axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title='')
    # also need to create the layout for our plot
    layout = go.Layout( showlegend=False,
                        scene=dict(xaxis=dict(axis),
                                yaxis=dict(axis),
                                zaxis=dict(axis),),
                        hovermode='closest')
    fig = go.Figure(data=data, layout=layout)
    if coord.size(0) > 0:
        X_range = torch.abs(coord[:, 0].max() - coord[:, 0].min()).item()/100
        Y_range = torch.abs(coord[:, 1].max() - coord[:, 1].min()).item()/100
        Z_range = torch.abs(coord[:, 2].max() - coord[:, 2].min()).item()/100
        # Ensure non-zero ranges
        X_range = max(X_range, 1e-4)
        Y_range = max(Y_range, 1e-4)
        Z_range = max(Z_range, 1e-4)
    else:
        # Default ranges for empty data
        X_range = Y_range = Z_range = 1.0
    fig.update_layout(scene_aspectmode='manual',
                      scene_aspectratio=dict(x=X_range, y=Y_range, z=Z_range))
    return fig