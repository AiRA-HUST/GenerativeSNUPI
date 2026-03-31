

# generate pdb file for visualization
def gen_pdb(coords, bp_Hindex, file_name):
    num_bps = coords.size(0)
    with open("pdb/" + file_name + ".pdb", "w") as f:
        f.write("MODEL        1\n")
        for i in range(num_bps):
            b = i//100
            h = bp_Hindex[i]
            x = 10 * coords[i,0] # convert to Angstrong
            y = 10 * coords[i,1]
            z = 10 * coords[i,2]
            se = 1
            f.write("ATOM  {:5d} B{:<3d} H{:<3d} {:4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}\n".format(i, b, h, b,  x, y, z, se))
            
        f.write("TER\nENDMDL\n")
        
import torch        
def ins_inner_nodes(coords, src_ID, dst_ID, num_ins=None):
    
    src_coords = coords[src_ID: src_ID+1,:]
    dst_coords = coords[dst_ID: dst_ID+1,:]
    relvec = ( dst_coords - src_coords )
    length = relvec.norm(dim=-1, keepdim=False)
    if num_ins is None:
        num_ins = int(torch.round(length/0.34) + 1)
    
    pts_idx = torch.linspace(0., 1.0, steps=num_ins, device=coords.device)
    inner_pts = pts_idx[1:-1,None] * relvec + src_coords
    coords = torch.vstack((coords, inner_pts))
    
    return coords


def ins_extra_nodes(coords, src_ID, dst_ID, num_ins):
    
    src_coords = coords[src_ID: src_ID+1,:]
    dst_coords = coords[dst_ID: dst_ID+1,:]
    relvec = ( dst_coords - src_coords )
    length = relvec.norm(dim=-1, keepdim=False)
    univec = relvec/length

    pts_idx = torch.arange(num_ins, device=coords.device) + 2
    extra_pts = 0.34 * pts_idx[:,None] * univec + src_coords
    coords = torch.vstack((coords, extra_pts))
    
    return coords

def del_nodes(coords, remove_ID):
    node_mask = torch.ones(coords.size(0), dtype=torch.bool, device=coords.device)
    node_mask[remove_ID] = False
    coords = coords[node_mask,:]
    return coords

def swap_nodes(tensor, row1, row2):
    """
    Swaps two rows in a given PyTorch tensor.

    Parameters:
    tensor (torch.Tensor): The input tensor.
    row1 (int): The index of the first row to swap.
    row2 (int): The index of the second row to swap.

    Returns:
    torch.Tensor: A new tensor with the rows swapped.
    """
    # Create a copy of the tensor to avoid modifying the original tensor
    tensor_copy = tensor.clone()

    # Swap the rows
    tensor_copy[row1], tensor_copy[row2] = tensor_copy[row2].clone(), tensor_copy[row1].clone()

    return tensor_copy

def move_node_to_last(tensor, row_idx):
    """
    Moves a specific row in a given PyTorch tensor to the last row.

    Parameters:
    tensor (torch.Tensor): The input tensor.
    row_idx (int): The index of the row to move.

    Returns:
    torch.Tensor: A new tensor with the specified row moved to the last row.
    """
    # Ensure row_idx is within the valid range
    if row_idx < 0 or row_idx >= tensor.size(0):
        raise IndexError("row_idx is out of bounds for the tensor.")

    # Extract the row to move and the remaining tensor
    row_to_move = tensor[row_idx].clone()
    remaining_tensor = torch.cat((tensor[:row_idx], tensor[row_idx+1:]), dim=0)

    # Concatenate the remaining tensor with the row moved to the last position
    new_tensor = torch.cat((remaining_tensor, row_to_move.unsqueeze(0)), dim=0)

    return new_tensor

def gen_xyz(coords, file_name):
    num_bps = coords.size(0)
    outFile = open(file_name + ".xyz", 'w')
    outFile.write(str(num_bps) + "\n")
    outFile.write("Geometry in xyz format\n")
    
    for i in range(num_bps):
        atom = "C"
        x = coords[i,0] # convert to Angstrong
        y = coords[i,1]
        z = coords[i,2]
        outFile.write("%s\t %d\t %d\t %d\t \n" %(atom, x, y, z))
        
    outFile.close()

def calculate_edge_lengths(points: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """
    Calculate the edge lengths for a 3D point cloud.

    Args:
        points (torch.Tensor): A tensor of shape [num_nodes, 3] representing the 3D points.
        edge_index (torch.Tensor): A tensor of shape [2, num_edges] representing the edges 
                                   as pairs of indices into the points tensor.

    Returns:
        torch.Tensor: A tensor of shape [num_edges] containing the lengths of the edges.
    """
    # Get the indices of the source and destination nodes
    src, dst = edge_index

    # Compute the differences between connected nodes
    edge_vectors = points[dst] - points[src]

    # Compute the Euclidean length of each edge
    edge_lengths = torch.norm(edge_vectors, dim=1)

    return edge_lengths

# Visualize the distribution of edge lengths
import matplotlib.pyplot as plt
def visualize_edge_length_distribution(edge_lengths: torch.Tensor, bins: int = 10):
    """
    Visualize the distribution of edge lengths using a histogram.

    Args:
        edge_lengths (torch.Tensor): A tensor containing the lengths of edges.
        bins (int): Number of bins for the histogram.
    """
    # Convert edge_lengths to a numpy array for compatibility with Matplotlib
    edge_lengths_np = edge_lengths.cpu().numpy()

    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.hist(edge_lengths_np, bins=bins, color='blue', alpha=0.7, edgecolor='black')
    plt.title('Distribution of Edge Lengths')
    plt.xlabel('Edge Length')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()