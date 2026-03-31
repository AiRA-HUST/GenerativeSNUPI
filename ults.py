import torch
import scipy.io as scio
from torch_geometric.data import Data


def mat_to_tensor_graph(file_name: str, device: torch.device) -> Data:
    """
    Convert mat file design to a graph representation that can be fed to the model.
    
    Args:
        file_name (str): Path to the .mat file
        device (torch.device): Device to load tensors onto
        
    Returns:
        Data: PyTorch Geometric Data object containing graph representation
    """
    try:
        data = scio.loadmat(file_name)
    except Exception as e:
        raise ValueError(f"Failed to load mat file {file_name}: {str(e)}")
    
    # Extract data from mat file
    conn = data["elem_conn"]
    icoord = data["init_coord"]
    prop = data["prop"]
    
    # Convert to graph data
    x = torch.tensor(icoord, dtype=torch.float, device=device)
    edge_index = torch.tensor(conn, dtype=torch.long, device=device)
    edge_attr = torch.tensor(prop, dtype=torch.float, device=device)
    
    # Handle coordinate data with or without PL
    if x[:, -1].amax() > 0:
        icoord_wo_pl = data["init_coord_woPL"]
        x_wo_pl = torch.tensor(icoord_wo_pl, dtype=torch.float, device=device)
    else:
        x_wo_pl = x[:, 0:3]
    
    # Create graph data with optional final coordinates
    if "finl_coord" in data:
        fcoord = data["finl_coord"]
        y = torch.tensor(fcoord, dtype=torch.float, device=device)
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, cadnano_coord=x_wo_pl)
    else:
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, cadnano_coord=x_wo_pl)
    
    return graph


def remap_tensor_values(tensor: torch.Tensor, old_values: torch.Tensor, new_values: torch.Tensor) -> torch.Tensor:
    """
    Remap tensor values according to old_values -> new_values mapping.
    
    Args:
        tensor (torch.Tensor): Input tensor to remap
        old_values (torch.Tensor): Original values to be replaced
        new_values (torch.Tensor): New values to replace with
        
    Returns:
        torch.Tensor: Tensor with remapped values
    """
    old_values = old_values.type(torch.long).to(tensor.device)
    new_values = new_values.type(torch.long).to(tensor.device)
    
    # Create mapping array
    mapping = torch.empty(tensor.max() + 1, dtype=new_values.dtype, device=new_values.device)
    mapping[old_values] = new_values
    
    return mapping[tensor]


def gradient_flow(X_i: torch.Tensor, Y_j: torch.Tensor, loss_fn, lr: float = 0.01) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Flows along the gradient of the cost function using a simple Euler scheme.
    
    Args:
        X_i (torch.Tensor): Source point cloud
        Y_j (torch.Tensor): Target point cloud
        loss_fn (callable): Loss function that takes (x_i, y_j) and returns a scalar
        lr (float, default=0.01): Learning rate (time step)
        
    Returns:
        tuple[torch.Tensor, torch.Tensor]: Updated source and target point clouds
    """
    # Parameters for the gradient descent
    n_steps = int(10 / lr) + 1
    
    # Make sure that we won't modify the reference samples
    x_i, y_j = X_i.clone(), Y_j.clone()
    
    # Perform gradient descent on Loss(α, β) w.r.t. the positions x_i
    x_i.requires_grad = True
    
    for i in range(n_steps):
        # Compute cost and gradient
        loss = loss_fn(x_i, y_j)
        print(f"Step {i+1}/{n_steps}, Loss: {loss:.4f}")
        
        [gradient] = torch.autograd.grad(loss, [x_i])
        
        # In-place modification of the tensor's values
        x_i.data -= lr * len(x_i) * gradient
    
    return x_i.detach().clone(), y_j.detach().clone()

    