from typing import Optional, Dict
import torch
import warnings
import torch.nn as nn
import dgl
from dgl import DGLGraph
from torch import Tensor

from se3_transformer.basis import get_basis, update_basis_with_fused
from se3_transformer.layers.attention import AttentionBlockSE3
from se3_transformer.layers.convolution import ConvSE3, ConvSE3FuseLevel
from se3_transformer.layers.norm import NormSE3
from se3_transformer.runtime.utils import str2bool
from se3_transformer.fiber import Fiber
from torch_geometric.nn import knn_graph

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_POSITIONAL_ENCODING_BASE = 10000
MIN_DISTANCE_CLAMP = 1e-3
GUMBEL_NOISE_EPSILON = 1e-20


def create_model(num_layers,
                 time_emb_dim,
                 fiber_hidden,
                 num_heads,
                 channels_div,
                 num_nearest_neighbors,
                 num_random_neighbors,
                 temp,
                 shape_condition,
                 tensor_cores,
                 low_memory,):
    """
    Factory function to create a DnaTransformer model with specified parameters.
    
    Args:
        num_layers: Number of transformer layers
        time_emb_dim: Dimension of time embedding
        fiber_hidden: Hidden fiber configuration
        num_heads: Number of attention heads
        channels_div: Channel division factor
        num_nearest_neighbors: Number of nearest neighbors for graph construction
        num_random_neighbors: Number of random neighbors for graph construction
        temp: Temperature parameter for Gumbel sampling
        shape_condition: Whether to use shape conditioning
        tensor_cores: Whether to use tensor cores optimization
        low_memory: Whether to use low memory mode
    
    Returns:
        DnaTransformer: Configured transformer model
    """
    return DnaTransformer(  
        num_layers=num_layers, 
        fiber_in=Fiber([(1, 1)]),
        fiber_hidden=fiber_hidden,
        fiber_out=Fiber([(1, 1)]),
        fiber_edge=Fiber({}),
        time_emb_dim=time_emb_dim, 
        num_nearest_neighbors=num_nearest_neighbors,
        num_random_neighbors=num_random_neighbors,
        temp=temp,
        shape_condition=shape_condition,
        num_heads=num_heads, 
        channels_div=channels_div,
        tensor_cores=tensor_cores,
        low_memory=low_memory,)


class User_Sequential(nn.Sequential):
    """
    Sequential module with arbitrary forward args and kwargs.
    Used to pass graph, basis and edge features through the network.
    """
    
    def forward(self, input, *args, **kwargs):
        """
        Forward pass that propagates additional arguments through all modules.
        
        Args:
            input: Primary input tensor
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        
        Returns:
            Tensor: Output after passing through all modules
        """
        for module in self:
            input = module(input, *args, **kwargs)
        return input


def residual_update(node_feats: Dict[str, Tensor], res_node_feats: Dict[str, Tensor], fiber: Fiber):
    """
    Update node features with residual connections.
    
    Args:
        node_feats: Current node features dictionary
        res_node_feats: Residual node features to add
        fiber: Fiber configuration specifying degrees and channels
    
    Returns:
        Dict[str, Tensor]: Updated node features with residual connections
    """
    for i, degree in enumerate(fiber.degrees):
        node_feats[str(degree)] = res_node_feats[str(degree)] + node_feats[str(degree)][:,:fiber.channels[i],:]
    return node_feats


def get_populated_edge_features(relative_pos: Tensor, edge_features: Optional[Dict[str, Tensor]] = None):
    """
    Add relative positions to existing edge features.
    
    Args:
        relative_pos: Relative position vectors between nodes
        edge_features: Optional existing edge features dictionary
    
    Returns:
        Dict[str, Tensor]: Edge features with relative positions added
    """
    edge_features = edge_features.copy() if edge_features else {}
    
    # Compute distance from relative positions
    r = relative_pos.norm(dim=-1, keepdim=True)
    
    # Add distance to edge features
    if '0' in edge_features:
        edge_features['0'] = torch.cat([edge_features['0'][..., None], r[..., None]], dim=1)
    else:
        edge_features['0'] = r[..., None]
    
    return edge_features


def time_encode_node_features(node_feats: Dict[str, Tensor], time_emb: Tensor):
    """
    Add time embedding to node features.
    
    Args:
        node_feats: Current node features dictionary
        time_emb: Time embedding tensor
    
    Returns:
        Dict[str, Tensor]: Node features with time embedding concatenated
    """
    node_feats = node_feats.copy() if node_feats else {}
    node_feats['0'] = torch.cat([node_feats['0'], time_emb[:, :, None]], dim=1)
    return node_feats


def reduce_graph(seg_edge_index, seg_coords):
    """
    Create a reduced graph from edge indices and coordinates.
    
    Args:
        seg_edge_index: Edge indices for the segment
        seg_coords: Coordinates of nodes in the segment
    
    Returns:
        DGLGraph: Graph with relative positions as edge data
    """
    # Extract source and destination nodes
    src, dst = seg_edge_index
    
    # Create DGL graph and add relative positions
    seg_graph = dgl.graph((src, dst)).to(seg_edge_index.device)
    seg_graph.edata['rel_pos'] = seg_coords[src] - seg_coords[dst]
    
    return seg_graph


class DnaTransformer(nn.Module):
    """
    Transformer model for DNA structure prediction using SE(3) equivariant layers.
    
    This model processes 3D molecular coordinates and generates predictions while
    maintaining rotational and translational equivariance.
    """
    
    def __init__(self,
        num_layers: int,
        fiber_in: Fiber,
        fiber_hidden: Fiber,
        fiber_out: Fiber,
        fiber_edge: Fiber = Fiber({}),
        time_emb_dim: int = 128,
        num_heads: int = 8,
        channels_div: int = 2,
        num_nearest_neighbors: int = 20,
        num_random_neighbors: int = 10,
        temp: float = 1.0,
        shape_condition: bool = False,
        norm: bool = True,
        use_layer_norm: bool = True,
        tensor_cores: bool = False,
        low_memory: bool = False,
        **kwargs):
        """
        Initialize the DnaTransformer.
        
        Args:
            num_layers: Number of attention layers
            fiber_in: Input fiber description
            fiber_hidden: Hidden fiber description
            fiber_out: Output fiber description
            fiber_edge: Input edge fiber description
            time_emb_dim: Dimension of time embedding
            num_heads: Number of attention heads
            channels_div: Channels division before feeding to attention layer
            num_nearest_neighbors: Number of nearest neighbors for graph construction
            num_random_neighbors: Number of random neighbors for graph construction
            temp: Temperature parameter for Gumbel sampling
            shape_condition: Whether to use shape conditioning
            norm: Apply normalization layer after each attention block
            use_layer_norm: Apply layer normalization between MLP layers
            tensor_cores: True if using Tensor Cores (affects fully fused convs and padded bases)
            low_memory: If True, will use slower ops that use less memory
        """
        super().__init__()
        
        # Store configuration parameters
        self.num_layers = num_layers
        self.fiber_edge = fiber_edge
        self.fiber_hidden = fiber_hidden
        self.shape_condition = shape_condition
        self.num_heads = num_heads
        self.channels_div = channels_div
        self.time_emb_dim = time_emb_dim
        self.max_degree = max(*fiber_in.degrees, *fiber_hidden.degrees, *fiber_out.degrees)
        self.tensor_cores = tensor_cores
        self.low_memory = low_memory
        self.num_nearest_neighbors = num_nearest_neighbors
        self.num_random_neighbors = num_random_neighbors
        self.temp = temp
        
        # Configure fusion level based on memory and tensor core settings
        if low_memory:
            self.fuse_level = ConvSE3FuseLevel.NONE
        else:
            # Fully fused convolutions when using Tensor Cores (and not low memory mode)
            self.fuse_level = ConvSE3FuseLevel.FULL if tensor_cores else ConvSE3FuseLevel.PARTIAL

        # Modify input fiber for shape conditioning
        if self.shape_condition:
            fiber_in = fiber_in.__add__(Fiber([(1, 1)]))

        # Build the sequence of graph processing modules
        graph_modules = []
        fiber_edge = Fiber([(0, time_emb_dim)])  # Add time embedding to edge features
        
        # Add attention blocks with optional normalization
        for i in range(num_layers):
            graph_modules.append(AttentionBlockSE3(
                fiber_in=fiber_in,
                fiber_out=fiber_hidden,
                fiber_edge=fiber_edge,
                num_heads=num_heads,
                channels_div=channels_div,
                use_layer_norm=use_layer_norm,
                max_degree=self.max_degree,
                fuse_level=self.fuse_level,
                low_memory=low_memory))
            
            if norm:
                graph_modules.append(NormSE3(fiber_hidden))
            
            fiber_in = fiber_hidden
        
        # Add final convolution layer
        graph_modules.append(ConvSE3(
            fiber_in=fiber_in,
            fiber_out=fiber_out,
            fiber_edge=fiber_edge,
            self_interaction=True,
            use_layer_norm=use_layer_norm,
            max_degree=self.max_degree,
            fuse_level=self.fuse_level,
            low_memory=low_memory))
            
        self.graph_modules = User_Sequential(*graph_modules)
        
        
    def pos_encoding(self, t, channels):
        """
        Generate positional encoding for time embeddings.
        
        Args:
            t: Time tensor
            channels: Number of channels for encoding
        
        Returns:
            Tensor: Positional encoding tensor
        """
        # Create inverse frequency components
        inv_freq = 1.0 / (DEFAULT_POSITIONAL_ENCODING_BASE ** (
            torch.arange(0, channels, 2, device=t.device).float() / channels))
        
        # Generate sine and cosine components
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        
        # Concatenate components
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc
    
    def forward(self, 
                node_feats: Dict[str, Tensor],
                t: Tensor,
                batch: Tensor,
                shape_node_feats: Optional[Tensor] = None,
                basis: Optional[Dict[str, Tensor]] = None,):
        """
        Forward pass of the DnaTransformer.
        
        Args:
            node_feats: Dictionary of node features by degree
            t: Time tensor for diffusion process
            batch: Batch indices for each node
            shape_node_feats: Optional shape conditioning features
            basis: Optional precomputed basis functions
        
        Returns:
            Tensor: Predicted node coordinates
        """
        # Extract coordinates from node features
        coords = node_feats['1'].squeeze()
        
        # Generate graph connectivity
        g = graph_gen(coords, batch, self.num_nearest_neighbors, self.num_random_neighbors, self.temp)
        src, _ = g.edges()
        
        # Prepare time embedding
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_emb_dim) 
        t = t[batch[src], :]
        
        # Handle shape conditioning
        if self.shape_condition:
            if shape_node_feats is None:
                shape_node_feats = torch.zeros_like(node_feats['1'], device=node_feats['1'].device)
            else:
                shape_node_feats = shape_node_feats[:, None, :]
            node_feats['1'] = torch.cat([node_feats['1'], shape_node_feats], dim=1)
            
        # Compute basis functions if not provided
        basis = basis or get_basis(
            g.edata['rel_pos'], 
            max_degree=self.max_degree, 
            compute_gradients=False,
            use_pad_trick=self.tensor_cores and not self.low_memory,
            amp=torch.is_autocast_enabled())

        # Add fused bases for optimization
        basis = update_basis_with_fused(
            basis, 
            self.max_degree, 
            use_pad_trick=self.tensor_cores and not self.low_memory,
            fully_fused=self.fuse_level == ConvSE3FuseLevel.FULL)

        # Prepare edge features with time embedding
        edge_feats = {'0': t}
        edge_feats = get_populated_edge_features(g.edata['rel_pos'], edge_feats)
        
        # Process through graph modules
        node_feats = self.graph_modules(node_feats, edge_feats, graph=g, basis=basis)
        
        return node_feats['1'].squeeze()


# =============================================================================
# Graph Generation Functions
# =============================================================================

def gumbel_sampling(coords, temp):
    """
    Generate sampling probabilities using Gumbel distribution based on distances.
    
    This function creates a probability distribution that favors closer nodes
    while introducing randomness through Gumbel noise.
    
    Args:
        coords: Node coordinates tensor
        temp: Temperature parameter controlling randomness
    
    Returns:
        Tensor: Sampling propensities for node pairs
    """
    temp = torch.tensor(temp, device=coords.device)
    
    # Compute pairwise distances
    pairwise_dist = torch.cdist(coords, coords, p=2.)
    pairwise_dist = torch.clamp(pairwise_dist, min=MIN_DISTANCE_CLAMP)
    
    # Create inverse cubic distance scaling
    cubic_dist_inv = temp / (pairwise_dist.pow(3))
    
    # Add Gumbel noise for stochastic sampling
    U = torch.rand_like(pairwise_dist, device=coords.device)
    propensity = cubic_dist_inv - torch.log(-torch.log(U + GUMBEL_NOISE_EPSILON) + GUMBEL_NOISE_EPSILON)
    
    # Zero out diagonal (self-connections)
    return propensity.fill_diagonal_(0.0)


def graph_gen(coords, batch, num_nearest=20, num_rand=10, temp=1.0):
    """
    Generate graph connectivity combining nearest neighbors and random sampling.
    
    This function creates a graph that includes both k-nearest neighbors for
    local connectivity and Gumbel-sampled random edges for global connectivity.
    
    Args:
        coords: Node coordinates
        batch: Batch indices for each node
        num_nearest: Number of nearest neighbors to connect
        num_rand: Number of random neighbors to sample
        temp: Temperature for Gumbel sampling
    
    Returns:
        DGLGraph: Graph with edge connectivity and relative positions
    """
    device = coords.device
    batch = batch.to(device)
    num_nodes = coords.size(0)
    
    # Split coordinates by batch
    _, segs = torch.unique(batch, sorted=True, return_counts=True)
    offset = torch.cat([torch.tensor([0], device=device), torch.cumsum(segs, dim=0)], dim=-1)
    coords_list = torch.split(coords, segs.tolist(), 0)
    
    # Generate random edges using Gumbel sampling
    src = []
    for i, x in enumerate(coords_list):
        x = x.to(device)
        
        # Sample random neighbors based on distance-weighted probabilities
        _, indices = torch.topk(gumbel_sampling(x, temp), k=num_rand, dim=1)
        indices = indices + offset[i].to(torch.long)
        src.append(indices)
    
    # Flatten random edge sources
    src = (torch.cat(src, dim=0)).reshape(-1,)
    dst = torch.arange(0, num_nodes, device=device).repeat_interleave(num_rand)
    edge_index_rand = torch.vstack((src, dst))
    
    # Generate nearest neighbor edges
    edge_index_nearest = knn_graph(coords, num_nearest, batch, loop=False, flow='source_to_target')
    
    # Combine both types of edges
    edge_index = torch.hstack((edge_index_nearest, edge_index_rand))
    src, dst = edge_index
    
    # Verify edges don't cross batch boundaries
    if (batch[src] != batch[dst]).sum().item() != 0:
        warnings.warn("Warning: error in graph generation")
    
    # Create DGL graph with relative positions
    graph = dgl.graph((src, dst)).to(device)
    graph.edata['rel_pos'] = coords[src] - coords[dst]
    
    return graph.to(device)




