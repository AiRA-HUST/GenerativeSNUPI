# Import necessary libraries
# Standard library imports
import math
from collections import namedtuple
from random import random

# Third-party imports
import torch
from torch import nn
import torch.nn.functional as F
import roma
from tqdm.auto import tqdm
from einops import rearrange
from geomloss import SamplesLoss
from torch_geometric.utils import scatter

# Constants
ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# Default values for training parameters
DEFAULT_SHAPE_CONDITION_PROB = 0.9  # Probability of applying shape conditioning during training
DEFAULT_GEOLOSS_SCALE = 1e2  # Scale factor for geometric loss gradient
DEFAULT_SINKHORN_BLUR = 0.05  # Blur parameter for Sinkhorn loss


def exists(x):
    """Check if a value exists (is not None).
    
    Args:
        x: Value to check
        
    Returns:
        bool: True if x is not None, False otherwise
    """
    return x is not None


def default(val, d):
    """Return val if it exists, otherwise return d.
    
    Args:
        val: Primary value to return
        d: Default value or callable to return if val doesn't exist
        
    Returns:
        val if it exists, otherwise d (or d() if d is callable)
    """
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(input, t, batch=None):
    """Extract values from input tensor at indices specified by t.
    
    Args:
        input (torch.Tensor): Input tensor to extract from
        t (torch.Tensor): Indices to extract
        batch (torch.Tensor, optional): Batch indices for further indexing
        
    Returns:
        torch.Tensor: Extracted values
    """
    out = torch.gather(input, 0, t)
    if batch is not None:
        out = out[batch]
    return out


def linear_beta_schedule(timesteps):
    """Create a linear beta schedule for diffusion.
    
    Linear schedule as proposed in the original DDPM paper.
    
    Args:
        timesteps (int): Number of timesteps
        
    Returns:
        torch.Tensor: Beta values for each timestep
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """Create a cosine beta schedule for diffusion.
    
    Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    
    Args:
        timesteps (int): Number of timesteps
        s (float): Small offset to prevent singularities
        
    Returns:
        torch.Tensor: Beta values for each timestep
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """Create a sigmoid beta schedule for diffusion.
    
    Sigmoid schedule as proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    
    Args:
        timesteps (int): Number of timesteps
        start (float): Starting value for sigmoid
        end (float): Ending value for sigmoid
        tau (float): Temperature parameter
        clamp_min (float): Minimum value for clamping
        
    Returns:
        torch.Tensor: Beta values for each timestep
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class Diffusion(nn.Module):
    """Gaussian diffusion model for generating DNA nanostructures.
    
    This class implements a diffusion process for generating DNA nanostructure coordinates
    with optional shape conditioning. It supports various noise schedules and objectives.
    """
    
    def __init__(
        self,
        model,
        *,
        timesteps=1000,
        sampling_timesteps=None,
        loss_type='l2',
        objective='pred_noise',
        beta_schedule='linear',
        schedule_fn_kwargs=dict(),
        ddim_sampling_eta=0.,
        min_snr_loss_weight=False,  # https://arxiv.org/abs/2303.09556
        min_snr_gamma=5,
    ):
        """Initialize the Diffusion model.
        
        Args:
            model: The neural network model to use for denoising
            timesteps (int): Number of diffusion timesteps
            sampling_timesteps (int, optional): Number of sampling timesteps (defaults to training timesteps)
            loss_type (str): Type of loss function ('l2')
            objective (str): Training objective ('pred_noise' or 'pred_x0')
            beta_schedule (str): Type of beta schedule ('linear', 'cosine', 'sigmoid')
            schedule_fn_kwargs (dict): Additional arguments for schedule function
            ddim_sampling_eta (float): Eta parameter for DDIM sampling
            min_snr_loss_weight (bool): Whether to use minimum SNR loss weighting
            min_snr_gamma (float): Gamma parameter for minimum SNR loss weighting
        """
        super().__init__()

        self.model = model
        self.shape_condition = self.model.shape_condition
        self.objective = objective

        # Select beta schedule function
        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        # Generate beta schedule
        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        # Calculate alpha values and cumulative products
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        # Store basic parameters
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # Sampling parameters
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # Helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        # Register basic schedules
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Register diffusion process parameters for q(x_t | x_{t-1})
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # Register posterior parameters for q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)

        # Log posterior variance (clipped to prevent numerical issues)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # Calculate loss weights based on Signal-to-Noise Ratio (SNR)
        snr = alphas_cumprod / (1 - alphas_cumprod)

        # Apply minimum SNR loss weighting if specified
        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max=min_snr_gamma)

        # Set loss weights based on objective
        if objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)

    def predict_noise_from_start(self, x_t, t, x0, batch):
        """Predict noise from the starting point x0.
        
        Args:
            x_t (torch.Tensor): Noisy data at time t
            t (torch.Tensor): Time indices
            x0 (torch.Tensor): Clean data
            batch (torch.Tensor): Batch indices
            
        Returns:
            torch.Tensor: Predicted noise
        """
        return ((extract(self.sqrt_recip_alphas_cumprod, t, batch)[:, None] * x_t[:, 0:3] - x0[:, 0:3]) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, batch)[:, None])

    def predict_start_from_noise(self, x_t, t, noise, batch):
        """Predict the starting point x0 from noise.
        
        Args:
            x_t (torch.Tensor): Noisy data at time t
            t (torch.Tensor): Time indices
            noise (torch.Tensor): Predicted noise
            batch (torch.Tensor): Batch indices
            
        Returns:
            torch.Tensor: Predicted starting point
        """
        x_start = (extract(self.sqrt_recip_alphas_cumprod, t, batch)[:, None] * x_t[:, 0:3] -
                   extract(self.sqrt_recipm1_alphas_cumprod, t, batch)[:, None] * noise[:, 0:3])
        return x_start

    def q_posterior(self, x_start, x_t, t, batch):
        """Calculate posterior q(x_{t-1} | x_t, x_0).
        
        Args:
            x_start (torch.Tensor): Clean data
            x_t (torch.Tensor): Noisy data at time t
            t (torch.Tensor): Time indices
            batch (torch.Tensor): Batch indices
            
        Returns:
            tuple: (posterior_mean, posterior_variance, posterior_log_variance_clipped)
        """
        posterior_mean = (extract(self.posterior_mean_coef1, t, batch)[:, None] * x_start[:, 0:3] +
                          extract(self.posterior_mean_coef2, t, batch)[:, None] * x_t[:, 0:3])
        posterior_variance = extract(self.posterior_variance, t, batch)[:, None]
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, batch)[:, None]
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x_t, t, batch, shape_cond=None):
        """Get model predictions for noise and starting point.
        
        Args:
            x_t (torch.Tensor): Noisy data at time t
            t (torch.Tensor): Time indices
            batch (torch.Tensor): Batch indices
            shape_cond (torch.Tensor, optional): Shape conditioning information
            
        Returns:
            ModelPrediction: Named tuple containing predicted noise and starting point
        """
        # Get model output
        if self.shape_condition:
            model_output = self.model({'1': x_t[:, None, :]}, t, batch, shape_cond)
        else:
            model_output = self.model({'1': x_t[:, None, :]}, t, batch)

        # Interpret model output based on objective
        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x_t, t, pred_noise, batch)
        elif self.objective == 'pred_x0':
            x_start = model_output
            pred_noise = self.predict_noise_from_start(x_t, t, x_start, batch)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x_t, t, batch, shape_cond=None):
        """Calculate mean and variance for sampling step.
        
        Args:
            x_t (torch.Tensor): Noisy data at time t
            t (torch.Tensor): Time indices
            batch (torch.Tensor): Batch indices
            shape_cond (torch.Tensor, optional): Shape conditioning information
            
        Returns:
            tuple: (model_mean, posterior_variance, posterior_log_variance, x_start)
        """
        # Get model predictions
        preds = self.model_predictions(x_t, t, batch, shape_cond)
        x_start = preds.pred_x_start
        
        # Calculate posterior parameters
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start, x_t, t, batch)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x_t, t, batch, shape_cond=None):
        """Sample from p(x_{t-1} | x_t).
        
        Args:
            x_t (torch.Tensor): Noisy data at time t
            t (torch.Tensor): Time indices
            batch (torch.Tensor): Batch indices
            shape_cond (torch.Tensor, optional): Shape conditioning information
            
        Returns:
            tuple: (predicted_x, x_start)
        """
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x_t, t, batch, shape_cond)
        
        # Add noise only if not at the last step
        noise = torch.randn_like(x_t[:, 0:3]) if t[0] > 0 else 0.
        pred_x = model_mean[:, 0:3] + (0.5 * model_log_variance).exp() * noise
        
        return pred_x, x_start

    def p_sample_loop(self, batch, x_shape, x_shape_batch, cfg_scale=0,
                      return_all_timesteps=False, silent=False):
        """Main sampling loop for generating samples.
        
        Args:
            batch (torch.Tensor): Batch indices
            x_shape (torch.Tensor): Target shape coordinates
            x_shape_batch (torch.Tensor): Target shape batch indices
            cfg_scale (float): Classifier-free guidance scale
            return_all_timesteps (bool): Whether to return all intermediate timesteps
            silent (bool): Whether to suppress progress bar
            
        Returns:
            torch.Tensor: Generated samples
        """
        batch, device = batch, self.betas.device
        _, sample_sizes = torch.unique(batch, return_counts=True)
        num_nodes = sample_sizes.sum().int()
        num_samples = sample_sizes.size(0)

        # Initialize with random noise
        x_t = torch.randn(num_nodes, 3, device=device)
        x_t = CoM(x_t, batch)  # Center of mass normalization
        X = [x_t]

        # Set up progress bar
        if silent:
            pbar = reversed(range(0, self.num_timesteps))
        else:
            pbar = tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling', total=self.num_timesteps)
            pbar.set_postfix({'N': num_nodes.item()})

        # Sampling loop
        shape_cond = None
        for t in pbar:
            t = (torch.ones(1) * t).long().to(device)
            t = t.repeat(num_samples)
            
            # Apply shape conditioning if enabled
            if self.shape_condition:
                shape_cond = grad_geoloss(x_t, x_shape, batch, x_shape_batch)
                shape_cond.detach()
                
            # Perform sampling step
            x_t, _ = self.p_sample(x_t, t, batch, shape_cond)

            # Apply classifier-free guidance if specified
            if cfg_scale > 0:
                uncond_x_t, _ = self.p_sample(x_t, t, batch)
                x_t = torch.lerp(uncond_x_t, x_t, cfg_scale)

            # Re-center coordinates
            x_t = CoM(x_t, batch)
            X.append(x_t)

        # Return final result or all timesteps
        ret = x_t if not return_all_timesteps else torch.stack(X, dim=1)
        return ret.detach()

    def sample(self, batch, x_shape, x_shape_batch, cfg_scale=0,
               return_all_timesteps=False, silent=False):
        """Generate samples using the trained diffusion model.
        
        Args:
            batch (torch.Tensor): Batch indices
            x_shape (torch.Tensor): Target shape coordinates
            x_shape_batch (torch.Tensor): Target shape batch indices
            cfg_scale (float): Classifier-free guidance scale
            return_all_timesteps (bool): Whether to return all intermediate timesteps
            silent (bool): Whether to suppress progress bar
            
        Returns:
            torch.Tensor: Generated samples
        """
        sample_fn = self.p_sample_loop
        return sample_fn(batch, x_shape, x_shape_batch, cfg_scale=cfg_scale,
                         return_all_timesteps=return_all_timesteps, silent=silent)

    def q_sample(self, x_start, t, batch, noise=None):
        """Sample from q(x_t | x_0) - the forward diffusion process.
        
        Args:
            x_start (torch.Tensor): Clean data
            t (torch.Tensor): Time indices
            batch (torch.Tensor): Batch indices
            noise (torch.Tensor, optional): Noise to add (generated if None)
            
        Returns:
            torch.Tensor: Noisy data at time t
        """
        noise = torch.randn_like(x_start[:, 0:3]) if noise is None else noise
        x_t = (extract(self.sqrt_alphas_cumprod, t, batch)[..., None] * x_start[:, 0:3] +
               extract(self.sqrt_one_minus_alphas_cumprod, t, batch)[..., None] * noise)
        return x_t

    @property
    def loss_fn(self):
        """Get the loss function used for training.
        
        Returns:
            function: Loss function (MSE loss)
        """
        return F.mse_loss

    def p_losses(self, dnas, t, noise=None):
        """Calculate training losses.
        
        Args:
            dnas: Data object containing DNA coordinates and shape information
            t (torch.Tensor): Time indices
            noise (torch.Tensor, optional): Noise to add (generated if None)
            
        Returns:
            torch.Tensor: Calculated loss
        """
        # Extract data
        x_start = dnas.x_s
        batch = dnas.x_s_batch
        x_shape = dnas.x_t
        x_shape_batch = dnas.x_t_batch

        # Generate noise and create noisy samples
        noise = torch.randn_like(x_start[:, 0:3]) if noise is None else noise
        x_t = self.q_sample(x_start, t, batch, noise)
        x_t = CoM(x_t, batch)  # Center of mass normalization

        # Apply shape conditioning with probability
        shape_cond = None
        if self.shape_condition and random() < DEFAULT_SHAPE_CONDITION_PROB:
            shape_cond = grad_geoloss(x_t, x_shape, batch, x_shape_batch)
            shape_cond.detach()

        # Get model predictions
        preds = self.model_predictions(x_t, t, batch, shape_cond)
        noise_pred = preds.pred_noise
        x_start_pred = preds.pred_x_start

        # Calculate loss based on objective
        loss_weight = extract(self.loss_weight, t, batch)
        
        if self.objective == 'pred_noise':
            loss = self.loss_fn(noise, noise_pred, reduction='none').mean(dim=-1)
            loss = (loss * loss_weight).mean()
        elif self.objective == 'pred_x0':
            loss = self.loss_fn(x_start, x_start_pred, reduction='none').mean(dim=-1)
            loss = (loss * loss_weight).mean()

        return loss

    def forward(self, dnas, *args, **kwargs):
        """Forward pass for training.
        
        Args:
            dnas: Data object containing DNA coordinates and shape information
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            torch.Tensor: Training loss
        """
        batch_size = dnas.num_graphs
        device = dnas.x_s.device
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()
        return self.p_losses(dnas, t, *args, **kwargs)


def CoM(pos, batch):
    """Center coordinates to zero center of mass (CoM).
    
    This function shifts the coordinates so that the center of mass
    for each batch is at the origin.
    
    Args:
        pos (torch.Tensor): Coordinates to center
        batch (torch.Tensor): Batch indices
        
    Returns:
        torch.Tensor: Centered coordinates
    """
    return pos - scatter(pos, batch, dim=0, reduce='mean')[batch]


def grad_geoloss(src, target, src_batch, target_batch):
    """Calculate correspondence loss using geometric loss.
    
    This function computes the gradient of the Sinkhorn (Wasserstein) distance
    between source and target point clouds, which is used for shape conditioning.
    
    Args:
        src (torch.Tensor): Source point cloud
        target (torch.Tensor): Target point cloud  
        src_batch (torch.Tensor): Source batch indices
        target_batch (torch.Tensor): Target batch indices
        
    Returns:
        torch.Tensor: Gradient of geometric loss scaled by DEFAULT_GEOLOSS_SCALE
    """
    src.requires_grad = True
    geomloss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=DEFAULT_SINKHORN_BLUR)
    g_x = torch.zeros_like(src, device=src.device)
    batch_size = target_batch.amax() + 1
    
    # Calculate gradient for each batch separately
    for i in range(batch_size):
        x = src[src_batch == i, :]
        y = target[target_batch == i, :]
        geomloss = geomloss_fn(x, y)
        g_x[src_batch == i, :], = torch.autograd.grad(geomloss, [x])
    
    return DEFAULT_GEOLOSS_SCALE * g_x


    