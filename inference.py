"""
Inference module for DNA diffusion model.

This module provides functionality for loading trained diffusion models
and performing inference to generate DNA sequences.
"""

import torch
from pathlib import Path
from typing import Optional, Tuple, Any, Dict
from tqdm import tqdm

from accelerate import Accelerator
from ema_pytorch import EMA



def exists(x: Any) -> bool:
    """Check if a value exists (is not None).
    
    Args:
        x: Value to check
        
    Returns:
        bool: True if x is not None, False otherwise
    """
    return x is not None


class Inference:
    """
    Inference class for DNA diffusion model.
    
    This class handles loading trained diffusion models from checkpoints
    and provides methods for generating DNA sequences.
    
    Attributes:
        model: The diffusion model
        accelerator: Accelerator for distributed inference
        ema: Exponential moving average model (if available)
        step: Current training step from checkpoint
    """
    
    def __init__(
        self,
        diffusion_model,
        *,
        ema_decay: float = 0.995,
        split_batches: bool = True,
        fp16: bool = True,
        amp: bool = False,
    ):
        """
        Initialize the inference class.
        
        Args:
            diffusion_model: The diffusion model to use for inference
            ema_decay: Decay rate for exponential moving average (default: 0.995)
            split_batches: Whether to split batches across devices (default: True)
            fp16: Whether to use FP16 precision (default: True)
            amp: Whether to use automatic mixed precision (default: False)
        """
        super().__init__()
        
        # Initialize accelerator for distributed inference
        mixed_precision = 'fp16' if fp16 else 'no'
        self.accelerator = Accelerator(
            split_batches=split_batches, 
            mixed_precision=mixed_precision
        )
        self.accelerator.native_amp = amp
        
        # Store model
        self.model = diffusion_model
        
        # Initialize EMA if this is the main process
        self.ema = None
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay)
        
        # Initialize step counter
        self.step = 0
        
        # Prepare model with accelerator
        self.model = self.accelerator.prepare(self.model)
        
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load model weights and state from a checkpoint file.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If checkpoint loading fails
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        try:
            device = self.accelerator.device
            print(f"Loading checkpoint from: {checkpoint_path}")
            
            # Load checkpoint data
            checkpoint_data = torch.load(checkpoint_path, map_location=device)
            
            # Load model state
            model = self.accelerator.unwrap_model(self.model)
            if 'model' in checkpoint_data:
                model.load_state_dict(checkpoint_data['model'])
                print("✓ Model weights loaded successfully")
            else:
                raise RuntimeError("No 'model' key found in checkpoint")
            
            # Load training step
            if 'step' in checkpoint_data:
                self.step = checkpoint_data['step']
                print(f"✓ Loaded from training step: {self.step}")
            
            # Load EMA state if available
            if self.ema is not None and 'ema' in checkpoint_data:
                self.ema.load_state_dict(checkpoint_data['ema'])
                print("✓ EMA weights loaded successfully")
            
            # Load scaler state if available
            if (exists(self.accelerator.scaler) and 
                exists(checkpoint_data.get('scaler'))):
                self.accelerator.scaler.load_state_dict(checkpoint_data['scaler'])
                print("✓ Scaler state loaded successfully")
            
            # Print version info if available
            if 'version' in checkpoint_data:
                print(f"✓ Checkpoint version: {checkpoint_data['version']}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {str(e)}")
    
    @torch.no_grad()
    def generate_samples(
        self, 
        num_samples: int = 1,
        batch_size: Optional[int] = None,
        use_ema: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate DNA sequence samples using the diffusion model.
        
        Args:
            num_samples: Number of samples to generate
            batch_size: Batch size for generation (default: same as num_samples)
            use_ema: Whether to use EMA model if available
            **kwargs: Additional arguments passed to the model's sample method
            
        Returns:
            torch.Tensor: Generated DNA sequences
        """
        if batch_size is None:
            batch_size = num_samples
            
        # Choose which model to use for inference
        model_to_use = self.model
        if use_ema and self.ema is not None:
            model_to_use = self.ema.ema_model
            
        # Set model to evaluation mode
        model_to_use.eval()
        
        print(f"Generating {num_samples} samples...")
        
        all_samples = []
        remaining_samples = num_samples
        
        with tqdm(total=num_samples, desc="Generating samples") as pbar:
            while remaining_samples > 0:
                current_batch_size = min(batch_size, remaining_samples)
                
                # Generate samples for current batch
                samples = model_to_use.sample(batch_size=current_batch_size, **kwargs)
                all_samples.append(samples)
                
                remaining_samples -= current_batch_size
                pbar.update(current_batch_size)
        
        # Concatenate all samples
        if len(all_samples) == 1:
            return all_samples[0]
        else:
            return torch.cat(all_samples, dim=0)[:num_samples]
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dict containing model information
        """
        model = self.accelerator.unwrap_model(self.model)
        
        info = {
            'model_class': model.__class__.__name__,
            'training_step': self.step,
            'has_ema': self.ema is not None,
            'device': str(self.accelerator.device),
            'mixed_precision': self.accelerator.mixed_precision,
        }
        
        # Add parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        info.update({
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
        })
        
        return info
    
    def print_model_info(self) -> None:
        """Print detailed information about the loaded model."""
        info = self.get_model_info()
        
        print("\n" + "="*50)
        print("MODEL INFORMATION")
        print("="*50)
        print(f"Model Class: {info['model_class']}")
        print(f"Training Step: {info['training_step']:,}")
        print(f"Device: {info['device']}")
        print(f"Mixed Precision: {info['mixed_precision']}")
        print(f"Has EMA: {info['has_ema']}")
        print(f"Total Parameters: {info['total_parameters']:,}")
        print(f"Trainable Parameters: {info['trainable_parameters']:,}")
        print("="*50 + "\n")


def load_inference_model(checkpoint_path: str, diffusion_model, **kwargs) -> Inference:
    """
    Convenience function to create and load an inference model.
    
    Args:
        checkpoint_path: Path to model checkpoint
        diffusion_model: Diffusion model instance
        **kwargs: Additional arguments for Inference class
        
    Returns:
        Inference: Loaded inference model ready for generation
    """
    inference = Inference(diffusion_model, **kwargs)
    inference.load_checkpoint(checkpoint_path)
    return inference

