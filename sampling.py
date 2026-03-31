#!/usr/bin/env python3
"""
DNA Structure Generation using Diffusion Models

This module provides functionality for generating DNA structures using
pre-trained diffusion models with various geometric shape conditions.
"""

import os
import sys
import json
import torch
import argparse
import warnings
import logging
import random
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Dict, Any

# Core modules
from inference import Inference
from visual import plot_dna
from diffusion import Diffusion
from se3_transformer.fiber import Fiber
from se3_transformer.transformer import create_model

# Shape generation modules
from shape.shape_ults import *

# Routing module
from routing import route_generated_design


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")


class DNAStructureGenerator:
    """
    A class for generating DNA structures using diffusion models.
    
    This class handles model loading, shape generation, and sampling
    of DNA structures based on geometric shape conditions.
    """
    
    # Constants
    DEFAULT_CONFIG_PATH = "configs/Condition.txt"
    DEFAULT_CHECKPOINT_PATH = "pretrained_models/Condition.pt"
    DEFAULT_OUTPUT_DIR = "generated_designs"
    DEFAULT_DEVICE = "cpu"  # Use CPU by default for compatibility
    
    def __init__(self, config_path: str = None, checkpoint_path: str = None, device: str = None):
        """
        Initialize the DNA Structure Generator.
        
        Args:
            config_path: Path to configuration file
            checkpoint_path: Path to model checkpoint
            device: Device to use for computation ('cpu' or 'cuda')
        """
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.checkpoint_path = checkpoint_path or self.DEFAULT_CHECKPOINT_PATH
        self.device = torch.device(device or self.DEFAULT_DEVICE)
        
        self.args = None
        self.model = None
        self.diffusion = None
        self.inference = None
        
        logger.info(f"Initializing DNA Structure Generator on device: {self.device}")
        self._load_config()
        self._setup_model()
        self._load_checkpoint()
    
    def _load_config(self) -> None:
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                config_dict = json.load(f)
            
            parser = argparse.ArgumentParser()
            self.args = parser.parse_args([])  # Pass empty list to avoid parsing sys.argv
            self.args.fiber_hidden = None
            self.args.__dict__.update(config_dict)
            
            logger.info(f"Configuration loaded from: {self.config_path}")
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file: {e}")
            raise
    
    def _setup_model(self) -> None:
        """Setup the diffusion model and related components."""
        try:
            # Create the transformer model
            self.model = create_model(
                num_layers=self.args.num_layers,
                time_emb_dim=self.args.time_emb_dim,
                fiber_hidden=Fiber(self.args.fiber_hidden),
                num_heads=self.args.num_heads,
                channels_div=self.args.channels_div,
                num_nearest_neighbors=self.args.num_nearest_neighbors,
                num_random_neighbors=self.args.num_random_neighbors,
                temp=self.args.temp,
                shape_condition=self.args.shape_condition,
                tensor_cores=self.args.tensor_cores,
                low_memory=self.args.low_memory
            )
            
            # Create diffusion wrapper
            self.diffusion = Diffusion(
                self.model,
                timesteps=self.args.timesteps,
                sampling_timesteps=self.args.sampling_timesteps,
                objective=self.args.objective,
                beta_schedule=self.args.beta_schedule
            )
            
            # Create inference wrapper
            self.inference = Inference(
                self.diffusion,
                ema_decay=self.args.ema_decay,
                split_batches=self.args.split_batches,
                fp16=self.args.fp16
            )
            
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Model created with {total_params:,} parameters")
            
        except Exception as e:
            logger.error(f"Failed to setup model: {e}")
            raise
    
    def _load_checkpoint(self) -> None:
        """Load model checkpoint."""
        try:
            if not os.path.exists(self.checkpoint_path):
                logger.error(f"Checkpoint file not found: {self.checkpoint_path}")
                raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
            
            self.inference.load_checkpoint(str(self.checkpoint_path))
            logger.info(f"Checkpoint loaded from: {self.checkpoint_path}")
            
            # Print model information
            self.inference.print_model_info()
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def generate_shape(self, shape_type: str, **kwargs) -> Tuple[torch.Tensor, int]:
        """
        Generate a target shape based on the specified type.
        
        Args:
            shape_type: Type of shape to generate ('circle', 'svg', 'pt', 'mesh')
            **kwargs: Additional parameters for shape generation
                For 'svg': filename (required), design_scaffold_length (default: 7249), max_scaf_len (default: 7249),
                          radius_diff (default: 2.0), radius_shape (default: 2.0), 
                          add_verts (default: False), zero_gravity (default: True)
                          Note: Internally mapped to max_scaffold_length and add_vertices
                For 'pt': filename (required)
                For 'mesh': filename (required - supports .ply and .stl files), 
                           design_scaffold_length (default: 7429), max_scaf_len (default: 7429),
                           radius_diff (default: 2.0), radius_shape (default: 2.0), 
                           add_verts (default: False), zero_gravity (default: True)
                           Note: Files searched in ply/2D/, ply/3D/, ply/, and stl/ directories
                For 'circle': radius, x0, y0, z0
            
        Returns:
            Tuple of (shape_tensor, num_base_pairs)
        """
        if shape_type == 'circle':
            try:
                shape, num_bps = circle2D(
                    radius=kwargs.get('radius', 10.0),
                    x0=kwargs.get('x0', 0.0),
                    y0=kwargs.get('y0', 0.0),
                    z0=kwargs.get('z0', 0.0),
                    return_num_bps=True
                )
                logger.info(f"Generated {shape_type} shape with {num_bps} base pairs")
                return shape, num_bps
            except Exception as e:
                logger.error(f"Failed to generate {shape_type} shape: {e}")
                raise
                
        elif shape_type == 'svg':
            filename = kwargs.get('filename')
            if not filename:
                raise ValueError("filename parameter is required for SVG shape type")
            
            svg_path = Path("target_shapes/svg") / filename
            if not svg_path.exists():
                raise FileNotFoundError(f"SVG file not found: {svg_path}")
            
            try:
                # Import svg_to_shape function
                from shape.svg_to_shape import svg_to_shape
                
                shape, num_bps = svg_to_shape(
                    svg_file=str(svg_path),
                    design_scaffold_length=kwargs.get('design_scaffold_length', 7249),
                    max_scaffold_length=kwargs.get('max_scaf_len', 7249),
                    radius_diff=kwargs.get('radius_diff', 2.0),
                    radius_shape=kwargs.get('radius_shape', 2.0),
                    add_vertices=kwargs.get('add_verts', False),
                    zero_gravity=kwargs.get('zero_gravity', True)
                )
                
                logger.info(f"Loaded SVG shape from {filename} with {num_bps} base pairs")
                return shape, num_bps
            except Exception as e:
                logger.error(f"Failed to load SVG shape from {filename}: {e}")
                raise
                
        elif shape_type == 'mesh':
            filename = kwargs.get('filename')
            if not filename:
                raise ValueError("filename parameter is required for mesh shape type")
            
            # Check PLY (2D and 3D) and STL directories for the mesh file
            mesh_paths = [
                Path("target_shapes/ply/2D") / filename,
                Path("target_shapes/ply/3D") / filename,
                Path("target_shapes/ply") / filename,
                Path("target_shapes/stl") / filename
            ]
            
            mesh_path = None
            for path in mesh_paths:
                if path.exists():
                    mesh_path = path
                    break
            
            if mesh_path is None:
                raise FileNotFoundError(f"Mesh file not found: {filename}. Searched in: {[str(p) for p in mesh_paths]}")
            
            try:
                # Import mesh_to_shape function
                from shape.mesh_to_shape import mesh_to_shape
                
                shape, num_bps = mesh_to_shape(
                    file_name=str(mesh_path),
                    design_scaffold_length=kwargs.get('design_scaffold_length', 7429),
                    max_scaffold_length=kwargs.get('max_scaf_len', 7429),
                    radius_diff=kwargs.get('radius_diff', 2.0),
                    radius_shape=kwargs.get('radius_shape', 2.0),
                    add_verts=kwargs.get('add_verts', False),
                    zero_gravity=kwargs.get('zero_gravity', True)
                )
                
                logger.info(f"Loaded mesh shape from {filename} with {num_bps} base pairs")
                return shape, num_bps
            except Exception as e:
                logger.error(f"Failed to load mesh shape from {filename}: {e}")
                raise
                
        elif shape_type == 'pt':
            filename = kwargs.get('filename')
            if not filename:
                raise ValueError("filename parameter is required for PT shape type")
            
            pt_path = Path("target_shapes/pt") / filename
            if not pt_path.exists():
                raise FileNotFoundError(f"PT file not found: {pt_path}")
            
            try:
                shape = torch.load(str(pt_path))
                if not isinstance(shape, torch.Tensor):
                    raise ValueError(f"Loaded PT file does not contain a torch.Tensor: {type(shape)}")
                
                num_bps = 6 * shape.size(0)
                logger.info(f"Loaded PT shape from {filename} with {num_bps} base pairs (6 * {shape.size(0)} points)")
                return shape, num_bps
            except Exception as e:
                logger.error(f"Failed to load PT shape from {filename}: {e}")
                raise
        
        else:
            available_shapes = ['circle', 'svg', 'pt', 'mesh']
            raise ValueError(f"Unknown shape type: {shape_type}. Available: {available_shapes}")
    
    def generate_samples(
        self,
        target_shape: torch.Tensor,
        num_base_pairs: int,
        num_samples: int = 5,
        output_dir: str = None,
        design_name: str = "sample",
        save_coordinates: bool = True,
        show_plots: bool = True,
        variation_range: float = 0.05
    ) -> Dict[str, Any]:
        """
        Generate DNA structure samples based on target shape.
        
        Args:
            target_shape: Target shape tensor
            num_base_pairs: Number of base pairs in the structure
            num_samples: Number of samples to generate
            output_dir: Output directory (auto-generated if None)
            design_name: Name prefix for output files
            save_coordinates: Whether to save coordinate files
            show_plots: Whether to display plots
            variation_range: Range for random variation in base pairs count
            
        Returns:
            Dictionary containing generation results and paths
        """
        # Setup output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_folder_name = f"{design_name}_{timestamp}"
            output_path = Path(self.DEFAULT_OUTPUT_DIR) / output_folder_name
        else:
            output_path = Path(output_dir)
        
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving results to: {output_path}")
        
        # Prepare shape for generation
        x_shape = target_shape.to(self.device)
        x_shape_batch = torch.zeros(target_shape.size(0), dtype=torch.long, device=self.device)
        
        # Get the model for inference
        diffusion_model = (
            self.inference.ema.ema_model if self.inference.ema 
            else self.inference.model
        )
        diffusion_model = self.inference.accelerator.unwrap_model(diffusion_model).to(self.device)
        
        results = {
            'output_path': str(output_path),
            'design_name': design_name,
            'num_samples': num_samples,
            'generated_files': []
        }
        
        # Generate samples
        for i in range(num_samples):
            try:
                # Add random variation to base pairs count
                variation = random.randint(
                    -int(num_base_pairs * variation_range),
                    int(num_base_pairs * variation_range)
                )
                batch_size = num_base_pairs + variation
                batch = torch.zeros(batch_size, dtype=torch.long, device=self.device)
                
                logger.info(f"Generating sample {i+1}/{num_samples} with {batch_size} base pairs")
                
                # Generate coordinates through diffusion
                coords = diffusion_model.sample(
                    batch, x_shape, x_shape_batch,
                    return_all_timesteps=True,
                    silent=False
                )
                
                # Extract final coordinates
                final_coords = coords[:, -1, :]

                
                # Create and optionally show plot
                if show_plots:
                    fig = plot_dna(final_coords)
                    fig.show()
                
                # Save coordinates if requested
                if save_coordinates:
                    coords_filename = f"{design_name}_sample_{i+1:03d}.pt"
                    coords_path = output_path / coords_filename
                    torch.save(final_coords, str(coords_path))
                    results['generated_files'].append(coords_filename)
                    logger.info(f"Coordinates saved: {coords_filename}")
                    
                    # Try to route the generated design
                    logger.info(f"Attempting to route generated design: {coords_filename}")
                    routing_success = route_generated_design(str(coords_path), show=show_plots)
                    
                    if routing_success:
                        logger.info(f"\n🎉 SUCCESS! DNA origami design routing completed successfully!")
                        logger.info(f"✓ Generated design: {coords_filename}")
                        logger.info(f"✓ All design files exported to: {output_path}")
                        logger.info(f"🚀 Program terminated after successful routing.")
                        
                        # Update results with successful routing info
                        results['routing_success'] = True
                        results['successful_file'] = coords_filename
                        
                        # Exit the program successfully
                        sys.exit(0)
                    else:
                        logger.warning(f"⚠ Routing failed for {coords_filename}, trying next sample...")
                
            except Exception as e:
                logger.error(f"Failed to generate sample {i+1}: {e}")
                continue
        
        # This line will only be reached if routing failed for all samples
        logger.warning(f"Generation complete! Generated {len(results['generated_files'])} samples, but routing failed for all.")
        results['routing_success'] = False
        return results
    
    def visualize_shape(self, shape: torch.Tensor, title: str = "Target Shape") -> None:
        """
        Visualize a target shape.
        
        Args:
            shape: Shape tensor to visualize
            title: Title for the plot
        """
        try:
            fig = plot_dna(shape)
            fig.show()
        except Exception as e:
            logger.error(f"Failed to visualize shape: {e}")

    def generate_samples_unconditional(
        self,
        num_base_pairs: int,
        num_samples: int = 5,
        output_dir: str = None,
        design_name: str = "unconditional_sample",
        save_coordinates: bool = True,
        show_plots: bool = True,
        variation_range: float = 0.01
    ) -> Dict[str, Any]:
        """
        Generate DNA structure samples unconditionally (without target shape conditioning).
        
        Args:
            num_base_pairs: Number of base pairs in the structure
            num_samples: Number of samples to generate
            output_dir: Output directory (auto-generated if None)
            design_name: Name prefix for output files
            save_coordinates: Whether to save coordinate files
            show_plots: Whether to display plots
            variation_range: Range for random variation in base pairs count
            
        Returns:
            Dictionary containing generation results and paths
        """
        # Check if the model supports unconditional generation
        if self.args.shape_condition:
            logger.warning("Warning: This model was trained with shape conditioning but being used unconditionally. Results may be suboptimal.")
        
        # Setup output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_folder_name = f"{design_name}_{timestamp}"
            output_path = Path(self.DEFAULT_OUTPUT_DIR) / output_folder_name
        else:
            output_path = Path(output_dir)
        
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving unconditional results to: {output_path}")
        
        # Get the model for inference
        diffusion_model = (
            self.inference.ema.ema_model if self.inference.ema 
            else self.inference.model
        )
        diffusion_model = self.inference.accelerator.unwrap_model(diffusion_model).to(self.device)
        
        results = {
            'output_path': str(output_path),
            'design_name': design_name,
            'num_samples': num_samples,
            'generated_files': []
        }
        
        # Generate samples
        for i in range(num_samples):
            try:
                # Add random variation to base pairs count
                variation = random.randint(
                    -int(num_base_pairs * variation_range),
                    int(num_base_pairs * variation_range)
                )
                batch_size = num_base_pairs + variation
                batch = torch.zeros(batch_size, dtype=torch.long, device=self.device)
                
                logger.info(f"Generating unconditional sample {i+1}/{num_samples} with {batch_size} base pairs")
                
                # For unconditional generation, we create dummy shape parameters
                # These won't be used since shape_condition is False in the unconditional model
                dummy_shape = torch.zeros((1, 3), device=self.device)
                dummy_shape_batch = torch.zeros(1, dtype=torch.long, device=self.device)
                
                # Generate coordinates through diffusion
                coords = diffusion_model.sample(
                    batch, dummy_shape, dummy_shape_batch,
                    return_all_timesteps=True,
                    silent=False
                )
                
                # Extract final coordinates
                final_coords = coords[:, -1, :]
                
                # Create and optionally show plot
                if show_plots:
                    fig = plot_dna(final_coords)
                    fig.show()
                
                # Save coordinates if requested
                if save_coordinates:
                    coords_filename = f"{design_name}_sample_{i+1:03d}.pt"
                    coords_path = output_path / coords_filename
                    torch.save(final_coords, str(coords_path))
                    results['generated_files'].append(coords_filename)
                    logger.info(f"Coordinates saved: {coords_filename}")
                    
                    # Try to route the generated design
                    logger.info(f"Attempting to route unconditional generated design: {coords_filename}")
                    routing_success = route_generated_design(str(coords_path), show=True)
                    
                    if routing_success:
                        logger.info(f"\n🎉 SUCCESS! Unconditional DNA origami design routing completed successfully!")
                        logger.info(f"✓ Generated design: {coords_filename}")
                        logger.info(f"✓ All design files exported to: {output_path}")
                        logger.info(f"🚀 Program terminated after successful routing.")
                        
                        # Update results with successful routing info
                        results['routing_success'] = True
                        results['successful_file'] = coords_filename
                        
                        # Exit the program successfully
                        sys.exit(0)
                    else:
                        logger.warning(f"⚠ Routing failed for {coords_filename}, trying next sample...")

                    
                
            except Exception as e:
                logger.error(f"Failed to generate unconditional sample {i+1}: {e}")
                continue
        
        # This line will only be reached if routing failed for all samples  
        logger.warning(f"Unconditional generation complete! Generated {len(results['generated_files'])} samples, but routing failed for all.")
        results['routing_success'] = False
        return results


def main():
    """Main function demonstrating usage of the DNA Structure Generator."""
    try:
        print("🧬 DNA Structure Generator with Automatic Routing")
        print("="*60)
        print("ℹ️  This program will:")
        print("   1. Generate DNA structure coordinates from target shapes")
        print("   2. Automatically attempt DNA origami routing for each sample")
        print("   3. Stop and exit when the first successful routing is completed")
        print("   4. Export all design files (.cndo, .bild, .dat, .top, .pdb) to the same folder")
        print("="*60)
        
        # Initialize generator
        generator = DNAStructureGenerator()
        

        
        # Example 1: Generate from SVG file (uncomment to use)
        '''
        print("\n" + "="*60)
        print("Example 2: SVG Shape")
        print("="*60)
        svg_design_name = "cat_nazca"
        svg_params = {
            'filename': 'cat_nazca.svg',
            'design_scaffold_length': 7249,
            'radius_diff': 2.0,
            'radius_shape': 2.0,
            'add_verts': False,
            'zero_gravity': True
        }
        
        logger.info(f"Loading SVG shape: {svg_design_name}")
        svg_shape, svg_num_bps = generator.generate_shape('svg', **svg_params)
        
        # Visualize SVG shape
        generator.visualize_shape(svg_shape, f"SVG Shape: {svg_design_name}")
        
        # Generate samples from SVG
        svg_results = generator.generate_samples(
            target_shape=svg_shape,
            num_base_pairs=svg_num_bps,
            num_samples=2,
            design_name=svg_design_name,
            save_coordinates=True,
            show_plots=True
        )
        '''
        
        # Example 2: Generate from mesh file (PLY)
        print("\n" + "="*60)
        print("Example 3: Mesh Shape")
        print("="*60)
        mesh_design_name = "triangle"
        mesh_params = {
            'filename': '01_triangle.ply',
            'design_scaffold_length': 7429,
            'max_scaf_len': 7429,
            'radius_diff': 2.0,
            'radius_shape': 2.0,
            'add_verts': False,
            'zero_gravity': True
        }
        
        logger.info(f"Loading mesh shape: {mesh_design_name}")
        mesh_shape, mesh_num_bps = generator.generate_shape('mesh', **mesh_params)
        
        # Visualize mesh shape
        generator.visualize_shape(mesh_shape, f"Mesh Shape: {mesh_design_name}")
        
        # Generate samples from mesh
        mesh_results = generator.generate_samples(
            target_shape=mesh_shape,
            num_base_pairs=mesh_num_bps,
            num_samples=2,
            design_name=mesh_design_name,
            save_coordinates=True,
            show_plots=True
        )
        
        # Print results summary - this will only be reached if routing failed for all samples
        print("\n" + "="*60)
        print("⚠ GENERATION COMPLETE - ROUTING FAILED")
        print("="*60)
        print(f"📁 Results saved in: {mesh_results['output_path']}")
        print(f"📊 Generated {len(mesh_results['generated_files'])} samples for design: {mesh_design_name}")
        print(f"❌ Note: All generated samples failed DNA origami routing")
        print(f"💡 You can manually try routing using the generated .pt files")
        print("\n📂 Generated coordinate files:")
        for filename in mesh_results['generated_files']:
            print(f"  • {filename}")
        print("="*60)
        
        # Example 3: Generate from STL file (uncomment to use)
        '''
        print("\n" + "="*60)
        print("Example 4: STL Shape")
        print("="*60)
        stl_design_name = "miura_origami"
        stl_params = {
            'filename': 'miura_ori_3x3_40PercentFolded.stl',
            'design_scaffold_length': 3000,  # Smaller scaffold length for detailed mesh
            'max_scaf_len': 7429,
            'radius_diff': 2.0,
            'radius_shape': 2.0,
            'add_verts': False,
            'zero_gravity': True
        }
        
        logger.info(f"Loading STL shape: {stl_design_name}")
        stl_shape, stl_num_bps = generator.generate_shape('mesh', **stl_params)
        
        # Visualize STL shape
        generator.visualize_shape(stl_shape, f"STL Shape: {stl_design_name}")
        
        # Generate samples from STL
        stl_results = generator.generate_samples(
            target_shape=stl_shape,
            num_base_pairs=stl_num_bps,
            num_samples=2,
            design_name=stl_design_name,
            save_coordinates=True,
            show_plots=True
        )
        
        # Print results summary
        print("\n" + "="*60)
        print("🎉 STL GENERATION COMPLETE!")
        print("="*60)
        print(f"📁 Results saved in: {stl_results['output_path']}")
        print(f"📊 Generated {len(stl_results['generated_files'])} samples for design: {stl_design_name}")
        print("\n📂 Files saved:")
        for filename in stl_results['generated_files']:
            print(f"  • {filename}")
        print("="*60)
        '''
        
        # Example 4: Generate from PT file (uncomment to use when PT files are available)
        '''
        print("\n" + "="*60)
        print("Example 4: PT Shape")
        print("="*60)
        pt_design_name = "Circle"
        pt_params = {
            'filename': 'Circle.pt'
        }
        
        logger.info(f"Loading PT shape: {pt_design_name}")
        pt_shape, pt_num_bps = generator.generate_shape('pt', **pt_params)
        
        # Visualize PT shape
        generator.visualize_shape(pt_shape, f"PT Shape: {pt_design_name}")
        
        # Generate samples from PT
        pt_results = generator.generate_samples(
            target_shape=pt_shape,
            num_base_pairs=pt_num_bps,
            num_samples=2,
            design_name=pt_design_name,
            save_coordinates=True,
            show_plots=True
        )
        '''
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise


if __name__ == "__main__":
    main()