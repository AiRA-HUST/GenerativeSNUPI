#!/usr/bin/env python3
"""
DNA Structure Sampling Runner

A command-line interface for generating DNA structures from target shapes or unconditionally.
Supports both conditional generation (SVG, PLY, STL, and PT file formats) and unconditional generation.

Conditional Usage:
    python run_sampling.py --target_shape path/to/shape.svg --num_samples 5
    python run_sampling.py --target_shape triangle.ply --num_samples 3 --output_dir my_results
    python run_sampling.py --target_shape Circle.pt --num_samples 2 --show_plots

Unconditional Usage:
    python run_sampling.py --unconditional --num_base_pairs 1000 --num_samples 3
    python run_sampling.py --unconditional --num_base_pairs 2000 --num_samples 5 --design_name my_design
"""

import argparse
import sys
import os
from pathlib import Path
import logging

# Import the DNA structure generator from sampling.py
from sampling import DNAStructureGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def detect_shape_type(file_path: str) -> str:
    """
    Detect the shape type from file extension.
    
    Args:
        file_path: Path to the shape file
        
    Returns:
        Shape type ('svg', 'pt', 'mesh')
    """
    file_path = Path(file_path)
    extension = file_path.suffix.lower()
    
    if extension == '.svg':
        return 'svg'
    elif extension == '.pt':
        return 'pt'
    elif extension in ['.ply', '.stl']:
        return 'mesh'
    else:
        raise ValueError(f"Unsupported file type: {extension}. Supported: .svg, .pt, .ply, .stl")


def get_shape_params(shape_type: str, file_path: str, args: argparse.Namespace) -> dict:
    """
    Get shape parameters based on shape type and command line arguments.
    
    Args:
        shape_type: Type of shape ('svg', 'pt', 'mesh')
        file_path: Path to the shape file
        args: Command line arguments
        
    Returns:
        Dictionary of shape parameters
    """
    filename = Path(file_path).name
    
    if shape_type == 'svg':
        return {
            'filename': filename,
            'design_scaffold_length': args.design_scaffold_length,
            'max_scaf_len': args.max_scaf_len,
            'radius_diff': args.radius_diff,
            'radius_shape': args.radius_shape,
            'add_verts': args.add_verts,
            'zero_gravity': args.zero_gravity
        }
    elif shape_type == 'pt':
        return {
            'filename': filename
        }
    elif shape_type == 'mesh':
        return {
            'filename': filename,
            'design_scaffold_length': args.design_scaffold_length,
            'max_scaf_len': args.max_scaf_len,
            'radius_diff': args.radius_diff,
            'radius_shape': args.radius_shape,
            'add_verts': args.add_verts,
            'zero_gravity': args.zero_gravity
        }
    else:
        raise ValueError(f"Unknown shape type: {shape_type}")


def validate_file_path(file_path: str, shape_type: str) -> bool:
    """
    Validate that the target shape file exists in the appropriate directory.
    
    Args:
        file_path: Path to the shape file
        shape_type: Type of shape ('svg', 'pt', 'mesh')
        
    Returns:
        True if file exists, False otherwise
    """
    filename = Path(file_path).name
    
    if shape_type == 'svg':
        full_path = Path("target_shapes/svg") / filename
    elif shape_type == 'pt':
        full_path = Path("target_shapes/pt") / filename
    elif shape_type == 'mesh':
        # Check multiple possible locations for mesh files
        possible_paths = [
            Path("target_shapes/ply/2D") / filename,
            Path("target_shapes/ply/3D") / filename,
            Path("target_shapes/ply") / filename,
            Path("target_shapes/stl") / filename
        ]
        return any(path.exists() for path in possible_paths)
    
    return full_path.exists() if 'full_path' in locals() else False


def main():
    """Main function to run the DNA structure sampling."""
    parser = argparse.ArgumentParser(
        description="Generate DNA structures from target shapes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""

Examples:
    # Generate 5 samples from SVG file (conditional mode)
    python run_sampling.py --target_shape cat_nazca.svg --num_samples 5

    # Generate 5 samples from SVG file with custom scaffold length (conditional mode)
    python run_sampling.py --target_shape cat_nazca.svg --num_samples 5 --design_scaffold_length 6000

    # Generate 3 samples from PLY file with custom scaffold length (conditional mode)
    python run_sampling.py --target_shape 01_triangle.ply --num_samples 3 --design_scaffold_length 6000

    # Generate 2 samples from PT file and save to custom directory (conditional mode)
    python run_sampling.py --target_shape Circle.pt --num_samples 2 --output_dir custom_results

    # Generate samples from PT file (conditional mode)
    python run_sampling.py --target_shape Circle.pt --num_samples 3

    # Generate samples without showing plots (conditional mode)
    python run_sampling.py --target_shape hexagon.ply --num_samples 4 --no_plots

    # Generate 3 unconditional samples with 1000 base pairs
    python run_sampling.py --unconditional --num_base_pairs 1000 --num_samples 3

    # Generate 5 unconditional samples with 2000 base pairs and custom name
    python run_sampling.py --unconditional --num_base_pairs 2000 --num_samples 5 --design_name my_unconditional

    # Generate unconditional samples without plots
    python run_sampling.py --unconditional --num_base_pairs 1500 --num_samples 2 --no_plots

    # Generate unconditional samples to custom directory
    python run_sampling.py --unconditional --num_base_pairs 800 --num_samples 4 --output_dir unconditional_results
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--target_shape', 
        help='Path to target shape file (SVG, PLY, STL, or PT). Required for conditional mode, ignored in unconditional mode.'
    )
    parser.add_argument(
        '--num_samples', 
        type=int, 
        default=2,
        help='Number of DNA structure samples to generate (default: 2)'
    )
    
    # Mode selection
    parser.add_argument(
        '--unconditional',
        action='store_true',
        help='Use unconditional generation mode (no target shape required)'
    )
    parser.add_argument(
        '--num_base_pairs',
        type=int,
        help='Number of base pairs for unconditional generation (required when using --unconditional)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output_dir',
        help='Custom output directory (default: auto-generated based on design name and timestamp)'
    )
    parser.add_argument(
        '--design_name',
        help='Name prefix for output files (default: derived from filename or "unconditional" for unconditional mode)'
    )
    parser.add_argument(
        '--config_path',
        help='Path to configuration file (default: configs/Condition.txt for conditional, configs/Uncondition.txt for unconditional)'
    )
    parser.add_argument(
        '--checkpoint_path',
        help='Path to model checkpoint (default: pretrained_models/Condition.pt for conditional, pretrained_models/Uncondition.pt for unconditional)'
    )
    parser.add_argument(
        '--device',
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use for computation (default: cpu)'
    )
    
    # Shape generation parameters
    parser.add_argument(
        '--design_scaffold_length',
        type=int,
        default=7249,
        help='Design scaffold length for SVG and mesh files - automatically calculates scale (default: 7249)'
    )
    parser.add_argument(
        '--max_scaf_len',
        type=int,
        default=7429,
        help='Maximum scaffold length (default: 7429)'
    )
    parser.add_argument(
        '--radius_diff',
        type=float,
        default=2.0,
        help='Radius difference parameter (default: 2.0)'
    )
    parser.add_argument(
        '--radius_shape',
        type=float,
        default=2.0,
        help='Radius shape parameter (default: 2.0)'
    )
    parser.add_argument(
        '--add_verts',
        action='store_true',
        help='Add vertices to shape (default: False)'
    )
    parser.add_argument(
        '--zero_gravity',
        action='store_true',
        default=True,
        help='Use zero gravity mode (default: True)'
    )
    
    # Sampling parameters
    parser.add_argument(
        '--variation_range',
        type=float,
        default=0.01,
        help='Range for random variation in base pairs count (default: 0.01)'
    )
    parser.add_argument(
        '--no_save',
        action='store_true',
        help='Do not save coordinate files (default: False)'
    )
    parser.add_argument(
        '--no_plots',
        action='store_true',
        help='Do not show plots (default: False)'
    )
    
    args = parser.parse_args()
    
    try:
        # Validate inputs
        if args.num_samples <= 0:
            raise ValueError("Number of samples must be positive")
        
        # Validate mode-specific arguments
        if args.unconditional:
            # Unconditional mode validation
            if args.num_base_pairs is None:
                raise ValueError("--num_base_pairs is required when using --unconditional mode")
            if args.num_base_pairs <= 0:
                raise ValueError("Number of base pairs must be positive")
            
            # Set defaults for unconditional mode
            config_path = args.config_path or 'configs/Uncondition.txt'
            checkpoint_path = args.checkpoint_path or 'pretrained_models/Uncondition.pt'
            design_name = args.design_name or "unconditional"
            
            logger.info("Running in UNCONDITIONAL mode")
            logger.info(f"Number of base pairs: {args.num_base_pairs}")
            
        else:
            # Conditional mode validation
            if args.target_shape is None:
                raise ValueError("--target_shape is required for conditional mode (or use --unconditional)")
            
            # Set defaults for conditional mode
            config_path = args.config_path or 'configs/Condition.txt'
            checkpoint_path = args.checkpoint_path or 'pretrained_models/Condition.pt'
            
            # Detect shape type from file extension
            shape_type = detect_shape_type(args.target_shape)
            logger.info(f"Detected shape type: {shape_type}")
            
            # Validate file exists
            if not validate_file_path(args.target_shape, shape_type):
                raise FileNotFoundError(f"Target shape file not found: {args.target_shape}")
            
            # Get design name from filename if not provided
            design_name = args.design_name or Path(args.target_shape).stem
            
            logger.info("Running in CONDITIONAL mode")
            logger.info(f"Target shape: {args.target_shape}")
        
        # Initialize the DNA structure generator
        logger.info("Initializing DNA Structure Generator...")
        generator = DNAStructureGenerator(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=args.device
        )
        
        if args.unconditional:
            # Unconditional generation workflow
            logger.info(f"Generating {args.num_samples} unconditional DNA structure samples...")
            results = generator.generate_samples_unconditional(
                num_base_pairs=args.num_base_pairs,
                num_samples=args.num_samples,
                output_dir=args.output_dir,
                design_name=design_name,
                save_coordinates=not args.no_save,
                show_plots=not args.no_plots,
                variation_range=args.variation_range
            )
            
            # Print results summary
            print("\n" + "="*60)
            print("🎉 UNCONDITIONAL DNA STRUCTURE GENERATION COMPLETE!")
            print("="*60)
            print(f"📁 Results saved in: {results['output_path']}")
            print(f"📊 Generated {len(results['generated_files'])} samples for design: {design_name}")
            print(f"🧬 Base pairs: {args.num_base_pairs}")
            
        else:
            # Conditional generation workflow
            # Generate shape
            logger.info(f"Loading target shape: {args.target_shape}")
            shape_params = get_shape_params(shape_type, args.target_shape, args)
            target_shape, num_base_pairs = generator.generate_shape(shape_type, **shape_params)
            
            # Visualize target shape
            if not args.no_plots:
                logger.info("Visualizing target shape...")
                generator.visualize_shape(target_shape, f"Target Shape: {design_name}")
            
            # Generate samples
            logger.info(f"Generating {args.num_samples} conditional DNA structure samples...")
            results = generator.generate_samples(
                target_shape=target_shape,
                num_base_pairs=num_base_pairs,
                num_samples=args.num_samples,
                output_dir=args.output_dir,
                design_name=design_name,
                save_coordinates=not args.no_save,
                show_plots=not args.no_plots,
                variation_range=args.variation_range,
                target_shape_path=args.target_shape
            )

            # Print results summary
            print("\n" + "="*60)
            print("🎉 CONDITIONAL DNA STRUCTURE GENERATION COMPLETE!")
            print("="*60)
            print(f"📁 Results saved in: {results['output_path']}")
            print(f"📊 Generated {len(results['generated_files'])} samples for design: {design_name}")
            print(f"🧬 Target shape: {args.target_shape}")
            print(f"🔬 Base pairs: {num_base_pairs}")
        
        if results['generated_files']:
            print("\n📂 Generated files:")
            for filename in results['generated_files']:
                print(f"  • {filename}")
        
        print("="*60)
        print("✅ Generation successful!")
        
    except KeyboardInterrupt:
        logger.info("Generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 