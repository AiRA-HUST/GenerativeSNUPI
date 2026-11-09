import os
import torch
import glob
from pathlib import Path
from datetime import date
from automated_design.gen_bild import *
from automated_design.gen_cando import *
from automated_design.gen_pyg import *
from automated_design.auto_design import automated_DnaOrigami
from tacox.CanDo_oxDNA import load_and_convert as CanDo_oxDNA
from tacox.oxDNA_PDB import oxDNA_PDB

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def route_generated_design(pt_file_path, 
                          min_COs_distance=15,
                          min_pairing_len=4,
                          additional_SSseq=1,
                          adaptive_staples=False,
                          adjust_short_edges=4,
                          adjust_long_edges=4,
                          show=False,
                          enable_tgavr=True,
                          target_shape=None,
                          tgavr_config=None,
                          tgavr_seed=0):
    """
    Process a single .pt file and generate DNA origami design files.

    Args:
        pt_file_path (str): Path to the .pt file to process
        min_COs_distance (int): Minimum crossover distance parameter
        min_pairing_len (int): Minimum pairing length parameter
        additional_SSseq (int): Additional single-strand sequence parameter
        adaptive_staples (bool): Whether to use adaptive staples
        adjust_short_edges (int): Short edge adjustment parameter
        adjust_long_edges (int): Long edge adjustment parameter
        show (bool): Whether to show design states/visualization
        enable_tgavr (bool): Target-Guided Adaptive Vertex Repair (default True).  The
            original routing runs first; only if it raises or fails validation (e.g. a
            staple > 50 bp) is the design repaired and routed again.  Designs the
            original routing handles are therefore exported exactly as before.
            False -> original routing only.
        target_shape (str): Target of the design: a .ply/.stl file, a .pt centre-line
            target (topology recovered), or a file name in target_shapes/.  None ->
            no vertex repair; the router-guided repairs (no target needed) still run.
        tgavr_config (automated_design.tgavr.TGAVRConfig): TGAVR parameters; default
            TGAVRConfig.recommended() (benchmark method M8, 15 routing calls)
        tgavr_seed (int): router seed used by TGAVR (0 = the benchmark setting)

    Returns:
        bool: True if successful, False if failed
    """
    
    # Check if the input file exists
    if not os.path.exists(pt_file_path):
        print(f"Error: File {pt_file_path} does not exist!")
        return False
    
    # Check if it's a .pt file
    if not pt_file_path.endswith('.pt'):
        print(f"Error: File {pt_file_path} is not a .pt file!")
        return False
    
    # Get the directory of the input file for exporting
    export_folder = os.path.dirname(pt_file_path)
    file_basename = os.path.splitext(os.path.basename(pt_file_path))[0]
    
    print(f"Processing file: {pt_file_path}")
    print(f"Export folder: {export_folder}")
    print(f"Base filename: {file_basename}")
    
    try:
        # Load coordinates
        coords = torch.load(pt_file_path, map_location=torch.device('cpu'))
        coords = coords[:,-1,:] if len(coords.size()) > 2 else coords
        
        print(f"Loaded coordinates with shape: {coords.shape}")
        
        if enable_tgavr:
            # Original routing first; TGAVR repair only if it fails validation
            dnaGeom, dnaTop, \
            coords, triads, bp_Hindex, \
            coords_scaf, scaf_route, coords_stap, node_scaf_seq, \
            stap_routes, unbound_nucle_coords = _route_with_tgavr(
                coords, target_shape, tgavr_config, export_folder, file_basename, tgavr_seed,
                show,
                dict(min_COs_distance=min_COs_distance,
                     min_pairing_len=min_pairing_len,
                     additional_SSseq=additional_SSseq,
                     adaptive_staples=adaptive_staples,
                     adjust_short_edges=adjust_short_edges,
                     adjust_long_edges=adjust_long_edges))
        else:
            # Process through automated DNA origami design
            dnaGeom, dnaTop, \
            coords, triads, bp_Hindex, \
            coords_scaf, scaf_route, coords_stap, node_scaf_seq, \
            stap_routes, unbound_nucle_coords = automated_DnaOrigami(coords,
                                               min_COs_distance=min_COs_distance,
                                               min_pairing_len=min_pairing_len,
                                               additional_SSseq=additional_SSseq,
                                               adaptive_staples=adaptive_staples,
                                               adjust_short_edges=adjust_short_edges,
                                               adjust_long_edges=adjust_long_edges,
                                               show=show)
        
        print("✓ DNA origami design completed successfully")
        
        # Generate PyG data
        dna_mean, dna_seq = gen_pyg(scaf_route, stap_routes, bp_Hindex, 
                                    coords_scaf, coords_stap, 
                                    coords, triads, node_scaf_seq, 
                                    add_seqlen=1)
        
        print("✓ PyG data generation completed")
        
        # Save mean and sequence tensors
        pt_mean_path = os.path.join(export_folder, f"{file_basename}_mean.pt")
        torch.save(dna_mean, pt_mean_path)
        print(f"✓ Saved: {pt_mean_path}")
        
        pt_seq_path = os.path.join(export_folder, f"{file_basename}_seq.pt")
        torch.save(dna_seq, pt_seq_path)
        print(f"✓ Saved: {pt_seq_path}")
        
        # Generate output files
        cndo_file_path = os.path.join(export_folder, f"{file_basename}.cndo")
        save_dna_info_to_cando_file(cndo_file_path, dnaTop, dnaGeom)
        print(f"✓ Saved: {cndo_file_path}")
        
        bild_file_path = os.path.join(export_folder, f"{file_basename}.bild")
        save_dna_info_to_bild_file(bild_file_path, coords_scaf, scaf_route, coords_stap, stap_routes, unbound_nucle_coords)
        print(f"✓ Saved: {bild_file_path}")
        
        # Generate oxDNA files
        try:
            opts = {"cando_file" : os.path.join(export_folder, file_basename),
                    "box" : 800.,
                    "print_force_file" : False}
            CanDo_oxDNA(opts)
            print(f"✓ Generated oxDNA files: {file_basename}.dat, {file_basename}.top")
        except Exception as e:
            print(f"⚠ Warning: Could not generate oxDNA files: {e}")
        
        # Generate PDB file
        try:
            opts = {"configuration" : os.path.join(export_folder, f"{file_basename}.dat"),
                    "topology" : os.path.join(export_folder, f"{file_basename}.top"),
                    "oxDNA_direction" : False,
                    "print_hydrogens" : False,
                    "uniform_residue_names" : False,
                    "one_file_per_strand" : False,
                    "rmsf_bfactor" : ""}
            oxDNA_PDB(opts)
            print(f"✓ Generated PDB file: {file_basename}.dat.pdb")
        except Exception as e:
            print(f"⚠ Warning: Could not generate PDB file: {e}")
        
        print(f"\n🎉 SUCCESS: DNA origami routing completed for {os.path.basename(pt_file_path)}")
        print(f"All design files exported to: {export_folder}")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED: Could not process {os.path.basename(pt_file_path)}")
        print(f"Error details: {e}")
        return False


def _route_with_tgavr(coords, target_shape, tgavr_config, export_folder, file_basename,
                      seed, show, routing_kwargs):
    """Route with the unchanged automated_DnaOrigami; if the result is not valid, run
    TGAVR (repair + routing) and save its report.  Returns the router outputs."""
    import json
    from automated_design.tgavr import tgavr_route, TGAVRConfig, run_router, validate_routing
    # 1. original routing, exactly as without TGAVR (same call, no re-seeding)
    out0, info0 = run_router(coords, routing_kwargs, seed=None, show=show)
    val0 = validate_routing(out0, info0)
    if val0['success']:
        print("TGAVR: original routing is valid, no repair needed")
        return out0
    print(f"TGAVR: original routing failed ({val0['category']}), repairing the design")
    # 2. repair and route again
    cfg = tgavr_config or TGAVRConfig.recommended()
    try:
        out, coords_used, report = tgavr_route(coords, target_shape, routing_kwargs,
                                               cfg=cfg, seed=seed)
    except Exception as e:
        if out0 is None:
            raise
        print(f"⚠ Warning: TGAVR failed ({e}); exporting the original routing")
        return out0
    clean = {k: v for k, v in report.items() if not k.startswith('_')}
    report_path = os.path.join(export_folder, f"{file_basename}_tgavr.json")
    with open(report_path, 'w') as f:
        json.dump(clean, f, indent=1, default=str)
    val = clean.get('validation', {})
    print(f"TGAVR: vertex repair: {clean.get('reason')} | routing calls: "
          f"{clean.get('routing_calls')} | validation: {val.get('category')} "
          f"(max staple {val.get('max_staple')} bp, scaffold {val.get('scaffold_nt')} nt) | "
          f"report: {report_path}")
    if val.get('scaffold_fits_M13') is False:
        print(f"⚠ Warning: scaffold of {val.get('scaffold_nt')} nt is longer than the longest "
              f"shipped scaffold sequence (7560 nt)")
    if clean.get('geometry_changed'):
        torch.save(coords_used, os.path.join(export_folder, f"{file_basename}_tgavr_coords.pt"))
    if out is None:
        if out0 is not None:        # keep what the original routing produced
            print("⚠ Warning: TGAVR found no routable design; exporting the original routing "
                  f"({val0['category']})")
            return out0
        raise RuntimeError(f"routing failed after TGAVR: {clean['attempts'][-1].get('exception')}")
    if not val.get('success'):
        print(f"⚠ Warning: routed design fails validation ({val.get('category')})")
    return out


def route_generated_design_batch(subfolder_name,
                          min_COs_distance=15,
                          min_pairing_len=4,
                          additional_SSseq=1,
                          adaptive_staples=False,
                          adjust_short_edges=4,
                          adjust_long_edges=4,
                          show=False,
                          enable_tgavr=True,
                          target_shape=None,
                          tgavr_config=None,
                          tgavr_seed=0):
    """
    Process .pt files from a generated_designs subfolder (legacy function).
    
    Args:
        subfolder_name (str): Name of the subfolder in generated_designs 
                             (e.g., "01_tetrahedron_20250704_120712")
        Other parameters: DNA origami design parameters and the optional TGAVR
            settings of route_generated_design (enabled by default)
    
    Returns:
        bool: True if successful, False if all files failed
    """
    
    # Set up paths - handle both cases: running from dnadiff/ or from routing/
    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")
    
    # Check if we're in the routing directory or the parent dnadiff directory
    if os.path.basename(cwd) == "routing":
        # We're in routing/, go up one level to dnadiff/
        dnadiff_dir = os.path.dirname(cwd)
    else:
        # We're already in dnadiff/ or another directory
        dnadiff_dir = cwd
    
    print(f"Looking for generated_designs in: {dnadiff_dir}")
    
    generated_designs_path = os.path.join(dnadiff_dir, "generated_designs", subfolder_name)
    print(f"Full path: {generated_designs_path}")
    
    if not os.path.exists(generated_designs_path):
        print(f"Error: Folder {generated_designs_path} does not exist!")
        print(f"Checking directory structure:")
        try:
            if os.path.exists(dnadiff_dir):
                items = os.listdir(dnadiff_dir)
                print(f"Contents of {dnadiff_dir}: {items}")
                
                gen_designs_full = os.path.join(dnadiff_dir, "generated_designs")
                if os.path.exists(gen_designs_full):
                    subfolders = os.listdir(gen_designs_full)
                    print(f"Available subfolders in generated_designs: {subfolders}")
                else:
                    print(f"generated_designs directory does not exist in {dnadiff_dir}")
            else:
                print(f"Directory {dnadiff_dir} does not exist")
        except Exception as e:
            print(f"Could not list directory contents: {e}")
        return False
    
    # Find all .pt files in the subfolder
    pt_files = glob.glob(os.path.join(generated_designs_path, "*.pt"))
    
    if not pt_files:
        print(f"Error: No .pt files found in {generated_designs_path}")
        return False
    
    print(f"Found {len(pt_files)} .pt files in {subfolder_name}")
    
    # Try each .pt file until one succeeds
    for i, pt_file in enumerate(pt_files):
        print(f"\nTrying file {i+1}/{len(pt_files)}: {os.path.basename(pt_file)}")
        
        success = route_generated_design(pt_file, 
                                       min_COs_distance=min_COs_distance,
                                       min_pairing_len=min_pairing_len,
                                       additional_SSseq=additional_SSseq,
                                       adaptive_staples=adaptive_staples,
                                       adjust_short_edges=adjust_short_edges,
                                       adjust_long_edges=adjust_long_edges,
                                       show=show,
                                       enable_tgavr=enable_tgavr,
                                       target_shape=target_shape,
                                       tgavr_config=tgavr_config,
                                       tgavr_seed=tgavr_seed)
        
        if success:
            print(f"\n✓ Successfully processed {os.path.basename(pt_file)} and exported design files!")
            return True
    
    print(f"\n✗ All {len(pt_files)} files failed to process successfully.")
    return False

# Example usage - you can uncomment and modify this section
if __name__ == "__main__":
    # Example 1: Process a single .pt file
    pt_file = "generated_designs/01_tetrahedron_20250704_120712/01_tetrahedron_sample_002.pt"
    success = route_generated_design(pt_file, show=True)  # show=True to display design states
    
    if success:
        print(f"✓ Successfully generated DNA design from {pt_file}")
    else:
        print(f"✗ Failed to generate DNA design from {pt_file}")
    
    # Example 2: Process all files from a subfolder (batch processing)
    # subfolder = "01_tetrahedron_20250704_120712"
    # success = route_generated_design_batch(subfolder, show=False)
    # 
    # if success:
    #     print(f"✓ Successfully generated DNA design from {subfolder}")
    # else:
    #     print(f"✗ Failed to generate DNA design from {subfolder}")

