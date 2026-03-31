import torch
import roma
import pandas as pd
import random
import warnings
import numpy as np
from typing import Dict, Set, List
from collections import defaultdict
from torch_geometric.utils import degree, coalesce, remove_self_loops
from automated_design.visual import plot_dna
from automated_design.routing import *
from automated_design.ults import *
from automated_design.snupi_energy import *

def gen_pyg(scaf_route, stap_routes, bp_Hindex,
            sc_coords, st_coords, 
            coords, triads, node_scaf_seq, 
            add_seqlen=1):
    dev = scaf_route.device
    
    # get scaffold ID for each base pair
    _, bp_scafID = torch.sort(scaf_route, )
    
    
    # generate full edges
    edge_index_scaf = torch.vstack((scaf_route[:-1], scaf_route[1:]))
    edge_index_stap = []
    for i in range(len(stap_routes)):
        st_route = stap_routes[i]
        edge_index_stap.append(torch.vstack((st_route[:-1], st_route[1:])))
    edge_index_stap = torch.hstack(edge_index_stap)
    # edge_index = torch.hstack((edge_index_scaf, edge_index_stap))
    # fig = plot_dna(coords, edge_index=edge_index)
    # fig.show()
    
    # generate physical edges following SNUPI definition
    edge_index = torch.hstack((edge_index_scaf, edge_index_stap.flip([0]))) # Flip_edge_index_stap to count double edges
    edge_index_np = edge_index.cpu().detach().numpy()
    edge_index_np, counts = np.unique(edge_index_np, return_counts=True, axis=1)
    edge_index = torch.from_numpy(edge_index_np)
    edge_index = edge_index.to(dev)
    
    # classify edge types
    deg = degree(edge_index[0])
    src, dst = edge_index
    mask_helix_edges = bp_Hindex[src] == bp_Hindex[dst]
    mask_double_edges = torch.from_numpy(counts == 2).to(device=dev)
    deg_edge = deg[src] + deg[dst]
    deg_edge = deg_edge.to(device=dev)
    
    Regular_BP_edges_mask = torch.logical_and(mask_helix_edges, mask_double_edges)
    Nicked_edges_mask = torch.logical_and(mask_helix_edges, ~mask_double_edges)
    HJnick_mask = torch.logical_and(Nicked_edges_mask, deg_edge > 2)
    Nicked_BP_mask = torch.logical_and(Nicked_edges_mask, deg_edge < 3)
    COedges_mask = torch.logical_and(~mask_helix_edges, ~mask_double_edges)
    SSedges_mask = torch.logical_and(~mask_helix_edges, mask_double_edges) 
    
    # fig = plot_dna(coords, edge_index=edge_index[:,Regular_BP_edges_mask])
    # fig.show()
    
    # 1: Regular Base-Pair steps 
    RBedge_index = edge_index[:, Regular_BP_edges_mask]
    RBedge_type = torch.ones(RBedge_index.size(1), dtype=torch.long, device=dev)
    RBedge_numNT = torch.zeros(RBedge_index.size(1), dtype=torch.long, device=dev)
    
    # Nicked Base-Pair steps: 2: staple Nicked BP steps | 3: scaffold Nicked BP steps
    Nickedge_index = edge_index[:, Nicked_BP_mask]
    Nickedge_numNT = torch.zeros(Nickedge_index.size(1), dtype=torch.long, device=dev)
    src_scafID, dst_scafID = bp_scafID[Nickedge_index]
    staple_edge_mask = torch.abs(src_scafID - dst_scafID) != 1.0
    Nickedge_type = 3*torch.ones(Nickedge_index.size(1), dtype=torch.long, device=dev)
    Nickedge_type[staple_edge_mask] = 2 

    # HJ steps: 4: staple | 5: scaffold
    HJedge_index = edge_index[:, HJnick_mask]
    HJedge_numNT = torch.zeros(HJedge_index.size(1), dtype=torch.long, device=dev)
    src_scafID, dst_scafID = bp_scafID[HJedge_index]
    staple_edge_mask = torch.abs(src_scafID - dst_scafID) != 1.0
    HJedge_type = 5*torch.ones(HJedge_index.size(1), dtype=torch.long, device=dev)
    HJedge_type[staple_edge_mask] = 4 
    
    # Double Cross-over steps: 6: scaffold double COs | 7: staple double COs
    COedge_index = edge_index[:, COedges_mask]
    COsrc_scafID, COdst_scafID = bp_scafID[COedge_index]
    stapCO_mask = torch.abs(COsrc_scafID - COdst_scafID) != 1.0
    COedge_type = 6*torch.ones(COedge_index.size(1), dtype=torch.long, device=dev)
    COedge_type[stapCO_mask] = 7
    COedge_numNT = torch.zeros(COedge_index.size(1), dtype=torch.long, device=dev)
    
    # Single Cross-over steps: 8: scaffold single COs | 9: staple single COs
    
    # Single-strand sequence analysis: 10: single-strand
    SSedge_index = edge_index[:, SSedges_mask]
    SSedge_type = 10*torch.ones(SSedge_index.size(1), dtype=torch.long, device=dev)
    src, dst = SSedge_index
    scaf_relvec = sc_coords[dst,:] - sc_coords[src, :]
    scaf_lens = scaf_relvec.norm(dim=-1, keepdim=False) 
    
    stap_relvec = st_coords[dst,:] - st_coords[src, :]
    stap_lens = stap_relvec.norm(dim=-1, keepdim=False) 
    
    # approximate single-strand sequence length
    num_scafNT_SS = torch.ceil(scaf_lens / 0.42) - 1 + add_seqlen
    num_stapNT_SS = torch.ceil(stap_lens / 0.42) - 1 + add_seqlen
    
    # Make sure the minimum value is 1
    num_scafNT_SS[num_scafNT_SS < 1] = 1
    num_stapNT_SS[num_stapNT_SS < 1] = 1
    
    # Stack all 
    edge_index = torch.hstack((RBedge_index, Nickedge_index, HJedge_index, 
                               COedge_index, SSedge_index, SSedge_index))
    edge_type = torch.hstack((RBedge_type, Nickedge_type, HJedge_type, COedge_type, SSedge_type, SSedge_type))
    
    edge_num_NT = torch.hstack((RBedge_numNT, Nickedge_numNT, HJedge_numNT, COedge_numNT, num_scafNT_SS, num_stapNT_SS))   
    
    # Generate edge_attr with mean properties SNUPI
    edge_attr_mean, edge_attr_seq = get_edge_prop(edge_index, edge_type, edge_num_NT, node_scaf_seq)
    
    # fig = plot_dna(coords, triads=triads[:,0,:], edge_index=edge_index)
    # fig.show()
    
    triadsvec = roma.rotmat_to_rotvec(triads.permute(0,2,1)) # 
    x = torch.hstack((coords, triadsvec, bp_Hindex[:,None]))
    g_mean = Data(x=x.to(dev), edge_index=edge_index.to(dev), edge_attr=edge_attr_mean.to(dev), edge_type=edge_type.to(dev), device=dev)
    g_seq = Data(x=x.to(dev), edge_index=edge_index.to(dev), edge_attr=edge_attr_seq.to(dev), edge_type=edge_type.to(dev), device=dev)  

    SE = mech_PE(g_mean.x, g_mean.edge_index, g_mean.edge_attr, g_mean.edge_type)
    print("Initial Strain Energy (mean properties): %.3f" %SE)
    
    SE = mech_PE(g_seq.x, g_seq.edge_index, g_seq.edge_attr, g_seq.edge_type)
    print("Initial Strain Energy (seq properties): %.3f" %SE)

    return g_mean, g_seq


def get_edge_prop(edge_index, edge_type, edge_num_NT, node_scaf_seq):
    
    dev = edge_index.device
    num_edges = edge_index.size(1)
    edge_attr_mean = torch.zeros((num_edges,30), dtype=torch.float32, device=dev)
    edge_attr_seq = torch.zeros((num_edges,30), dtype=torch.float32, device=dev)
    
    prop_mean = pd.read_csv("automated_design/mean_prop_snupi.csv")
    prop_seq = pd.read_csv("automated_design/seq_prop_snupi.csv")
    
    # assign mean properties to edge_attr
    edge_attr_mean[edge_type==1,:] = torch.tensor(prop_mean['1'], dtype=torch.float32, device=dev)
    edge_attr_mean[torch.logical_or(edge_type==2, edge_type==3),:] = torch.tensor(prop_mean['2'], dtype=torch.float32, device=dev)
    edge_attr_mean[torch.logical_or(edge_type==4, edge_type==5),:] = torch.tensor(prop_mean['3'], dtype=torch.float32, device=dev)
    edge_attr_mean[torch.logical_or(edge_type==6, edge_type==7),:] = torch.tensor(prop_mean['11'], dtype=torch.float32, device=dev)
    edge_attr_mean[torch.logical_or(edge_type==8, edge_type==9),:] = torch.tensor(prop_mean['12'], dtype=torch.float32, device=dev)
    
    # assign seq properties to edge_attr
    for i in range(num_edges):
        if edge_type[i] < 6:
            seq = str(edge_type[i].item()) + node_scaf_seq[edge_index[0,i]] + node_scaf_seq[edge_index[1,i]]
        elif edge_type[i] < 8:
            seq = "6" + node_scaf_seq[edge_index[0,i]] + node_scaf_seq[edge_index[1,i]]
        elif edge_type[i] < 10:
            seq = "7" + node_scaf_seq[edge_index[0,i]] + node_scaf_seq[edge_index[1,i]]
            
        edge_attr_seq[i,:] = torch.tensor(prop_seq[seq], dtype=torch.float32, device=dev)
    
    # assign single-strand properties to edge_attr
    SS_prop = get_SSedge_PROP(edge_num_NT[edge_type==10], sampling=False)
    edge_attr_mean[edge_type==10,:] = SS_prop.to(dev)
    edge_attr_seq[edge_type==10,:] = SS_prop.to(dev)
    
    return edge_attr_mean, edge_attr_seq


def get_SSedge_PROP(SS_NT_NUM, sampling=False):
    """_summary_

    Args:
        SS_NT_NUM (numpy array): Number of nt per Single-strand edges (num_SS,)

    Returns:
        SS_PROP_EQ (numpy array): Equilibrium properties for each Single-strand edges (num_SS, 30)
    """
    SS_NT_NUM = SS_NT_NUM.cpu().detach().numpy()
    
    num_SS = SS_NT_NUM.shape[0]
    
    BP_NT_NUM = SS_NT_NUM + 2
    sHJ_IND = np.zeros(num_SS, dtype=np.float32)

    # Default SNUPI parameters
    kB = (1.3806504*(10**(-23))) * ((10**9)*(10**12))
    kBT = kB*300
    SS_EA_ka = 80
    SS_EA_kb = 0.072
    SS_EA_kc = 1.16
    SS_EI_SS = 2.5
    SS_GA = 0
    SS_CF_IND = 1
    
    if sampling:
        SS_LCT1_S = [0.3800, 0.11]
        SS_LCT1_L = [0.6800, 0.29]
        SS_LCT1_k = [0.2000, 0.02]
        SS_LPB_L = [0.6700, 0.15]
        SS_LPB_ka = [5.400, 0.3]
        SS_LPB_kb = [0.2100, 0.02]
        SS_EA_L = [15.0, 2.8]
        SS_EA_H = [710.0, 60]
        SS_GJ_L = [2.0, 1.2]
        SS_GJ_H = [15.0, 3.6]
        SS_GJ_k = [0.3000, 0.12]
    else:
        SS_LCT1_S = [0.3800, 0.0]
        SS_LCT1_L = [0.6800, 0.0]
        SS_LCT1_k = [0.2000, 0.0]
        SS_LPB_L = [0.6700, 0.0]
        SS_LPB_ka = [5.400, 0.0]
        SS_LPB_kb = [0.2100, 0.0]
        SS_EA_L = [15.0, 0.]
        SS_EA_H = [710.0, 0.]
        SS_GJ_L = [2.0, 0.]
        SS_GJ_H = [15.0, 0.]
        SS_GJ_k = [0.3000, 0.]

    # 
    L_CT1_S = get_NRND_INTV(mean=SS_LCT1_S[0], std=SS_LCT1_S[1], intv=SS_LCT1_S[1], num=num_SS)
    L_CT1_L = get_NRND_INTV(mean=SS_LCT1_L[0], std=SS_LCT1_L[1], intv=SS_LCT1_L[1], num=num_SS)
    L_CT1_k = get_NRND_INTV(mean=SS_LCT1_k[0], std=SS_LCT1_k[1], intv=SS_LCT1_k[1], num=num_SS)

    L_PB_L  = get_NRND_INTV(mean=SS_LPB_L[0], std=SS_LPB_L[1], intv=SS_LPB_L[1],  num=num_SS)
    L_PB_ka = get_NRND_INTV(mean=SS_LPB_ka[0], std=SS_LPB_ka[1], intv=SS_LPB_ka[1], num=num_SS)
    L_PB_kb = get_NRND_INTV(mean=SS_LPB_kb[0], std=SS_LPB_kb[1], intv=SS_LPB_kb[1], num=num_SS)

    EA_L = get_NRND_INTV(mean=SS_EA_L[0], std=SS_EA_L[1], intv=SS_EA_L[1], num=num_SS)
    EA_H = get_NRND_INTV(mean=SS_EA_H[0], std=SS_EA_H[1], intv=SS_EA_H[1], num=num_SS)

    EA_ka = SS_EA_ka
    EA_kb = SS_EA_kb
    EA_kc = SS_EA_kc

    EI_SS = SS_EI_SS

    GJ_L = get_NRND_INTV(mean=SS_GJ_L[0], std=SS_GJ_L[1], intv=SS_GJ_L[1], num=num_SS)
    GJ_H = get_NRND_INTV(mean=SS_GJ_H[0], std=SS_GJ_H[1], intv=SS_GJ_H[1], num=num_SS)
    GJ_k = get_NRND_INTV(mean=SS_GJ_k[0], std=SS_GJ_k[1], intv=SS_GJ_k[1], num=num_SS)

    GA = SS_GA

    # Contour length per nucleotide
    L_CT1 = L_CT1_S * np.exp(-L_CT1_k*(BP_NT_NUM - 3)) + L_CT1_L * (1 - np.exp(-L_CT1_k*(BP_NT_NUM - 3)))

    # Contour length
    H_RAD = 1
    L_CT_sHJ = (L_CT1 + 3*H_RAD)/2
    L_CT_SS = L_CT1 * (BP_NT_NUM-1)
    L_CT = (sHJ_IND == 1)*L_CT_sHJ + (sHJ_IND == 0)*L_CT_SS       
    
    # Persistence length
    L_PB_SS = L_PB_ka * (L_CT - L_CT1_S) * np.exp(-L_PB_kb * (L_CT - L_CT1_S)) + L_PB_L
    L_PB_sHJ = L_PB_L
    L_PB = (sHJ_IND == 1)*L_PB_sHJ + (sHJ_IND == 0)*L_PB_SS

    # Equilibrated normalized length
    eta = 5/2 - 3*L_CT/(4*L_PB_L)
    L_NM_EQ = np.sqrt((eta + np.sqrt(eta**2 + 14))/7)

    # Equilibrated end-to-end length
    L_EE_EQ = L_NM_EQ*L_CT

    # Bending rigidity
    SS_EI = kBT*L_PB;

    # Torsional rigidity
    SS_GJ = (GJ_H - GJ_L) * np.exp(-GJ_k*(L_CT - 2*L_CT1_S)) + GJ_L

    # Stretching rigidity
    EA_k = EA_ka * np.log(EA_kb * L_PB + EA_kc)
    EA_EQ = EA_L / (1 + np.exp(-EA_k * (1 - L_NM_EQ)))   + EA_H/(1 + np.exp(-EA_k * (L_NM_EQ - 1)))

    # Get full single-strand properties
    SS_PROP_EQ = np.zeros((num_SS,30), dtype=np.float32)
    SS_PROP_EQ[:,0] = L_EE_EQ
    SS_PROP_EQ[:,9] = SS_GJ
    SS_PROP_EQ[:,10] = SS_EI
    SS_PROP_EQ[:,11] = SS_EI
    SS_PROP_EQ[:,12] = EA_EQ
    SS_PROP_EQ[:,13] = GA
    SS_PROP_EQ[:,14] = GA
    
    SS_PROP_EQ = torch.from_numpy(SS_PROP_EQ)
    return SS_PROP_EQ

def get_NRND_INTV(mean, std, intv, num):
    nrnd = np.random.normal(mean, std, num)
    out_ind = (nrnd > mean + intv) | (nrnd < mean - intv)
    while np.sum(out_ind) > 0:
        tmp = np.random.normal(mean, std, np.sum(out_ind))
        nrnd[out_ind] = tmp
        out_ind = (nrnd > mean + intv) | (nrnd < mean - intv)
    return nrnd