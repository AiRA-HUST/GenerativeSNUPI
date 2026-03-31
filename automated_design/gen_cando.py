import torch
import random
import warnings
import numpy as np
from typing import Dict, Set, List
from collections import defaultdict
from torch_geometric.utils import degree, coalesce, remove_self_loops
from automated_design.visual import plot_dna
from automated_design.routing import *
from automated_design.ults import *
    
#%%    
def gen_dnageom(coords, triads):
    """_summary_

    Args:
        coords (Tensor): Base-pairs coordinates [Nx3]
        triads (Tensor): Base-pairs orientation [Nx3x3] 

    Returns:
        dnaGeom (struc): Dna structure geometry
    """
    num_bps = coords.size(0)
    coords_np = coords.detach().to('cpu').numpy() # covert to angstroms
    triads_np = triads.detach().to('cpu').numpy()

    # dna structure geometry
    dnaGeom = DNAGeom(num_bps=num_bps)
    for bp_id in range(num_bps):
        dnaGeom.dNode[bp_id, :] = coords_np[bp_id,:]
        dnaGeom.triad[:, 2, bp_id] = triads_np[bp_id, 0, :] 
        dnaGeom.triad[:, 0, bp_id] = triads_np[bp_id, 1, :] # Please check it later (Maybe we need minus (-) for right side)
        dnaGeom.triad[:, 1, bp_id] = triads_np[bp_id, 2, :]
        dnaGeom.id_nt[bp_id] = [int(1 + 2*bp_id), int(2*(bp_id + 1))]
    return dnaGeom

#%%
def gen_dnatop(SCroute, STroutes, SSedge_index, 
               coords_scaf, coords_stap, coords=None, 
               add_seqlen=0, scaf_unbound=True):
    
    SCroute_Nidx, STroutes_Nidx, \
    SCroute_bpidx, STroutes_bpidx, \
    scaffold_seq, staples_seq, BP_scaf_nucle, \
        scaf_SS_coords, stap_SS_coords = assign_nucleidx_to_route(  SCroute, 
                                                                    STroutes, 
                                                                    SSedge_index,
                                                                    coords_scaf, 
                                                                    coords_stap, 
                                                                    add_seqlen=add_seqlen,
                                                                    scaf_unbound=scaf_unbound)
    # fig = plot_dna(coords, edge_index=torch.vstack((SCroute[:-1], SCroute[1:])), node_labels=BP_scaf_nucle)
    # fig.show()
    
    dnaTop = []
    ids = []
    # scaffold
    scaf_len = len(SCroute_Nidx)
    for i in range(scaf_len):
        id = SCroute_Nidx[i]
        up = -1 if i == 0 else SCroute_Nidx[i-1]
        down = -1 if i == scaf_len - 1 else SCroute_Nidx[i+1]
        across = 2*(SCroute_bpidx[i] + 1) if SCroute_bpidx[i] > -1 else -1
        seq = scaffold_seq[i]
        dnaTop.append(DnaTop(id, up, down, across, seq))
        ids.append(id)

    # staples
    num_staples = len(STroutes_Nidx)
    for i in range(num_staples):
        
        stap_len = len(STroutes_Nidx[i])
        stap_seq = staples_seq[i]
        stap_Nidx = STroutes_Nidx[i]
        stap_bpidx = STroutes_bpidx[i]
        
        for j in range(stap_len):
            id = stap_Nidx[j]
            up = -1 if j == 0 else stap_Nidx[j-1]
            down = -1 if j == (stap_len-1) else stap_Nidx[j+1]
            across = (1 + 2*stap_bpidx[j]) if stap_bpidx[j] > -1 else -1
            seq = stap_seq[j]  
            dnaTop.append(DnaTop(id, up, down, across, seq))
            ids.append(id)
            
    # # sorted id        
    # ids = torch.hstack(ids)
    # _, indices = torch.sort(ids)
    # sorted_dnaTop = []
    # for i in range(ids.size(0)):
    #     sorted_dnaTop.append(dnaTop[indices[i]])
        
    return dnaTop, scaf_SS_coords, stap_SS_coords, BP_scaf_nucle


def save_dna_info_to_cando_file(cando_filename, dnaTop, dnaGeom):
    """
    Converts dnaInfo into a CanDo text file format.
    See http://cando-dna-origami.org/cndo-file-converter/ for file format

    WARNING!  While all of the internal bits here are 0-indexed, the
    output file needs to be 1-indexed.
    """

    def handle_1_indexing(val):
        one_indexed = True
        if one_indexed:
            val = int(val) + 1 if val >= 0 else val
        return val

    fid = open(cando_filename, 'w')

    # File header
    fid.write(
        '"CanDo (.cndo) file format version 1.0, Keyao Pan, Laboratory '
        'for Computational Biology and Biophysics, Massachusetts '
        'Institute of Technology, November 2015"\n')
    fid.write('\n')

    # dnaInfo.dnaTop
    fid.write('dnaTop,id,up,down,across,seq\n')
    for i, topology in enumerate(dnaTop):
        fid.write('{},{},{},{},{},{}\n'.format(
            handle_1_indexing(i),
            topology.id,
            topology.up,
            topology.down,
            topology.across,
            topology.seq))
    fid.write('\n')

    # dnaInfo.dnaGeom.dNode
    fid.write('dNode,"e0(1)","e0(2)","e0(3)"\n')
    for i, node in enumerate(dnaGeom.dNode):
        fid.write('{},{},{},{}\n'.format(
            handle_1_indexing(i),
            node[0],
            node[1],
            node[2]))
    fid.write('\n')

    # dnaInfo.dnaGeom.triad
    fid.write('triad,"e1(1)","e1(2)","e1(3)","e2(1)",'
                '"e2(2)","e2(3)","e3(1)","e3(2)","e3(3)"\n')
    for i in range(dnaGeom.triad.shape[-1]):
        triad = dnaGeom.triad[:, :, i]
        fid.write('{},{},{},{},{},{},{},{},{},{}\n'.format(
            handle_1_indexing(i),
            triad[0, 0],
            triad[1, 0],
            triad[2, 0],
            
            triad[0, 1],
            triad[1, 1],
            triad[2, 1],
            
            triad[0, 2],
            triad[1, 2],
            triad[2, 2]))
    fid.write('\n')

    # dnaInfo.dnaGeom.id_nt
    fid.write('id_nt,id1,id2\n')
    for i, id_nt in enumerate(dnaGeom.id_nt):
        fid.write('{},{},{}\n'.format(
            handle_1_indexing(i),
            int(id_nt[0]),
            int(id_nt[1])))

    fid.close()
    
#%%
class DNAGeom(object):
    def __init__(self, num_bps):
        self.dNode = np.zeros(shape=(num_bps, 3))
        self.triad = np.zeros(shape=(3, 3, num_bps))
        self.id_nt = np.zeros(shape=(num_bps, 2))

class DnaTop(object):
    def __init__(self, ident, up, down, across, seq):
        self.id = ident
        self.up = up
        self.down = down
        self.across = across
        self.seq = seq

    def __str__(self):
        s = "<DnaTop Object. id={}, up={}, down={}, across={}, seq={}>".format(
            self.id, self.up, self.down, self.across, self.seq
        )
        return s

    def __repr__(self):
        return self.__str__()
