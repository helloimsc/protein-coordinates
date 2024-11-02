import torch, typing, warnings, io, subprocess
import torch.nn as nn
import numpy as np
from Bio.PDB import PDBIO, Chain, Residue, Polypeptide, Atom, PDBParser, Selection
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import logging, socket, os
from protein_residues import normal as RESIDUES
from Bio.SeqUtils import seq1, seq3
import dgl
parser = PDBParser()

#Main Function
def generate_graph(pdb_path, k):
    #PDB to Tensors
    frames, seq = get_backbone(pdb_path)
    frames = torch.tensor(np.array(frames[0]))
    i_seq = encode(seq[0]) #This is a tensor of integers size N
    forward, reverse, side = generate_vector(frames)

    #Construct DGL Graph
    ca = frames[:,1,:]
    dist = torch.cdist(frames[,ca,p=2.0)
    dist, idx=dist.sort()
    idx = idx[:,:k]

    
    # Generate Graph (N.B. Need to remove edge to self)
    n = ca.shape[0]
    row = torch.arange(0, n).repeat(k)
    col = idx.flatten()
    graph = dgl.graph((row,col))

    #Generate Node Features
    forward, reverse, side = generate_vectors(frames)
    graph.ndata['forward'] = forward
    graph.ndata['reverse'] = reverse
    graph.ndata['side'] = side
    
    #Generate Edge Features
    #Euclidean Distance Embedding
    rbf_dist = _rbf(dist[:,:k])
    graph.edata['rbf_dist']=rbf_dist.flatten()

    #Sequential Distance Sinusiodal Embedding
    t = row-col
    t = sc_embed(t.unsqueeze(dim=-1))
    graph.edata['seq_embed'] = t 
    
    # Unit Vector 
    j = ca[row,:]
    i = ca[col,:]
    direction = j-i
    direction = direction/torch.norm(direction,p=2.0,dim=-1)
    graph.edata['u_vec'] = direction
    
    return graph

#Create Pre-defined Variables (Need these to run during import, I'll format later)
extended_protein_letters = "ACDEFGHIKLMNPQRSTVWYBXZJUO"
aa_to_int = { ch:i for i,ch in enumerate(extended_protein_letters) }
int_to_aa = { i:ch for i,ch in enumerate(extended_protein_letters) }
encode = lambda s: torch.tensor([aa_to_int[c] for c in s])
decode = lambda l: ''.join([int_to_aa[i] for i in l])
RESIDUES = PROCESS_RESIDUES(RESIDUES)

#Sub Functions

def PROCESS_RESIDUES(d):
    #X is unknown AA I should check this function to incorporate X
    #Additional AA, B - Asparagine, U - selenocysteine, Z - glutamic
    #acid, and O - ornithine (Doesn't clash with IUPAC convention)
    d['HIS'] = d['HIP']
    d = {key: val for key, val in d.items() if seq1(key) != 'X'}
    for key in d:
        atoms = d[key]['atoms']
        d[key] = {'CA': 'C'} | {key: val['symbol'] for key, val in atoms.items() if val['symbol'] != 'H' and key != 'CA'}
    return d

def get_backbone(pdb_path,model_num=0):
    #Returns a tuple (frames,seq) frames being a list of list of np arrays where each np array is one chain of size n_aa x 4. seq is a list of     #strings where each string is the one letter representation of each amino acid
#in each chain

    model = parser.get_structure('',pdb_path)[model_num]
    seq = []
    frames = []
    for chain in model.child_list:
        chain_coord = []
        chain_seq = ''
        for res in chain:
            chain_seq+=res.resname
            #Might change the Cb portion of Gly
            if res.resname=="GLY":
                if (res.id[0] != ' ') or ('CA' not in res.child_dict) or (res.resname not in RESIDUES): continue
                bb = np.stack((res["N"].coord,res["CA"].coord,res["C"].coord,np.array([0,0,0])),axis=0)
                chain_coord.append(bb)
            else:
                if (res.id[0] != ' ') or ('CA' not in res.child_dict) or (res.resname not in RESIDUES): continue
                bb = np.stack((res["N"].coord,res["CA"].coord,res["C"].coord,res["CB"].coord),axis=0)
                chain_coord.append(bb)
        chain_coord=np.array(chain_coord)
        frames.append(chain_coord)
        seq.append(seq1(chain_seq))
    return frames,seq

def generate_vectors(frame):
    #The frames tensor is of shape (N x 4 x 3) Where for the -2 dim is N, CA, C, CB
    forward = frame[1:,1,:] - frame[:-1,1,:] #This doesn't include the foward vector for the nth aa
    reverse = frame[:-1,1,:] - frame[1:,1,:] #This doesn't include the foward vector for the 1th aa
    side = frame[:,3,:] - frame [:,1,:]
    #Normalizing the vectors
    forward = forward/torch.norm(forward,p=2, dim=-1)
    reverse = reverse/torch.norm(forward,p=2, dim=-1)
    side = side/torch.norm(side,p=2,dim=-1)
    #Padding, might alter it as these aren't unit vectors
    forward = torch.cat((forward,torch.zeros((1,3))),axis=0)
    reverse = torch.cat((torch.zeros((1,3)),reverse),axis=0)
    return (forward,reverse,side)
    
#Amino acids usually have an L chirality
#Nelson, Lehninger; et al. (2008). Lehninger Principles of Biochemistry. Macmillan. p. 474.
# The R/S system also has no fixed relation to the D/L system. For example, the side-chain one of serine contains a hydroxyl group, −OH. If a thiol group, −SH, were swapped in for it, the D/L labeling would, by its definition, not be affected by the substitution. But this substitution would invert the molecule's R/S labeling, because the CIP priority of CH2OH is lower than that for CO2H but the CIP priority of CH2SH is higher than that for CO2H. For this reason, the D/L system remains in common use in certain areas of biochemistry, such as amino acid and carbohydrate chemistry, because it is convenient to have the same chiral label for the commonly occurring structures of a given type of structure in higher organisms. In the D/L system, nearly all naturally occurring amino acids are all L, while naturally occurring carbohydrates are nearly all D. All proteinogenic amino acids are S, except for cysteine, which is R. 

def sc_embed(t,embed_dim,n):
    #NB t should be of shape (n,1) for broadcasting purposes
    sin_embd = torch.sin(t/n**(torch.arange(embed_dim)/(embed_dim)))[None,:]
    cos_embd = torch.cos(t/n**(torch.arange(embed_dim)/(embed_dim)))[None,:]
    #Output = (n,embed_dim)
    return torch.cat((sin_embd,cos_embd),dim =-1).squeeze(dim=0)

def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design
    
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF