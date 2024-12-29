import os, json, pdb
import numpy as np
import torch
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
import networkx as nx
import re
from .eden import vectorize
from sklearn.metrics.pairwise import pairwise_kernels

DATA_DIR = os.environ['DATA_DIR']
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}
bond_decoder = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE, 4: Chem.rdchem.BondType.AROMATIC}
AN_TO_SYMBOL = {6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}


def load_mol(filepath):
    print(f'Loading file {filepath}')
    if not os.path.exists(filepath):
        raise ValueError(f'Invalid filepath {filepath} for dataset')
    load_data = np.load(filepath)
    result = []
    i = 0
    while True:
        key = f'arr_{i}'
        if key in load_data.keys():
            result.append(load_data[key])
            i += 1
        else:
            break
    return list(map(lambda x, a: (x, a), result[0], result[1]))

def compute_laplacian(adjacency, adj_format, normalize):
    """
    adjacency : batch of annotated adjacency matrix (bs, n, n), elements contain values > 1
    normalize: can be None, 'sym' or 'rw' for the combinatorial, symmetric normalized or random walk Laplacians
    adj_format: bool
        if only use adj without annotated edge info.
    Return:
        L (n x n ndarray): combinatorial or symmetric normalized Laplacian.
    """    
    if adj_format:
        adjacency = adjacency.gt(0.5)*1.0

    diag = torch.sum(adjacency, dim=-1)     # (bs, n)
    n = diag.shape[-1]
    D = torch.diag_embed(diag)      # Degree matrix      # (bs, n, n)
    combinatorial = D - adjacency                        # (bs, n, n)

    if not normalize:
        return (combinatorial + combinatorial.transpose(1, 2)) / 2
    
    diag0 = diag.clone()
    diag[diag == 0] = 1e-12
    diag_norm = 1 / torch.sqrt(diag)            # (bs, n)
    D_norm = torch.diag_embed(diag_norm)        # (bs, n, n)
    L = torch.eye(n, device=adjacency.device).unsqueeze(0) - D_norm @ adjacency @ D_norm
    L[diag0 == 0] = 0
    return (L + L.transpose(1, 2)) / 2

def get_eigenvalues_features(eigenvalues, k):
    """
    values : eigenvalues -- (bs, n)
    node_mask: (bs, n)
    k: num of non zero eigenvalues to keep
    """
    ev = eigenvalues
    bs, n = ev.shape
    n_connected_components = (ev < 1e-5).sum(dim=-1)
    assert (n_connected_components > 0).all(), (n_connected_components, ev)
    
    if k == -1:
        k = n - max(n_connected_components) # take all except egval=0

    to_extend = max(n_connected_components) + k - n
    
    if to_extend > 0:
        eigenvalues = torch.hstack((eigenvalues, 2 * torch.ones(bs, to_extend).type_as(eigenvalues)))
    indices = torch.arange(k).type_as(eigenvalues).long().unsqueeze(0) + n_connected_components.unsqueeze(1)
    first_k_ev = torch.gather(eigenvalues, dim=1, index=indices)
    return n_connected_components.unsqueeze(-1), first_k_ev

def get_eigenvectors_features(vectors, node_mask, n_connected, k):
    """
    vectors (bs, n, n) : eigenvectors of Laplacian IN COLUMNS
    returns:
        not_lcc_indicator : indicator vectors of largest connected component (lcc) for each graph  -- (bs, n, 1)
        k_lowest_eigvec : k first eigenvectors for the largest connected component   -- (bs, n, k)
    """
    bs, n = vectors.size(0), vectors.size(1)

    # Create an indicator for the nodes outside the largest connected components
    first_ev = torch.round(vectors[:, :, 0], decimals=3) * node_mask                        # bs, n
    
    # Add random value to the mask to prevent 0 from becoming the mode
    random = torch.randn(bs, n, device=node_mask.device) * (~node_mask)                                   # bs, n
    first_ev = first_ev + random
    most_common = torch.mode(first_ev, dim=1).values                                    # values: bs -- indices: bs
    mask = ~ (first_ev == most_common.unsqueeze(1))
    not_lcc_indicator = (mask * node_mask).unsqueeze(-1).float()

    # Get the eigenvectors corresponding to the first nonzero eigenvalues
    if k == -1:
        k = (n - max(n_connected)).item() # take all except egval=0

    to_extend = max(n_connected) + k - n
    
    if to_extend > 0:
        vectors = torch.cat((vectors, torch.zeros(bs, n, to_extend).type_as(vectors)), dim=2)   # bs, n , n + to_extend
    indices = torch.arange(k).type_as(vectors).long().unsqueeze(0).unsqueeze(0) + n_connected.unsqueeze(2)    # bs, 1, k
    indices = indices.expand(-1, n, -1)                                               # bs, n, k
    first_k_ev = torch.gather(vectors, dim=2, index=indices)       # bs, n, k
    first_k_ev = first_k_ev * node_mask.unsqueeze(2)

    return not_lcc_indicator, first_k_ev   

def get_keigenvecs(adj_batch, mask, adj_format, normalize, k):
    '''compute laplacian matrix for batched graphs
    @param  adj_batch: {0,..,N}^{b x n_max x n_max}
    @param mask: {False,True}^(b x n_max)
    @param adj_format: bool
        use adj without annotated edge info
    @param normalize: bool
        use normalized laplacian
    @param k: num  of lowest eigvec to keep
    '''
    L = compute_laplacian(adj_batch, adj_format, normalize)
    mask_diag = 2 * L.shape[-1] * torch.eye(adj_batch.shape[-1]).type_as(L).unsqueeze(0)
    mask_diag = mask_diag * (~mask.unsqueeze(1)) * (~mask.unsqueeze(2))
    L = L * mask.unsqueeze(1) * mask.unsqueeze(2) + mask_diag
    eigvals, eigvectors = torch.linalg.eigh(L)

    n_connected_comp, k_lowest_eigenvalues = get_eigenvalues_features(eigenvalues=eigvals, k=k)
    # Retrieve eigenvectors features
    nonlcc_indicator, k_lowest_eigenvector = get_eigenvectors_features(vectors=eigvectors,
                                                                        node_mask=mask,
                                                                        n_connected=n_connected_comp,
                                                                        k=k)
    return k_lowest_eigenvalues, k_lowest_eigenvector


def get_transform_fn(dataset):
    if dataset == 'QM9':
        def transform(g, adj_format, normalize, k):
            # # 6: C, 7: N, 8: O, 9: F
            # # x: (N), adj: (4, N, N) no AROMATIC bond
            x, adj = g
            
            adj = np.concatenate([adj[:3], 1 - np.sum(adj[:3], axis=0, keepdims=True)], axis=0).astype(np.float32)
            adj = adj.argmax(axis=0)
            adj =  torch.tensor(adj)
            # 0:vitrual-edge, 1:S, 2:D, 3:T
            adj = torch.where(adj == 3, 0, adj + 1).to(torch.float32)
            mask_edge = adj.sum(-1) > 0
            
            mask_atom =  x > 0
            num_atom = sum(mask_atom)
            
            # only consider molecule having more than one atom,
            # and have at least one bond type.
            if num_atom > 1 and (mask_atom == mask_edge.numpy()).all():
                x_ = np.zeros((9, 5))
                assert len(x) == 9 & len(adj) == 9
                mask = x >= 6
                indices = x[mask] - 6 + 1
                x_[mask, indices] = 1.
                x_[~mask, 0] = 1. # 0's indx for vitrual node.
                x = torch.tensor(x_).to(torch.float32)
                adj = adj.to(torch.float32)
                # one-hot encoding x
                keigval, keigvec = get_keigenvecs(adj.unsqueeze(dim=0), mask_edge.unsqueeze(dim=0), adj_format, normalize, k) # bs, n, k
                data = [x, adj, keigval.squeeze(), keigvec.squeeze()]
            else:
                data = None
            return data
        
    elif dataset == 'ZINC250k':
        def transform(g, adj_format, normalize, k):
            """
            @param: g
                graph tuple
            @param: k
                k lowest eigvec to keep (-1 to take all)
            """
            x, adj = g
            adj = np.concatenate([adj[:3], 1 - np.sum(adj[:3], axis=0, keepdims=True)], axis=0).astype(np.float32)
            adj = adj.argmax(axis=0)
            adj = torch.tensor(adj)
            
            # 0:vitrual-edge, 1:S, 2:D, 3:T
            adj = torch.where(adj == 3, 0, adj + 1).to(torch.float32)
            mask_edge = adj.sum(-1) > 0
            
            mask_atom =  x > 0
            num_atom = sum(mask_atom)
            
            # 6: C, 7: N, 8: O, 9: F, 15: P, 16: S, 17: Cl, 35: Br, 53: I
            zinc250k_atomic_num_list = [0, 6, 7, 8, 9, 15, 16, 17, 35, 53] # 0's ind for virt node
            if num_atom > 1 and (mask_atom == mask_edge.numpy()).all():
                x_ = np.zeros((38, 10), dtype=np.float32)
                for i in range(38):
                    # zero-vector for virt node
                    if x[i] in zinc250k_atomic_num_list:
                        ind = zinc250k_atomic_num_list.index(x[i])
                        x_[i, ind] = 1.
                    else:
                        x_[i, 0] = 1.
                # one-hot encoding x
                x = torch.tensor(x_).to(torch.float32) # bs, n
                adj = adj.to(torch.float32) # bs, n, n 
                keigval, keigvec = get_keigenvecs(adj.unsqueeze(dim=0), mask_edge.unsqueeze(dim=0), adj_format, normalize, k) # bs, n, k
                data = [x, adj, keigval.squeeze(), keigvec.squeeze()]
            else:
                data = None
            return data 
    return transform


def process_mol_raw(dataset):
    '''
    '''
    dataset = dataset.split('_')[0]
    
    mols = load_mol(os.path.join(DATA_DIR, f'{dataset.lower()}_kekulized.npz'))
    
    with open(os.path.join(DATA_DIR, f'valid_idx_{dataset.lower()}.json')) as f:
        test_idx = json.load(f)
            
    if dataset == 'QM9':
        test_idx = test_idx['valid_idxs']
        test_idx = [int(i) for i in test_idx]
        
    train_idx = [i for i in range(len(mols)) if i not in test_idx]
    print(f'Number of training mols: {len(train_idx)} | Number of test mols: {len(test_idx)}')
    
    train_mols = [mols[i] for i in train_idx]
    test_mols = [mols[i] for i in test_idx]
    
    train_list = []
    test_list = []
    
    transform = get_transform_fn(dataset)
    
    idx_train = 0
    for g in train_mols:
        data = transform(g, adj_format=False, normalize=False, k=-1)
        if data is not None:
            data  =  data + [idx_train,]
            train_list.append(data)
            idx_train += 1

    idx_test = 0 
    for g in test_mols:
        data = transform(g, adj_format=False, normalize=False, k=-1)
        if data is not None:
            data  =  data + [idx_test,]
            test_list.append(data)
            idx_test += 1
    print(f'Number of processed training mols: {len(train_list)} | Number of processed test mols: {len(test_list)}')
    return train_list, test_list

def load_smiles(data_dir, dataset='QM9'):
    if dataset == 'QM9':
        col = 'SMILES1'
    elif dataset == 'ZINC250k':
        col = 'smiles'
    else:
        raise ValueError('wrong dataset name in load_smiles')
    
    df = pd.read_csv(f'{data_dir}/{dataset.lower()}.csv')
    with open(f'{data_dir}/valid_idx_{dataset.lower()}.json') as f:
        test_idx = json.load(f)
        
    if dataset == 'QM9':
        test_idx = test_idx['valid_idxs']
        test_idx = [int(i) for i in test_idx]
    
    train_idx = [i for i in range(len(df)) if i not in test_idx]
    return list(df[col].loc[train_idx]), list(df[col].loc[test_idx])


def canonicalize_smiles(smiles):
    return [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in smiles]

def check_valency(mol):
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find('#')
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r'\d+', e_sub)))
        return False, atomid_valence

def valid_mol_can_with_seg(m, largest_connected_comp=True):
    if m is None:
        return None
    sm = Chem.MolToSmiles(m, isomericSmiles=True)
    if largest_connected_comp and '.' in sm:
        vsm = [(s, len(s)) for s in sm.split('.')]  # 'C.CC.CCc1ccc(N)cc1CCC=O'.split('.')
        vsm.sort(key=lambda tup: tup[1], reverse=True)
        mol = Chem.MolFromSmiles(vsm[0][0])
    else:
        mol = Chem.MolFromSmiles(sm)
    return mol

def correct_mol(m):
    # xsm = Chem.MolToSmiles(x, isomericSmiles=True)
    mol = m
    #####
    no_correct = False
    flag, _ = check_valency(mol)
    if flag:
        no_correct = True

    while True:
        flag, atomid_valence = check_valency(mol)
        if flag:
            break
        else:
            assert len(atomid_valence) == 2
            idx = atomid_valence[0]
            v = atomid_valence[1]
            queue = []
            for b in mol.GetAtomWithIdx(idx).GetBonds():
                queue.append((b.GetIdx(), int(b.GetBondType()), b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
            queue.sort(key=lambda tup: tup[1], reverse=True)
            if len(queue) > 0:
                start = queue[0][2]
                end = queue[0][3]
                t = queue[0][1] - 1
                mol.RemoveBond(start, end)
                if t >= 1:
                    mol.AddBond(start, end, bond_decoder[t])
    return mol, no_correct

def construct_mol(atoms, adj, atomic_num_list):
    '''construct molecule
    @params
        atoms: {R}^(N)
            (0, ..., num_atom_types: vitrual)
        adj: {R}^(N, N)
            (0: vitrual, 1:S, 2:D, 3:T)
            predicted class for each bond
    '''
    mol = Chem.RWMol() 
    atoms_exist = (atoms != 0) 
    atoms = atoms[atoms_exist] 
   
    for atom in atoms:
        mol.AddAtom(Chem.Atom(int(atomic_num_list[atom])))
        
    adj = adj[atoms_exist, :][:, atoms_exist]
    diag_matrix = np.eye(sum(atoms_exist)) < 0.5
    adj = adj * diag_matrix

    for start, end in zip(*np.nonzero(adj)):
        if start > end:
            mol.AddBond(int(start), int(end), bond_decoder[adj[start, end]])
            # add formal charge to atom: e.g. [O+], [N+], [S+]
            # not support [O-], [N-], [S-], [NH+] etc.
            flag, atomid_valence = check_valency(mol)
            if flag:
                continue
            else:
                assert len(atomid_valence) == 2
                idx = atomid_valence[0]
                v = atomid_valence[1]
                an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
                    mol.GetAtomWithIdx(idx).SetFormalCharge(1)
    return mol

def gen_mol(x, adj, dataset, largest_connected_comp=True):    
    '''
    @params
         x_gen: {R}^(B, N) 
            (..., d_x - 1: vitrual)
            generated class probabilities of each class.
        adj_gen: {R}^(B, N, N) 
            (0:S, 1:D, 2:T, 3: virtual)
            generated class probabilities of each class.
    '''
    RDLogger.DisableLog('rdApp.*')
    if dataset == 'QM9':
        atomic_num_list = [0, 6, 7, 8, 9]
    elif dataset == 'ZINC250k':
        atomic_num_list = [0, 6, 7, 8, 9, 15, 16, 17, 35, 53]
    elif dataset == 'MOSES':
        atomic_num_list =[0, 6, 7, 16, 8, 9, 17, 35]
    else:
        raise ValueError(f'dataset {dataset} invalid!')
        
    mols, num_no_correct = [], 0
    for x_elem, adj_elem in zip(x, adj):
        
        x_elem = x_elem.numpy()
        adj_elem = adj_elem.numpy()

        mol = construct_mol(x_elem, adj_elem, atomic_num_list)
        cmol, no_correct = correct_mol(mol)
        if no_correct: num_no_correct += 1
        vcmol = valid_mol_can_with_seg(cmol, largest_connected_comp=largest_connected_comp)
        mols.append(vcmol)
    mols = [mol for mol in mols if mol is not None]
    return mols, num_no_correct

def mols_to_smiles(mols):
    return [Chem.MolToSmiles(mol) for mol in mols]

def mols_to_nx(mols):
    nx_graphs = []
    for mol in mols:
        G = nx.Graph()

        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(),
                       label=atom.GetSymbol())
                    
        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(),
                       bond.GetEndAtomIdx(),
                       label=int(bond.GetBondTypeAsDouble()))
        
        nx_graphs.append(G)
    return nx_graphs

### code adapted from https://github.com/idea-iitd/graphgen/blob/master/metrics/mmd.py
def compute_nspdk_mmd(samples1, samples2, metric, is_hist=True, n_jobs=None):
    def kernel_compute(X, Y=None, is_hist=True, metric='linear', n_jobs=None):
        X = vectorize(X, complexity=4, discrete=True)
        if Y is not None:
            Y = vectorize(Y, complexity=4, discrete=True)
        return pairwise_kernels(X, Y, metric='linear', n_jobs=n_jobs)

    X = kernel_compute(samples1, is_hist=is_hist, metric=metric, n_jobs=n_jobs)
    Y = kernel_compute(samples2, is_hist=is_hist, metric=metric, n_jobs=n_jobs)
    Z = kernel_compute(samples1, Y=samples2, is_hist=is_hist, metric=metric, n_jobs=n_jobs)
    return np.average(X) + np.average(Y) - 2 * np.average(Z)

def nspdk_stats(graph_ref_list, graph_pred_list):
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]
    num_threads =  min(torch.get_num_threads(), 20)
    mmd_dist = compute_nspdk_mmd(graph_ref_list, graph_pred_list_remove_empty, metric='nspdk', is_hist=False, n_jobs=num_threads)
    return mmd_dist

def mask_minibatch(data):
    '''return the current data with the corresponding max
    number of nodes of the minibatch
    '''
    _node_gt, _edge_gt, _, _eigvec, _ = data
    mask = _node_gt[:,:,1:].sum(-1).gt(0) # bs, n
    max_node = mask.sum(dim=-1).max()
    node_gt = _node_gt[:,:max_node,:]
    edge_gt = _edge_gt[:,:max_node,:max_node]
    eigvec = _eigvec[:,:max_node,:]
    return (node_gt, edge_gt, eigvec)

def create_exp_dir(path):
    '''
    '''
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    print('Experiment dir : {}'.format(path))

def check_ckpt_exist(path, is_last=True):
    '''check checkpoint exist
    @params:
        is_last: bool
            check last or best ckpt
    '''
    prefix = 'save_last' if is_last else 'best'
    ckpt_list = [x for x in os.listdir(path) if prefix in x]

    if len(ckpt_list) == 0:
        cont_train = False
        last_ckpt = None
    else:
        cont_train = True
        last_ckpt = prefix+f'-v{len(ckpt_list)-1}.ckpt' if len(ckpt_list) > 1 else ckpt_list[0]

    return cont_train, last_ckpt

def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
  betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float32)
  warmup_time = int(num_diffusion_timesteps * warmup_frac)
  betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float32)
  return betas

def get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps):
  if beta_schedule == 'quad':
    betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float32) ** 2
  elif beta_schedule == 'linear':
    betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float32)
  elif beta_schedule == 'warmup10':
    betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
  elif beta_schedule == 'warmup50':
    betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
  elif beta_schedule == 'const':
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float32)
  elif beta_schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
    betas = 1. / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float32)
  else:
    raise NotImplementedError(beta_schedule)
  assert betas.shape == (num_diffusion_timesteps,)
  return betas