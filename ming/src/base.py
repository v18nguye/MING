import os
import json
import warnings
import pickle
import pandas as pd
import torch
import torch.nn as nn
from .molsets import get_all_metrics
from torch.utils.data import TensorDataset
from torch_geometric.data import InMemoryDataset
from src.utils import load_smiles, canonicalize_smiles, gen_mol, mols_to_smiles, mols_to_nx, \
nspdk_stats
from rdkit import Chem

# suppress networkx future warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
DATA_DIR = os.environ['DATA_DIR']

class GraphBase(InMemoryDataset):
    def __init__(self, is_train=False, root=None, name=None):
        self.is_train = is_train
        self.name = name
        root = root + name        
        super().__init__(root)
        self.data = torch.load(self.processed_paths[0])
       
    @property
    def processed_file_names(self):
        if self.is_train:
            return ['train_data.pt']
        else:
            return ['test_data.pt']
    
    def process(self):
        raise NotImplementedError
    
    def store_data(self, data):
        x_tensor = torch.cat([i[0].unsqueeze(0) for i in data], dim=0)
        adj_tensor = torch.cat([i[1].unsqueeze(0) for i in data], dim=0) # with annotated edges
        all_eigval = torch.cat([i[2].unsqueeze(0) for i in data], dim=0) # except eigvals = 0
        all_eigvec = torch.cat([i[3].unsqueeze(0) for i in data], dim=0) # except eigvals = 0
        indices = torch.cat([torch.Tensor([[i[4]]]) for i in data], dim=0) 
        torch.save(TensorDataset(x_tensor, adj_tensor, all_eigval, all_eigvec, indices), self.processed_paths[0])

def remove_invalid_and_H(test_smiles):
    '''removing Hydrogen and invalid smiles on the test set, sparsediff
    applied on moses guacamol
    https://github.com/qym7/SparseDiff/blob/main/sparse_diffusion/metrics/molecular_metrics.py
    '''
    test_smiles_no_h = []
    total_count = 0
    valid_count = 0
    for smile in list(test_smiles):
        total_count += 1
        mol = Chem.MolFromSmiles(smile, sanitize=True)
        if mol is not None:  # wierd thing happens to test_smiles
            valid_count += 1
            for a in mol.GetAtoms():
                if a.GetNumImplicitHs():
                    a.SetNumRadicalElectrons(a.GetNumImplicitHs())
                    a.SetNoImplicit(True)
                    a.UpdatePropertyCache()
            molRemoveAllHs = Chem.RemoveAllHs(mol)
            test_smiles_no_h.append(Chem.MolToSmiles(molRemoveAllHs))
    print('among {} test smiles, {} are valid'.format(total_count, valid_count))
    return test_smiles_no_h

class MolMetric(nn.Module):
    def __init__(self, dataset, data_dir=None):
        super(MolMetric, self).__init__()
        self.dataset = dataset
        if data_dir is None:
            self.data_dir = DATA_DIR
        else:
            self.data_dir = data_dir
        self.train_smiles, self.test_smiles, self.test_graph_list = self._load_data()
        
    def _load_data(self):
        '''load canonicalized smiles
        '''
        if self.dataset != 'MOSES':
            train, test = load_smiles(self.data_dir, self.dataset)
            train = canonicalize_smiles(train)
            test = canonicalize_smiles(test)
            with open(f'{self.data_dir}/{self.dataset.lower()}_test_nx.pkl', 'rb') as f:
                test_graph_list = pickle.load(f) # for NSPDK MMD, smiles -> nx, no kekulization, no canoncalization.
        else:
            train = list(pd.read_csv(f'{self.data_dir}/raw/train_moses.csv')['SMILES'].values)
            
            try:
                with open(f'{self.data_dir}/processed/processed_test_smiles.json') as f:
                    test = json.load(f)
            except:
                test = list(pd.read_csv(f'{self.data_dir}/raw/test_moses.csv')['SMILES'].values)
                test = remove_invalid_and_H(test)
                with open(f'{self.data_dir}/processed/processed_test_smiles.json', 'w') as f:
                    json.dump(test, f)
                       
            test_graph_list = None
            
        return train, test, test_graph_list
    
    def forward(self, x_gen, adj_gen):
        '''
        @params
            x_gen: list (mol's atoms) 
                (0, ..., num_atoms) # 0 -> vitrual
            
            adj_gen: list( mol's adj)
                (0: vitrual, 1:S, 2:D, 3:T)
        '''
        result = {}
        gen_mols, num_mols_wo_correction = gen_mol(x_gen, adj_gen, self.dataset)
        num_mols = len(gen_mols)
        val_no_correct = num_mols_wo_correction / num_mols
        result['valid_wo_correct'] = val_no_correct
        
        gen_smiles = mols_to_smiles(gen_mols)
        gen_smiles = [smi for smi in gen_smiles if len(smi)]
        
        num_threads =  min(torch.get_num_threads(), 20)
        scores = get_all_metrics(gen=gen_smiles, dataset=self.dataset, k=len(gen_smiles), device='cuda', n_jobs=num_threads, test=self.test_smiles, train=self.train_smiles)
        
        if self.dataset != 'MOSES':
            scores_nspdk = nspdk_stats(self.test_graph_list, mols_to_nx(gen_mols))
            result['nspdk_mmd'] = scores_nspdk 
            key_metrics = ['FCD/Test', 'Novelty', 'FCD/Test', 'SNN/Test', 'Frag/Test', 'Scaf/Test']
        else:
             key_metrics = ['FCD/Test', 'Novelty']
             
        for key in scores.keys():
            if 'unique' in key:
                key_metrics.append(key)

        for metric in key_metrics:
            result[metric] = scores[metric]

        return val_no_correct, result