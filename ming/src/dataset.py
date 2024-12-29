from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
from typing import Any, Sequence
import os.path as osp
import torch
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_dense_batch, remove_self_loops
from src.utils import get_keigenvecs
from torch_geometric.data import Data, InMemoryDataset
from torch.utils.data import TensorDataset

from .base import GraphBase
from .utils import process_mol_raw
    
class MolDataset(GraphBase):
    def __init__(self, is_train, root, name):
        '''
        @params
            num_debug: int
                num of samples to debug
        '''
        super().__init__(is_train, root, name)

    def process(self):
        train_list, test_list = process_mol_raw(self.name)
        if self.is_train:
            self.store_data(train_list)
        else:
            self.store_data(test_list)   

# from DiGress     
atom_decoder = ['C', 'N', 'S', 'O', 'F', 'Cl', 'Br']
num_atom_type_and_vitrual = 8
max_node_num = 27

def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]
      
def convert2dense(x, edge_index, edge_attr, batch):
    # convert to dense data
    X, mask_node = to_dense_batch(x=x, batch=batch, max_num_nodes=max_node_num)
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    E = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_node_num)
    E = E.argmax(-1).squeeze()  # (n_max, n_max)
    X = F.one_hot(X.squeeze().argmax(-1), num_classes=num_atom_type_and_vitrual).float() # (n_max, 8), including vitr=1 at index=0
    
    return X, E, mask_node # (n_max, 8), (n_max, n_max)

def mol_to_torch_geometric(mol, atom_encoder, smiles=None):
    '''Code from sparsediff
    '''
    adj = torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True))
    edge_index = adj.nonzero().contiguous().T
    bond_types = adj[edge_index[0], edge_index[1]]
    bond_types[bond_types == 1.5] = 4
    edge_attr = bond_types.long()

    node_types = []
    for atom in mol.GetAtoms():
        node_types.append(atom_encoder[atom.GetSymbol()])

    node_types = torch.Tensor(node_types).long()

    data = Data(
        x=node_types,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    return data

class MOSESDataset(InMemoryDataset):
    def __init__(self, root, stage, name):
        self.stage = stage
        self.atom_decoder =  ["C", "N", "S", "O", "F", "Cl", "Br"]
        self.atom_encoder = {"C": 0, "N": 1, "S": 2, "O": 3, "F": 4, "Cl": 5, "Br": 6}
        self.raw_moses_dir = root + '/raw/'
        
        if self.stage == 'train':
            self.file_idx = 0
        elif self.stage == 'val':
            self.file_idx = 1
        else:
            self.file_idx = 2
        
        root = root + name      
        super().__init__(root)
        self.data = torch.load(self.processed_paths[self.file_idx])
        
    @property
    def split_file_name(self):
        return ['train_moses.csv']

    @property
    def split_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        splitting."""
        files = to_list(self.split_file_name)
        return [osp.join(self.raw_moses_dir, f) for f in files]

    @property
    def processed_file_names(self):
        return ['train.pt']

    def process(self):
        RDLogger.DisableLog('rdApp.*')
        types =  {'null': 0, 'C': 1, 'N': 2, 'S': 3, 'O': 4, 'F': 5, 'Cl': 6, 'Br': 7}
        bonds = {'null': 0, BT.SINGLE: 1, BT.DOUBLE: 2, BT.TRIPLE: 3, BT.AROMATIC: 4} # moses having BT.AROMATIC

        path = self.split_paths[self.file_idx]
        smiles_list = pd.read_csv(path)['SMILES'].values

        data_list = []
        for i, smile in enumerate(tqdm(smiles_list)):
            
            mol = Chem.MolFromSmiles(smile)
            
            # removing hydrogen
            mol = Chem.RemoveAllHs(mol)
            # apply Kekulization removing Aromaticity flags 
            Chem.Kekulize(mol)
            
            if mol is not None:
                _data_check = mol_to_torch_geometric(mol, self.atom_encoder)
            
            N = mol.GetNumAtoms()

            type_idx = []
            for atom in mol.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])

            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()]]

            if len(row) == 0:
                continue

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = F.one_hot(edge_type, num_classes=len(bonds)).to(torch.float)
            
            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_attr = edge_attr[perm]
            
            if ((_data_check.x +1) == torch.tensor(type_idx)).all() and (_data_check.edge_attr.sum() == edge_type.sum()): # check code processing from sparsediff

                x = F.one_hot(torch.tensor(type_idx), num_classes=len(types)).float()
                _data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

                X, E, mask_node = convert2dense(_data.x, _data.edge_index, _data.edge_attr, _data.batch) # (n_max, 8), (n_max, n_max)
                mask_edge = E.sum(-1) > 0
                
                assert (mask_node == mask_edge).all(), ValueError('mask_node and mask_edge should be consistent!')

                keigval, keigvec = get_keigenvecs(E.unsqueeze(dim=0), mask_edge.unsqueeze(dim=0), adj_format=False, normalize=False, k=-1)
                data = [X, E, keigval.squeeze(), keigvec.squeeze()]
                
                data  =  data + [i,]
                data_list.append(data)
                
        print(f'total data {len(smiles_list)}, sucessfully processed {len(data_list)}')  
        # legacy code
        x_tensor = torch.cat([i[0].unsqueeze(0) for i in data_list], dim=0)
        adj_tensor = torch.cat([i[1].unsqueeze(0) for i in data_list], dim=0) # with annotated edges
        all_eigval = torch.cat([i[2].unsqueeze(0) for i in data_list], dim=0) # except eigvals = 0
        all_eigvec = torch.cat([i[3].unsqueeze(0) for i in data_list], dim=0) # except eigvals = 0
        indices = torch.cat([torch.Tensor([[i[4]]]) for i in data_list], dim=0) 
        
        torch.save(TensorDataset(x_tensor, adj_tensor, all_eigval, all_eigvec, indices), self.processed_paths[self.file_idx])     