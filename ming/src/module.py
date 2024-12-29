import pdb
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def exists(val):
    return val is not None

def cast_tuple(val, repeat = 1):
    return val if isinstance(val, tuple) else ((val,) * repeat)

def geometric_initializer(layer, in_dim):
    nn.init.normal_(layer.weight, mean=np.sqrt(np.pi) / np.sqrt(in_dim), std=0.00001)
    nn.init.constant_(layer.bias, -1)


def first_layer_sine_initializer(layer):
    with torch.no_grad():
        if hasattr(layer, "weight"):
            num_input = layer.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            layer.weight.uniform_(-1 / num_input, 1 / num_input)

def sine_initializer(layer):
    with torch.no_grad():
        if hasattr(layer, "weight"):
            num_input = layer.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            layer.weight.uniform_(
                -np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30
            )

# sin activation
class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)
    
class Model(nn.Module):
    """ x_0-parameterization
    Arguments:
        inp_dim_s: int, synthetic-net's input dimension,  x
        out_dim_s: int, synthetic-out's output dimension, y_noise or y_0
        hidden_dim: int = 512, number of neurons in hidden layers
        inp_dim_m: int, modulation input dim, z or z + t
        n_layers: int = 4, number of layers (total, including first and last)
        geometric_init: bool = True, initialize weights so that output is spherical
        beta: int = 0, if positive, use SoftPlus(beta) instead of ReLU activations
        sine: bool = True, use SIREN activation in the first layer
        all_sine: bool = True, use SIREN activations in all other layers
        skip: bool = True, add a skip connection to the middle layer
        bn: bool = True, use batch normalization
        dropout: float = 0.0, dropout rate
    """
    def __init__(
        self,
        inp_dim_s: int,
        outp_dim_s: int,
        hidden_dim: int,
        inp_dim_m: int,
        n_layers: int,
        geometric_init: bool=True,
        beta: int=0,
        sine: bool=True,
        all_sine: bool=True,
        skip: bool=True,
        bn: bool=True,
        dropout: float=0.0,
    ):
        super().__init__()

        self.n_layers = n_layers
        self.geometric_init = geometric_init
        self.beta = beta
        self.sine = sine
        self.all_sine = all_sine
        self.skip = skip
        self.bn = bn
        self.dropout = dropout

        # Modules
        self.syn = nn.ModuleList()
        self.mod = nn.ModuleList()
        in_dim_s = inp_dim_s 
        out_dim_s = hidden_dim
        in_dim_m = inp_dim_m
        out_dim_m = hidden_dim
        self.modulate_at = []
        
        for i in range(n_layers):
            layer_s = nn.Linear(in_dim_s, out_dim_s) 
            layer_m = nn.Linear(in_dim_m, out_dim_m) 
            # Custom initializations
            if geometric_init:
                if i == n_layers - 1:
                    geometric_initializer(layer_s, in_dim_s)
                    geometric_initializer(layer_m, in_dim_m)
            elif sine:
                if i == 0:
                    first_layer_sine_initializer(layer_s)
                elif all_sine:
                    sine_initializer(layer_s)
                    
            self.syn.append(layer_s)
            if i < n_layers - 1:
                self.mod.append(layer_m)

            # Activation, BN, and dropout
            if i < n_layers - 1:
                if sine:
                    if i == 0:
                        act_s = Sine()
                    else:
                        act_s = Sine() if all_sine else nn.Tanh()
                elif beta > 0:
                    act_s = nn.Softplus(beta=beta)  # IGR uses Softplus with beta=100
                else:
                    act_s = nn.ReLU(inplace=True)
                self.syn.append(act_s)
                
                if beta > 0:
                    act_m = nn.Softplus(beta=beta)  # IGR uses Softplus with beta=100
                else:
                    act_m = nn.ReLU(inplace=True)
                self.mod.append(act_m)
                
                if bn:
                    self.syn.append(nn.LayerNorm(out_dim_s))
                    self.mod.append(nn.LayerNorm(out_dim_m))
                if dropout > 0:
                    self.syn.append(nn.Dropout(dropout))
                    self.mod.append(nn.Dropout(dropout))

            in_dim_m = hidden_dim + inp_dim_m
            in_dim_s = hidden_dim
            # Skip connection
            if i + 1 == int(np.ceil(n_layers / 2)) and skip:
                self.skip_at = len(self.syn)
                in_dim_s += inp_dim_s

            out_dim_s = hidden_dim
            out_dim_m = hidden_dim
            
            self.modulate_at.append(len(self.syn)-1)
            
            if i + 1 == n_layers - 1:
                out_dim_s = outp_dim_s

    def forward(self, feat_s, feat_m):
        '''
        @params
            feat_s: synthetic feat,  x
            feat_m: mod feat, z + time or z
        '''
        s_in = feat_s
        m_in = feat_m
        for i, layer_s in enumerate(self.syn):
            if i+1 != len(self.syn):
                layer_m = self.mod[i]
            else:
                layer_m =  None
            if i == self.skip_at:
                feat_s = torch.cat([feat_s, s_in], dim=-1)
            feat_s = layer_s(feat_s)
            feat_m = layer_m(feat_m) if layer_m is not None else None
            if i in self.modulate_at and feat_m is not None:
                feat_s = feat_s*feat_m
                feat_m = torch.cat([feat_m, m_in], dim=-1)
        return feat_s