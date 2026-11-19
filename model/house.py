# -*- coding: utf-8 -*-
'''
@Time    : 2024/10/25 21:32
@Author  : Zekun Cai
@File    : house.py
@Software: PyCharm
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

sys.path.append("..")
from param import *

data_dim = SETTING['House']['data_dim']
gene_dim = SETTING['House']['gene_dim']
embed_dim = SETTING['House']['embed_dim']
n_train = SETTING['House']['n_train']
ode_method = SETTING['House']['ode_method']
rk_step = SETTING['House']['rk_step']

backbone_model = nn.Sequential()

# 原始结构：30 → 400 → 400 → 1（无 house_mlp_hidden）；gene_dim 须与 param 一致
predictive_model = nn.ModuleList([nn.Sequential(
    nn.Linear(data_dim, 400),
    nn.ReLU(),
    nn.Linear(400, 400),
    nn.ReLU(),
    nn.Linear(400, 1)) for i in range(n_train)])

_house_cfg = SETTING['House']
if _house_cfg.get('use_cia', False):
    from model.cia_tabular import CIAFeatureAugment

    _cia_in = predictive_model[0][0].in_features
    cia_module = CIAFeatureAugment(
        _cia_in,
        hidden_dim=_house_cfg.get('cia_hidden_dim', 256),
        reserve_ratio=_house_cfg.get('cia_reserve_ratio', 0.35),
        concrete_temp=_house_cfg.get('cia_concrete_temp', 1.0),
        task_fusion=_house_cfg.get('cia_task_fusion', False),
        fusion_init_logit=_house_cfg.get('cia_fusion_init_logit', -2.0),
        input_layernorm=_house_cfg.get('cia_input_layernorm', False),
    )
else:
    cia_module = None


# Define the Generalized Model, it receives the domain data x and the generalized model parameters,
# then computes the domain data label y.
# The model structure is the same as the predictive model.
def generalized_model(domain_x, domain_param):
    # Build parameter structures for generalized model
    weights = {}
    biases = {}
    start_idx = 0
    for name, p in predictive_model[0].state_dict().items():
        end_idx = start_idx + p.numel()
        if name.endswith("bias"):
            biases[name] = domain_param[start_idx:end_idx].view(p.shape)
        elif name.endswith("weight"):
            weights[name] = domain_param[start_idx:end_idx].view(p.shape)
        else:
            raise ValueError('Not defined layer!')
        start_idx = end_idx

    # Forward pass through the generalized parameters
    x = domain_x
    for name, layer in predictive_model[0].named_children():
        if isinstance(layer, nn.Linear):
            weight_name = f"{name}.weight"
            bias_name = f"{name}.bias"
            x = F.linear(x, weights[weight_name], biases[bias_name])
        elif isinstance(layer, nn.ReLU):
            x = F.relu(x)
        elif isinstance(layer, nn.Sigmoid):
            x = torch.sigmoid(x)
        elif isinstance(layer, nn.LogSoftmax):
            x = F.log_softmax(x, dim=layer.dim)
        elif isinstance(layer, nn.Dropout):
            pass  # Dropout is ignored during manual computation
        else:
            raise ValueError('Not defined layer!')

    domain_y = x
    return domain_y


# Define the Encoder and the Decoder
encoder = nn.Sequential(
    nn.Linear(gene_dim, 1024),
    nn.ReLU(),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Linear(128, embed_dim))

decoder = nn.Sequential(
    nn.Linear(embed_dim, 128),
    nn.ReLU(),
    nn.Linear(128, 512),
    nn.ReLU(),
    nn.Linear(512, 1024),
    nn.ReLU(),
    nn.Linear(1024, gene_dim))


# Define the Koopman Operator
class dyna_f(nn.Module):
    def __init__(self):
        super(dyna_f, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim, bias=False),
        )

    def forward(self, t, y):
        return self.net(y)
