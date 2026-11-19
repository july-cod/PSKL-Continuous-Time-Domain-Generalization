# -*- coding: utf-8 -*-
SETTING = {
    'Moons': {
        'name': '2-Moons',
        'data_file': './data/Moons50/dataset.pkl',
        'data_dim': 2,
        'data_size': 1000,
        'n_domains': 50,  # Number of domains
        'task': 'binary_classification',

        'n_train': 35,  # Number of training domains
        'seg_len': 10,  # Max length of domain sequence
        'n_val_seg': 5,  # Number of validation sequence

        'epoch': 300,
        'batch': 10,
        'pred_learning_rate': 1e-2,
        'coder_learning_rate': 1e-3,
        'dyn_learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'grad_clip_norm': 5.0,

        'alpha': 1,
        'beta': 100,
        'gamma': 10,

        'gene_dim': 2751,  # Raw parameter dimensions of the generalized model
        'embed_dim': 32,  # Embedding dimensions of the generalized model
        'ode_method': 'rk4',
        'rk_step': 0.2,

        'use_cia': True,
        'cia_lambda': 0.1,
        'cia_mix_weight': 0.5,
        'cia_spu_entropy_weight': 0.02,
        'cia_reserve_ratio': 0.35,
        'cia_min_keep_dims': 2,
        'cia_hidden_dim': 256,
        'cia_concrete_temp': 1.0,
        'cia_task_fusion': False,
        'cia_fusion_init_logit': -2.0,
        'cia_input_layernorm': False,
        'cia_main_use_invariant': True,
        'cia_skip_causal_aux': True,
        'random_seed': True,
    },

    'MNIST': {
        'name': 'MNIST',
        'data_file': './data/MNIST50/dataset.pkl',
        'data_dim': 28,
        'data_size': 1000,
        'n_domains': 50,
        'task': 'classification',

        'n_train': 35,
        'seg_len': 10,
        'n_val_seg': 5,

        'epoch': 380,
        'batch': 4,
        'pred_learning_rate': 8e-4,
        'coder_learning_rate': 8e-4,
        'dyn_learning_rate': 8e-4,
        'weight_decay': 2e-4,
        'grad_clip_norm': 3.0,

        'alpha': 1,
        'beta': 100,
        'gamma': 8,

        'gene_dim': 75146,
        'embed_dim': 32,
        'ode_method': 'rk4',
        'rk_step': 0.2,

        'use_cia': True,
        'cia_lambda': 0.08,
        'cia_mix_weight': 0.45,
        'cia_spu_entropy_weight': 0.015,
        'cia_reserve_ratio': 0.45,
        'cia_hidden_dim': 256,
        'cia_concrete_temp': 1.0,
        'cia_task_fusion': False,
        'cia_fusion_init_logit': -2.0,
        'cia_input_layernorm': False,
        'cia_main_use_invariant': True,
        'cia_skip_causal_aux': True,

        'backbone_dropout': 0.5,
        'head_dropout': 0.5,

        'ckpt_val_plain_nll': True,
        'nll_class_weight': True,
        'nll_class_weight_power': 0.15,
        'nll_num_classes': 10,
        'nll_focal_gamma': 0.0,
        'nll_tail_boost_classes': [],
        'nll_tail_boost_factor': 1.0,

        'random_seed': True,
    },

    'Twitter': {
        'name': 'Twitter',
        'data_file': './data/Twitter/dataset.pkl',
        'data_dim': 526,
        'data_size': None, # Variable
        'n_domains': 50,
        'task': 'binary_classification',

        'n_train': 35,
        'seg_len': 10,
        'n_val_seg': 5,

        'epoch': 300,
        'batch': 2,
        'pred_learning_rate': 1e-3,
        'coder_learning_rate': 1e-3,
        'dyn_learning_rate': 1e-3,
        'weight_decay': 0,

        'alpha': 1,
        'beta': 10,
        'gamma': 10,

        'gene_dim': 8385,
        'embed_dim': 32,
        'ode_method': 'rk4',
        'rk_step': 0.2,

        'use_cia': True,
        'cia_lambda': 0.1,
        'cia_mix_weight': 0.5,
        'cia_spu_entropy_weight': 0.02,
        'cia_reserve_ratio': 0.35,
        'cia_hidden_dim': 256,
        'cia_concrete_temp': 1.0,
        'cia_task_fusion': False,
        'cia_fusion_init_logit': -2.0,
        'cia_input_layernorm': False,
        'cia_main_use_invariant': True,
        'cia_skip_causal_aux': True,
        'random_seed': True,
    },

    'YearBook': {
        'name': 'YearBook',
        'data_file': './data/YearBook/dataset.pkl',
        'data_dim': 32,
        'data_size': None,
        'n_domains': 40,
        'task': 'binary_classification',

        'n_train': 28,
        'seg_len': 10,
        'n_val_seg': 5,

        'epoch': 200,
        'batch': 1,
        'pred_learning_rate': 1e-3,
        'coder_learning_rate': 1e-3,
        'dyn_learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'grad_clip_norm': 5.0,

        'alpha': 0.1,
        'beta': 100,
        'gamma': 10,

        'gene_dim': 135361,
        'embed_dim': 32,
        'ode_method': 'rk4',
        'rk_step': 0.2,

        'use_cia': True,
        'cia_lambda': 0.1,
        'cia_mix_weight': 0.5,
        'cia_spu_entropy_weight': 0.02,
        'cia_reserve_ratio': 0.35,
        'cia_hidden_dim': 256,
        'cia_concrete_temp': 1.0,
        'cia_task_fusion': False,
        'cia_fusion_init_logit': -2.0,
        'cia_input_layernorm': False,
        'cia_main_use_invariant': True,
        'cia_skip_causal_aux': True,
        'random_seed': True,
    },

    'Cyclone': {
        'name': 'Cyclone',
        'data_file': './data/Cyclone/dataset.pkl',
        'data_dim': 64,
        'data_size': None,
        'n_domains': 72,
        'task': 'regression',

        'n_train': 50,
        'seg_len': 10,
        'n_val_seg': 5,

        'epoch': 300,
        'batch': 1,
        'pred_learning_rate': 1e-3,
        'coder_learning_rate': 1e-3,
        'dyn_learning_rate': 1e-3,
        'weight_decay': 2e-4,
        'grad_clip_norm': 5.0,

        'alpha': 1,
        'beta': 100,
        'gamma': 100,

        'gene_dim': 135361,
        'embed_dim': 32,
        'ode_method': 'rk4',
        'rk_step': 0.2,

        'backbone_norm': True,
        'feature_layernorm': True,
        'backbone_dropout': 0.55,
        'head_dropout': 0.35,

        'use_cia': True,
        'cia_lambda': 0.028,
        'cia_mix_weight': 0.22,
        'cia_spu_entropy_weight': 0.008,
        'cia_reserve_ratio': 0.55,
        'cia_hidden_dim': 256,
        'cia_concrete_temp': 1.3,
        'cia_task_fusion': False,
        'cia_fusion_init_logit': -2.0,
        'cia_input_layernorm': False,
        'cia_main_use_invariant': True,
        'cia_skip_causal_aux': True,

        'random_seed': True,
    },

    'House': {
        'name': 'House',
        'data_file': './data/House/dataset.pkl',
        'data_dim': 30,
        'data_size': None,
        'n_domains': 40,
        'task': 'regression',

        'n_train': 28,
        'seg_len': 10,
        'n_val_seg': 5,

        'epoch': 370,
        'batch': 2,
        'pred_learning_rate': 8e-4,
        'coder_learning_rate': 8e-4,
        'dyn_learning_rate': 8e-4,
        'weight_decay': 2.5e-4,
        'grad_clip_norm': 5.0,

        'alpha': 1,
        'beta': 15,
        'gamma': 10,

        'gene_dim': 173201,
        'embed_dim': 32,
        'ode_method': 'rk4',
        'rk_step': 0.2,

        'use_cia': True,
        'cia_lambda': 0.005,
        'cia_mix_weight': 0.16,
        'cia_spu_entropy_weight': 0.004,
        'cia_reserve_ratio': 0.45,
        'cia_hidden_dim': 256,
        'cia_concrete_temp': 1.0,
        'cia_task_fusion': False,
        'cia_fusion_init_logit': -2.0,
        'cia_input_layernorm': False,
        'cia_main_use_invariant': True,
        'cia_skip_causal_aux': True,
        'random_seed': True,
    }
}
