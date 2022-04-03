import torch

d_model = 256 # input_size
multi_heads = 8
num_layers_encoder = 2
num_layers_decoder = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
rand_unif_init_mag = 0.02
trunc_norm_init_std = 1e-4