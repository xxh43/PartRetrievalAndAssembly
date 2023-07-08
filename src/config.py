from cgi import test
import torch
import numpy as np

import os
import shutil
import joblib


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def to_numpy(item):
    if torch.is_tensor(item):
        if item.is_cuda:
            return item.cpu().detach().numpy()
        else:
            return item.detach().numpy()
    else:
        return item

def to_tensor(item):
    return torch.tensor(item, device=device)

max_part_num = 5
part_point_num = 512
vae_lr = 0.0005
vae_batch_size = 512
shape_lr = 0.008
collision_weight = 0.0002
latent_dist_weight = 0.002
part_recon_weight = 0.4
retrieval_lr = 0.02
post_lr = 0.0005
enc_dim = 64
overlap_threshold = 0.5
symmetry_threshold = 0.004

debug = False
to_vis = False
to_fig = False
use_parallel = False
pretrain_max_epoch = 3001
retrieval_max_iteration = 200 #1000
post_max_iteration = 150 #1000
init_epoch = 10001
train_epoch = 401
supervise_threshold = 0.004
random_train_epoch = 100
supervise_max_iteration = 3001
cluster_max_iteration = 100


import argparse
global_parser = argparse.ArgumentParser()
global_parser.add_argument("--data_dir", type=str, default='../../data')
global_parser.add_argument("--exp_dir", type=str, default='ours')
global_parser.add_argument("--split_file", type=str, default='split_partnet_faucet.csv')

global_parser.add_argument("--dataset_option", type=str, default='')
global_parser.add_argument("--part_dataset", type=str, default='partnet')
global_parser.add_argument("--part_category", type=str, default='faucet')
global_parser.add_argument("--part_count", type=int, default=2000)

global_parser.add_argument("--shape_dataset", type=str, default='partnet')
global_parser.add_argument("--shape_category", type=str, default='faucet')
global_parser.add_argument("--source_shape_count", type=int, default=500)
global_parser.add_argument("--train_shape_count", type=int, default=250)
global_parser.add_argument("--test_shape_count", type=int, default=120)
global_parser.add_argument("--val_shape_count", type=int, default=80)
global_parser.add_argument("--eval_on_train_shape_count", type=int, default=100)

global_parser.add_argument("--k", type=int, default=6)
global_parser.add_argument("--use_shift", type=int, default=1)
global_parser.add_argument("--use_borrow", type=int, default=0)

global_args = global_parser.parse_args()

if global_args.use_shift == 1:
    use_shift = True
else:
    use_shift = False

if global_args.use_borrow == 1:
    use_borrow = True
else:
    use_borrow = False
