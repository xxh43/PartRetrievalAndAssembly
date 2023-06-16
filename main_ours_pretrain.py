
import argparse
import copy
import gc
import math
import os
import shutil
import time
from operator import pos
from pathlib import Path

#import sklearn.external.joblib as extjoblib
#import sklearn.external.joblib as extjoblib
import joblib
import matplotlib.pyplot as plt
from networkx.algorithms.centrality import group
import numpy as np
import scipy
from scipy.spatial import distance
import torch
import torch.nn as nn
import torch.nn.functional as F
from networkx.algorithms import similarity
from networkx.algorithms.distance_measures import center
from networkx.algorithms.planarity import top_of_stack
from networkx.algorithms.smallworld import sigma
from numpy.core.fromnumeric import argmax, argmin, shape
from scipy import spatial, stats
from scipy.spatial.distance import braycurtis, cdist
#from chamfer_distance import *
#from chamfer_distance import *
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from torch import norm, tensor
from torch._C import dtype
from torch.autograd import Variable
from torch.utils.data import DataLoader

#from trimesh import *
#from dataset_partnet import *
from util_file import *
from util_motion import *
from util_vis import *

torch.set_printoptions(precision=10)

import argparse
import copy
import gc
import math
import os
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader
from util_file import *
from util_vis import *


import argparse
import copy
import gc
import math
import os
import shutil
import time
from operator import pos
from pathlib import Path

#import sklearn.external.joblib as extjoblib
#import sklearn.external.joblib as extjoblib
import joblib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from networkx.algorithms.distance_measures import center
from networkx.algorithms.planarity import top_of_stack
from networkx.algorithms.smallworld import sigma
from numpy.core.fromnumeric import argmin, shape
#from pytorch3d.loss.chamfer import chamfer_distance_one_direction
from scipy import spatial, stats
from scipy.spatial.distance import braycurtis, cdist
#from chamfer_distance import *
#from chamfer_distance import *
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from torch import nn, tensor
from torch._C import dtype
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader

from util_file import *
from util_motion import *
from util_vis import *

torch.set_printoptions(precision=10)

import argparse
import copy
import gc
import math
import os
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
import torch
from config import *
import torchvision
from main_common import *
import pytorch3d
from pytorch3d.loss import chamfer_distance

class VAE(nn.Module):
    def __init__(self, part_point_num, latent_dim, kld_weight):
        super(VAE, self).__init__()

        self.part_point_num = part_point_num
        self.latent_dim = latent_dim
        self.kld_weight = kld_weight

        modules = []
        hidden_dims = [3, 32, 64, 64, self.latent_dim]

        for i in range(len(hidden_dims)-1):
            modules.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dims[i], hidden_dims[i+1], 1),
                    nn.BatchNorm1d(hidden_dims[i+1]),
                    nn.LeakyReLU())
            )

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], self.latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], self.latent_dim)

        modules = []

        self.decoder_input = nn.Linear(self.latent_dim, hidden_dims[-1])

        hidden_dims = [self.latent_dim, 128, 512]

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Linear(hidden_dims[-1], part_point_num*3)

    def encode(self, input):
        result = self.encoder(input)
        result = torch.max(result, 2, keepdim=True)[0]
        result = torch.flatten(result, 1, -1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def decode(self, z):
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        result = result.view(-1, self.part_point_num, 3)
        return result

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  self.decode(z), z, mu, log_var

    def loss_function(self, input, recon, mu, log_var):
        recons_loss, _ = chamfer_distance(recon, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss = recons_loss + self.kld_weight * kld_loss
        return loss

bce_loss = torch.nn.BCELoss()
def pre_train(exp_folder, part_vol_pcs, enc_dim, vae_lr):

    part_vol_pcs = to_numpy(part_vol_pcs)
    base_kld_weight = 0.00001
    vae = VAE(part_vol_pcs.shape[1], enc_dim, base_kld_weight) 
    vae.cuda()
    
    optimizer = torch.optim.Adam(list(vae.parameters()), lr=vae_lr)

    for epoch in range(pretrain_max_epoch):

        epoch_loss = 0
        print('epoch', epoch)
        part_encs = None
        part_recons = None
        for batch_start in range(0, len(part_vol_pcs), vae_batch_size):            
            batched_part_pcs = torch.tensor(part_vol_pcs[batch_start:batch_start+vae_batch_size], device=device, dtype=torch.float)
            recon, sampled_enc, mu, log_var = vae(batched_part_pcs.transpose(1, 2))
            vae_loss = vae.loss_function(batched_part_pcs, recon, mu, log_var)

            optimizer.zero_grad()
            vae_loss.backward()
            optimizer.step()
            epoch_loss += vae_loss.item()

            if part_encs is None:
                part_encs = mu
            else:
                part_encs = torch.cat((part_encs, mu))

            if part_recons is None:
                part_recons = recon
            else:
                part_recons = torch.cat((part_recons, recon), dim=0)

        print('epoch loss', epoch_loss)
    
    return vae, part_encs

