
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

print(torch.__version__)
print(torchvision.__version__)
import threading, queue

import pytorch3d
from pytorch3d.loss import chamfer_distance

from torch.nn.functional import normalize

class InfoVAE(nn.Module):
    def __init__(self, part_point_num, latent_dim, kld_weight):
        super(InfoVAE, self).__init__()

        self.part_point_num = part_point_num
        self.latent_dim = latent_dim
        self.kld_weight = kld_weight

        modules = []
        hidden_dims = [3, 32, 64, 128, 256, 512]

        # Build Encoder
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

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(self.latent_dim, hidden_dims[-1])

        hidden_dims = [512, 512, 1024, 1024]

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
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """

        result = self.encoder(input)
        result = torch.max(result, 2, keepdim=True)[0]
        result = torch.flatten(result, 1, -1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
       
        return mu, log_var

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        result = result.view(-1, self.part_point_num, 3)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  self.decode(z), z, mu, log_var

    def compute_kernel(self, x, y):
        #https://github.com/pratikm141/MMD-Variational-Autoencoder-Pytorch-InfoVAE/blob/master/mmd_vae_pytorchver.ipynb
        x_size = x.shape[0]
        y_size = y.shape[0]
        dim = x.shape[1]

        tiled_x = x.view(x_size,1,dim).repeat(1, y_size,1)
        tiled_y = y.view(1,y_size,dim).repeat(x_size, 1,1)

        return torch.exp(-torch.mean((tiled_x - tiled_y)**2,dim=2)/dim*1.0)

    def compute_mmd(self, x, y):
        #https://github.com/pratikm141/MMD-Variational-Autoencoder-Pytorch-InfoVAE/blob/master/mmd_vae_pytorchver.ipynb
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        return torch.mean(x_kernel) + torch.mean(y_kernel) - 2*torch.mean(xy_kernel)

    def loss_function(self, input, recon, mu, log_var, z):

        recons_loss, _ = chamfer_distance(recon, input)
        true_samples = torch.randn((len(input), self.latent_dim), device=device)
        mmd_loss = self.compute_mmd(true_samples, z)
        loss = recons_loss + self.kld_weight * mmd_loss
        return loss

    def sample(self, num_samples):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)
        #z = z.to('cuda:0')

        samples = self.decode(z)
        return samples

    def generate(self, x):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]



class VAE(nn.Module):
    def __init__(self, part_point_num, latent_dim, kld_weight):
        super(VAE, self).__init__()

        self.part_point_num = part_point_num
        self.latent_dim = latent_dim
        self.kld_weight = kld_weight

        modules = []
        hidden_dims = [3, 32, 64, 64, self.latent_dim]

        # Build Encoder
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

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(self.latent_dim, hidden_dims[-1])

        #hidden_dims = [64, 128, 512, 512]

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
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """

        result = self.encoder(input)
        result = torch.max(result, 2, keepdim=True)[0]
        result = torch.flatten(result, 1, -1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
       
        return mu, log_var

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        result = result.view(-1, self.part_point_num, 3)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  self.decode(z), z, mu, log_var

    def loss_function(self, input, recon, mu, log_var):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """

        recons_loss, _ = chamfer_distance(recon, input)
        #recons_loss = F.mse_loss(recon, input)
        
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss = recons_loss + self.kld_weight * kld_loss
        return loss


    def sample(self, num_samples):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)
        #z = z.to('cuda:0')

        samples = self.decode(z)
        return samples

    def generate(self, x):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

bce_loss = torch.nn.BCELoss()
def pre_train(exp_folder, part_vol_pcs, enc_dim, vae_lr):

    pre_train_folder = os.path.join(exp_folder, 'pre_train')
    if not os.path.exists(pre_train_folder):
        #shutil.rmtree(pre_train_folder)
        os.makedirs(pre_train_folder)

    if os.path.isfile(os.path.join(pre_train_folder, 'vae'+str(len(part_vol_pcs))+'.joblib')) and os.path.isfile(os.path.join(pre_train_folder, 'part_encs'+str(len(part_vol_pcs))+'.joblib')):
        vae = joblib.load(os.path.join(pre_train_folder, 'vae'+str(len(part_vol_pcs))+'.joblib'))
        part_encs = joblib.load(os.path.join(pre_train_folder, 'part_encs'+str(len(part_vol_pcs))+'.joblib'))
        return vae, to_numpy(part_encs)

    part_vol_pcs = to_numpy(part_vol_pcs)

    base_kld_weight = 0.00001
    #vae = VAE(part_vol_pcs.shape[1], enc_dim, 0.001) # not good
    vae = VAE(part_vol_pcs.shape[1], enc_dim, base_kld_weight) # best so far

    #vae = InfoVAE(part_vol_pcs.shape[1], enc_dim, 0.5)
    vae.cuda()
    
    optimizer = torch.optim.Adam(list(vae.parameters()), lr=vae_lr)

    #vis_part_indices = random.sample(range(0, len(part_vol_pcs)), 10)

    for epoch in range(pretrain_max_epoch):

        epoch_loss = 0
        print('epoch', epoch)
        part_encs = None
        part_recons = None
        for batch_start in range(0, len(part_vol_pcs), vae_batch_size):            
            batched_part_pcs = torch.tensor(part_vol_pcs[batch_start:batch_start+vae_batch_size], device=device, dtype=torch.float)
            recon, sampled_enc, mu, log_var = vae(batched_part_pcs.transpose(1, 2))
            #vae_loss = vae.loss_function(batched_part_pcs, recon, mu, log_var, sampled_enc)
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

        #if to_vis:
            #if epoch == 0:
                #visualize_part_encs(part_encs, labels, os.path.join(pre_train_folder, 'encs_before.png'))
                #for i in vis_part_indices:
                    #display_pcs([part_vol_pcs[i]], os.path.join(pre_train_folder, str(i)+'pc_before.png'), True)

        #if epoch == pretrain_max_epoch - 1:
            #visualize_part_encs(part_encs, os.path.join(pre_train_folder, 'encs_after.png'))
            #for i in range(10):
                #display_pcs([part_vol_pcs[i]], os.path.join(pre_train_folder, str(i)+'real_pc.png'), True, True)
                #display_pcs([part_recons[i]], os.path.join(pre_train_folder, str(i)+'pc_dec.png'), True, True)

        print('epoch loss', epoch_loss)

    if to_vis:
        vae.eval()
        visulize_interpolation(part_vol_pcs, part_encs, vae, pre_train_folder)
    print('training finish !')

    part_encs = to_numpy(part_encs)
    
    joblib.dump(vae, os.path.join(pre_train_folder, 'vae'+str(len(part_vol_pcs))+'.joblib'))
    joblib.dump(part_encs, os.path.join(pre_train_folder, 'part_encs'+str(len(part_vol_pcs))+'.joblib'))
    
    return vae, part_encs


def generate_interpolations(enc1, enc2):
    inter_count = 10
    interpolations = []
    for i in range(inter_count+1):
        interpolation = enc1 + (enc2 - enc1)/inter_count * i
        interpolations.append(interpolation)
    return interpolations

def visulize_interpolation(part_pcs, part_encs, vae, pre_train_folder):

    inter_folder = os.path.join(pre_train_folder, 'latent_interpolations')
    if not os.path.exists(inter_folder):
        os.makedirs(inter_folder)

    pc1s = []
    enc1s = []
    for i in range(0, 5):
        enc1 = part_encs[i]
        enc1s.append(enc1)
        pc1s.append(part_pcs[i])
    
    pc2s = []
    enc2s = []
    for i in range(5, 10):
        enc2 = part_encs[i]
        enc2s.append(enc2)
        pc2s.append(part_pcs[i])

    for i in range(len(enc1s)):
        inter_i_folder = os.path.join(inter_folder, str(i))
        if not os.path.exists(inter_i_folder):
            os.makedirs(inter_i_folder)
        display_pcs([pc1s[i]], os.path.join(inter_i_folder, 'start.png'), save=True)
        display_pcs([pc2s[i]], os.path.join(inter_i_folder, 'end.png'), save=True)
        interpolations = generate_interpolations(enc1s[i], enc2s[i])
        for j in range(len(interpolations)):
            inter_recon = vae.decode(torch.stack([interpolations[j]]))[0]
            display_pcs([inter_recon], os.path.join(inter_i_folder, str(j)+'.png'), save=True)
    