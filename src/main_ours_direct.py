
import argparse
import copy
import os
import shutil
import time
import time
from pathlib import Path
import gc
import joblib
import matplotlib.pyplot as plt
import numpy as np
from numpy import arange
import scipy
from sklearn import cluster, get_config
from main_ours_pretrain import *
from joblib import Parallel, delayed
from util_collision import *

from util_motion import *
from util_vis import *
from scipy.spatial.transform import Rotation as R
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


from util_vis import *

bce_loss = torch.nn.BCELoss()
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
from scipy import spatial, stats


from util_motion import *
from util_vis import *

import time


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



import torch
import torchvision
from scipy.spatial.transform import Rotation as R
from torch import nn
from torch.nn import functional as F
from pytorch3d import *
from pytorch3d.loss import chamfer_distance
from main_ours_pretrain import *
from util_collision import *
from config import *
from main_common import *
from data_manager import *
import time

print('pytorch version', torch.__version__)

def get_distance_to_nearest_part(shape_encs, part_encs, k):
    shape_encs = shape_encs.reshape(-1, enc_dim)
    dists = torch.cdist(shape_encs, part_encs)
    min_dists = torch.min(dists, dim=1).values
    min_dists = torch.mean(min_dists.reshape(-1, k), dim=1)
    return min_dists

def forward_with_symmetry(k, shape_encs, part_vae, translations, scales, up_angles):

    translations = translations.reshape(-1, 3)
    up_angles = (up_angles.reshape(-1, 1)).squeeze(dim=1)

    enc_dim = part_vae.latent_dim
    shape_encs = shape_encs.reshape(-1, enc_dim)

    part_recons = part_vae.decode(shape_encs)
    part_recons = up_rotate_parts(part_recons, up_angles)
    part_recons = translate_parts(part_recons, translations)

    normals1 = torch.tensor([1,0,0], device=device, dtype=torch.float).unsqueeze(dim=0).repeat_interleave(len(part_recons), dim=0).unsqueeze(dim=1)
    vecs = (part_recons).permute(0, 2, 1)
    dots = torch.bmm(normals1, vecs)
    temp = torch.bmm(dots.permute(0, 2, 1), normals1)
    part_recons_sym = part_recons - 2 * temp
    reflected_part_recons = torch.cat((part_recons, part_recons_sym), dim=1)

    reflected_part_recons = reflected_part_recons.reshape(-1, 2, part_point_num, 3)
    reflected_part_recons = reflected_part_recons.reshape(-1, k, 2, part_point_num, 3)
    reflected_part_recons = reflected_part_recons.permute(0, 2, 1, 3, 4)
    reflected_part_recons = reflected_part_recons.flatten(1, 2)
    reflected_part_recons = reflected_part_recons.reshape(-1, part_point_num, 3)
    
    return reflected_part_recons

def forward(shape_encs, part_vae, translations, scales, up_angles):

    translations = translations.reshape(-1, 3)
    up_angles = (up_angles.reshape(-1, 1)).squeeze(dim=1)

    enc_dim = part_vae.latent_dim
    shape_encs = shape_encs.reshape(-1, enc_dim)
    
    part_recons = part_vae.decode(shape_encs)
    part_recons = up_rotate_parts(part_recons, up_angles)
    part_recons = translate_parts(part_recons, translations)
    return part_recons

def cluster_targets(shape_pcs, exp_folder, use_symmetry):

    if use_symmetry:
        distmat_filename = os.path.join(exp_folder, 'sym_train_target_distance_matrix.joblib')
    else:
        distmat_filename = os.path.join(exp_folder, 'nonsym_train_target_distance_matrix.joblib')

    if os.path.isfile(distmat_filename):
        shape_distance_mat = joblib.load(distmat_filename)
        return shape_distance_mat

    shape_pcs = torch.tensor(shape_pcs, device=device, dtype=torch.float)

    cluster_lr = 0.01

    distance_matrix = np.zeros((len(shape_pcs), len(shape_pcs)))    
    for i in range(len(shape_pcs)):
        print('i', i, '----------------------------------')
        ref_shape_pc = shape_pcs[i].unsqueeze(dim=0)
        other_shape_pcs = shape_pcs
        part_translations = torch.zeros((len(other_shape_pcs), 3), device=device, requires_grad=True)
        part_rotations = torch.tensor(torch.cat((torch.ones((len(other_shape_pcs), 1)), torch.zeros((len(other_shape_pcs), 3))), dim=1), device=device, requires_grad=True)
        part_scales = torch.ones((len(other_shape_pcs), 3), device=device, requires_grad=True)
        optimizer = torch.optim.Adam([part_translations]+[part_scales]+[part_rotations], lr=cluster_lr)
        repeated_shape_pcs = ref_shape_pc.repeat_interleave(len(other_shape_pcs), dim=0)
        for iteration in range(500):
            transformed_other_shape_pcs = other_shape_pcs
            transformed_other_shape_pcs = scale_parts(transformed_other_shape_pcs, part_scales)
            transformed_other_shape_pcs = translate_parts(transformed_other_shape_pcs, part_translations)
            errors, _ = chamfer_distance(transformed_other_shape_pcs, repeated_shape_pcs, batch_reduction=None)
            loss = torch.mean(errors)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('loss', loss.item())
        
        distance_matrix[i] = to_numpy(errors)
    
    joblib.dump(distance_matrix, distmat_filename)
    return distance_matrix

def get_nearest_train_neighbor(pc, other_pcs):

    print('getting nearest neighbor...........')

    ref_pc = torch.tensor(pc, device=device, dtype=torch.float).unsqueeze(dim=0)
    other_pcs = torch.tensor(other_pcs, device=device, dtype=torch.float)

    translations = torch.zeros((len(other_pcs), 3), device=device, requires_grad=True)
    angles = torch.zeros(len(other_pcs), device=device, requires_grad=True)
    scales = torch.ones((len(other_pcs), 3), device=device, requires_grad=True)
    optimizer = torch.optim.Adam([translations]+[scales]+[angles], lr=0.005)
    repeated_pcs = ref_pc.repeat_interleave(len(other_pcs), dim=0)
    for iteration in range(50):
        transformed_other_pcs = other_pcs
        transformed_other_pcs = scale_parts(transformed_other_pcs, scales)
        #transformed_other_pcs = up_rotate_parts(transformed_other_pcs, angles)
        transformed_other_pcs = translate_parts(transformed_other_pcs, translations)
        
        errors, _ = chamfer_distance(transformed_other_pcs, repeated_pcs, batch_reduction=None)
        loss = torch.mean(errors)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print('loss', loss.item())
        
    nb_index = [x for _, x in sorted(zip(to_numpy(errors).tolist(), arange(0, len(other_pcs))), key=lambda pair: pair[0])][0]

    return nb_index


def learn_from_neighbor(shape_index, nb_distances, all_shape_errors, all_shape_centers, all_translations, all_scales, all_up_angles):
    
    all_shape_errors = torch.tensor(all_shape_errors, device=device)
    all_shape_centers = torch.tensor(all_shape_centers, device=device)
    all_translations = torch.tensor(all_translations, device=device)
    all_scales = torch.tensor(all_scales, device=device)
    all_up_angles = torch.tensor(all_up_angles, device=device)

    sorted_shape_errors = sorted(all_shape_errors)
    good_recon_index = int(0.05*len(all_shape_errors))
    bad_recon_index = int(0.2*len(all_shape_errors))
    good_recon_error = sorted_shape_errors[good_recon_index]
    bad_recon_error = sorted_shape_errors[bad_recon_index]

    recon_error = all_shape_errors[shape_index]
    sorted_nb_indices = [x for _, x in sorted(zip(nb_distances, np.arange(0, len(nb_distances))))]

    if recon_error > bad_recon_error:
        count = 0
        for nb_index in sorted_nb_indices:
            count += 1
            if all_shape_errors[nb_index] < good_recon_error:
                return True, nb_index, sorted_nb_indices
            
            if count > 5:
                return True, -1, sorted_nb_indices

    return False, -1, sorted_nb_indices

def swap_recon_symmetry(shape_pc, part_recon_pcs, k):

    if k == 1:
        return part_recon_pcs

    target_to_regions_dists = None
    for i in range(len(part_recon_pcs)):
        t_to_r_dists = torch.cdist(torch.stack([shape_pc]), torch.stack([part_recon_pcs[i]]))[0]
        min_t_to_r_dists = torch.min(t_to_r_dists, dim=1).values
        if target_to_regions_dists is None:
            target_to_regions_dists = min_t_to_r_dists.unsqueeze(dim=1)
        else:
            target_to_regions_dists = torch.cat((target_to_regions_dists, min_t_to_r_dists.unsqueeze(dim=1)), dim=1)

    min_target_to_region_dists = torch.min(target_to_regions_dists, dim=1).values
    shape_uncovered_region_indices = torch.topk(min_target_to_region_dists, k=int(0.05*len(shape_pc)), largest=True).indices
    if len(shape_uncovered_region_indices) == 0:
        return part_recon_pcs

    shape_uncovered_region = shape_pc[shape_uncovered_region_indices]

    eps = 0.1
    region_dists = torch.cdist(shape_uncovered_region.unsqueeze(dim=0), shape_uncovered_region.unsqueeze(dim=0))[0]
    region_adjacency = to_numpy(region_dists < eps)
    region_graph = nx.from_numpy_matrix(region_adjacency)

    max_point_count = 0 
    largest_cc_region = None
    largest_cc_region_index = None
    for cc in list(nx.connected_components(region_graph)):            
        cc = np.array(list(cc))
        cc_region = shape_uncovered_region[cc]
        if len(cc_region) > max_point_count:
            max_point_count = len(cc_region)
            largest_cc_region = cc_region
            largest_cc_region_index = cc
    
    min_after_swap_value = torch.mean(min_target_to_region_dists)
    
    swap_index = -1
    for i in range(k):
        target_to_regions_dists_without_i = torch.cat((target_to_regions_dists[:, 0:i], target_to_regions_dists[:, i+1:i+k], target_to_regions_dists[:, i+k+1:]), dim=1)
        min_target_to_region_dists_without_i = torch.min(target_to_regions_dists_without_i, dim=1).values
        min_target_to_region_dists_without_i[largest_cc_region_index] = 0.0
        value_after_swap = torch.mean(min_target_to_region_dists_without_i)
    
        if value_after_swap < min_after_swap_value:
            min_after_swap_value = value_after_swap
            swap_index = i

    if swap_index >= 0:
        new_part_recon_pcs = [None]*len(part_recon_pcs)
        
        for i in range(k):
            if i != swap_index:
                new_part_recon_pcs[i] = copy.deepcopy(part_recon_pcs[i])
                new_part_recon_pcs[i+k] = copy.deepcopy(part_recon_pcs[i+k])
            else:
                region_i = copy.deepcopy(largest_cc_region).unsqueeze(dim=0)
                normals1 = torch.tensor([1,0,0], device=device, dtype=torch.float).unsqueeze(dim=0).repeat_interleave(len(region_i), dim=0).unsqueeze(dim=1)
                vecs = (region_i).permute(0, 2, 1)
                dots = torch.bmm(normals1, vecs)
                temp = torch.bmm(dots.permute(0, 2, 1), normals1)
                reflected_region_i = region_i - 2 * temp
                new_part_recon_pcs[i] = copy.deepcopy(region_i[0])
                new_part_recon_pcs[i+k] = copy.deepcopy(reflected_region_i[0])

        return new_part_recon_pcs
    else:
        return part_recon_pcs

def swap_recon(shape_pc, part_recon_pcs, k):

    target_to_regions_dists = None
    for i in range(len(part_recon_pcs)):
        t_to_r_dists = torch.cdist(torch.stack([shape_pc]), torch.stack([part_recon_pcs[i]]))[0]
        min_t_to_r_dists = torch.min(t_to_r_dists, dim=1).values
        if target_to_regions_dists is None:
            target_to_regions_dists = min_t_to_r_dists.unsqueeze(dim=1)
        else:
            target_to_regions_dists = torch.cat((target_to_regions_dists, min_t_to_r_dists.unsqueeze(dim=1)), dim=1)

    min_target_to_region_dists = torch.min(target_to_regions_dists, dim=1).values
    shape_uncovered_region_indices = torch.topk(min_target_to_region_dists, k=int(0.05*len(shape_pc)), largest=True).indices
    shape_uncovered_region = shape_pc[shape_uncovered_region_indices]

    eps = 0.1
    region_dists = torch.cdist(shape_uncovered_region.unsqueeze(dim=0), shape_uncovered_region.unsqueeze(dim=0))[0]
    region_adjacency = to_numpy(region_dists < eps)
    region_graph = nx.from_numpy_matrix(region_adjacency)

    max_point_count = 0 
    largest_cc_region = None
    largest_cc_region_index = None
    for cc in list(nx.connected_components(region_graph)):            
        cc = np.array(list(cc))
        cc_region = shape_uncovered_region[cc]
        if len(cc_region) > max_point_count:
            max_point_count = len(cc_region)
            largest_cc_region = cc_region
            largest_cc_region_index = cc
    
    min_after_swap_value = torch.mean(min_target_to_region_dists)
    
    swap_index = -1
    for i in range(len(part_recon_pcs)):
        target_to_regions_dists_without_i = torch.cat((target_to_regions_dists[:, 0:i], target_to_regions_dists[:, i+1:]), dim=1)
        #print('target_to_regions_dists_without_i shape', target_to_regions_dists_without_i.shape)
        min_target_to_region_dists_without_i = torch.min(target_to_regions_dists_without_i, dim=1).values
        min_target_to_region_dists_without_i[largest_cc_region_index] = 0.0
        value_after_swap = torch.mean(min_target_to_region_dists_without_i)
        #print('value_after_swap', value_after_swap)
        if value_after_swap < min_after_swap_value:
            #print('swap !')
            min_after_swap_value = value_after_swap
            swap_index = i

    if swap_index >= 0:
        new_part_recon_pcs = [None]*len(part_recon_pcs)

        for i in range(len(part_recon_pcs)):
            if i != swap_index:
                new_part_recon_pcs[i] = copy.deepcopy(part_recon_pcs[i])
            else:
                new_part_recon_pcs[i] = copy.deepcopy(largest_cc_region)

        return new_part_recon_pcs
    else:
        return part_recon_pcs

def process_shape_new(shape_pc, part_recon_pcs, part_vae, k, use_symmetry):

    shape_pc = torch.tensor(shape_pc, device=device, dtype=torch.float)
    part_recon_pcs = torch.tensor(part_recon_pcs, device=device, dtype=torch.float)

    region_min_point_count = 40
    uncover_ratio = 0.7

    eps = 0.1
    
    part_recon_pcs = part_recon_pcs.reshape(-1, part_point_num, 3)

    if use_symmetry:
        part_recon_pcs = swap_recon_symmetry(shape_pc, part_recon_pcs, k)
    else:
        part_recon_pcs = swap_recon(shape_pc, part_recon_pcs, k)

    target_to_regions_dists = None
    for i in range(len(part_recon_pcs)):
        t_to_r_dists = torch.cdist(torch.stack([shape_pc]), torch.stack([part_recon_pcs[i]]))[0]
        min_t_to_r_dists = torch.min(t_to_r_dists, dim=1).values
        if target_to_regions_dists is None:
            target_to_regions_dists = min_t_to_r_dists.unsqueeze(dim=1)
        else:
            target_to_regions_dists = torch.cat((target_to_regions_dists, min_t_to_r_dists.unsqueeze(dim=1)), dim=1)
    
    min_target_to_region_dists = torch.min(target_to_regions_dists, dim=1).values
    min_target_to_region_indices = torch.min(target_to_regions_dists, dim=1).indices

    regions_before = []
    regions_uncovered = []
    regions_after = []

    part_region_pcs = []
    updated_part_region_pcs = []
    part_region_coverages = []
    for j in range(len(part_recon_pcs)):
        part_region_pc = shape_pc[min_target_to_region_indices == j]
        part_region_index = (min_target_to_region_indices == j).nonzero().squeeze(dim=1)
        part_region_coverage = min_target_to_region_dists[min_target_to_region_indices == j]
        
        if len(part_region_index) < region_min_point_count:
            temp_t_to_r_dists = target_to_regions_dists[:, j]
            temp_t_to_r_min_indices = torch.topk(temp_t_to_r_dists, dim=0, k=region_min_point_count, largest=False).indices
            part_region_pc = shape_pc[temp_t_to_r_min_indices[0:region_min_point_count]]
            part_region_index = temp_t_to_r_min_indices[0:region_min_point_count].nonzero().squeeze(dim=1)
            
            min_target_to_region_indices[temp_t_to_r_min_indices[0:region_min_point_count]] = j
            min_target_to_region_dists[temp_t_to_r_min_indices[0:region_min_point_count]] = temp_t_to_r_dists[temp_t_to_r_min_indices[0:region_min_point_count]]
            part_region_coverage = min_target_to_region_dists[temp_t_to_r_min_indices[0:region_min_point_count]]

        part_region_pcs.append(part_region_pc)
        updated_part_region_pcs.append(part_region_pc)
        part_region_coverages.append(part_region_coverage)
        regions_before.append(copy.deepcopy(part_region_pc))


    part_vae.eval()
    part_region_encs = []
    part_region_translations = []
    part_region_scales = []
    part_region_angles = []

    for i in range(k):

        if use_symmetry:
            candidate_region = part_region_pcs[i]
            reflected_candidate_region = part_region_pcs[i+k]

            overlap = get_inside_dist( candidate_region.unsqueeze(dim=0), reflected_candidate_region.unsqueeze(dim=0), 5)[0]

            if overlap > overlap_threshold:
                candidate_region_pc = torch.cat((candidate_region, reflected_candidate_region), dim=0)
                candidate_region_weight = torch.cat((part_region_coverages[i], part_region_coverages[i+k]))
                top_region_indices = torch.topk(candidate_region_weight, int(uncover_ratio*len(candidate_region_weight)), largest=True).indices
                
                other_region_pcs = []
                for j in range(len(updated_part_region_pcs)):
                    if j != i and j != i+k:
                        other_region_pcs.append(updated_part_region_pcs[j])

                uncovered_region = candidate_region_pc[top_region_indices]
                regions_uncovered.append(copy.deepcopy(uncovered_region))
                best_part_region = get_best_connected_components_new(uncovered_region, other_region_pcs, shape_pc, None, None, region_min_point_count, eps)

                updated_part_region_pcs[i] = best_part_region
                updated_part_region_pcs[i+k] = best_part_region

            else:
                candidate_region_pc = part_region_pcs[i]
                top_region_indices = torch.topk(part_region_coverages[i], int(uncover_ratio*len(part_region_coverages[i])), largest=True).indices
                
                other_region_pcs = []
                for j in range(len(updated_part_region_pcs)):
                    if j != i:
                        other_region_pcs.append(updated_part_region_pcs[j])

                uncovered_region = candidate_region_pc[top_region_indices]
                regions_uncovered.append(copy.deepcopy(uncovered_region))
                best_part_region = get_best_connected_components_new(uncovered_region, other_region_pcs, shape_pc, None, None, region_min_point_count, eps)

                updated_part_region_pcs[i] = best_part_region

        else:
            candidate_region_pc = part_region_pcs[i] 
            top_region_indices = torch.topk(part_region_coverages[i], int(uncover_ratio*len(part_region_coverages[i])), largest=True).indices
            
            other_region_pcs = []
            for j in range(len(updated_part_region_pcs)):
                if j != i:
                    other_region_pcs.append(updated_part_region_pcs[j])
                    
            uncovered_region = candidate_region_pc[top_region_indices]
            regions_uncovered.append(copy.deepcopy(uncovered_region))
            best_part_region = get_best_connected_components_new(uncovered_region, other_region_pcs, shape_pc, None, None, region_min_point_count, eps)
            updated_part_region_pcs[i] = best_part_region

        regions_after.append(copy.deepcopy(best_part_region))

        calibrate_translation = -torch.mean(best_part_region, dim=0)
        min_values = torch.min(best_part_region, dim=0).values
        max_values = torch.max(best_part_region, dim=0).values
        calibrate_angle = torch.tensor(get_y_aligned_reset_angle(best_part_region), device=device, dtype=torch.float)
        input_part_region = best_part_region + calibrate_translation
        input_part_region = up_rotate_parts(torch.stack([input_part_region]), torch.stack([calibrate_angle]))[0]
        part_region_enc_i, _ = part_vae.encode(torch.stack([input_part_region]).transpose(1, 2))
        part_region_enc = part_region_enc_i[0]
        part_region_encs.append(part_region_enc)

        part_region_translation = -calibrate_translation
        part_region_angle = -calibrate_angle

        part_region_scale = torch.tensor([1.0, 1.0, 1.0], device=device, dtype=torch.float)

        part_region_translations.append(part_region_translation)
        part_region_scales.append(part_region_scale)
        part_region_angles.append(part_region_angle)


    part_region_encs = torch.stack(part_region_encs[0:k])
    part_region_translations = torch.stack(part_region_translations[0:k])
    part_region_scales = torch.stack(part_region_scales[0:k])
    part_region_angles = torch.stack(part_region_angles[0:k])
    
    return part_region_encs, part_region_translations, part_region_scales, part_region_angles, part_recon_pcs, regions_before, regions_uncovered, regions_after


def get_best_connected_components_new(region_pc, other_region_pcs, target_pc, target_graph, target_graph_weight, region_min_point_count, eps):
    
    if len(region_pc) < region_min_point_count:
        return region_pc

    region_dists = torch.cdist(region_pc.unsqueeze(dim=0), region_pc.unsqueeze(dim=0))[0]
    region_adjacency = to_numpy(region_dists < eps)
    region_graph = nx.from_numpy_matrix(region_adjacency)

    best_region = None
    use_euclidean = True

    if use_euclidean:
        best_region = None
        max_sum_distance = -np.inf
        for cc in list(nx.connected_components(region_graph)):            
            if len(list(cc)) < region_min_point_count:
                continue

            cc = np.array(list(cc))
            cc_region = region_pc[cc]

            sum_distance = 0
            for i in range(len(other_region_pcs)):
                cc_to_other_dists = torch.cdist(cc_region, other_region_pcs[i])
                dist = torch.max(torch.min(cc_to_other_dists, dim=1).values)
                sum_distance += dist

            if sum_distance > max_sum_distance:
                max_sum_distance = sum_distance
                best_region = cc_region
    
    if best_region is None or len(best_region) < region_min_point_count:
        return region_pc

    return best_region


def compute_symmetry_infos(target_pcs, is_train, exp_folder, category):

    normals1 = torch.tensor([1,0,0], device=device, dtype=torch.float).unsqueeze(dim=0)

    sym_shape_indices = []
    nonsym_shape_indices = []

    for target_index in range(len(target_pcs)):
        
        target_pc = torch.tensor(target_pcs[target_index], device=device, dtype=torch.float)
        dotvals1 = torch.matmul(target_pc, normals1.transpose(0,1)).squeeze(dim=1)

        positive_pc1 = target_pc[dotvals1 > 0]
        negative_pc1 = target_pc[dotvals1 < 0]

        vecs = positive_pc1.permute(1, 0)
        dots = torch.matmul(normals1, vecs)
        temp = torch.matmul(dots.permute(1, 0), normals1)
        reflected_positive_pc1 = positive_pc1 - 2 * temp

        error, _ = chamfer_distance(torch.stack([reflected_positive_pc1]), torch.stack([negative_pc1]))
        if error < symmetry_threshold:
            sym_shape_indices.append(target_index)
        else:
            nonsym_shape_indices.append(target_index)

    return sym_shape_indices, nonsym_shape_indices



def direct_test(test_folder, train_folder, test_target_pcs, test_target_indices, train_target_pcs, train_target_indices, part_encs, part_vae, part_num, use_symmetry):

    print('test...')

    all_centers = []
    all_translations = []
    all_scales = []
    all_angles = []
    for i in range(len(test_target_pcs)):
        print('getting nearest neighbor for target: ', i)
        index = get_nearest_train_neighbor(test_target_pcs[i], train_target_pcs)
        nb_index = train_target_indices[index]
        all_centers.append( torch.tensor(joblib.load(os.path.join(train_folder, str(nb_index)+'center.joblib')), device=device) )
        all_translations.append(torch.tensor(joblib.load(os.path.join(train_folder, str(nb_index)+'translation.joblib')), device=device) )
        all_scales.append(torch.tensor(joblib.load(os.path.join(train_folder, str(nb_index)+'scale.joblib')), device=device) )
        all_angles.append(torch.tensor(joblib.load(os.path.join(train_folder, str(nb_index)+'angle.joblib')), device=device) )
    all_centers = torch.stack(all_centers)
    all_translations = torch.stack(all_translations)
    all_scales = torch.stack(all_scales)
    all_angles = torch.stack(all_angles)

    all_centers = all_centers.clone().detach().requires_grad_(True)
    all_translations = all_translations.clone().detach().requires_grad_(True)
    all_scales = all_scales.clone().detach().requires_grad_(True)
    all_angles = all_angles.clone().detach().requires_grad_(True)
    shape_optimizer = torch.optim.Adam([all_centers]+[all_translations]+[all_angles], lr=shape_lr)

    part_vae.eval()

    best_errors = [np.inf]*len(test_target_pcs)
    best_recons = [None]*len(test_target_pcs)
    best_centers = [None]*len(test_target_pcs)
    best_translations = [None]*len(test_target_pcs)
    best_scales = [None]*len(test_target_pcs)
    best_angles = [None]*len(test_target_pcs)

    for iteration in range(0, 501):
        print('part_num', part_num, 'test iteration', iteration, '-----------------------------------')
        
        #append_items_to_file(summary, ['-----------------test optim iteration  ', str(iteration)])

        iteration_shape_loss = 0

        all_target_errors = None
        all_target_recons = None

        batch_size = 4
        if len(test_target_pcs) % batch_size == 1:
            batch_size = batch_size-1
        
        for batch_start in range(0, len(test_target_pcs), batch_size):
            
            batched_target_pcs = torch.tensor(test_target_pcs[batch_start:batch_start+batch_size], device=device, dtype=torch.float)
            
            batched_centers = all_centers[batch_start:batch_start+batch_size]
            batched_translations = all_translations[batch_start:batch_start+batch_size]
            batched_scales = all_scales[batch_start:batch_start+batch_size]
            batched_angles = all_angles[batch_start:batch_start+batch_size]

            if use_symmetry:
                batched_part_recons = forward_with_symmetry(part_num, batched_centers, part_vae, batched_translations, batched_scales, batched_angles)
                batched_reshaped_part_recons = batched_part_recons.reshape(-1, 2*part_num*part_point_num, 3)
                collision_errors = get_collision_distance(batched_reshaped_part_recons.reshape(-1, 2*part_num, part_point_num, 3), part_num, use_symmetry)
            else:
                batched_part_recons = forward(batched_centers, part_vae, batched_translations, batched_scales, batched_angles)
                batched_reshaped_part_recons = batched_part_recons.reshape(-1, part_num*part_point_num, 3)
                collision_errors = get_collision_distance(batched_reshaped_part_recons.reshape(-1, part_num, part_point_num, 3), part_num, use_symmetry)
            
            #min_dists = get_distance_to_nearest_part(batched_centers, part_encs, part_num)
            batched_target_recons = batched_reshaped_part_recons
            avg_recon_errors, _ = chamfer_distance(batched_target_recons, batched_target_pcs, batch_reduction=None)
            batched_target_errors = avg_recon_errors + collision_weight * collision_errors 

            loss = torch.mean(batched_target_errors)
            shape_optimizer.zero_grad()
            loss.backward()
            shape_optimizer.step()
            iteration_shape_loss += loss.item()

            if all_target_errors is None:
                all_target_errors = to_numpy(batched_target_errors)
            else:
                all_target_errors = np.concatenate((all_target_errors, to_numpy(batched_target_errors)), axis=0)

            if all_target_recons is None:
                all_target_recons = to_numpy(batched_target_recons)
            else:
                all_target_recons = np.concatenate((all_target_recons, to_numpy(batched_target_recons)), axis=0)

        print('epoch loss', iteration_shape_loss/len(test_target_pcs))

        #main_mem, cuda_mem = get_free_mem()
        #append_items_to_file(summary, ['free main mem', str(main_mem), 'free cuda mem', str(cuda_mem)])

        # post_iteration --------------------------------------------------------------------------------------------------

        for i in range(len(test_target_pcs)):
            if all_target_errors[i] < best_errors[i]:
                best_recons[i] = copy.deepcopy(to_numpy(all_target_recons[i]))
                best_errors[i] = copy.deepcopy(to_numpy(all_target_errors[i]))
                best_centers[i] = copy.deepcopy(to_numpy(all_centers[i])) 
                best_translations[i] = copy.deepcopy(to_numpy(all_translations[i])) 
                best_scales[i] = copy.deepcopy(to_numpy(all_scales[i]))
                best_angles[i] = copy.deepcopy(to_numpy(all_angles[i]))

    #if to_vis:
        #vis_folder = os.path.join(test_folder, 'vis'+str(iteration))
        #if not os.path.exists(vis_folder):
            #os.makedirs(vis_folder)
        #for i in range(min(len(test_target_pcs), batch_size)):
            #display_pcs([test_target_pcs[i]], os.path.join(vis_folder, str(i)+'target.png'), True)
            #display_pcs(best_recons[i].reshape(-1, part_point_num, 3), os.path.join(vis_folder, str(i)+'_best_recon.png'), True)
   
    for i in range(len(test_target_indices)):
        joblib.dump(test_target_pcs[i], os.path.join(test_folder, str(test_target_indices[i])+'target_pc.joblib'))
        joblib.dump(best_recons[i], os.path.join(test_folder, str(test_target_indices[i])+'recon.joblib'))
        joblib.dump(best_centers[i], os.path.join(test_folder, str(test_target_indices[i])+'center.joblib'))
        joblib.dump(best_translations[i], os.path.join(test_folder, str(test_target_indices[i])+'translation.joblib'))
        joblib.dump(best_scales[i], os.path.join(test_folder, str(test_target_indices[i])+'scale.joblib'))
        joblib.dump(best_angles[i], os.path.join(test_folder, str(test_target_indices[i])+'angle.joblib'))


def direct_train(exp_folder, folder, category, target_pcs, target_indices, part_encs, part_vae, part_num, use_symmetry):

    process_interval = 60
    process_count = 6

    if use_borrow:
        shape_distance_mat = cluster_targets(target_pcs, exp_folder, use_symmetry)

    best_errors = [np.inf]*len(target_pcs)
    best_recons = [None]*len(target_pcs)
    best_centers = [None]*len(target_pcs)
    best_translations = [None]*len(target_pcs)
    best_scales = [None]*len(target_pcs)
    best_angles = [None]*len(target_pcs)

    all_centers = torch.rand((len(target_pcs), part_num, enc_dim), device=device, dtype=torch.float, requires_grad=True)
    all_translations = torch.rand((len(target_pcs), part_num, 3), device=device, dtype=torch.float, requires_grad=True)
    all_angles = (-np.pi + 2 * np.pi * torch.rand((len(target_pcs), part_num), device=device)).clone().detach().requires_grad_(True)
    all_scales = torch.ones((len(target_pcs), part_num, 3), device=device, dtype=torch.float, requires_grad=True)
    shape_optimizer = torch.optim.Adam([all_centers]+[all_translations]+[all_angles], lr=shape_lr)
    process_records = defaultdict(list)

    part_vae.eval()

    #main_mem, cuda_mem = get_free_mem()
    #append_items_to_file(summary, ['free main mem', str(main_mem), 'free cuda mem', str(cuda_mem)])

    start = time.time()

    max_iteration = 1400
    for iteration in range(0, max_iteration):
        
        print('part_num', part_num, 'train iteration', iteration, '-----------------------------------')

        #append_items_to_file(summary, ['-----------------train optim iteration  ', str(iteration)])

        iteration_shape_loss = 0

        all_target_errors = None
        all_target_recons = None

        batch_size = 16

        for batch_start in range(0, len(target_pcs), batch_size):
            
            batched_target_pcs = torch.tensor(np.array(target_pcs[batch_start:batch_start+batch_size]), device=device, dtype=torch.float)

            batched_centers = all_centers[batch_start:batch_start+batch_size]
            batched_translations = all_translations[batch_start:batch_start+batch_size]
            batched_scales = all_scales[batch_start:batch_start+batch_size]
            batched_angles = all_angles[batch_start:batch_start+batch_size]

            if use_symmetry:
                batched_part_recons = forward_with_symmetry(part_num, batched_centers, part_vae, batched_translations, batched_scales, batched_angles)
                batched_reshaped_part_recons = batched_part_recons.reshape(-1, 2*part_num*part_point_num, 3)
                collision_errors = get_collision_distance(batched_reshaped_part_recons.reshape(-1, 2*part_num, part_point_num, 3), part_num, use_symmetry)
            else:
                batched_part_recons = forward(batched_centers, part_vae, batched_translations, batched_scales, batched_angles)
                batched_reshaped_part_recons = batched_part_recons.reshape(-1, part_num*part_point_num, 3)
                collision_errors = get_collision_distance(batched_reshaped_part_recons.reshape(-1, part_num, part_point_num, 3), part_num, use_symmetry)

            #print('collision_errors', collision_errors)
            batched_target_recons = batched_reshaped_part_recons
            avg_recon_errors, _ = chamfer_distance(batched_target_recons, batched_target_pcs, batch_reduction=None)
            batched_target_errors = avg_recon_errors + collision_weight * collision_errors
            
            loss = torch.mean(batched_target_errors)
            shape_optimizer.zero_grad()
            loss.backward()
            shape_optimizer.step()
            iteration_shape_loss += loss.item()
            
            if all_target_errors is None:
                all_target_errors = to_numpy(batched_target_errors)
            else:
                all_target_errors = np.concatenate((all_target_errors, to_numpy(batched_target_errors)), axis=0)

            if all_target_recons is None:
                all_target_recons = to_numpy(batched_target_recons)
            else:
                all_target_recons = np.concatenate((all_target_recons, to_numpy(batched_target_recons)), axis=0)

        # post_iteration --------------------------------------------------------------------------------------------------
        print('epoch loss', iteration_shape_loss/len(target_pcs))

        for i in range(len(target_pcs)):
            if all_target_errors[i] < best_errors[i]:
                best_recons[i] = copy.deepcopy(to_numpy(all_target_recons[i]))
                best_errors[i] = copy.deepcopy(to_numpy(all_target_errors[i]))
                best_centers[i] = copy.deepcopy(to_numpy(all_centers[i])) 
                best_translations[i] = copy.deepcopy(to_numpy(all_translations[i])) 
                best_scales[i] = copy.deepcopy(to_numpy(all_scales[i]))
                best_angles[i] = copy.deepcopy(to_numpy(all_angles[i]))

        if iteration >= 30: 
            new_all_centers = []
            new_all_translations = []
            new_all_scales = []
            new_all_angles = []
            for i in range(len(target_pcs)):
                if use_shift and ((len(process_records[i])==0) or (len(process_records[i]) < process_count and (iteration - process_records[i][-1] == process_interval))):
                    process_records[i].append(iteration) 
                    new_shape_centers_i, new_translations_i, new_scales_i, new_angle_i, new_recons, regions_before, regions_uncover, regions_after = process_shape_new(target_pcs[i], all_target_recons[i], part_vae, part_num, use_symmetry)
                    new_all_centers.append(new_shape_centers_i)
                    new_all_translations.append(new_translations_i)
                    new_all_scales.append(new_scales_i)
                    new_all_angles.append(new_angle_i)

                else:
                    new_all_centers.append(all_centers[i])
                    new_all_translations.append(all_translations[i])
                    new_all_scales.append(all_scales[i])
                    new_all_angles.append(all_angles[i])

            updated_all_centers = []
            updated_all_translations = []
            updated_all_scales = []
            updated_all_angles = []

            for i in range(len(target_pcs)):
                if len(target_pcs) > 1 and use_borrow and (len(process_records[i]) >= process_count and iteration - process_records[i][-1] == 100):
                    reset, index, sorted_nb_indices = learn_from_neighbor(i, shape_distance_mat[i], best_errors, best_centers, best_translations, best_scales, best_angles)
                    
                    if reset == True:
                        if index < 0:
                            updated_center = torch.rand(best_centers[0].shape, device=device)
                            updated_translation = torch.rand(best_translations[0].shape, device=device)
                            updated_scale = torch.ones(best_scales[0].shape, device=device)
                            updated_angle = -np.pi + 2 * np.pi * torch.rand(best_angles[0].shape, device=device)
                        else:
                            updated_center = torch.tensor(best_centers[index], device=device)
                            updated_translation = torch.tensor(best_translations[index], device=device)
                            updated_scale = torch.tensor(best_scales[index], device=device)
                            updated_angle = torch.tensor(best_angles[index], device=device)
                        process_records[i].clear()
                    else:
                        updated_center = new_all_centers[i]
                        updated_translation = new_all_translations[i]
                        updated_scale = new_all_scales[i]
                        updated_angle = new_all_angles[i]
                else:
                    updated_center = new_all_centers[i]
                    updated_translation = new_all_translations[i]
                    updated_scale = new_all_scales[i]
                    updated_angle = new_all_angles[i]

                updated_all_centers.append(updated_center)
                updated_all_translations.append(updated_translation)
                updated_all_scales.append(updated_scale)
                updated_all_angles.append(updated_angle)

            all_centers = torch.stack(updated_all_centers)
            all_centers = all_centers.clone().detach().requires_grad_(True)
            all_translations = torch.stack(updated_all_translations)
            all_translations = all_translations.clone().detach().requires_grad_(True)
            all_scales = torch.stack(updated_all_scales)
            all_scales = all_scales.clone().detach().requires_grad_(True)
            all_angles = torch.stack(updated_all_angles)
            all_angles = all_angles.clone().detach().requires_grad_(True)
            shape_optimizer = torch.optim.Adam([all_centers]+[all_translations]+[all_angles], lr=shape_lr)
    
    for i in range(len(target_pcs)):
        joblib.dump(target_pcs[i], os.path.join(folder, str(target_indices[i])+'target_pc.joblib'))
        joblib.dump(best_recons[i], os.path.join(folder, str(target_indices[i])+'recon.joblib'))
        joblib.dump(best_centers[i], os.path.join(folder, str(target_indices[i])+'center.joblib'))
        joblib.dump(best_translations[i], os.path.join(folder, str(target_indices[i])+'translation.joblib'))
        joblib.dump(best_scales[i], os.path.join(folder, str(target_indices[i])+'scale.joblib'))
        joblib.dump(best_angles[i], os.path.join(folder, str(target_indices[i])+'angle.joblib'))
        

def run(data_dir, exp_folder, part_dataset, part_category, part_count, shape_dataset, shape_category, train_shape_count, test_shape_count, eval_on_train_shape_count, k):

    _, train_shape_ids, test_shape_ids, _ = read_split(global_args.split_file)
    
    source_shape_ids, _, _, _ = read_split(global_args.split_file)
    
    print('train_shape_ids', train_shape_ids)

    print('part_dataset', part_dataset)
    print('part_category', part_category)
    print('max_part_count', part_count)
    print('shape_category', shape_category)
    print('train_shape_count', train_shape_count)
    print('source shape count', len(source_shape_ids))
    print('train shape count', len(train_shape_ids))
    print('test shape count', len(test_shape_ids))
    print('k', k)

    part_meshes, part_vol_pcs, part_sur_pcs = get_parts(data_dir, part_dataset, part_category, part_count, source_shape_ids, False)

    part_vol_pcs, part_sur_pcs, part_meshes = calibrate_parts(part_vol_pcs, part_sur_pcs, part_meshes)

    train_shape_meshes, train_shape_vol_pcs, train_shape_sur_pcs = get_shapes(data_dir, shape_dataset, shape_category, train_shape_ids, train_shape_count)
    test_shape_meshes, test_shape_vol_pcs, test_shape_sur_pcs = get_shapes(data_dir, shape_dataset, shape_category, test_shape_ids, test_shape_count)

    init_part_vae, init_part_encs = pre_train(exp_folder, np.array(part_vol_pcs), enc_dim, vae_lr)

    part_vae = copy.deepcopy(init_part_vae)
    part_encs = copy.deepcopy(init_part_encs)
    part_encs = torch.tensor(part_encs, device=device, dtype=torch.float)

    direct_train_folder = os.path.join(exp_folder, 'direct_train', str(k))
    if not os.path.exists(direct_train_folder):
        os.makedirs(direct_train_folder)
    
    train_result_folder = os.path.join(exp_folder, 'train_results', str(k))
    if not os.path.exists(train_result_folder):
        os.makedirs(train_result_folder)

    train_sym_shape_indices, train_nonsym_shape_indices = compute_symmetry_infos(train_shape_vol_pcs, True, exp_folder, shape_category)
    if len(train_sym_shape_indices) > 0:
        direct_train(exp_folder, direct_train_folder, shape_category, [train_shape_vol_pcs[idx] for idx in train_sym_shape_indices], train_sym_shape_indices, part_encs, part_vae, int(k*0.5), True)
    if len(train_nonsym_shape_indices) > 0:
        direct_train(exp_folder, direct_train_folder, shape_category, [train_shape_vol_pcs[idx] for idx in train_nonsym_shape_indices], train_nonsym_shape_indices, part_encs, part_vae, k, False)

    retrieve_train_ids = []
    retrieve_train_indices = []
    for idx in train_sym_shape_indices:
        if train_shape_ids[idx] in train_shape_ids[0:eval_on_train_shape_count]:
            retrieve_train_indices.append(idx)
            retrieve_train_ids.append(train_shape_ids[idx])

    if use_parallel:
        Parallel(n_jobs=2)(delayed(retrieve_symmetry)(train_shape_vol_pcs[retrieve_train_indices[i]], part_vol_pcs, int(k*0.5), retrieve_train_indices[i], retrieve_train_ids[i], part_vae, part_encs, direct_train_folder, train_result_folder) for i in range(len(retrieve_train_indices)))
    else:
        for i in range(len(retrieve_train_indices)):
            retrieve_symmetry(train_shape_vol_pcs[retrieve_train_indices[i]], part_vol_pcs, int(k*0.5), retrieve_train_indices[i], retrieve_train_ids[i], part_vae, part_encs, direct_train_folder, train_result_folder)
        
    retrieve_train_ids = []
    retrieve_train_indices = []
    for idx in train_nonsym_shape_indices:
        if train_shape_ids[idx] in train_shape_ids[0:eval_on_train_shape_count]:
            retrieve_train_indices.append(idx)
            retrieve_train_ids.append(train_shape_ids[idx])
    if use_parallel:
        Parallel(n_jobs=2)(delayed(retrieve_single_ours)(train_shape_vol_pcs[retrieve_train_indices[i]], part_vol_pcs, int(k), retrieve_train_indices[i], retrieve_train_ids[i], part_vae, part_encs, direct_train_folder, train_result_folder) for i in range(len(retrieve_train_indices)))
    else:
        for i in range(len(retrieve_train_indices)):
            retrieve_single_ours(train_shape_vol_pcs[retrieve_train_indices[i]], part_vol_pcs, int(k), retrieve_train_indices[i], retrieve_train_ids[i], part_vae, part_encs, direct_train_folder, train_result_folder)
    
    direct_test_folder = os.path.join(exp_folder, 'direct_test', str(k))
    if not os.path.exists(direct_test_folder):
        os.makedirs(direct_test_folder)
    
    test_result_folder = os.path.join(exp_folder, 'test_results', str(k))
    if not os.path.exists(test_result_folder):
        os.makedirs(test_result_folder)

    test_sym_shape_indices, test_nonsym_shape_indices = compute_symmetry_infos(test_shape_vol_pcs, False, exp_folder, shape_category)

    if len(test_sym_shape_indices) > 0:
        direct_test(direct_test_folder, direct_train_folder, [test_shape_vol_pcs[idx] for idx in test_sym_shape_indices], test_sym_shape_indices, [train_shape_vol_pcs[idx] for idx in train_sym_shape_indices] ,train_sym_shape_indices,  part_encs, part_vae, int(k*0.5), True)
    if len(test_nonsym_shape_indices) > 0:
        direct_test(direct_test_folder, direct_train_folder, [test_shape_vol_pcs[idx] for idx in test_nonsym_shape_indices], test_nonsym_shape_indices, [train_shape_vol_pcs[idx] for idx in train_nonsym_shape_indices], train_nonsym_shape_indices, part_encs, part_vae, k, False)

    retrieve_test_ids = []
    retrieve_test_indices = []
    for idx in test_sym_shape_indices:
        retrieve_test_indices.append(idx)
        retrieve_test_ids.append(test_shape_ids[idx])
    if use_parallel:
        Parallel(n_jobs=2)(delayed(retrieve_symmetry)(test_shape_vol_pcs[retrieve_test_indices[i]], part_vol_pcs, int(k*0.5), retrieve_test_indices[i], retrieve_test_ids[i], part_vae, part_encs, direct_test_folder, test_result_folder) for i in range(len(retrieve_test_indices)))
    else:
        for i in range(len(retrieve_test_indices)):
            retrieve_symmetry(test_shape_vol_pcs[retrieve_test_indices[i]], part_vol_pcs, int(k*0.5), retrieve_test_indices[i], retrieve_test_ids[i], part_vae, part_encs, direct_test_folder, test_result_folder)

    retrieve_test_ids = []
    retrieve_test_indices = []
    for idx in test_nonsym_shape_indices:
        retrieve_test_indices.append(idx)
        retrieve_test_ids.append(test_shape_ids[idx])
    if use_parallel:
        Parallel(n_jobs=2)(delayed(retrieve_single_ours)(test_shape_vol_pcs[retrieve_test_indices[i]], part_vol_pcs, int(k), retrieve_test_indices[i], retrieve_test_ids[i], part_vae, part_encs, direct_test_folder, test_result_folder) for i in range(len(retrieve_test_indices)))
    else:
        for i in range(len(retrieve_test_indices)):
            retrieve_single_ours(test_shape_vol_pcs[retrieve_test_indices[i]], part_vol_pcs, int(k), retrieve_test_indices[i], retrieve_test_ids[i], part_vae, part_encs, direct_test_folder, test_result_folder)
    

if __name__ == "__main__":

    exp_folder = os.path.join(global_args.exp_dir, global_args.part_dataset + global_args.part_category + '_to_' + global_args.shape_dataset + global_args.shape_category + str(global_args.train_shape_count) + 'shift' + str(use_shift) + 'borrow' + str(use_borrow))
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)

    run(global_args.data_dir, exp_folder, global_args.part_dataset, global_args.part_category, global_args.part_count, global_args.shape_dataset, global_args.shape_category, global_args.train_shape_count, global_args.test_shape_count, global_args.eval_on_train_shape_count, global_args.k)   
