


from audioop import avg
import numpy as np
import trimesh
import torch
import json

from collections import defaultdict
import networkx as nx
from scipy.spatial.distance import cdist
from util_vis import *
from sklearn.decomposition import PCA

part_point_num = 1024

def get_inside_dist(pcs1, pcs2, k=500):
    r = 0.1
    dist1to2 = torch.cdist(pcs1, pcs2)
    dists = torch.sum(torch.topk(torch.sum(torch.topk(torch.relu(r - dist1to2), k=k, dim=2).values, dim=2), k = k, dim=1).values, dim=1)
    return dists

def get_outside_dist(pcs1, pcs2):
    dist1to2 = torch.cdist(pcs1, pcs2)
    min_dists = torch.min(dist1to2, dim=2)
    dists = torch.min(min_dists, dim=0)
    return dists

def get_detach_distance_explicit(pcs):

    part_num = pcs.shape[0]
    target_num = pcs.shape[1]
    part_point_num = pcs.shape[2]

    detach_errors = torch.zeros(target_num, device=device)
    for i in range(part_num):
        repeated_part_recons_i = pcs[i].repeat(part_num-1, 1, 1)
        concatenated_other_recons = torch.cat((pcs[0:i], pcs[i+1:]), dim=0).reshape(-1, part_point_num, 3)
        detach_errors_i = get_outside_dist(repeated_part_recons_i, concatenated_other_recons)
        detach_errors_i = torch.mean(detach_errors_i.reshape(len(pcs[i]), -1), dim=1)    
    detach_errors = detach_errors/part_num
    return detach_errors

def get_collision_distance_explicit(pcs):

    if len(pcs) == 1:
      return torch.tensor([0.0], device=device, dtype=torch.float)  

    #print('pcs shape', pcs.shape)

    part_num = pcs.shape[0]
    target_num = pcs.shape[1]
    part_point_num = pcs.shape[2]

    collision_errors = torch.zeros(target_num, device=device)
    for i in range(part_num):
        repeated_part_recons_i = pcs[i].repeat(part_num-1, 1, 1)
        concatenated_other_recons = torch.cat((pcs[0:i], pcs[i+1:]), dim=0).reshape(-1, part_point_num, 3)
        collision_errors_i = get_inside_dist(repeated_part_recons_i, concatenated_other_recons)
        collision_errors_i = torch.mean(collision_errors_i.reshape(len(pcs[i]), -1), dim=1)
        max_coords = torch.max(pcs[i], dim=1).values
        min_coords = torch.min(pcs[i], dim=1).values
        diag_i = torch.norm(max_coords - min_coords, dim=1)
        collision_errors += collision_errors_i
    collision_errors = collision_errors/part_num

    return collision_errors

def get_collision_distance(pcs, k, use_symmetry):

    pcs = pcs.permute(1,0,2,3)
    part_num = pcs.shape[0]
    target_num = pcs.shape[1]
    part_point_num = pcs.shape[2]
    collision_errors = torch.zeros(target_num, device=device)

    all_part_recons = None
    all_other_recons = None

    for i in range(k):
        
        if use_symmetry:
            part_recons_i = pcs[i].repeat(part_num-2, 1, 1)
            other_recons_i = torch.cat((pcs[0:i], pcs[i+1:i+k], pcs[i+k+1:]), dim=0).reshape(-1, part_point_num, 3)
        else:
            part_recons_i = pcs[i].repeat(part_num-1, 1, 1)
            other_recons_i = torch.cat((pcs[0:i], pcs[i+1:]), dim=0).reshape(-1, part_point_num, 3)

        if all_part_recons == None:
            all_part_recons = part_recons_i
        else:
            all_part_recons = torch.cat((all_part_recons, part_recons_i), dim=0)
        
        if all_other_recons == None:
            all_other_recons = other_recons_i
        else:
            all_other_recons = torch.cat((all_other_recons, other_recons_i), dim=0)

    if len(all_part_recons) == 0 or len(all_other_recons) == 0:
        return collision_errors

    collision_errors = get_inside_dist(all_part_recons, all_other_recons)
    collision_errors = collision_errors.reshape(k, -1)
    collision_errors = collision_errors.reshape(k, target_num, -1)
    collision_errors = collision_errors.permute(1, 0, 2)
    collision_errors = torch.mean(torch.mean(collision_errors, dim=2), dim=1)
    collision_errors = collision_errors/part_num
    return collision_errors
