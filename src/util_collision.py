


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

'''
def pc_to_bbox_collision_distances_batched(pcs, handle_zs, handle_ts, handle_rotmats):

    standarized_pcs = pcs - handle_ts.unsqueeze(dim=1)
    standarized_pcs = torch.bmm(handle_rotmats.inverse(), standarized_pcs.transpose(1,2)).transpose(1,2)
    repeated_handle_zs = handle_zs.unsqueeze(dim=1).repeat_interleave(standarized_pcs.shape[1], dim=1)

    dists = repeated_handle_zs - torch.abs(standarized_pcs) 

    threshold = ((torch.min(handle_zs, dim=1).values*0.2).unsqueeze(dim=1).unsqueeze(dim=2)).repeat((1, pcs.shape[1], 3))

    #threshold = 0
    dists[dists > threshold] = dists[dists > threshold]
    dists[dists < threshold] = -99999

    final_dists = torch.relu(torch.min(dists, dim=2).values)
    final_dists = final_dists * final_dists
    final_dists = torch.sum(final_dists, dim=1)
    #print('final_dists shape', final_dists.shape)
    
    return torch.mean(final_dists)

def pc_to_bbox_collision_distances(pc, z, t, rotmat):

    z = torch.tensor(z, device=device, dtype=torch.float)
    t = torch.tensor(t, device=device, dtype=torch.float)
    rotmat = torch.tensor(rotmat, device=device, dtype=torch.float)

    #print('pc shape', pc.shape)
    #print('z shape', z.shape)
    #print('t shape', t.shape)
    #print('rotmat shape', rotmat.shape)

    standarized_pc = pc - t
    #print('standarized_pc shape', standarized_pc.shape)

    standarized_pc = torch.matmul(rotmat.inverse(), standarized_pc.transpose(0,1)).transpose(0,1)
    #print('standarized_pc shape', standarized_pc.shape)
    
    repeated_zs = z.unsqueeze(dim=0).repeat_interleave(standarized_pc.shape[0], dim=0)
    #print('repeated_zs shape', repeated_zs.shape)

    dists = repeated_zs - torch.abs(standarized_pc) 
    #print('dists shape', dists.shape)

    #threshold = ((torch.min(handle_zs, dim=1).values*0.2).unsqueeze(dim=1).unsqueeze(dim=2)).repeat((1, pcs.shape[1], 3))

    threshold = 0.0

    #threshold = 0
    dists[dists > threshold] = dists[dists > threshold]
    dists[dists < threshold] = -99999

    #print('dists shape', dists.shape)

    final_dists = torch.relu(torch.min(dists, dim=1).values)
    
    #print('final_dists shape', final_dists.shape)

    #exit()

    final_dists = final_dists * final_dists
    #print('final_dists shape', final_dists.shape)
    summed_final_dist = torch.sum(final_dists, dim=0)

    #print('final_dists', final_dists)

    return summed_final_dist

def get_collision_distance(pc1, pc2):

    #print('pc1 shape', pc1.shape)
    rotmat1, t1, z1 = get_bbox(to_numpy(pc1))
    rotmat2, t2, z2 = get_bbox(to_numpy(pc2))
   
    collision_1to2 = pc_to_bbox_collision_distances(pc1, z2, t2, rotmat2)
    #print('collision_1to2', collision_1to2)
    
    collision_2to1 = pc_to_bbox_collision_distances(pc2, z1, t1, rotmat1)
    
    return collision_1to2+collision_2to1

'''

def get_inside_dist(pcs1, pcs2, k=500):
    r = 0.1
    dist1to2 = torch.cdist(pcs1, pcs2)
    #print('torch.relu(r - dist1to2) shape', torch.relu(r - dist1to2).shape)
    #print(torch.topk(torch.relu(r - dist1to2), k=50, dim=2).values.shape)
    #print(torch.topk(torch.sum(torch.topk(torch.relu(r - dist1to2), k=50, dim=2).values, dim=2), k = 50, dim=1).values.shape)

    dists = torch.sum(torch.topk(torch.sum(torch.topk(torch.relu(r - dist1to2), k=k, dim=2).values, dim=2), k = k, dim=1).values, dim=1)
    return dists

def get_outside_dist(pcs1, pcs2):
    dist1to2 = torch.cdist(pcs1, pcs2)
    min_dists = torch.min(dist1to2, dim=2)
    dists = torch.min(min_dists, dim=0)
    return dists

'''
def get_collision_distance_explicit(pcs):

    part_num = pcs.shape[0]
    target_num = pcs.shape[1]
    part_point_num = pcs.shape[2]

    collision_errors = torch.zeros(target_num, device=device)
    for i in range(part_num-1):
        repeated_part_recons_i = pcs[i].repeat(part_num-i-1, 1, 1)
        concatenated_other_recons = pcs[i+1:].reshape(-1, part_point_num, 3)
        collision_errors_i = get_inside_dist(repeated_part_recons_i, concatenated_other_recons)
        collision_errors_i = torch.mean(collision_errors_i.reshape(len(pcs[i]), -1), dim=1)
        collision_errors += collision_errors_i    
    collision_errors = collision_errors/part_num

    return collision_errors
'''

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
        #print('repeated_part_recons_i shape', repeated_part_recons_i.shape)
        #concatenated_other_recons = pcs[i+1:].reshape(-1, part_point_num, 3)
        concatenated_other_recons = torch.cat((pcs[0:i], pcs[i+1:]), dim=0).reshape(-1, part_point_num, 3)
        #print('concatenated_other_recons shape', concatenated_other_recons.shape)
        collision_errors_i = get_inside_dist(repeated_part_recons_i, concatenated_other_recons)
        collision_errors_i = torch.mean(collision_errors_i.reshape(len(pcs[i]), -1), dim=1)
        #print('collision_errors_i  shape', collision_errors_i.shape)
        max_coords = torch.max(pcs[i], dim=1).values
        #print('max_coords', max_coords)
        min_coords = torch.min(pcs[i], dim=1).values
        #print('min_coords', min_coords)
        diag_i = torch.norm(max_coords - min_coords, dim=1)
        #print('diag_i', diag_i.shape)
        collision_errors += collision_errors_i
    collision_errors = collision_errors/part_num

    return collision_errors

def get_collision_distance(pcs, k, use_symmetry):

    pcs = pcs.permute(1,0,2,3)
    #print('pcs shape', pcs.shape)

    part_num = pcs.shape[0]
    target_num = pcs.shape[1]
    part_point_num = pcs.shape[2]
    collision_errors = torch.zeros(target_num, device=device)

    #print('k', k)
    #print('part_num', part_num)

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

    #exit()

    if len(all_part_recons) == 0 or len(all_other_recons) == 0:
        return collision_errors

    #print('all_part_recons shape', all_part_recons.shape)
    #print('all_other_recons shape', all_other_recons.shape)
    
    collision_errors = get_inside_dist(all_part_recons, all_other_recons)
    #print('collision_errors shape', collision_errors.shape)
    collision_errors = collision_errors.reshape(k, -1)
    #print('collision_errors_i', collision_errors_i)
    #print('collision_errors shape', collision_errors.shape)

    collision_errors = collision_errors.reshape(k, target_num, -1)
    #print('collision_errors_i', collision_errors_i)
    #print('collision_errors shape', collision_errors.shape)
    #exit()

    collision_errors = collision_errors.permute(1, 0, 2)
    #print('collision_errors_i', collision_errors_i)
    #print('collision_errors shape', collision_errors.shape)

    collision_errors = torch.mean(torch.mean(collision_errors, dim=2), dim=1)

    #print('collision_errors_i  shape', collision_errors_i.shape)
    #exit()

    #max_coords = torch.max(pcs[i], dim=1).values
    #print('max_coords', max_coords)
    #min_coords = torch.min(pcs[i], dim=1).values
    #print('min_coords', min_coords)
    #diag_i = torch.norm(max_coords - min_coords, dim=1)
    #print('diag_i', diag_i.shape)

    #print('collision_errors shape', collision_errors.shape)
    #exit()

    collision_errors = collision_errors/part_num
    return collision_errors

    return collision_errors

def get_pred_distances(pcs1, pc_encs1, pcs2, pc_encs2, im_net):
    dist2to1 = im_net(pc_encs1, pcs2)
    dist1to2 = im_net(pc_encs2, pcs1)
    return torch.mean(torch.mean(dist1to2, dim=1),dim=0) + torch.mean(torch.mean(dist2to1, dim=1),dim=0)

def get_collision_distance_implicit(pcs1, pc_encs1, pcs2, pc_encs2, pcs3, pc_encs3, im_net):

    dist_1to2 = get_pred_distances(pcs1, pc_encs1, pcs2, pc_encs2, im_net)
    dist_1to3 = get_pred_distances(pcs1, pc_encs1, pcs3, pc_encs3, im_net)
    dist_2to3 = get_pred_distances(pcs2, pc_encs2, pcs3, pc_encs3, im_net)

    collisions = torch.stack([dist_1to2, dist_1to3, dist_2to3])
    return torch.mean(collisions)

'''
def get_collision_distance_explicit_symmetry(pcs, k, sym_labels):

    print('pcs shape', pcs.shape)

    if pcs.shape[1] <= 1:
      return torch.tensor([0.0], device=device, dtype=torch.float) 

    part_num = pcs.shape[0]
    target_num = pcs.shape[1]
    part_point_num = pcs.shape[2]

    collision_errors = torch.zeros(target_num, device=device)
    for i in range(part_num):

        print('i', i)

        sep0 = i-4*k
        sep1 = i-3*k
        sep2 = i-2*k
        sep3 = i-1*k
        sep4 = i
        sep5 = i+1*k
        sep6 = i+2*k
        sep7 = i+3*k
        sep8 = i+4*k

        seps = [sep0, sep1, sep2, sep3, sep4, sep5, sep6, sep7, sep8]

        print('seps', seps)

        repeated_part_recons_i = pcs[i].repeat(part_num-4, 1, 1)

        concatenated_other_recons = None
        for j in range(len(seps)):
            if seps[j] < part_num and seps[j] + k > 0:
                low = seps[j]+1
                hi = seps[j]+k
            else:
                low = np.inf
                hi = -np.inf

            if low != np.inf and hi != -np.inf:

                print('low', low, 'hi', hi)
                low = max(0, low)
                if concatenated_other_recons == None:
                    concatenated_other_recons = pcs[low:hi]
                else:
                    concatenated_other_recons = torch.cat((concatenated_other_recons, pcs[low:hi]), dim=0)
                print('concatenated_other_recons shape', concatenated_other_recons.shape)
        concatenated_other_recons = concatenated_other_recons.reshape(-1, part_point_num, 3)

        print('repeated_part_recons_i shape', repeated_part_recons_i.shape)
        print('concatenated_other_recons shape', concatenated_other_recons.shape)
        

        #print('concatenated_other_recons shape', concatenated_other_recons.shape)
        collision_errors_i = get_inside_dist(repeated_part_recons_i, concatenated_other_recons)
        collision_errors_i = torch.mean(collision_errors_i.reshape(len(pcs[i]), -1), dim=1)
        #print('collision_errors_i  shape', collision_errors_i.shape)
        max_coords = torch.max(pcs[i], dim=1).values
        #print('max_coords', max_coords)
        min_coords = torch.min(pcs[i], dim=1).values
        #print('min_coords', min_coords)
        #diag_i = torch.norm(max_coords - min_coords, dim=1)
        #print('diag_i', diag_i.shape)
        collision_errors += collision_errors_i
    collision_errors = collision_errors/part_num

    return collision_errors    
'''