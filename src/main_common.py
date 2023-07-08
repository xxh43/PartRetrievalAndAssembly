

import copy
import os
from re import A
import joblib
import numpy as np
from numpy import arange
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from main_ours_pretrain import *
from util_collision import *
from util_motion import *
from util_vis import *
from scipy.spatial.transform import Rotation as R
import copy
import gc
import os
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from util_vis import *
import copy
import gc
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import braycurtis, cdist
from scipy.spatial.transform import Rotation as R
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader
from util_motion import *
from util_vis import *
import time
from scipy.spatial.transform import Rotation
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
import matplotlib.pyplot as plt

torch.set_printoptions(precision=10)
bce_loss = torch.nn.BCELoss()

def get_y_aligned_bbox(pc):

    try:
        pc_2d = np.concatenate((np.expand_dims(pc[:, 0], axis=1), np.expand_dims(pc[:,2], axis=1)), axis=1)    
        to_origin, extents = trimesh.bounds.oriented_bounds(pc_2d, angle_digits=10)
        t_xz = to_origin[:2, :2].transpose().dot(-to_origin[:2, 2])
        size_y = np.max(pc[:,1]) - np.min(pc[:,1])
        t = np.array([t_xz[0], np.min(pc[:,1])+size_y*0.5, t_xz[1]])
        size = np.array([extents[0]*0.5, size_y*0.5, extents[1]*0.5])
        xdir = np.array([to_origin[0, 0], 0, to_origin[0, 1]])
        zdir = np.array([to_origin[1, 0], 0, to_origin[1, 1]])
        ydir = np.cross(xdir, zdir)
        rotmat = np.vstack([xdir, ydir, zdir]).T
    except:
        center = pc.mean(axis=0, keepdims=True)
        points = pc - center
        t = center[0, :]
        pca = PCA()
        pca.fit(points)
        pcomps = pca.components_
        points_local = np.matmul(pcomps, points.transpose()).transpose()
        size = (points_local.max(axis=0) - points_local.min(axis=0))*0.5
        xdir = pcomps[0, :]
        ydir = pcomps[1, :]
        xdir /= np.linalg.norm(xdir)
        ydir /= np.linalg.norm(ydir)
        zdir = np.cross(xdir, ydir)
        rotmat = np.vstack([xdir, ydir, zdir]).T

    return xdir, np.array([0, 1, 0]), zdir

def get_y_aligned_reset_angle(pc):

    try:
        pc = to_numpy(pc)
        xlocal, ylocal, zlocal = get_y_aligned_bbox(pc)
        local_mat = np.vstack([xlocal, ylocal, zlocal]).T
        xworld = [1.0, 0.0, 0.0]
        yworld = [0.0, 1.0, 0.0]
        zworld = [0.0, 0.0, 1.0]
        world_mat = np.vstack([xworld, yworld, zworld]).T
        rot_mat = np.matmul(np.linalg.inv(local_mat), world_mat)    
        r = Rotation.from_matrix(rot_mat)
        y_angle = r.as_euler('xyz', degrees=False)[1]
    except:
        return 0

    return y_angle

def reflect_region(region):

    print('region shape', region.shape)

    region = torch.tensor(region, device=device)

    region = copy.deepcopy(region).unsqueeze(dim=0)

    normals1 = torch.tensor([1,0,0], device=device, dtype=torch.float).unsqueeze(dim=0).repeat_interleave(len(region), dim=0).unsqueeze(dim=1)
    vecs = (region).permute(0, 2, 1)
    dots = torch.bmm(normals1, vecs)
    temp = torch.bmm(dots.permute(0, 2, 1), normals1)
    reflected_region = region - 2 * temp
    
    #print('region shape', region.shape)
    #print('reflected_region shape', reflected_region.shape)
    
    #merged_region = torch.cat((region[0], reflected_region[0]), dim=0)
    #print('merged_region shape', merged_region.shape)
    
    return to_numpy(reflected_region[0])

def translate_parts(part_pcs, translation_vectors):
    transformed_part_pcs = part_pcs
    transformed_part_pcs = translate_with_vector_batched(transformed_part_pcs, translation_vectors)
    return transformed_part_pcs

def up_rotate_parts(pcs, angles):    
    rigid_axes = torch.tensor([[0.0, 1.0, 0.0]], device=device, dtype=torch.float).repeat_interleave(len(pcs), dim=0)
    transformed_pcs = rotate_with_axis_center_angle_batched(pcs, rigid_axes, torch.mean(pcs, dim=1), angles)
    return transformed_pcs


def scale_parts(part_pcs, scales, is_isotropic=False):

    #lo = torch.tensor(lo, device=device, dtype=torch.float)
    #hi = torch.tensor(hi, device=device, dtype=torch.float)
    #scales = torch.clamp(torch.abs(scales), lo, hi)

    reset_vectors = torch.mean(part_pcs, dim=1).unsqueeze(dim=1).repeat_interleave(part_pcs.shape[1], dim=1)
    part_pcs = part_pcs - reset_vectors

    if is_isotropic:
        transformed_part_pcs = isotropic_scale_with_value_batched(part_pcs, torch.abs(scales[:, 0]))
    else:
        transformed_part_pcs = anisotropic_scale_with_value_batched(part_pcs, torch.abs(scales))
    
    transformed_part_pcs = transformed_part_pcs + reset_vectors
    return transformed_part_pcs

def reflect_parts(pcs, axes):    
    centers = torch.zeros(pcs.shape, device=device, dtype=torch.float)
    transformed_pcs = reflect_pc_batched(pcs, axes, centers)
    return transformed_pcs

def assemble_parts(part_pcs, scales, up_angles, translations):

    #print('part_pcs shape', part_pcs.shape)
    #print('scales shape', scales.shape)
    #print('up_angles shape', up_angles.shape)
    #print('translations shape', translations.shape)
    #exit()

    transformed_part_pcs = part_pcs
    #if scales != None:
        #transformed_part_pcs = scale_parts(transformed_part_pcs, scales)
    if up_angles != None:
        transformed_part_pcs = up_rotate_parts(transformed_part_pcs, up_angles)
    if translations != None:
        transformed_part_pcs = translate_parts(transformed_part_pcs, translations)
    return transformed_part_pcs

def calibrate_parts(part_vol_pcs, part_sur_pcs, part_meshes):
    calibrated_part_vol_pcs = []
    calibrated_part_sur_pcs = []
    calibrated_part_meshes = []
    for i in range(len(part_vol_pcs)):
        
        origin = np.array([0, 0, 0])
        calibrate_vector = torch.tensor(origin - np.mean(part_vol_pcs[i], axis=0), device=device, dtype=torch.float)
        calibrate_angle = torch.tensor(get_y_aligned_reset_angle(part_vol_pcs[i]), device=device, dtype=torch.float)
        min_values = np.min(part_vol_pcs[i], axis=0)
        max_values = np.max(part_vol_pcs[i], axis=0)
        diag = np.linalg.norm(max_values - min_values)
        calibrate_scales = torch.tensor(np.array([2.0/diag, 2.0/diag, 2.0/diag]), device=device, dtype=torch.float)
        part_vol_pc = torch.tensor(part_vol_pcs[i], device=device, dtype=torch.float)
        
        calibrated_part_vol_pc = part_vol_pc + calibrate_vector        
        calibrated_part_vol_pc = up_rotate_parts(torch.stack([calibrated_part_vol_pc]), torch.stack([calibrate_angle]))[0]
        #calibrated_part_vol_pc = anisotropic_scale_with_value_batched(torch.stack([calibrated_part_vol_pc]), torch.stack([calibrate_scales]))[0]
        calibrated_part_vol_pcs.append(to_numpy(calibrated_part_vol_pc))

        if len(part_sur_pcs) > 0:
            part_sur_pc = torch.tensor(part_sur_pcs[i], device=device, dtype=torch.float)
            calibrated_part_sur_pc = part_sur_pc + calibrate_vector        
            calibrated_part_sur_pc = up_rotate_parts(torch.stack([calibrated_part_sur_pc]), torch.stack([calibrate_angle]))[0]
            #calibrated_part_sur_pc = anisotropic_scale_with_value_batched(torch.stack([calibrated_part_sur_pc]), torch.stack([calibrate_scales]))[0]
            calibrated_part_sur_pcs.append(to_numpy(calibrated_part_sur_pc))

        if len(part_meshes) > 0:
            part_mesh_vertices = torch.tensor(part_meshes[i].vertices, device=device, dtype=torch.float)
            calibrated_part_mesh_vertices = part_mesh_vertices + calibrate_vector        
            calibrated_part_mesh_vertices = up_rotate_parts(torch.stack([calibrated_part_mesh_vertices]), torch.stack([calibrate_angle]))[0]
            #calibrated_part_mesh_vertices = anisotropic_scale_with_value_batched(torch.stack([calibrated_part_mesh_vertices]), torch.stack([calibrate_scales]))[0]
            calibrated_part_mesh = copy.deepcopy(part_meshes[i])
            calibrated_part_mesh.vertices = to_numpy(calibrated_part_mesh_vertices)
            calibrated_part_meshes.append(calibrated_part_mesh)
        
    return calibrated_part_vol_pcs, calibrated_part_sur_pcs, calibrated_part_meshes

def get_segment_enc(segment, part_vae):
    calibrate_translation = -torch.mean(segment, dim=0)
    calibrate_angle = torch.tensor(get_y_aligned_reset_angle(segment), device=device, dtype=torch.float)
    segment = segment + calibrate_translation
    segment = up_rotate_parts(torch.stack([segment]), torch.stack([calibrate_angle]))[0]
    segment_enc, _ = part_vae.encode(torch.stack([segment]).transpose(1, 2))
    segment_enc = segment_enc[0]
    return segment_enc

def get_nearest_part_indices(shape_enc, part_encs, n):
    #print('part_encs shape', part_encs.shape)
    shape_encs = shape_enc.reshape(-1, enc_dim)
    #print('shape_encs shape', shape_encs.shape)
    dists = torch.cdist(shape_encs, part_encs)
    #print('dists shape', dists.shape)

    #min_dists = torch.min(dists, dim=1).values
    mink_indices = torch.topk(dists, k=n, dim=1, largest=False).indices
    #print('mink_indices shape', mink_indices.shape)

    #min_dists = torch.mean(min_dists.reshape(-1, n), dim=1)
    #print('min_dists shape', min_dists.shape)
    return mink_indices[0]

def retrieve_single(pc, other_pc, pc_enc, shape_pc, part_pcs, part_vae, part_encs, is_conditioned=False):

    if pc_enc is not None:
        pc_enc = torch.tensor(pc_enc, device=device, dtype=torch.float)

    region_min_point_count = 20

    pcs = pc+other_pc
    #if other_pc != None:
        #pcs.append(other_pc)

  
    point_min_distances = torch.zeros(shape_pc.shape[0], device=device)
    point_min_distances[:] = np.inf
    point_part_indices = torch.zeros(shape_pc.shape[0], device=device)
    for j in range(len(pcs)):
        part_recon_pc = pcs[j]
        dists = torch.cdist(torch.stack([shape_pc]), torch.stack([part_recon_pc]))[0]
        min_values = torch.min(dists, dim=1).values
        #min_indices = torch.min(dists, dim=1).indices
        point_part_indices[min_values < point_min_distances] = j
        point_min_distances[min_values < point_min_distances] = min_values[min_values < point_min_distances]
    
    target_regions = []
    #region_encs = []
    for j in range(len(pcs)):
        part_region = shape_pc[point_part_indices == j]
        if len(part_region) < region_min_point_count:
            r_to_t_dists = torch.cdist(torch.stack([pcs[j]]), torch.stack([shape_pc]))[0]
            r_to_t_min_indices = torch.min(r_to_t_dists, dim=1).indices
            part_region = shape_pc[r_to_t_min_indices[0:region_min_point_count]]
        target_regions.append(part_region)
        #region_encs.append(get_segment_enc(part_region, part_vae))
    #region_encs = torch.stack(region_encs)
    #print('region_encs shape', region_encs.shape)

    nearest_part_indices = arange(0, len(part_pcs))
    #print('nearest_part_indices shape', nearest_part_indices.shape)
    nearest_part_pcs = torch.tensor(part_pcs, device=device, dtype=torch.float)

    for i in range(len(target_regions)):

        #nearest_part_indices = get_nearest_part_indices(pc_enc, part_encs, 40)
        #print('nearest_part_indices shape', nearest_part_indices.shape)
        #nearest_part_pcs = torch.tensor([part_pcs[v] for v in nearest_part_indices], device=device, dtype=torch.float)
        #print('nearest_part_pcs shape', nearest_part_pcs.shape)
        #exit()

        repeated_target_region = target_regions[i].unsqueeze(dim=0).repeat_interleave(len(nearest_part_pcs), dim=0)
        retrieval_translations = torch.zeros((len(nearest_part_pcs), 3), device=device, requires_grad=True)
        retrieval_scales = torch.ones((len(nearest_part_pcs), 3), device=device, requires_grad=True)
        retrieval_up_angles = torch.ones((len(nearest_part_pcs), 1), device=device, requires_grad=True)
        #retrieval_optimizer = torch.optim.Adam([retrieval_translations]+[retrieval_scales]+[retrieval_up_angles], lr=retrieval_lr)
        retrieval_optimizer = torch.optim.Adam([retrieval_translations]+[retrieval_up_angles], lr=retrieval_lr)
        for retrieval_iteration in range(retrieval_max_iteration):
            transformed_nearest_part_pcs = assemble_parts(nearest_part_pcs, None, retrieval_up_angles.squeeze(dim=1), retrieval_translations)
            candidate_errors, _ = chamfer_distance(transformed_nearest_part_pcs, repeated_target_region, batch_reduction=None)
            loss = torch.mean(candidate_errors)

            retrieval_optimizer.zero_grad()
            loss.backward()
            retrieval_optimizer.step()


        if is_conditioned:
            #return None, None, None, None
            min_retrieval_error = np.min(to_numpy(candidate_errors))
            print('min_retrieval_error', min_retrieval_error)
            if min_retrieval_error < 0.1:
                best_candidate_index = np.argmin(to_numpy(candidate_errors))
                return nearest_part_indices[best_candidate_index], retrieval_translations[best_candidate_index], retrieval_scales[best_candidate_index], retrieval_up_angles[best_candidate_index], pcs, target_regions
            else:
                return None, None, None, None, pcs, target_regions
        else:
            best_candidate_index = np.argmin(to_numpy(candidate_errors))
            return nearest_part_indices[best_candidate_index], retrieval_translations[best_candidate_index], retrieval_scales[best_candidate_index], retrieval_up_angles[best_candidate_index], pcs, target_regions

def retrieve_symmetry(shape_pc, part_pcs, k, idx, shape_id, part_vae, part_encs, direct_folder, result_folder):
    #print('retrieve symmetry ... ')

    '''
    print('retrieve symmetry ... ', shape_id)
    if os.path.isfile(os.path.join(result_folder, str(shape_id)+'_part_indices.joblib')) and \
        os.path.isfile(os.path.join(result_folder, str(shape_id)+'_part_syms.joblib')) and \
        os.path.isfile(os.path.join(result_folder, str(shape_id)+'_part_translations.joblib')) and \
        os.path.isfile(os.path.join(result_folder, str(shape_id)+'_part_scales.joblib')) and \
        os.path.isfile(os.path.join(result_folder, str(shape_id)+'_part_angles.joblib')) and \
        os.path.isfile(os.path.join(result_folder, str(shape_id)+'_pred_part_pcs.joblib')):
        return 
    '''

    if os.path.isfile(os.path.join(direct_folder, str(idx)+'center.joblib')):
        part_recon_encs = joblib.load(os.path.join(direct_folder, str(idx)+'center.joblib')).reshape(-1, enc_dim)
    else:
        part_recon_encs = [None]*10

    part_recon_pcs = joblib.load(os.path.join(direct_folder, str(idx)+'recon.joblib')).reshape(-1, part_point_num, 3)
    
    shape_pc = torch.tensor(shape_pc, device=device, dtype=torch.float)
    part_recon_pcs = torch.tensor(part_recon_pcs, device=device, dtype=torch.float)

    retrieved_part_indices = []
    retrieved_part_syms = []
    retrieved_part_translations = []
    retrieved_part_scales = []
    retrieved_part_up_angles = []

    vis_recon = []
    vis_seg = []
    print('k------------', k)
    print('part_recon_pcs shape', part_recon_pcs.shape)

    for i in range(k):

        overlap = get_inside_dist( part_recon_pcs[i].unsqueeze(dim=0), part_recon_pcs[i+k].unsqueeze(dim=0), 5)[0]
        #print('overlap', overlap)
        
        retrieved = False
        
        if overlap > overlap_threshold:
            sym_group_pc = [torch.cat((part_recon_pcs[i], part_recon_pcs[i+k]))]
            other_pc = []
            for j in range(len(part_recon_pcs)):
                if j != i or j != i+k:
                    other_pc.append(part_recon_pcs[j])
            sym_group_index, sym_group_translation, sym_group_scale, sym_group_angle, temp_recons, temp_segs = retrieve_single(sym_group_pc, other_pc, part_recon_encs[i], shape_pc, part_pcs, part_vae, part_encs, False)
            if sym_group_index is not None:
                retrieved = True
                retrieved_part_indices += [sym_group_index]
                retrieved_part_syms += [False]
                retrieved_part_translations += [sym_group_translation]
                retrieved_part_scales += [sym_group_scale]
                retrieved_part_up_angles += [sym_group_angle]

            vis_recon += sym_group_pc
            vis_seg.append(temp_segs[0])

        if retrieved is False:
            print('retrieved  is False')
            
            pc = []
            other_pc = []
            for j in range(len(part_recon_pcs)):
                if j == i:
                    pc.append(part_recon_pcs[j])
                else:
                    other_pc.append(part_recon_pcs[j])

            sym_index, sym_translation, sym_scale, sym_angle, temp_recons, temp_segs = retrieve_single(pc, other_pc, part_recon_encs[i], shape_pc, part_pcs, part_vae, part_encs, False)
            
            vis_recon += [part_recon_pcs[i], part_recon_pcs[i+k]]
            vis_seg.append(temp_segs[0])
            vis_reflected = reflect_region(temp_segs[0])
            vis_seg.append(vis_reflected)

            #merge_threshold = 0.0005
            #overlap, _ = chamfer_distance(torch.stack([part_recon_pcs[i]]), torch.stack([part_recon_pcs[i+k]]))

            #if overlap < merge_threshold:
                #retrieved_part_indices += [sym_index]
                #retrieved_part_syms += [False]
                #retrieved_part_translations += [sym_translation]
                #retrieved_part_scales += [sym_scale]
                #retrieved_part_up_angles += [sym_angle]
            #else:
            retrieved_part_indices += [sym_index, sym_index]
            retrieved_part_syms += [False, True]
            retrieved_part_translations += [sym_translation, sym_translation]
            retrieved_part_scales += [sym_scale, sym_scale]
            retrieved_part_up_angles += [sym_angle, sym_angle]

    # debug code
    #retrieved_part_pcs = torch.stack([torch.tensor(part_pcs[v], device=device) for v in retrieved_part_indices])
    #print('retrieved_part_indices', retrieved_part_indices)
    #display_pcs(retrieved_part_pcs)
    #retrieved_part_scales = torch.stack(retrieved_part_scales)
    #retrieved_part_up_angles = torch.stack(retrieved_part_up_angles)
    #retrieved_part_translations = torch.stack(retrieved_part_translations)
    #transformed_nearest_part_pcs = assemble_parts(retrieved_part_pcs, retrieved_part_scales, retrieved_part_up_angles.squeeze(dim=1), retrieved_part_translations)
    #display_pcs([shape_pc])
    #display_pcs(transformed_nearest_part_pcs)
    #exit()

    #merged_part_recon_pcs= []
    #for j in range(k):
        #merged_part_recon_pcs.append(torch.cat((part_recon_pcs[j], part_recon_pcs[j+k])))    

    #display_pcs(vis_recon, os.path.join(result_folder, str(shape_id)+'recon.png'), True)

    recon_blender_folder = os.path.join(result_folder, str(shape_id)+'recon_blender'+str(k))
    if not os.path.exists(recon_blender_folder):
        os.makedirs(recon_blender_folder)
    for j in range(len(vis_recon)):
        joblib.dump(vis_recon[j], os.path.join(recon_blender_folder, str(j)+'.joblib'))
    
    #display_pcs(vis_seg, os.path.join(result_folder, str(shape_id)+'segs.png'), True)

    seg_blender_folder = os.path.join(result_folder, str(shape_id)+'seg_blender'+str(k))
    if not os.path.exists(seg_blender_folder):
        os.makedirs(seg_blender_folder)
    for j in range(len(vis_seg)):
        joblib.dump(vis_seg[j], os.path.join(seg_blender_folder, str(j)+'.joblib'))

    #joblib.dump(retrieved_part_indices, os.path.join(result_folder, str(shape_id)+'_.joblib'))
    joblib.dump(retrieved_part_indices, os.path.join(result_folder, str(shape_id)+'_part_indices.joblib'))
    joblib.dump(retrieved_part_syms, os.path.join(result_folder, str(shape_id)+'_part_syms.joblib'))
    joblib.dump(retrieved_part_translations, os.path.join(result_folder, str(shape_id)+'_part_translations.joblib'))
    joblib.dump(retrieved_part_scales, os.path.join(result_folder, str(shape_id)+'_part_scales.joblib'))
    joblib.dump(retrieved_part_up_angles, os.path.join(result_folder, str(shape_id)+'_part_angles.joblib'))
    joblib.dump(part_pcs, os.path.join(result_folder, str(shape_id)+'_pred_part_pcs.joblib'))

    #return retrieved_part_indices, retrieved_part_syms, retrieved_part_translations, retrieved_part_scales, retrieved_part_up_angles

def retrieve_single_ours(shape_pc, part_pcs, k, idx, shape_id, part_vae, part_encs, direct_folder, result_folder):
    
    '''
    if os.path.isfile(os.path.join(result_folder, str(shape_id)+'_part_indices.joblib')) and \
        os.path.isfile(os.path.join(result_folder, str(shape_id)+'_part_syms.joblib')) and \
        os.path.isfile(os.path.join(result_folder, str(shape_id)+'_part_translations.joblib')) and \
        os.path.isfile(os.path.join(result_folder, str(shape_id)+'_part_scales.joblib')) and \
        os.path.isfile(os.path.join(result_folder, str(shape_id)+'_part_angles.joblib')):
        return 
    '''

    part_recon_pcs = joblib.load(os.path.join(direct_folder, str(idx)+'recon.joblib')).reshape(-1, part_point_num, 3)
    if os.path.isfile(os.path.join(direct_folder, str(idx)+'center.joblib')):
        part_recon_encs = joblib.load(os.path.join(direct_folder, str(idx)+'center.joblib')).reshape(-1, enc_dim)
    else:
        part_recon_encs = [None]*10

    retrieve_core(shape_pc, part_recon_pcs, part_recon_encs, part_pcs, k, shape_id, part_vae, part_encs, result_folder)

def retrieve_core(shape_pc, part_recon_pcs, part_recon_encs, part_pcs, k, shape_id, part_vae, part_encs, result_folder):

    print('retrieve nonsym....', shape_id)

    shape_pc = torch.tensor(shape_pc, device=device, dtype=torch.float)
    
    part_recon_pcs = torch.tensor(part_recon_pcs, device=device, dtype=torch.float)


    retrieved_part_indices = []
    retrieved_part_syms = []
    retrieved_part_translations = []
    retrieved_part_scales = []
    retrieved_part_up_angles = []

    vis_recon = []
    vis_seg = []
    for i in range(k):

        pc = []
        other_pc = []
        for j in range(len(part_recon_pcs)):
            if j == i:
                pc.append(part_recon_pcs[j])
            else:
                other_pc.append(part_recon_pcs[j])
        
        sym_index, sym_translation, sym_scale, sym_angle, temp_recons, temp_segs = retrieve_single(pc, other_pc, part_recon_encs[i], shape_pc, part_pcs, part_vae, part_encs, False)
        retrieved_part_indices += [sym_index]
        retrieved_part_syms += [False]
        retrieved_part_translations += [sym_translation]
        retrieved_part_scales += [sym_scale]
        retrieved_part_up_angles += [sym_angle]

        vis_recon += pc
        vis_seg.append(temp_segs[0])

    recon_blender_folder = os.path.join(result_folder, str(shape_id)+'recon_blender'+str(k))
    if not os.path.exists(recon_blender_folder):
        os.makedirs(recon_blender_folder)
    for j in range(len(vis_recon)):
        joblib.dump(to_numpy(vis_recon[j]), os.path.join(recon_blender_folder, str(j)+'.joblib'))
    #display_pcs(vis_recon, os.path.join(result_folder, str(shape_id)+'recon.png'), True)
    #display_pcs(vis_seg, os.path.join(result_folder, str(shape_id)+'segs.png'), True)
    joblib.dump(retrieved_part_indices, os.path.join(result_folder, str(shape_id)+'_part_indices.joblib'))
    joblib.dump(retrieved_part_syms, os.path.join(result_folder, str(shape_id)+'_part_syms.joblib'))
    joblib.dump(retrieved_part_translations, os.path.join(result_folder, str(shape_id)+'_part_translations.joblib'))
    joblib.dump(retrieved_part_scales, os.path.join(result_folder, str(shape_id)+'_part_scales.joblib'))
    joblib.dump(retrieved_part_up_angles, os.path.join(result_folder, str(shape_id)+'_part_angles.joblib'))

    #return retrieved_part_indices, retrieved_part_syms, retrieved_part_translations, retrieved_part_scales, retrieved_part_up_angles


