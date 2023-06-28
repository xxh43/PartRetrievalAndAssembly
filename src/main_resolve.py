
import argparse
from cgi import test
import copy
from dis import dis
from functools import partial
import gc
import math
import os
import shutil
import time
from operator import pos

#import sklearn.external.joblib as extjoblib
#import sklearn.external.joblib as extjoblib
import joblib
import matplotlib.pyplot as plt
import numpy as np
from numpy import arange
from numpy import real
from numpy import require
from numpy import concatenate
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.spatial.transform import Rotation as R
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from torch import tensor
from torch._C import dtype
from torch.autograd import Variable
from torch.utils.data import DataLoader
#from main_common import post_process
from main_ours_pretrain import *
from joblib import Parallel, delayed
#from trimesh import *
from util_collision import *
#from util_mesh import *
from util_file import *
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

bce_loss = torch.nn.BCELoss()
import argparse
import copy
import gc
import math
import os
import shutil
import time
from operator import pos

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

from scipy import spatial, stats
from scipy.spatial.distance import braycurtis, cdist
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from torch import nn, tensor
from torch._C import dtype
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader

#from trimesh import *
from util_file import *
from util_motion import *
from util_vis import *

import time
from util_mesh_surface import *
from util_mesh_volume import *

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

bce_loss = torch.nn.BCELoss()
import torch
import torchvision
from scipy.spatial.transform import Rotation as R
from torch import nn
from torch.nn import functional as F

from pytorch3d import *
from pytorch3d.loss import chamfer_distance

from networkx.algorithms import bipartite
from main_ours_pretrain import *

from main_common import *
from scipy.spatial import KDTree
from util_collision import *
from config import *

from scipy.spatial import distance
from data_manager import *

from config import *


def get_final_eval_score(chamfer_error, k):
    complexity_weight = 0.00015
    final_score = chamfer_error + k * complexity_weight 
    return final_score

def adjust_filter(part_meshes, part_syms, part_translations, part_scales, part_angles, shape_mesh, shape_vol_pc, shape_id, use_remove=True):

    retrieval_dict = {}
    retrieval_dict['k'] = len(part_meshes)

    retrieval_dict['shape_mesh'] = shape_mesh

    sym_info = [0]*len(part_meshes)
    part_meshes_before = copy.deepcopy(part_meshes)
    for i in range(len(part_meshes_before)):
        transformed_retrieved_part_vertices = torch.tensor(part_meshes_before[i].vertices, device=device, dtype=torch.float)
        retrieved_part_scale = part_scales[i]
        print('retrieved_part_scale', retrieved_part_scale)
        retrieved_part_translation = part_translations[i]       
        retrieved_part_angle = part_angles[i]
        transformed_retrieved_part_vertices = assemble_parts(torch.stack([transformed_retrieved_part_vertices]), torch.stack([retrieved_part_scale]), torch.stack([retrieved_part_angle]), torch.stack([retrieved_part_translation]))[0]
        
        retrieved_part_sym = part_syms[i]  
        if retrieved_part_sym:
            sym_info[i-1]=1        
            transformed_retrieved_part_vertices = reflect_parts(torch.stack([transformed_retrieved_part_vertices]), torch.stack([torch.tensor([1.0,0.0,0.0], device=device, dtype=torch.float)]))[0]
            
        part_meshes_before[i].vertices = to_numpy(transformed_retrieved_part_vertices)
    retrieval_dict['recon_part_meshes_before'] = part_meshes_before

    part_meshes_after = copy.deepcopy(part_meshes_before)

    use_vol_filter = True

    if use_remove:

        if use_vol_filter:
            recon_shape_mesh_before = merge_meshes(part_meshes_before)
            recon_shape_vol_pc_before = volumetric_sample_mesh(recon_shape_mesh_before, shape_vol_pc.shape[0])
            chamfer_error_before, _ = chamfer_distance(torch.stack([torch.tensor(recon_shape_vol_pc_before, device=device, dtype=torch.float)]), torch.stack([torch.tensor(shape_vol_pc, device=device, dtype=torch.float)]))
        else:
            recon_shape_mesh_before = merge_meshes(part_meshes_before)
            shape_sur_pc, _, _ = surface_sample_mesh(shape_mesh, shape_vol_pc.shape[0])
            recon_shape_sur_pc_before, _, _ = surface_sample_mesh(recon_shape_mesh_before, shape_vol_pc.shape[0])
            chamfer_error_before, _ = chamfer_distance(torch.stack([torch.tensor(recon_shape_sur_pc_before, device=device, dtype=torch.float)]), torch.stack([torch.tensor(shape_sur_pc, device=device, dtype=torch.float)]))
        
        part_pcs_after = []
        for i in range(len(part_meshes_after)):
            if use_vol_filter:
                part_pc = volumetric_sample_mesh(part_meshes_after[i], part_point_num)
            else:
                part_pc, _, _ = surface_sample_mesh(part_meshes_after[i], part_point_num)
            part_pcs_after.append(part_pc)

        next_index = 0

        while True:
            is_updated = False
            
            if len(part_meshes_after) == 1:
                break

            for i in range(next_index, len(part_meshes_after)):
                #print('try to remove part ', i, ' --------------------------------------------')
                
                if sym_info[i] == 1:
                    part_meshes_without_i = copy.deepcopy(part_meshes_after[0:i] + part_meshes_after[i+2:])
                    part_pcs_without_i = copy.deepcopy(part_pcs_after[0:i] + part_pcs_after[i+2:])
                else:
                    part_meshes_without_i = copy.deepcopy(part_meshes_after[0:i] + part_meshes_after[i+1:])
                    part_pcs_without_i = copy.deepcopy(part_pcs_after[0:i] + part_pcs_after[i+1:])
                
                if len(part_meshes_without_i) == 0:
                    break

                #new_part_meshes_without_i = post_optim_mesh(part_meshes_without_i, part_pcs_without_i, shape_vol_pc, 1000)
                new_part_meshes_without_i = part_meshes_without_i
                recon_shape_mesh_without_i = merge_meshes(new_part_meshes_without_i)
                if use_vol_filter:
                    recon_shape_vol_pc_without_i = volumetric_sample_mesh(recon_shape_mesh_without_i, shape_vol_pc.shape[0], 60)
                    if recon_shape_vol_pc_without_i is None:
                        #print('volumetric sampling taking too long !!, shape', shape_id)
                        continue
                    chamfer_error, _ = chamfer_distance(torch.stack([torch.tensor(recon_shape_vol_pc_without_i, device=device, dtype=torch.float)]), torch.stack([torch.tensor(shape_vol_pc, device=device, dtype=torch.float)]))
                else:
                    recon_shape_sur_pc_without_i, _, _ = surface_sample_mesh(recon_shape_mesh_without_i, shape_vol_pc.shape[0])
                    chamfer_error, _ = chamfer_distance(torch.stack([torch.tensor(recon_shape_sur_pc_without_i, device=device, dtype=torch.float)]), torch.stack([torch.tensor(shape_sur_pc, device=device, dtype=torch.float)]))

                if chamfer_error < chamfer_error_before * 1.05:
                    print('confirm remove part ', i)
                    is_updated = True
                    next_index = i
                    break
            
            if is_updated:
                part_meshes_after = copy.deepcopy(part_meshes_without_i)
                part_pcs_after = copy.deepcopy(part_pcs_without_i)
            else:
                break
    
    if use_vol_filter:
        recon_shape_mesh_after = merge_meshes(part_meshes_after)
        recon_shape_vol_pc_after = volumetric_sample_mesh(recon_shape_mesh_after, shape_vol_pc.shape[0])
        chamfer_error_after, _ = chamfer_distance(torch.stack([torch.tensor(recon_shape_vol_pc_after, device=device, dtype=torch.float)]), torch.stack([torch.tensor(shape_vol_pc, device=device, dtype=torch.float)]))
    else:
        recon_shape_mesh_after = merge_meshes(part_meshes_after)
        recon_shape_sur_pc_after, _, _ = surface_sample_mesh(recon_shape_mesh_after, shape_vol_pc.shape[0])
        chamfer_error_after, _ = chamfer_distance(torch.stack([torch.tensor(recon_shape_sur_pc_after, device=device, dtype=torch.float)]), torch.stack([torch.tensor(shape_sur_pc, device=device, dtype=torch.float)]))
    
    retrieval_dict['chamfer_error'] = chamfer_error_after
    retrieval_dict['recon_part_meshes_after'] = part_meshes_after

    return retrieval_dict


def resolve_shape(shape_id, shape_ids, part_meshes, part_vol_pcs, part_sur_pcs, shape_meshes, shape_vol_pcs, shape_sur_pcs, summary_folder, results_folder):
    
    print('resolving shape ', str(shape_id))

    if os.path.isfile(os.path.join(summary_folder, str(shape_id)+'.joblib')):
        return

    #print('shape_ids', shape_ids)

    shape_folder = os.path.join(summary_folder, str(shape_id))
    if not os.path.exists(shape_folder):
        os.makedirs(shape_folder)
    
    shape_to_best = {}

    for k in list(os.listdir(results_folder)):

        k_folder = os.path.join(results_folder, k)

        print('k', k)

        print('shape_id', shape_id)
        
        part_indices = joblib.load(os.path.join(k_folder, str(shape_id)+'_part_indices.joblib'))
        if os.path.isfile(os.path.join(k_folder, str(shape_id)+'_part_syms.joblib')):
            part_syms = joblib.load(os.path.join(k_folder, str(shape_id)+'_part_syms.joblib'))
        else:
            part_syms = None
        part_translations = joblib.load(os.path.join(k_folder, str(shape_id)+'_part_translations.joblib'))
        part_scales = joblib.load(os.path.join(k_folder, str(shape_id)+'_part_scales.joblib'))
        part_angles = joblib.load(os.path.join(k_folder, str(shape_id)+'_part_angles.joblib'))

        retrieved_part_meshes = [copy.deepcopy(part_meshes[v]) for v in part_indices]
        retrieved_part_sur_pcs = [copy.deepcopy(part_sur_pcs[v]) for v in part_indices]
        retrieved_part_vol_pcs = [copy.deepcopy(part_vol_pcs[v]) for v in part_indices]

        shape_mesh = shape_meshes[shape_ids.index(shape_id)]
        shape_sur_pc = shape_sur_pcs[shape_ids.index(shape_id)]
        shape_vol_pc = shape_vol_pcs[shape_ids.index(shape_id)]

        retrieval_dict = adjust_filter(retrieved_part_meshes, part_syms, part_translations, part_scales, part_angles, shape_mesh, shape_vol_pc, shape_id, False)
        retrieval_dict['part_indices'] = part_indices
        joblib.dump(retrieval_dict, os.path.join(shape_folder, str(shape_id)+'_'+str(k)+'.joblib'))

def finalize_resolve_shape(shape_id, shape_ids, part_meshes, part_vol_pcs, part_sur_pcs, shape_meshes, shape_vol_pcs, shape_sur_pcs, summary_folder, results_folder):
    
    print('resolving shape ', str(shape_id))

    shape_folder = os.path.join(summary_folder, str(shape_id))
    
    best_res = None
    best_score = np.inf

    all_files = os.listdir(shape_folder)
    res_files = [f for f in all_files if f.endswith('.joblib')]

    for k_res_file in res_files:
        print('k_res_file', k_res_file)
        k_res = joblib.load(os.path.join(shape_folder, k_res_file))
        score = get_final_eval_score(k_res['chamfer_error'], len(k_res['recon_part_meshes_after']))
        k_res['score'] = score 
        joblib.dump(k_res, os.path.join(shape_folder, k_res_file))
        if score < best_score:
            best_score = score
            best_res = copy.deepcopy(k_res)
    
    joblib.dump(best_res, os.path.join(summary_folder, str(shape_id)+'.joblib'))


def resolve(data_dir, exp_folder, part_dataset, part_category, part_count, shape_dataset, shape_category, train_shape_count, test_shape_count, eval_on_train_shape_count):

    print('resolve .... ')

    _, train_shape_ids, test_shape_ids, _ = read_split(shape_dataset, shape_category, train_shape_count)
    if part_dataset == 'partnet':
        if part_category == shape_category:
            source_shape_ids, _, _, _ = read_split(part_dataset, part_category, train_shape_count)
        else:
            source_shape_ids, _, _, _ = read_split(part_dataset, part_category, None)
    else:
        source_shape_ids = []

    print('part_dataset', part_dataset)
    print('part_category', part_category)
    print('max_part_count', part_count)
    print('shape_category', shape_category)
    print('train_shape_count', train_shape_count)
    print('source shape count', len(source_shape_ids))
    print('train shape count', len(train_shape_ids))
    print('test shape count', len(test_shape_ids))

    train_shape_ids = train_shape_ids[0:train_shape_count]
    test_shape_ids = test_shape_ids[0:test_shape_count]

    part_meshes, part_vol_pcs, part_sur_pcs = get_parts(data_dir, part_dataset, part_category, part_count, source_shape_ids, True)
    part_vol_pcs, part_sur_pcs, part_meshes = calibrate_parts(part_vol_pcs, part_sur_pcs, part_meshes)

    train_shape_meshes, train_shape_vol_pcs, train_shape_sur_pcs = get_shapes(data_dir, shape_dataset, shape_category, train_shape_ids, train_shape_count, all_formats=True)
    eval_train_shape_ids = train_shape_ids[0:eval_on_train_shape_count]

    print('eval_train_shape_ids', eval_train_shape_ids)

    train_results_folder = os.path.join(exp_folder, 'train_results')
    train_summary_folder = os.path.join(exp_folder, 'train_summary')

    if not os.path.exists(train_summary_folder):
        os.makedirs(train_summary_folder)

    if use_parallel:
        results = Parallel(n_jobs=2)(delayed(resolve_shape)(shape_id, train_shape_ids, part_meshes, part_vol_pcs, part_sur_pcs, train_shape_meshes, train_shape_vol_pcs, train_shape_sur_pcs, train_summary_folder, train_results_folder) for shape_id in eval_train_shape_ids)
    else:
        for shape_id in eval_train_shape_ids:
            resolve_shape(shape_id, train_shape_ids, part_meshes, part_vol_pcs, part_sur_pcs, train_shape_meshes, train_shape_vol_pcs, train_shape_sur_pcs, train_summary_folder, train_results_folder)    
            finalize_resolve_shape(shape_id, train_shape_ids, part_meshes, part_vol_pcs, part_sur_pcs, train_shape_meshes, train_shape_vol_pcs, train_shape_sur_pcs, train_summary_folder, train_results_folder)    
    
    test_shape_meshes, test_shape_vol_pcs, test_shape_sur_pcs = get_shapes(data_dir, shape_dataset, shape_category, test_shape_ids, test_shape_count, all_formats=True)    
    test_results_folder = os.path.join(exp_folder, 'test_results')
    test_summary_folder = os.path.join(exp_folder, 'test_summary')
        
    if not os.path.exists(test_summary_folder):
        os.makedirs(test_summary_folder)

    if use_parallel:
        results = Parallel(n_jobs=2)(delayed(resolve_shape)(shape_id, test_shape_ids, part_meshes, part_vol_pcs, part_sur_pcs, test_shape_meshes, test_shape_vol_pcs, test_shape_sur_pcs, test_summary_folder, test_results_folder) for shape_id in test_shape_ids)
    else:
        for shape_id in test_shape_ids:
            resolve_shape(shape_id, test_shape_ids, part_meshes, part_vol_pcs, part_sur_pcs, test_shape_meshes, test_shape_vol_pcs, test_shape_sur_pcs, test_summary_folder, test_results_folder)    
            finalize_resolve_shape(shape_id, test_shape_ids, part_meshes, part_vol_pcs, part_sur_pcs, test_shape_meshes, test_shape_vol_pcs, test_shape_sur_pcs, test_summary_folder, test_results_folder)    

if __name__ == "__main__":

    exp_folder = os.path.join(global_args.exp_dir, global_args.part_dataset + global_args.part_category + '_to_' + global_args.shape_dataset + global_args.shape_category + str(global_args.train_shape_count) + 'shift' + str(use_shift) + 'borrow' + str(use_borrow))
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)
    
    resolve(global_args.data_dir, exp_folder, global_args.part_dataset, global_args.part_category, global_args.part_count, global_args.shape_dataset, global_args.shape_category, global_args.train_shape_count, global_args.test_shape_count, global_args.eval_on_train_shape_count)


    