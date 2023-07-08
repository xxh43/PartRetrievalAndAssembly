
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

#from trimesh import *
from util_collision import *
#from util_mesh import *

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
from util_mesh_surface import *
from util_mesh_volume import *
from main_ours_pretrain import *

from main_common import *
from scipy.spatial import KDTree
from util_collision import *
from config import *

from scipy.spatial import distance
from data_manager import *
from config import *

def render_shape(shape_id, dict, folder, spec, is_summary):
    if is_summary:
        render_meshes([dict['shape_mesh']], True, os.path.join(folder, str(shape_id)+'zgt_shape_mesh.png'))

        if 'recon_part_meshes_after' in dict:
            render_meshes(dict['recon_part_meshes_after'], False, os.path.join(folder, str(shape_id)+'retrieved_parts_mesh_after.png'))
        
        if 'recon_shape_mesh' in dict:
            render_meshes([dict['recon_shape_mesh']], True, os.path.join(folder, str(shape_id)+'recon_shape_mesh.png'))

        if 'recon_part_meshes_after' in dict:
            blender_recon_folder = os.path.join(folder, 'blender', str(shape_id)+'recon')
            if not os.path.exists(blender_recon_folder):
                os.makedirs(blender_recon_folder)
            for i in range(len(dict['recon_part_meshes_after'])):
                dict['recon_part_meshes_after'][i].export(os.path.join(blender_recon_folder, str(i)+'.obj'), file_type='obj')
                f = open(os.path.join(blender_recon_folder, str(i)+'_'+str(dict['part_indices'][i])+".txt"), "w")
                f.close()
            
            blender_target_folder = os.path.join(folder, 'blender', str(shape_id)+'target')
            if not os.path.exists(blender_target_folder):
                os.makedirs(blender_target_folder)
            dict['shape_mesh'].export(os.path.join(blender_target_folder, 'target.obj'), file_type='obj')
            
            shape_vol_pc = volumetric_sample_mesh(dict['shape_mesh'], 4096)
            joblib.dump(shape_vol_pc, os.path.join(blender_target_folder, 'target_pc.joblib'))
    
    else:
        render_meshes([dict['shape_mesh']], True, os.path.join(folder, 'zgt_shape_mesh.png'))

        if 'recon_part_meshes_before' in dict:
            render_meshes(dict['recon_part_meshes_before'], False, os.path.join(folder, str(spec)+'retrieved_parts_mesh_before.png'))
        if 'recon_part_meshes_after' in dict:
            render_meshes(dict['recon_part_meshes_after'], False, os.path.join(folder, str(spec)+'_'+str(dict['score'])+'retrieved_parts_mesh_after.png'))
            #render_meshes(dict['recon_part_meshes_after'], False, os.path.join(folder, str(spec)+'_retrieved_parts_mesh_after.png'))

            blender_recon_folder = os.path.join(folder, 'blender', str(shape_id)+str(len(dict['recon_part_meshes_after']))+'recon_mesh')
            if not os.path.exists(blender_recon_folder):
                os.makedirs(blender_recon_folder)
            for i in range(len(dict['recon_part_meshes_after'])):
                dict['recon_part_meshes_after'][i].export(os.path.join(blender_recon_folder, str(i)+'.obj'), file_type='obj')
            
        if 'recon_shape_mesh' in dict:
            render_meshes([dict['recon_shape_mesh']], True, os.path.join(folder, str(shape_id)+'recon_shape_mesh.png'))

        if 'shape_vol_pc' in dict:
            display_pcs([dict['shape_vol_pc']], os.path.join(folder, str(spec)+'shape_vol_pc.png'), True)

        if 'shape_recon' in dict:
            display_pcs(dict['shape_recon'], os.path.join(folder, str(spec)+'shape_recon.png'), True)
        if 'shape_seg' in dict:
            display_pcs(dict['shape_seg'], os.path.join(folder, str(spec)+'shape_seg.png'), True)
        if 'pred_part_meshes' in dict and dict['pred_part_meshes'] is not None:
            render_meshes(dict['pred_part_meshes'], False, os.path.join(folder, str(spec)+'pred_part_meshes.png'))
        

def visulize_shape(shape_id, summary_folder):

    print('visulize shape ', shape_id)
    shape_folder = os.path.join(summary_folder, str(shape_id))
    for shape_file in list(os.listdir(shape_folder)):
        if shape_file.endswith('.joblib'):
            print('shape_file', shape_file)
            shape_dict = joblib.load(os.path.join(shape_folder, str(shape_file)))
            render_shape(shape_id, shape_dict, shape_folder, shape_file.split('.')[0], False)
            #os.remove(os.path.join(shape_folder, str(shape_file)))
    shape_summary_dict = joblib.load(os.path.join(summary_folder, str(shape_id)+'.joblib'))
    render_shape(shape_id, shape_summary_dict, summary_folder, '', True)

def generate_visulizations(data_dir, exp_folder, part_dataset, part_category, part_count, shape_dataset, shape_category, train_shape_count, test_shape_count, eval_on_train_shape_count):

    source_shape_ids, train_shape_ids, test_shape_ids, _ = read_split(global_args.split_file)

    eval_train_shape_ids = train_shape_ids[0:eval_on_train_shape_count]
    summary_folder = os.path.join(exp_folder, 'train_summary')
    if use_parallel:
        results = Parallel(n_jobs=2)(delayed(visulize_shape)(shape_id, summary_folder) for shape_id in eval_train_shape_ids)
    else:
        for shape_id in eval_train_shape_ids:
            visulize_shape(shape_id, summary_folder)

    test_shape_ids = test_shape_ids[0:test_shape_count]
    summary_folder = os.path.join(exp_folder, 'test_summary')
    if use_parallel:
        results = Parallel(n_jobs=2)(delayed(visulize_shape)(shape_id, summary_folder) for shape_id in test_shape_ids)
    else:
        for shape_id in test_shape_ids:
            visulize_shape(shape_id, summary_folder)


if __name__ == "__main__":

    exp_folder = os.path.join(global_args.exp_dir, global_args.part_dataset + global_args.part_category + '_to_' + global_args.shape_dataset + global_args.shape_category + str(global_args.train_shape_count) + 'shift' + str(use_shift) + 'borrow' + str(use_borrow))
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)
    
    generate_visulizations(global_args.data_dir, exp_folder, global_args.part_dataset, global_args.part_category, global_args.part_count, global_args.shape_dataset, global_args.shape_category, global_args.train_shape_count, global_args.test_shape_count, global_args.eval_on_train_shape_count)
