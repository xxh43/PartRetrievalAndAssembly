
import torch
import numpy as np
import os
import joblib
from util_vis import *
from config import * 
from util_motion import *
import joblib
sys.modules['sklearn.externals.joblib'] = joblib
import six
sys.modules['sklearn.externals.six'] = six
import csv
from pytorch3d.loss import chamfer_distance

def align_shape_by_sym_plane(target_pc, ref_normal, up_axis, origin):
    
    target_pc = torch.tensor(target_pc, dtype=torch.float)

    max_length = 0
    max_angle = 0
    max_plane_normal = None
    is_symmetrical = False

    test_count = 20

    for i in range(0, test_count+1):
        angle = i*np.pi/test_count
        print('angle', angle)
        plane_normal = rotate_with_axis_center_angle(torch.stack([torch.tensor(ref_normal, dtype=torch.float)]), torch.tensor(up_axis, dtype=torch.float), torch.tensor(origin, dtype=torch.float), torch.tensor(angle))
        dotvals = torch.matmul(target_pc, plane_normal.transpose(0,1)).squeeze(dim=1)
        
        positive_pc1 = target_pc[dotvals > 0]
        negative_pc1 = target_pc[dotvals < 0]

        vecs = positive_pc1.permute(1, 0)
        dots = torch.matmul(plane_normal, vecs)
        temp = torch.matmul(dots.permute(1, 0), plane_normal)
        reflected_positive_pc1 = positive_pc1 - 2 * temp

        error, _ = chamfer_distance(torch.stack([reflected_positive_pc1]), torch.stack([negative_pc1]))
        #print('error', error)
        if error < symmetry_threshold:
            is_symmetrical = True
            min_proj = min(dotvals)
            max_proj = max(dotvals)
            print('min_proj', min_proj, 'max_proj', max_proj)
            if max_proj - min_proj > max_length:
                max_length = max_proj - min_proj          
                max_angle = angle
                max_plane_normal = plane_normal
    
    if is_symmetrical is False:
        return None
    else:
        print('max_angle', max_angle)
        print('max_plane_normal', max_plane_normal)
        return -max_angle

def align_shapes_by_sym_plane(meshes):

    print('len(meshes)', len(meshes))
    
    origin = np.array([0.0, 0.0, 0.0])
    up_axis = np.array([0.0, 1.0, 0.0])
    ref_normal = np.array([1.0, 0.0, 0.0])

    out_meshes = []

    for i in range(len(meshes)):
        
        align_angle = align_shape_by_sym_plane(meshes[i].vertices, ref_normal, up_axis, origin)

        out_mesh = copy.deepcopy(meshes[i])
        if align_angle is not None:
            out_mesh.vertices = to_numpy(rotate_with_axis_center_angle(torch.tensor(out_mesh.vertices), torch.tensor(up_axis), torch.tensor(origin), torch.tensor(align_angle)))

        out_meshes.append(out_mesh)

    return out_meshes
