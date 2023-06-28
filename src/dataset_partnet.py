from pathlib import Path
import json
from torch._C import dtype
import trimesh
import networkx as nx
from collections import defaultdict
import copy
from objects import *
import os
from util_mesh_surface import *
import joblib
#import shutil
import pytorch3d
from pytorch3d.loss import chamfer_distance 
from util_motion import *
from config import *
from util_mesh_surface import *
from util_mesh_volume import *
from util_vis import *

from joblib import Parallel, delayed

partnet_data_path = '../data/partnet_data'
processed_partnet_data_path = 'raw_partnet_data'

part_obj_count = 10

def get_raw_partnet_parts(json_filename, level, mesh_path):
    parts = []
    with open(json_filename) as json_file:
        jo = json.load(json_file)
        get_parts_dfs(jo[0], 0, level, parts, mesh_path)
    return parts

def get_parts_dfs(jo, depth, target_depth, parts, mesh_path):

    if depth == target_depth or (depth < target_depth and 'children' not in jo):
        objs = jo['objs']
        meshes = []
        for obj in objs:
            if len(meshes) < part_obj_count:
                mesh = load_mesh(os.path.join(mesh_path, obj + '.obj'))
                meshes.append(mesh)
            else:
                part_mesh = merge_meshes(meshes)
                part = Part()
                part.mesh = part_mesh
                part.pc, _, _ = surface_sample_mesh(part.mesh, 400)
                parts.append(part)
                meshes = []
        if len(meshes) > 0:
            part_mesh = merge_meshes(meshes)
            part = Part()
            part.mesh = part_mesh
            part.pc, _, _ = surface_sample_mesh(part.mesh, 400)
            parts.append(part)
        return 
        
    for child_jo in jo['children']:
        get_parts_dfs(child_jo, depth+1, target_depth, parts, mesh_path) 

def process_shape(shape_id, level):
    shape_folder = os.path.join(partnet_data_path, shape_id)
    parts = get_raw_partnet_parts(os.path.join(shape_folder, 'result_after_merging.json'), level, os.path.join(shape_folder, 'objs'))
    shape = Shape()
    shape.parts = parts
    meshes = []
    for part in shape.parts:
        meshes.append(part.mesh)
    shape.mesh = merge_meshes(meshes)
    
    shape.pc, _, _ = surface_sample_mesh(shape.mesh, 3000)
    return shape

def read_shape_ids(category):
    annotation_file = '../data/partnet_annotation/stats/all_valid_anno_info.txt'
    fin = open(annotation_file, 'r')
    lines = [line.rstrip() for line in fin]
    fin.close()
    
    cat_to_ids = defaultdict(list)

    for line in lines:
        strs = line.split(' ')
        cat = strs[2]
        id = strs[0]
        cat_to_ids[cat].append(id)

    return cat_to_ids[category]

def process_dataset(category, level):

    if category == 'faucet':
        ori_category = 'Faucet'

    shape_folder = os.path.join(processed_partnet_data_path, category)
    if not os.path.exists(shape_folder):
        os.makedirs(shape_folder)
    
    shape_vis_folder = os.path.join(processed_partnet_data_path, category + 'vis')
    if not os.path.exists(shape_vis_folder):
        os.makedirs(shape_vis_folder)

    all_shapes = []
    shape_ids = read_shape_ids(ori_category)
    for shape_id in shape_ids:

        print('processing shape:', shape_id)
        #if os.path.isfile(os.path.join(shape_folder, str(shape_id) + '.joblib')):
            #continue
        
        shape = process_shape(shape_id, level)
        
        valid = True
        if np.isnan(shape.pc).any() or np.isinf(shape.pc).any():
            valid = False
        for part in shape.parts:
            if np.isnan(part.pc).any() or np.isinf(part.pc).any():
                valid = False
        if not valid:
            continue             
        
        if len(shape.parts) <= 10:
            to_save = True
            for i in range(len(all_shapes)):
                distance, _ = chamfer_distance(torch.stack([torch.tensor(shape.pc, device=device, dtype=torch.float)]), torch.stack([torch.tensor(all_shapes[i].pc, device=device, dtype=torch.float)]))
                print('distance', distance)
                if distance < 0.0015:
                    to_save = False
                    break

            if to_save:                                
                joblib.dump(shape, os.path.join(shape_folder, str(shape_id) + '.joblib'))
                #display_pcs([shape.pc], os.path.join(shape_vis_folder, str(len(all_shapes))+'.png'), True)
                all_shapes.append(shape)
                part_pcs = []
                for i in range(len(shape.parts)):
                    part_pcs.append(shape.parts[i].pc)
                display_pcs(part_pcs, os.path.join(shape_vis_folder, str(len(all_shapes))+'seg.png'), True)

        if len(all_shapes) >= 5000:
            break

def prepare_intermediate_data(in_folder, intermediate_folder):

    if not os.path.exists(intermediate_folder):
        os.makedirs(intermediate_folder)

    raw_shapes_folder = os.path.join(intermediate_folder, 'raw_shapes')
    if not os.path.exists(raw_shapes_folder):
        os.makedirs(raw_shapes_folder)

    raw_shapes_parts_folder = os.path.join(intermediate_folder, 'raw_shapes_parts')
    if not os.path.exists(raw_shapes_parts_folder):
        os.makedirs(raw_shapes_parts_folder)

    for shape_file in os.listdir(in_folder):
        shape_id = shape_file.split('.')[0]
        shape = joblib.load(os.path.join(in_folder, shape_file))

        shape_folder = os.path.join(raw_shapes_folder, str(shape_id))
        if not os.path.exists(shape_folder):
            os.makedirs(shape_folder)
        
        shape_mesh_obj = trimesh.exchange.obj.export_obj(shape.mesh)
        shape_mesh_obj_file = open(os.path.join(shape_folder, "shape.obj"), "w")
        shape_mesh_obj_file.write(shape_mesh_obj)

        shape_parts_folder = os.path.join(raw_shapes_parts_folder, str(shape_id))
        if not os.path.exists(shape_parts_folder):
            os.makedirs(shape_parts_folder)

        part_meshes = [shape.parts[i].mesh for i in range(len(shape.parts))]
        for i in range(len(part_meshes)):
            part_mesh_obj = trimesh.exchange.obj.export_obj(part_meshes[i])
            part_mesh_obj_file = open(os.path.join(shape_parts_folder, str(i)+".obj"), "w")
            part_mesh_obj_file.write(part_mesh_obj)

def prepare_final_data(shape_id, in_folder, out_folder):

    print('final processing ', shape_id)

    converted_shapes_folder = os.path.join(in_folder, 'converted_shapes')
    if not os.path.exists(os.path.join(converted_shapes_folder, shape_id)):
        return
    
    converted_shapes_parts_folder = os.path.join(in_folder, 'converted_shapes_parts')
    if not os.path.exists(os.path.join(converted_shapes_parts_folder, shape_id)):
        return

    if not os.path.isfile(os.path.join(converted_shapes_folder, str(shape_id), 'shape.obj')):
        return 
    for i in range(len(list(os.listdir(os.path.join(converted_shapes_parts_folder, shape_id))))):
        if not os.path.isfile(os.path.join(converted_shapes_parts_folder, shape_id, str(i)+'.obj')):
            return
    
    final_shape_folder = os.path.join(out_folder, 'final_shapes', str(shape_id))
    if not os.path.exists(final_shape_folder):
        os.makedirs(final_shape_folder)

    final_parts_folder = os.path.join(out_folder, 'final_parts', str(shape_id))
    if not os.path.exists(final_parts_folder):
        os.makedirs(final_parts_folder)

    shape_mesh = trimesh.load(os.path.join(converted_shapes_folder, str(shape_id), 'shape.obj'), force='mesh')

    min_values = np.min(shape_mesh.vertices, axis=0)    
    max_values = np.max(shape_mesh.vertices, axis=0)    
    shape_diag = np.linalg.norm(max_values - min_values)
    scale = np.array([2.0/shape_diag, 2.0/shape_diag, 2.0/shape_diag])
    print('scale', scale)
    shape_mesh.vertices = to_numpy(anisotropic_scale_with_value_batched(torch.tensor([shape_mesh.vertices], device=device), torch.tensor([scale], device=device))[0])
    shape_mesh.vertices = shape_mesh.vertices - np.mean(shape_mesh.vertices, axis=0)
    vol_shape_pc = volumetric_sample_mesh(shape_mesh, part_point_num*max_part_num)
    sur_shape_pc, _, _ = surface_sample_mesh(shape_mesh, part_point_num*max_part_num)
    joblib.dump(shape_mesh, os.path.join(final_shape_folder, 'mesh.joblib'))
    joblib.dump(vol_shape_pc, os.path.join(final_shape_folder, 'vol_pc.joblib'))
    joblib.dump(sur_shape_pc, os.path.join(final_shape_folder, 'sur_pc.joblib'))

    for i in range(len(list(os.listdir(os.path.join(converted_shapes_parts_folder, shape_id))))):
    
        part_folder = os.path.join(final_parts_folder, str(i))
        if not os.path.exists(part_folder):
            os.makedirs(part_folder)

        part_mesh = trimesh.load(os.path.join(converted_shapes_parts_folder, shape_id, str(i)+'.obj'), force='mesh')
        part_mesh.vertices = to_numpy(anisotropic_scale_with_value_batched(torch.tensor([part_mesh.vertices], device=device), torch.tensor([scale], device=device))[0])
        part_mesh.vertices = part_mesh.vertices - np.mean(part_mesh.vertices, axis=0)
        part_vol_pc = volumetric_sample_mesh(part_mesh, part_point_num)
        part_sur_pc, _, _ = surface_sample_mesh(part_mesh, part_point_num)
        joblib.dump(part_mesh, os.path.join(part_folder, 'mesh.joblib'))
        joblib.dump(part_vol_pc, os.path.join(part_folder, 'vol_pc.joblib'))
        joblib.dump(part_sur_pc, os.path.join(part_folder, 'sur_pc.joblib'))

def prepare_NP_final_data(shape_id, in_folder, out_folder):

    print('final NP processing: ', shape_id)

    out_target_folder = os.path.join(out_folder, str(shape_id), str(0))
    if not os.path.exists(out_target_folder):
        os.makedirs(out_target_folder)

    '''
    image_folder = os.path.join(out_target_folder, 'images')
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    '''

    if os.path.isfile(os.path.join(out_target_folder, "model_watertight.obj")):
        return

    final_shape_folder = os.path.join(in_folder, 'final_shapes')
    shape_mesh = joblib.load(os.path.join(final_shape_folder, str(shape_id), 'mesh.joblib'))
    min_values = np.min(shape_mesh.vertices, axis=0)    
    max_values = np.max(shape_mesh.vertices, axis=0)    
    shape_diag = np.linalg.norm(max_values - min_values)
    scale = np.array([1.0/shape_diag, 1.0/shape_diag, 1.0/shape_diag])
    print('scale', scale)
    shape_mesh.vertices = to_numpy(anisotropic_scale_with_value_batched(torch.tensor([shape_mesh.vertices], device=device), torch.tensor([scale], device=device))[0])
    obj_mesh = trimesh.exchange.obj.export_obj(shape_mesh)
    obj_file = open(os.path.join(out_target_folder, "model_watertight.obj"), "w")
    obj_file.write(obj_mesh)
    
    '''
    images = generate_mesh_shots(shape_mesh, 24)
    for j in range(len(images)):
        plt.figure(figsize=(1.5, 1.5))
        #plt.subplot(1,2,1)
        plt.axis('off')
        plt.imshow(images[j])
        #plt.subplot(1,2,2)
        plt.axis('off')
        #plt.imshow(depth, cmap=plt.cm.gray_r)
        plt.savefig(os.path.join(image_folder, str(j) +'.jpg'))
        plt.close()
    '''

if __name__ == '__main__':
    
    if global_args.dataset_option == 'raw':
        level = 4
        process_dataset(global_args.shape_category, level)

    if global_args.dataset_option == 'intermediate':
        input_data_folder = os.path.join('raw_partnet_data', global_args.shape_category)
        itermediate_data_folder = os.path.join('intermediate_partnet_shapes', global_args.shape_category)
        prepare_intermediate_data(input_data_folder, itermediate_data_folder)

    if global_args.dataset_option == 'final':
        itermediate_data_folder = os.path.join('intermediate_partnet_shapes', global_args.shape_category)
        final_data_folder = os.path.join('../../../scratch/final_partnet_shapes', global_args.shape_category)
        shape_ids = list(os.listdir(os.path.join(itermediate_data_folder, 'converted_shapes')))
        #for shape_id in shape_ids:
            #prepare_final_data(shape_id, itermediate_data_folder, final_data_folder)
        results = Parallel(n_jobs=4)(delayed(prepare_final_data)(shape_id, itermediate_data_folder, final_data_folder) for shape_id in shape_ids)
    
    if global_args.dataset_option == 'NP_final':
        final_data_folder = os.path.join('../../../scratch/final_partnet_shapes', global_args.shape_category)
        neural_data_folder = os.path.join('../../../scratch/final_NP_partnet_shapes', global_args.shape_category)
        shape_ids = list(os.listdir(os.path.join(final_data_folder, 'final_shapes')))
        for shape_id in shape_ids:
            prepare_NP_final_data(shape_id, final_data_folder, neural_data_folder)
        #results = Parallel(n_jobs=4)(delayed(prepare_NP_final_data)(shape_id, final_data_folder, neural_data_folder) for shape_id in shape_ids)

