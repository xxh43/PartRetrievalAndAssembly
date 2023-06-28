
import copy
from platform import node
from time import time
import numpy as np
import trimesh
import math
import time
from mesh_contain.inside_mesh import check_mesh_contains
from util_mesh_surface import *
#from util_vis import display_meshes, display_pcs

def get_submesh(mesh, face_indices):
    submesh = copy.deepcopy(mesh)
    submesh.faces = mesh.faces[face_indices]
    return submesh

def load_mesh(obj_filename):
    tri_obj = trimesh.load_mesh(obj_filename)
    if tri_obj.is_empty:
        return None
    if type(tri_obj) is trimesh.scene.scene.Scene:
        tri_mesh = tri_obj.dump(True)
    else:
        tri_mesh = tri_obj

    return tri_mesh

def merge2mesh(mesh1, mesh2):
    new_mesh = copy.deepcopy(mesh1)
    shifted_mesh2_faces = copy.deepcopy(mesh2.faces) + copy.deepcopy(mesh1.vertices.shape[0])
    new_mesh.faces = np.concatenate((copy.deepcopy(mesh1.faces), copy.deepcopy(shifted_mesh2_faces)))
    new_mesh.vertices = np.concatenate((copy.deepcopy(mesh1.vertices), copy.deepcopy(mesh2.vertices)))
    return new_mesh

def merge_meshes(meshes):
    if len(meshes) == 0:
        return None
    base_mesh = meshes[0]
    for i in range(1, len(meshes)):
        base_mesh = merge2mesh(base_mesh, meshes[i])
    return base_mesh

def volumetric_sample_mesh_octree(mesh, sample_count):

    shape_pc, _, _ = surface_sample_mesh(mesh, 2048)
    tree = Octree()
    tree.build(shape_pc)
    bounds, weights = tree.get_leaf_node_bounds()
    total_weight = 0
    for i in range(len(weights)):
        total_weight += weights[i]

    all_pc = None

    print('number of nodes', len(bounds))
    
    for i in range(len(bounds)):

        print('processing node', i)

        bound = bounds[i]
        weight = weights[i]

        t = np.array([(bound[0]+bound[1])*0.5, (bound[2]+bound[3])*0.5, (bound[4]+bound[5])*0.5])
        t = np.expand_dims(t, axis=1)
        rotmat = np.eye(3)
        z = np.array([(bound[1]-bound[0])*0.5, (bound[3]-bound[2])*0.5, (bound[5]-bound[4])*0.5])
        
        extents = z * 2.01
        transform = np.concatenate((rotmat, t), axis=1)
        pad_vector = np.array([0,0,0,1])
        pad_vector = np.expand_dims(pad_vector, axis=0)
        transform = np.concatenate((transform, pad_vector), axis=0)
    
        node_sample_count = int(weight/total_weight*sample_count)+1
        print('node_sample_count', node_sample_count)

        node_pc = None
        node_pre_pc_count = 0
        while  node_pc is None or len(node_pc) <= node_sample_count:
            rec_pc = trimesh.sample.volume_rectangular(extents, node_sample_count*5, transform)
            labels = points_in_mesh(mesh, rec_pc)
            inside_pc = rec_pc[labels==True]

            if node_pc is None:
                node_pc = inside_pc
            else:
                node_pc = np.concatenate((node_pc, inside_pc), axis=0)
            
            if len(node_pc) == node_pre_pc_count:
                break

            node_pre_pc_count = len(node_pc)
                
        if all_pc is None:
            all_pc = node_pc
        else:
            all_pc = np.concatenate((all_pc, node_pc), axis=0)

    np.random.shuffle(all_pc)
    final_sampled_pc = all_pc[0:sample_count]
    return final_sampled_pc

def volumetric_sample_mesh(mesh, sample_count,wait_time = 600):

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    extents = extents * 1.5
    t = to_origin[:3, :3].transpose().dot(-to_origin[:3, 3])
    t = np.expand_dims(t, axis=1)
    xdir = to_origin[0, :3]
    ydir = to_origin[1, :3]
    zdir = np.cross(xdir, ydir)
    rotmat = np.vstack([xdir, ydir, zdir]).T
    transform = np.concatenate((rotmat, t), axis=1)
    pad_vector = np.array([0,0,0,1])
    pad_vector = np.expand_dims(pad_vector, axis=0)
    transform = np.concatenate((transform, pad_vector), axis=0)
    
    all_pc = None
    start = time.time()
    while all_pc is None or len(all_pc) < sample_count:
        rec_pc = trimesh.sample.volume_rectangular(extents, sample_count*50, transform)
        labels = points_in_mesh(mesh, rec_pc)
        inside_pc = rec_pc[labels==True]
        if all_pc is None:
            all_pc = inside_pc
        else:
            all_pc = np.concatenate((all_pc, inside_pc), axis=0)
        end = time.time()
        if end - start > wait_time:
            print('volumetric sampling taking too long !!', end - start)
            return None

    np.random.shuffle(all_pc)
    final_sampled_pc = all_pc[0:sample_count]
    return final_sampled_pc

def points_in_mesh(mesh, points):
    labels = check_mesh_contains(mesh, points)
    return labels
    