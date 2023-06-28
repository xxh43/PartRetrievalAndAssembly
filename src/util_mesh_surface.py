
import copy
import numpy as np
import trimesh
import math

from util_vis import display_pcs

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

def surface_sample_mesh(mesh, amount):
    if mesh is None:
        return np.zeros((amount, 3)), np.zeros((amount, 3)), None

    pc, indices = trimesh.sample.sample_surface(mesh, amount)
    
    pc_normals = []
    '''
    for i in range(len(pc)):
        point = pc[i]
        face_index = indices[i]
        point_normal = get_point_normal(point, face_index, mesh)
        pc_normals.append(point_normal)
    pc_normals = np.array(pc_normals)
    '''
    
    return pc, pc_normals, indices


def get_point_normal(point, face_index, mesh):

    #mesh.fix_normals()
    f = mesh.faces[face_index]
    f_normal = mesh.face_normals[face_index]
    return f_normal
    #print('f_normal', f_normal)
    #exit()

    normal_0 = mesh.vertex_normals[f[0]]
    normal_1 = mesh.vertex_normals[f[1]]
    normal_2 = mesh.vertex_normals[f[2]]

    p0 = mesh.vertices[f[0]]
    p1 = mesh.vertices[f[1]]
    p2 = mesh.vertices[f[2]]

    area_2 = get_area(point, p0, p1)
    area_0 = get_area(point, p1, p2)
    area_1 = get_area(point, p2, p0)

    weight_0 = area_0 / (area_0 + area_1 + area_2)
    weight_1 = area_1 / (area_0 + area_1 + area_2)
    weight_2 = area_2 / (area_0 + area_1 + area_2)

    point_normal = normal_0 * weight_0 + normal_1 * weight_1 + normal_2 * weight_2
    point_normal = point_normal/np.linalg.norm(point_normal)

    if np.dot(point_normal, f_normal) < 0:
        point_normal = -point_normal

    return point_normal

def get_area(p0, p1, p2):

    a = np.linalg.norm(p0 - p1)
    b = np.linalg.norm(p1 - p2)
    c = np.linalg.norm(p2 - p0)
    p = (a + b + c) / 2

    tmp = p * (p - a) * (p - b) * (p - c)

    tmp = max(0, tmp)

    area = math.sqrt(tmp)
    return area

def get_boundary_edges(mesh):

    visited_edges = []
    for edge in mesh.edges:
        v0 = edge[0]
        v1 = edge[1]
        visited_edges.append((v0,v1))

    boundary_edges = []
    for edge in mesh.edges:
        v0 = edge[0]
        v1 = edge[1]
        if (v1,v0) not in visited_edges:
            boundary_edges.append((v0,v1))
    
    print('number of all edges', len(mesh.edges))
    print('number of boundary edges', len(boundary_edges))
    return boundary_edges
