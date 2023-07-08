
import numpy as np
import os
import plotly.graph_objects as go
import pyrender
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt
import trimesh
from util_motion import rotate_with_axis_center_angle

print(torch.__version__)

'''
grey = (128/255, 128/255, 128/255)
black = (0, 0, 0)
red = (230/255, 25/255, 75/255)
green = (60/255, 180/255, 75/255)
blue = (0/255, 130/255, 200/255)
purple = (145/255, 30/255, 180/255)
orange = (245/255, 130/255, 48/255)
yellow = (255/255, 255/255, 25/255)
cyan = (70/255, 240/255, 240/255)
maroon = (128/255, 0, 0)
olive = (128/255, 128/255, 0)
teal = (0, 128/255, 128/255)
navy = (0, 0, 128/255)
lime = (210/255, 245/255, 60/255)
magenta = (240/255, 50/255, 230/255)
brown = (170/255, 110/255, 40/255)
pink = (250/255, 190/255, 212/255)
apricot = (255/255, 215/255, 180/255)
beige = (255/255, 250/255, 200/255)
mint = (170/255, 255/255, 195/255)
lavender = (220/255, 190/255, 255/255)
'''

alpha = 220
color_palette_rgba = [
                np.array([25, 180, 25, alpha]), 
                np.array([230,25,25, alpha]),
                np.array([0,130,200, alpha]),
                np.array([245,130,0, alpha]),
                np.array([170,0,100, alpha]),
                np.array([50,0,50, alpha]),
                np.array([0, 25, 25, alpha]), 
                np.array([50,10,25, alpha]),
                np.array([25,75,150, alpha]),
                np.array([245,20,50, alpha]),
                np.array([30,10,90, alpha]),
                np.array([100,0,50, alpha]),
                np.array([128, 128, 128, alpha])
                ]

color_palette_rgb_str = [
                'rgb(25, 180, 25)', 
                'rgb(230,25,25)', 
                'rgb(0,130,200)', 
                'rgb(245,130,0)', 
                'rgb(170,0,100)', 
                'rgb(50,0,50)', 
                'rgb(0, 25, 25)',
                'rgb(50,10,25)',
                'rgb(25,75,150)',
                'rgb(245,20,50)',
                'rgb(30,10,90)',
                'rgb(100,0,50)',    
                'rgb(128, 128, 128)'
                ]

import matplotlib._color_data as mcd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
def to_numpy(item):
    if torch.is_tensor(item):
        if item.is_cuda:
            return item.cpu().detach().numpy() 
        else:
            return item.detach().numpy()
    else:
        return item

def to_tensor(item):
    return torch.tensor(item, device=device)

def makeLookAt(position, target, up):
        
    forward = np.subtract(target, position)
    forward = np.divide( forward, np.linalg.norm(forward) )

    right = np.cross( forward, up )
    
    # if forward and up vectors are parallel, right vector is zero; 
    #   fix by perturbing up vector a bit
    if np.linalg.norm(right) < 0.001:
        epsilon = np.array( [0.001, 0, 0] )
        right = np.cross( forward, up + epsilon )
        
    right = np.divide( right, np.linalg.norm(right) )
    
    up = np.cross( right, forward )
    up = np.divide( up, np.linalg.norm(up) )
    
    return np.array([[right[0], up[0], -forward[0], position[0]], 
                        [right[1], up[1], -forward[1], position[1]], 
                        [right[2], up[2], -forward[2], position[2]],
                        [0, 0, 0, 1]]) 

def render_meshes(meshes, is_target, filename):
    scene = pyrender.Scene(ambient_light=[0.2, 0.2, 0.2], bg_color=(0,0,0,0))
    for i, recon_part_mesh in enumerate(meshes):

        trimesh.repair.fix_normals(recon_part_mesh)
        if is_target:
            color_index = -1
        else:
            color_index = i
            
        vertex_colors = np.array([color_palette_rgba[color_index]] * len(recon_part_mesh.vertices)).reshape((len(recon_part_mesh.vertices), 4))
        recon_part_mesh.visual.vertex_colors = vertex_colors
        mesh = pyrender.Mesh.from_trimesh(recon_part_mesh)
        scene.add(mesh)
    
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0, zfar = 10)
    camera_pose = makeLookAt(np.array([1.5,0.75,1.5]), np.array([0,0,0]), np.array([0,1,0]))
    scene.add(camera, pose=camera_pose)

    #light1 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    light1 = pyrender.SpotLight(color=[1.0, 1.0, 1.0], intensity=5.0, innerConeAngle=0.0001, outerConeAngle=1.0)
    #light2 = pyrender.SpotLight(color=[1.0, 1.0, 1.0], intensity=2.0, innerConeAngle=0.05, outerConeAngle=1.0)
    #light3 = pyrender.SpotLight(color=[1.0, 1.0, 1.0], intensity=2.0, innerConeAngle=0.05, outerConeAngle=1.0)
    #light4 = pyrender.SpotLight(color=[1.0, 1.0, 1.0], intensity=2.0, innerConeAngle=0.05, outerConeAngle=1.0)
    #light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.0)
    light_pose1 = makeLookAt(np.array([1.0,1.0,1.0]), np.array([0,0,0]), np.array([0,1,0]))
    #light_pose2 = makeLookAt(np.array([-1.0,1.0,1.0]), np.array([0,0,0]), np.array([0,1,0]))
    #light_pose3 = makeLookAt(np.array([1.0,1.0,-1.0]), np.array([0,0,0]), np.array([0,1,0]))
    #light_pose4 = makeLookAt(np.array([-1.0,1.0,-1.0]), np.array([0,0,0]), np.array([0,1,0]))
    scene.add(light1, pose=light_pose1)
    #scene.add(light2, pose=light_pose2)
    #scene.add(light3, pose=light_pose3)
    #scene.add(light4, pose=light_pose4)
    r = pyrender.OffscreenRenderer(1000, 1000)
    color, depth = r.render(scene, flags=224|2048)
    plt.figure()
    #plt.subplot(1,2,1)
    plt.axis('off')
    plt.imshow(color)
    #plt.subplot(1,2,2)
    plt.axis('off')
    #plt.imshow(depth, cmap=plt.cm.gray_r)
    plt.savefig(filename, transparent=True)
    plt.close()
    
def display_pcs(pcs, filename=None, save=False, is_grey=False, colors=None):

    if pcs is None:
        return 

    pcs = to_numpy(pcs)

    xmin = np.inf
    xmax = -np.inf
    ymin = np.inf
    ymax = -np.inf
    zmin = np.inf
    zmax = -np.inf

    traces = []
    for i in range(len(pcs)):
        pcs[i] = to_numpy(pcs[i])
        x = []
        y = []
        z = []
        c = []
        for p in pcs[i]:
            if is_grey:
                c.append(str(color_palette_rgb_str[-1]))
            else:
                if colors is None:
                    c.append(str(color_palette_rgb_str[i%len(color_palette_rgb_str)]))
                else:
                    c.append(str(colors[i]))
                    
            x.append(p[0])
            if p[0] < xmin:
                xmin = p[0]
            if p[0] > xmax:
                xmax = p[0] 
            y.append(p[1])
            if p[1] < ymin:
                ymin = p[1]
            if p[1] > ymax:
                ymax = p[1] 
            z.append(p[2])
            if p[2] < zmin:
                zmin = p[2]
            if p[2] > zmax:
                zmax = p[2] 

        trace = go.Scatter3d(
            x=x, 
            y=y, 
            z=z, 
            mode='markers', 
            marker=dict(
                size=3,
                color=c,                  
                opacity=1.0
            )
        )
        traces.append(trace)

    fig = go.Figure(data=traces)

    #range_min = min(xmin,min(ymin,zmin))
    #range_max = max(xmax,max(ymax,zmax))
    range_min = -1
    range_max = 1

    camera = dict(
    up=dict(x=0, y=1, z=0),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=1.5, y=1.0, z=1.5)
    )

    fig.update_layout(scene_camera=camera, showlegend=False)

    fig.update_layout(
        scene = dict(xaxis = dict(range=[range_min, range_max],),
                     yaxis = dict(range=[range_min, range_max],),
                     zaxis = dict(range=[range_min, range_max],),
                     aspectmode='cube'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        autosize=False,width=1000,height=1000,
        #title=filename
        )

    fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False )

    if save:
        fig.write_image(filename)
    else:
        fig.show()

def display_meshes(meshes, filename=' ', save=False):

    xmin = np.inf
    xmax = -np.inf
    ymin = np.inf
    ymax = -np.inf
    zmin = np.inf
    zmax = -np.inf

    traces = []
    for mesh_index in range(len(meshes)):
        print('mesh_index', mesh_index)
        color = color_palette_rgb_str[mesh_index%len(color_palette_rgb_str)]
        mesh = meshes[mesh_index]
        x = []
        y = []
        z = []
        i = []
        j = []
        k = []
        
        for v in mesh.vertices:
            x.append(v[0])
            y.append(v[1])
            z.append(v[2])
            if v[0] < xmin:
                xmin = v[0]
            if v[0] > xmax:
                xmax = v[0] 
            if v[1] < ymin:
                ymin = v[1]
            if v[1] > ymax:
                ymax = v[1] 
            if v[2] < zmin:
                zmin = v[2]
            if v[2] > zmax:
                zmax = v[2]
        
        for f in mesh.faces:
            i.append(f[0])
            j.append(f[1])
            k.append(f[2])

        trace = go.Mesh3d(
            x=x, 
            y=y, 
            z=z, 
            i = i,
            j = j,
            k = k,
            color = color,
            opacity = 1.0
        )
        traces.append(trace)

    fig = go.Figure(data=traces)

    #range_min = min(xmin,min(ymin,zmin))
    #range_max = max(xmax,max(ymax,zmax))
    range_min = -1.5
    range_max = 1.5

    camera = dict(
    up=dict(x=0, y=1, z=0),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=1.5, y=1.0, z=1.5)
    )

    fig.update_layout(scene_camera=camera)

    fig.update_layout(
        scene = dict(xaxis = dict(range=[range_min, range_max],),
                     yaxis = dict(range=[range_min, range_max],),
                     zaxis = dict(range=[range_min, range_max],),
                     aspectmode='cube'),
        #paper_bgcolor='rgba(0,0,0,0)',
        #plot_bgcolor='rgba(0,0,0,0)',
        autosize=False,width=1000,height=1000,
        title=filename
        )

    #fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False )

    if save:
        fig.write_image(filename)
    else:
        fig.show()

'''
grey = (128/255, 128/255, 128/255)
black = (0, 0, 0)
red = (230/255, 25/255, 75/255)
green = (60/255, 180/255, 75/255)
blue = (0/255, 130/255, 200/255)
purple = (145/255, 30/255, 180/255)
orange = (245/255, 130/255, 48/255)
yellow = (255/255, 255/255, 25/255)
cyan = (70/255, 240/255, 240/255)
maroon = (128/255, 0, 0)
olive = (128/255, 128/255, 0)
teal = (0, 128/255, 128/255)
navy = (0, 0, 128/255)
lime = (210/255, 245/255, 60/255)
magenta = (240/255, 50/255, 230/255)
brown = (170/255, 110/255, 40/255)
pink = (250/255, 190/255, 212/255)
apricot = (255/255, 215/255, 180/255)
beige = (255/255, 250/255, 200/255)
mint = (170/255, 255/255, 195/255)
lavender = (220/255, 190/255, 255/255)
'''

from matplotlib import cm
import matplotlib.colors as mcolors

enc_color_palette = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'cyan', 'olive', 'teal', 'lime', 'magenta', 'brown', 'pink']

css4_colors = list(mcolors.CSS4_COLORS.keys())
np.random.shuffle(css4_colors)
#print('css4_colors', css4_colors.keys())
#exit()

def generate_mesh_shots(in_mesh, shot_count):

    figs = []

    angle_delta = (2*np.pi)/shot_count

    eye_pos = torch.tensor([1.3, 1.0, 1.3], device=device, dtype=torch.float)
    up_axis = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float)
    origin = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float)
        
    for i in range(shot_count):

        #print('generating shot ', i)

        scene = pyrender.Scene(ambient_light=[0.2, 0.2, 0.2])
    
        trimesh.repair.fix_normals(in_mesh)
    
        vertex_colors = np.array([np.array([240, 240, 240])] * len(in_mesh.vertices)).reshape((len(in_mesh.vertices), 3))
        in_mesh.visual.vertex_colors = vertex_colors
        mesh = pyrender.Mesh.from_trimesh(in_mesh)
        scene.add(mesh)
        
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0, zfar = 10)

        rotated_eye_pos = to_numpy(rotate_with_axis_center_angle(torch.stack([eye_pos]), up_axis, origin, torch.tensor(i*angle_delta,device=device))[0])

        camera_pose = makeLookAt(rotated_eye_pos, np.array([0,0,0]), np.array([0,1,0]))
        scene.add(camera, pose=camera_pose)

        #light1 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        light1 = pyrender.SpotLight(color=[1.0, 1.0, 1.0], intensity=5.0, innerConeAngle=0.0001, outerConeAngle=1.0)
        #light2 = pyrender.SpotLight(color=[1.0, 1.0, 1.0], intensity=2.0, innerConeAngle=0.05, outerConeAngle=1.0)
        #light3 = pyrender.SpotLight(color=[1.0, 1.0, 1.0], intensity=2.0, innerConeAngle=0.05, outerConeAngle=1.0)
        #light4 = pyrender.SpotLight(color=[1.0, 1.0, 1.0], intensity=2.0, innerConeAngle=0.05, outerConeAngle=1.0)
        #light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.0)
        light_pose1 = makeLookAt(np.array([1.0,1.0,1.0]), np.array([0,0,0]), np.array([0,1,0]))
        #light_pose2 = makeLookAt(np.array([-1.0,1.0,1.0]), np.array([0,0,0]), np.array([0,1,0]))
        #light_pose3 = makeLookAt(np.array([1.0,1.0,-1.0]), np.array([0,0,0]), np.array([0,1,0]))
        #light_pose4 = makeLookAt(np.array([-1.0,1.0,-1.0]), np.array([0,0,0]), np.array([0,1,0]))
        scene.add(light1, pose=light_pose1)
        #scene.add(light2, pose=light_pose2)
        #scene.add(light3, pose=light_pose3)
        #scene.add(light4, pose=light_pose4)
        r = pyrender.OffscreenRenderer(512, 512)
        rendered_img, depth = r.render(scene, flags=224)
        #plt.figure()
        #plt.subplot(1,2,1)
        #plt.axis('off')
        #plt.imshow(color)
        #plt.subplot(1,2,2)
        #plt.axis('off')
        #plt.imshow(depth, cmap=plt.cm.gray_r)
        #plt.savefig(filename)
        #plt.close()

        figs.append(rendered_img)
    
    return figs


def get_free_mem():
    main_mem = get_free_main_mem()
    cuda_mem = get_free_cuda_mem()
    return main_mem, cuda_mem
    
def get_free_cuda_mem():

    mem = torch.cuda.mem_get_info(device=0)
    return mem

    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    return f

def get_free_main_mem():
    total_memory, used_memory, free_memory = map(
    int, os.popen('free -t -m').readlines()[-1].split()[1:])
    
    free_mem_percent = round((free_memory/total_memory) * 100, 2)
    
    return free_memory


def visualize_part_encs(encs, labels, filename):
    encs = to_numpy(encs)
    data_2d = TSNE(n_components=2, perplexity=50, n_iter=5000, learning_rate=200).fit_transform(encs)
    data_embedded_x = data_2d[:,0]
    data_embedded_y = data_2d[:,1]
    #data_embedded_x = encs[:, 0]
    #data_embedded_y = encs[:, 1]
    fig, ax = plt.subplots(figsize=(16,12))

    label_count = max(labels)

    print('label_count', label_count)
    
    for i in range(label_count):
        labeled_x = []
        labeled_y = []
        color_indices = []
        for j in range(len(encs)):
            if labels[j] == i:
                labeled_x.append(data_embedded_x[j])
                labeled_y.append(data_embedded_y[j])
                color_indices.append(labels[j])
        
        ax.scatter(labeled_x, labeled_y, c=css4_colors[i], label=i)

    #color_indices = []
    #for i in range(len(encs)):
        #color_indices.append(labels[i])
        #color = enc_color_palette[color_index%len(enc_color_palette)]
        #ax.scatter(data_embedded_x[i], data_embedded_y[i], c=color)
        #ax.annotate(labels[i], (data_embedded_x[i], data_embedded_y[i]))
    #ax.scatter(data_embedded_x[0:len(encs)], data_embedded_y[0:len(encs)], c=color_indices, cmap=plt.get_cmap('hsv'))

    ax.legend()
    fig.savefig(filename)


def visualize_part_encs_with_gaussians(encs, enc_labels, centers, filename):

    print('encs shape', encs.shape)
    encs = to_numpy(encs)
    centers = to_numpy(centers)
    
    combined_encs = np.concatenate((encs, centers),axis=0)

    data_2d = TSNE(n_components=2, perplexity=50, n_iter=5000, learning_rate=200).fit_transform(combined_encs)

    data_embedded_x = data_2d[:,0]
    data_embedded_y = data_2d[:,1]
    fig, ax = plt.subplots(figsize=(16,12))

    color_indices = []
    for i in range(len(encs)):
        color_indices.append(enc_labels[i])
        #color = enc_color_palette[color_index%len(enc_color_palette)]
        #ax.scatter(data_embedded_x[i], data_embedded_y[i], c=color)
        #ax.annotate(str(color_index), (data_embedded_x[i], data_embedded_y[i]), c=color)

    ax.scatter(data_embedded_x[0:len(encs)], data_embedded_y[0:len(encs)], c=color_indices, cmap=plt.get_cmap('hsv'))

    gaussian_index = 0
    for i in range(len(encs), len(encs)+len(centers)):
        color = 'black'
        ax.scatter(data_embedded_x[i], data_embedded_y[i], s=400, c=color, marker='x')
        ax.annotate(str(gaussian_index), (data_embedded_x[i], data_embedded_y[i]))
        gaussian_index += 1

    fig.savefig(filename)


def draw_error():

    #v_list = read_file_to_items('exp/validation.txt')
    v_list = [1.43, 1.1, 0.93, 0.68, 0.65, 0.82, 0.54, 0.99, 0.78, 0.71, 0.54, 0.50, 0.51, 0.52, 0.53, 0.54, 0.48, 0.54, 0.46, 0.52, 0.51, 0.56, 0.55, 0.58, 0.5, 0.54, 0.52, 0.53, 0.54, 0.58, 0.55, 0.53, 0.47,0.52,0.5,0.52]
    h_list = []
    for i in range(0, len(v_list)*32, 32):
        h_list.append(i)

    fig=plt.figure()
    ax=fig.add_axes([0.1,0.1,0.8,0.8])
    ax.set_xlabel('number of shapes trained')
    ax.set_ylabel('test shapes average reconstruction error')
    ax.set_title('testing')
    
    ax.scatter(h_list, v_list, color='r')
    ax.plot(h_list, v_list, color='r', label='model')
    ax.legend()

    plt.savefig('exp/error.png')
    plt.close()





def draw_bars_new(filename):
    items = read_file_to_items(filename)
    ids = []
    errors = []
    for item in items:
        print('item', item)
        errors.append(float(item.split(' ')[1]))
    
    print('errors', errors)

    xs = np.arange(0, 0.05, 0.0001)
    ys = [0]*len(xs)
    #print('x_locs', x_locs)
    
    for error in errors:
        for i in range(len(xs)-1):
            if error > xs[i] and error < xs[i+1]:
                ys[i] += 1
                break
    
    print('xs', xs)
    print('ys', ys)

    plt.bar(xs, ys, width=0.001)
    plt.savefig('test.png')

    exit()


def draw_lines():

    fig, ax = plt.subplots(figsize=(8,4))

    xs = [0, 1, 2, 3, 4]
    ys = [0.00411, 0.00386, 0.00372, 0.00376, 0.00380]

    x_names = ['iter0', 'iter1', 'iter2', 'iter3', 'iter4']
    plt.xticks(xs, x_names)

    ax.plot(xs, ys, label='Ours surface CD')

    ys1 = [0.00429] * len(ys)
    ax.plot(xs, ys1, label='NP surface CD')
    
    ys2 = [0.00256, 0.00221, 0.00228, 0.00220, 0.00221]
    ax.plot(xs, ys2, label='Ours Volume CD')

    ys3 = [0.00276] * len(ys)
    ax.plot(xs, ys3, label='NP volume CD')

    ax.set_ylim(0, 0.005)
    ax.legend()

    plt.savefig('test1.png')


if __name__ == "__main__":
    draw_bars_new('per_shape_chamfer_distance.txt')
    #draw_lines()