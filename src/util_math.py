
import sys
sys.path.append('..')
import numpy as np
import math
import torch

def iszero(v):
    if abs(v) < 10e-6:
        return True
    else:
        return False

def perpendicular_vector(v):
    if iszero(v[0]) and iszero(v[1]):
        if iszero(v[2]):
            # v is Vector(0, 0, 0)
            raise ValueError('zero vector')

        # v is Vector(0, 0, v.z)
        return np.array([0, 1, 0])
    return np.array([-v.y, v.x, 0])

def angle_between_vec(vec1, vec2):
    try:
        dotval = np.dot(vec1, vec2)
        angle = math.acos(dotval)
        return angle
    except:
        print('error')
        return 0.5 * np.pi

def point_to_line(p, p_on_line, line_dir):
    line_dir = line_dir / np.linalg.norm(line_dir)
    dis_vec = p - p_on_line
    distance = np.linalg.norm(dis_vec * np.dot(dis_vec, line_dir) - dis_vec)
    return distance

def cartesian_to_spherical(x, y, z):
    r = math.sqrt(x * x + y * y + z * z)
    theta = math.atan(y / x)
    phi = math.acos(z / r)

    return theta, phi, r

def spherical_to_cartesian(x, y, r):
    theta = x * math.pi * 2
    phi = math.pi * 1 * y
    x = r * torch.cos(theta) * torch.sin(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(phi)

    return [x, y, z]
