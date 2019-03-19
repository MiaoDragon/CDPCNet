"""
Generate point cloud from the standard cube
Acheive this by firstly fix one dim to be 0 or 1, and
generate the other two dims by uniform sampling in [0,1]
resulting shape: 1x3xN
"""
import numpy as np
from open3d import *
import math
def generate_cube(N, scale, move, rotation):
    """
    Generate point cloud from the standard cube
    Acheive this by firstly fix one dim to be 0 or 1, and
    generate the other two dims by uniform sampling in [0,1]
    resulting shape: 3xN
    input: scale -- scalar
           move  -- vector of size 3
           rotation  -- 3x3 matrix
    """
    pc = []
    for i in range(N):
        # firstly randomly select one dim to set to 0 or 1
        dim_1 = np.random.randint(low=0, high=3)
        val = np.random.uniform(low=0, high=1, size=3)
        val_1 = np.random.choice([0,1])
        val[dim_1] = val_1
        pc.append(val)
    # pc is of shape Nx3 now
    # scale, rotate and move each point
    pc = np.array(pc)
    # before rotation, center to origin
    pc = pc - 0.5
    # scale up
    pc = scale * pc
    # rotate
    pc = pc @ rotation
    # move
    pc = pc + move
    pc = pc.T
    return pc

def generate_solid_cube(N, scale, move, rotation):
    """
    Generate point cloud from the standard cube
    Acheive this by firstly fix one dim to be 0 or 1, and
    generate the other two dims by uniform sampling in [0,1]
    resulting shape: 3xN
    input: scale -- scalar
           move  -- vector of size 3
           rotation  -- 3x3 matrix
    """
    pc = np.random.uniform(low=0, high=1, size=(N,3))
    # pc is of shape Nx3 now
    # scale, rotate and move each point
    # before rotation, center to origin
    pc = pc - 0.5
    # scale up
    pc = scale * pc
    # rotate
    pc = pc @ rotation
    # move
    pc = pc + move
    pc = pc.T
    return pc

def generate_ball(N, scale, move):
    """
    Generate point cloud from the standard ball (radius=1)
    Achieve this by firstly using Gaussian distribution,
    then normalize the generated vectors
    resulting shape: 3xN
    input: scale -- scalar
           move  -- vector of size 3
           rotation  -- 3x3 matrix
    """
    pc = np.random.normal(loc=0, scale=1, size=(N,3))
    pc = (pc.T / np.linalg.norm(pc, axis=1)).T
    # pc is of shape Nx3 now
    # scale, and move each point
    # scale up
    pc = scale * pc
    # move
    pc = pc + move
    pc = pc.T
    return pc

def generate_solid_ball(N, scale, move):
    """
    Generate point cloud from the standard ball (radius=1)
    Achieve this by firstly using Gaussian distribution,
    then normalize the generated vectors
    resulting shape: 3xN
    input: scale -- scalar
           move  -- vector of size 3
           rotation  -- 3x3 matrix
    """
    pc = np.random.normal(loc=0, scale=1, size=(N,3))
    length = np.random.uniform(low=0, high=1, size=(N))
    pc = (pc.T / np.linalg.norm(pc, axis=1) * length).T

    # pc is of shape Nx3 now
    # scale, and move each point
    # scale up
    pc = scale * pc
    # move
    pc = pc + move
    pc = pc.T
    return pc

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    ref: https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
