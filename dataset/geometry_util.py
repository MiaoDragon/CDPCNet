"""
utility functions for generating simple geometries dataset
including functions for generating collision and no-collision data
"""
import math
import numpy as np
import klampt
import open3d
from klampt import vis as klvis
import time
import pandas as pd
from tqdm import tqdm
from pyntcloud import PyntCloud

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




def visualize(pcs):
    # visualize point cloud list
    pcd_list = []
    for pc in pcs:
        pcd = open3d.PointCloud()
        pcd.points = open3d.Vector3dVector(pc.T)
        pcd_list.append(pcd)
    vis = open3d.Visualizer()
    vis.create_window()
    for pcd in pcd_list:
        vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

def make_mesh(P):
    # given a point cloud in shape: 3xN
    # construct the mesh representation of the point cloud
    # for collision detection later
    pc = pd.DataFrame(P.T)
    pc.columns = ['x', 'y', 'z']
    pc = PyntCloud(pc)
    convex_hull_id = pc.add_structure("convex_hull")
    convex_hull = pc.structures[convex_hull_id]
    pc.mesh = convex_hull.get_mesh()
    #ax = fig.add_subplot(111, projection='3d')
    xs = P.T[pc.mesh['v1']]
    ys = P.T[pc.mesh['v2']]
    zs = P.T[pc.mesh['v3']]
    #pc1.mesh.plot()#backend='matplotlib')
    #pc1.plot(backend="matplotlib")
    pc = klampt.TriangleMesh()
    #pc2 = klampt.TriangleMesh()
    for i in range(len(xs)):
        for j in range(3):
            pc.vertices.append(xs[i,j])
        for j in range(3):
            pc.vertices.append(ys[i,j])
        for j in range(3):
            pc.vertices.append(zs[i,j])
        pc.indices.append(i*3)
        pc.indices.append(i*3+1)
        pc.indices.append(i*3+2)
    geo = klampt.Geometry3D(pc)
    return geo

def collision_check(P1, P2):
    # given point cloud representation, detect whether in collision
    # True if in collision, False otherwise
    geo1 = make_mesh(P1)
    geo2 = make_mesh(P2)
    return geo1.collides(geo2)

def generate_one_collision(N=2800, pert_ratio=0.1):
    # ******generate collision data******
    # randomly select ball or cube, but only move a little relatively
    while True:
        geo_type = np.random.randint(low=0, high=2)
        rotation_axis1 = np.random.normal(size=3)
        rotation_axis1 = rotation_axis1 / np.linalg.norm(rotation_axis1)
        rad1 = np.random.uniform(low=-np.pi, high=np.pi)
        R1 = rotation_matrix(rotation_axis1, rad1)
        scale1 = np.random.uniform(low=0.5, high=1.5)
        move_d1 = np.random.normal(size=3)
        move_d1 = move_d1 / np.linalg.norm(move_d1)
        move_s1 = np.random.uniform(low=0., high=10.)
        move1 = move_d1 * move_s1
        if geo_type == 0:
            P1 = generate_cube(N, scale1, move1, R1)
            P1_solid = generate_solid_cube(4*N, scale1, move1, R1)
        else:
            P1 = generate_ball(N, scale1, move1)
            P1_solid = generate_solid_ball(4*N, scale1, move1)

        geo_type = np.random.randint(low=0, high=2)
        rotation_axis2 = np.random.normal(size=3)
        rotation_axis2 = rotation_axis2 / np.linalg.norm(rotation_axis2)
        rad2 = np.random.uniform(low=-np.pi, high=np.pi)
        R2 = rotation_matrix(rotation_axis2, rad2)
        scale2 = np.random.uniform(low=0.5, high=1.5)
        move_d2 = np.random.normal(size=3)
        move_d2 = move_d2 / np.linalg.norm(move_d2) * pert_ratio
        move_d2 = move_d1 + move_d2
        move_d2 = move_d2 / np.linalg.norm(move_d2)
        move_s2 = np.random.uniform(low=move_s1-scale1, high=move_s1+scale1)
        move2 = move_d2 * move_s2
        if geo_type == 0:
            P2 = generate_cube(N, scale2, move2, R2)
            P2_solid = generate_solid_cube(4*N, scale2, move2, R2)
        else:
            P2 = generate_ball(N, scale2, move2)
            P2_solid = generate_solid_ball(4*N, scale2, move2)

        if collision_check(P1_solid, P2_solid):
            return P1, P2

def generate_collision(batch=1000, N=2800, pert_ratio=0.1):
    print('collision:')
    collision_data_P1 = []
    collision_data_P2 = []
    for i in tqdm(range(batch)):
        #print('collision: %d' % (i))
        P1, P2 = generate_one_collision(N, pert_ratio)
        collision_data_P1.append(P1)
        collision_data_P2.append(P2)
    collision_data_P1 = np.array(collision_data_P1)
    collision_data_P2 = np.array(collision_data_P2)
    return collision_data_P1, collision_data_P2

def generate_one_no_collision(N=2800):
    # ******generate collision data******
    # randomly select ball or cube, but only move a little relatively
    while True:
        geo_type = np.random.randint(low=0, high=2)
        rotation_axis1 = np.random.normal(size=3)
        rotation_axis1 = rotation_axis1 / np.linalg.norm(rotation_axis1)
        rad1 = np.random.uniform(low=-np.pi, high=np.pi)
        R1 = rotation_matrix(rotation_axis1, rad1)
        scale1 = np.random.uniform(low=0.5, high=1.5)
        move_d1 = np.random.normal(size=3)
        move_d1 = move_d1 / np.linalg.norm(move_d1)
        move_s1 = np.random.uniform(low=0., high=10.)
        move1 = move_d1 * move_s1
        if geo_type == 0:
            P1 = generate_cube(N, scale1, move1, R1)
            P1_solid = generate_solid_cube(4*N, scale1, move1, R1)
        else:
            P1 = generate_ball(N, scale1, move1)
            P1_solid = generate_solid_ball(4*N, scale1, move1)

        geo_type = np.random.randint(low=0, high=2)
        rotation_axis2 = np.random.normal(size=3)
        rotation_axis2 = rotation_axis2 / np.linalg.norm(rotation_axis2)
        rad2 = np.random.uniform(low=-np.pi, high=np.pi)
        R2 = rotation_matrix(rotation_axis2, rad2)
        scale2 = np.random.uniform(low=0.5, high=1.5)
        move_d2 = np.random.normal(size=3)
        move_d2 = move_d2 / np.linalg.norm(move_d2)
        move_s2 = np.random.uniform(low=0., high=10.)
        move2 = move_d2 * move_s2
        if geo_type == 0:
            P2 = generate_cube(N, scale2, move2, R2)
            P2_solid = generate_solid_cube(4*N, scale2, move2, R2)
        else:
            P2 = generate_ball(N, scale2, move2)
            P2_solid = generate_solid_ball(4*N, scale2, move2)

        if not collision_check(P1_solid, P2_solid):
            return P1, P2

def generate_no_collision(batch=1000, N=2800):
    print('no collision:')
    no_collision_data_P1 = []
    no_collision_data_P2 = []
    for i in tqdm(range(batch)):
        #print('no collision: %d' % (i))
        P1, P2 = generate_one_no_collision(N)
        no_collision_data_P1.append(P1)
        no_collision_data_P2.append(P2)
    no_collision_data_P1 = np.array(no_collision_data_P1)
    no_collision_data_P2 = np.array(no_collision_data_P2)
    return no_collision_data_P1, no_collision_data_P2


def test():
    P1, P2 = generate_one_no_collision()
    visualize([P1, P2])
    geo1 = make_mesh(P1_solid)
    geo2 = make_mesh(P2_solid)
    print(geo1.collides(geo2))
    def setup():
      klvis.show()

    def callback():
      #...do stuff to world... #this code is executed at approximately 10 Hz due to the sleep call
      time.sleep(0.1)
      if False:
        klvis.show(False)         #hides the window if not closed by user

    def cleanup():
      #can perform optional cleanup code here
      pass
    #geo1.drawGL()
    #app = klampt.Appearance()
    #app.drawGL(geo1)
    klvis.add('geo1',geo1)
    klvis.add('geo2',geo2)
    klvis.loop(setup=setup,callback=callback,cleanup=cleanup)

if __name__=='__main__':
    test()
