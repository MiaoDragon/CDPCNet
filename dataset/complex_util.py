"""
utility functions for generating simple geometries dataset
including functions for generating collision and no-collision data
"""
import math
import numpy as np
import klampt
#import open3d
from klampt import vis as klvis
import time
import pandas as pd
from tqdm import tqdm
from pyntcloud import PyntCloud
from pyntcloud.structures.delanuay import Delaunay3D
# globally load point clouds
geos = ['dragon', 'happy', 'horse', 'bunny']
geo_clouds = {}
geo_pcs = {}
geo_meshs = {}
for geo in geos:
    # use compressed version for collision detection
    cloud = PyntCloud.from_file('complex/'+geo+'_compressed.ply')
    geo_clouds[geo] = cloud
    geo_meshs[geo] = cloud.mesh
    mean_x = cloud.points['x'].mean()
    mean_y = cloud.points['y'].mean()
    mean_z = cloud.points['z'].mean()
    cloud.points['x'] = cloud.points['x'] - mean_x
    cloud.points['y'] = cloud.points['y'] - mean_y
    cloud.points['z'] = cloud.points['z'] - mean_z
    cloud.points['x'] = cloud.points['x'] * 10
    cloud.points['y'] = cloud.points['y'] * 10
    cloud.points['z'] = cloud.points['z'] * 10
    cloud.mesh = geo_meshs[geo]
    geo_clouds[geo] = cloud
    pc = PyntCloud.from_file('complex/'+geo+'_compressed.ply')
    geo_pcs[geo] = np.array([pc.points['x'], pc.points['y'], pc.points['z']])
    choices = np.random.choice(len(pc.points['x']), 2800)
    geo_pcs[geo] = geo_pcs[geo][..., choices]
    geo_pcs[geo][0,:] -= mean_x
    geo_pcs[geo][1,:] -= mean_y
    geo_pcs[geo][2,:] -= mean_z
    geo_pcs[geo] = 10 * geo_pcs[geo]
"""
def visualize(pcs):
    # visualize point cloud list
    pcd_list = []
    for pc in pcs:
        pcd = open3d.PointCloud()
        pcd.points = open3d.Vector3dVector(pc.T)
        print(pc.mean(axis=1))
        print(pc.std(axis=1))
        pcd_list.append(pcd)
    vis = open3d.Visualizer()
    vis.create_window()
    for pcd in pcd_list:
        vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
"""

def make_mesh(pc):
    # given PyntCloud
    # construct the mesh representation of the point cloud
    # for collision detection later
    P = pc.points
    P = np.array([P['x'], P['y'], P['z']])
    xs = P.T[pc.mesh['v1']].astype('d')
    ys = P.T[pc.mesh['v2']].astype('d')
    zs = P.T[pc.mesh['v3']].astype('d')
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


def collision_check(P1, P2):
    # given point cloud representation, detect whether in collision
    # True if in collision, False otherwise
    geo1 = make_mesh(P1)
    geo2 = make_mesh(P2)
    klvis.add('geo1', geo1)
    klvis.add('geo2', geo2)
    #klvis.show()
    return geo1.collides(geo2)

def generate_one_collision(N=2800, pert_ratio=0.1):
    # ******generate collision data******
    # randomly select ball or cube, but only move a little relatively
    while True:
        geo_type = np.random.choice(len(geos))
        rotation_axis1 = np.random.normal(size=3)
        rotation_axis1 = rotation_axis1 / np.linalg.norm(rotation_axis1)
        rad1 = np.random.uniform(low=-np.pi, high=np.pi)
        R1 = rotation_matrix(rotation_axis1, rad1)
        scale1 = np.random.uniform(low=0.6, high=1.4)
        move_d1 = np.random.normal(size=3)
        move_d1 = move_d1 / np.linalg.norm(move_d1)
        move_s1 = np.random.uniform(low=0., high=10.)
        move1 = move_d1 * move_s1
        P1 = geo_pcs[geos[geo_type]]
        P1 = (scale1 * P1.T @ R1 + move1).T
        cloud1 = geo_clouds[geos[geo_type]]
        p1 = np.array([cloud1.points['x'], cloud1.points['y'], cloud1.points['z']])
        p1 = (scale1 * p1.T @ R1 + move1)
        p1 = pd.DataFrame(p1)
        p1.columns = ['x', 'y', 'z']
        cloud1.points = p1
        cloud1.mesh = geo_meshs[geos[geo_type]]


        geo_type = np.random.choice(len(geos))
        rotation_axis2 = np.random.normal(size=3)
        rotation_axis2 = rotation_axis2 / np.linalg.norm(rotation_axis2)
        rad2 = np.random.uniform(low=-np.pi, high=np.pi)
        R2 = rotation_matrix(rotation_axis2, rad2)
        scale2 = np.random.uniform(low=0.6, high=1.4)
        move_d2 = np.random.normal(size=3)
        move_d2 = move_d2 / np.linalg.norm(move_d2) * pert_ratio
        move_d2 = move_d1 + move_d2
        move_d2 = move_d2 / np.linalg.norm(move_d2)
        move_s2 = np.random.uniform(low=move_s1-scale1, high=move_s1+scale1)
        move2 = move_d2 * move_s2
        P2 = geo_pcs[geos[geo_type]]
        P2 = (scale2 * P2.T @ R2 + move2).T
        cloud2 = geo_clouds[geos[geo_type]]
        p2 = np.array([cloud2.points['x'], cloud2.points['y'], cloud2.points['z']])
        p2 = (scale2 * p2.T @ R2 + move2)
        p2 = pd.DataFrame(p2)
        p2.columns = ['x', 'y', 'z']
        cloud2.points = p2
        cloud2.mesh = geo_meshs[geos[geo_type]]

        if collision_check(cloud1, cloud2):
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
        geo_type = np.random.choice(len(geos))
        rotation_axis1 = np.random.normal(size=3)
        rotation_axis1 = rotation_axis1 / np.linalg.norm(rotation_axis1)
        rad1 = np.random.uniform(low=-np.pi, high=np.pi)
        R1 = rotation_matrix(rotation_axis1, rad1)
        scale1 = np.random.uniform(low=0.5, high=1.5)
        move_d1 = np.random.normal(size=3)
        move_d1 = move_d1 / np.linalg.norm(move_d1)
        move_s1 = np.random.uniform(low=0., high=10.)
        move1 = move_d1 * move_s1
        P1 = geo_pcs[geos[geo_type]]
        P1 = (scale1 * P1.T @ R1 + move1).T
        cloud1 = geo_clouds[geos[geo_type]]
        p1 = np.array([cloud1.points['x'], cloud1.points['y'], cloud1.points['z']])
        p1 = (scale1 * p1.T @ R1 + move1)
        p1 = pd.DataFrame(p1)
        p1.columns = ['x', 'y', 'z']
        cloud1.points = p1
        cloud1.mesh = geo_meshs[geos[geo_type]]

        geo_type = np.random.choice(len(geos))
        rotation_axis2 = np.random.normal(size=3)
        rotation_axis2 = rotation_axis2 / np.linalg.norm(rotation_axis2)
        rad2 = np.random.uniform(low=-np.pi, high=np.pi)
        R2 = rotation_matrix(rotation_axis2, rad2)
        scale2 = np.random.uniform(low=0.5, high=1.5)
        move_d2 = np.random.normal(size=3)
        move_d2 = move_d2 / np.linalg.norm(move_d2)
        move_s2 = np.random.uniform(low=0., high=10.)
        move2 = move_d2 * move_s2
        P2 = geo_pcs[geos[geo_type]]
        P2 = (scale2 * P2.T @ R2 + move2).T
        cloud2 = geo_clouds[geos[geo_type]]
        p2 = np.array([cloud2.points['x'], cloud2.points['y'], cloud2.points['z']])
        p2 = (scale2 * p2.T @ R2 + move2)
        p2 = pd.DataFrame(p2)
        p2.columns = ['x', 'y', 'z']
        cloud2.points = p2
        cloud2.mesh = geo_meshs[geos[geo_type]]

        if not collision_check(cloud1, cloud2):
            return P1, P2

def generate_no_collision(batch=1000, N=2800):
    print('no collision:')
    no_collision_data_P1 = []
    no_collision_data_P2 = []
    for i in tqdm(range(batch)):
        #print('no collision: %d' % (i))
        P1, P2 = generate_one_no_collision(N)
    no_collision_data_P1 = np.array(no_collision_data_P1)
    no_collision_data_P2 = np.array(no_collision_data_P2)
    return no_collision_data_P1, no_collision_data_P2

"""
def test():
    P1, P2 = generate_one_collision()
    visualize([P1, P2])
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
    #klvis.add('geo1',geo1)
    #klvis.add('geo2',geo2)
    klvis.loop(setup=setup,callback=callback,cleanup=cleanup)

if __name__=='__main__':
    test()
"""
