from geometry_util import *
import numpy as np
import klampt
import open3d
from klampt import vis as klvis
import time
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
N = 2800
pert_ratio = 0.1
# ******generate collision data******
# randomly select ball or cube, but only move a little relatively
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

#visualize([P1, P2, P1_solid, P2_solid])

pc1 = klampt.PointCloud()
pc1.setPoints(4*N, P1_solid.flatten())
pc2 = klampt.PointCloud()
pc2.setPoints(4*N, P2_solid.flatten())

geo1 = klampt.Geometry3D(pc1)
geo2 = klampt.Geometry3D(pc2)
print(geo1.collides(geo2))

def setup():
  klvis.show()

def callback():
  #...do stuff to world... #this code is executed at approximately 10 Hz due to the sleep call
  time.sleep(0.1)
  if done():
    klvis.show(False)         #hides the window if not closed by user

def cleanup():
  #can perform optional cleanup code here
  pass
klvis.add('geo1',geo1)
klvis.add('geo2',geo2)
klvis.loop(setup=setup,callback=callback,cleanup=cleanup)

# ******generate collision-free data******
