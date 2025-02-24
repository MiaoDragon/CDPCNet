from klampt import *
from klampt import vis as vis
m = TriangleMesh()
m2 = TriangleMesh()
points = [0,0,0,1,0,0,1,1,0,0,0,1,1,0,1,1,1,1,0,1,1,.5,.5,1,.5,.5,0,.5,.5,0]
for p in points:
    m.vertices.append(p)
    m2.vertices.append(p)
scale = 3
m.indices.append(0)
m.indices.append(1)
m.indices.append(2)
m.indices.append(0)
m.indices.append(2)
m.indices.append(3)
m.indices.append(0)
m.indices.append(1)
m.indices.append(5)
m.indices.append(0)
m.indices.append(5)
m.indices.append(4)
m.indices.append(1)
m.indices.append(2)
m.indices.append(5)
m.indices.append(2)
m.indices.append(5)
m.indices.append(6)
m.indices.append(0)
m.indices.append(3)
m.indices.append(4)
m.indices.append(3)
m.indices.append(4)
m.indices.append(7)
m.indices.append(2)
m.indices.append(3)
m.indices.append(6)
m.indices.append(3)
m.indices.append(6)
m.indices.append(7)
m.indices.append(4)
m.indices.append(5)
m.indices.append(6)
m.indices.append(4)
m.indices.append(6)
m.indices.append(7)

m.indices.append(4)
m.indices.append(6)
m.indices.append(8)
m.indices.append(4)
m.indices.append(6)
m.indices.append(9)


m2.indices.append(0)
m2.indices.append(1)
m2.indices.append(2)
m2.indices.append(0)
m2.indices.append(2)
m2.indices.append(3)
m2.indices.append(0)
m2.indices.append(1)
m2.indices.append(5)
m2.indices.append(0)
m2.indices.append(5)
m2.indices.append(4)
m2.indices.append(1)
m2.indices.append(2)
m2.indices.append(5)
geo1 = Geometry3D(m)
geo2 = Geometry3D(m2)
#bb = geo1.getBB()
#print(bb)
pc = geo1.convert('PointCloud')
pc = pc.getPointCloud()
print(geo1.collides(geo2))
vis.add('geo1', geo1)
vis.add('geo2', geo2)
vis.run()
vis.kill()
