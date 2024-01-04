import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.spatial import Delaunay
from functools import reduce





def get_distance(x):
    return np.sqrt(np.sum(x**2))


pcd = o3d.io.read_point_cloud("test_ICP/seq24.ply")



plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=10000)

[a,b,c,d] = plane_model

removed_pcd = pcd.select_by_index(inliers)
removed_pcd.paint_uniform_color((0,1,0))                 #This is floor === GREEN COLOR
desired_object = pcd.select_by_index(inliers, invert=True)
desired_object.paint_uniform_color((0, 1, 0)) #this is BOX TO BE MEAUSRE
# o3d.visualization.draw_geometries([desired_object])


##################################################################
# LOGIC IMPLEMENTED HERe 
##################################################

# step1 seprate floor from truck
#get all points of desired object >>>>>>>>>>>>>>>>>>>>>>>>RANAC ALGORITHM

plane_model_, inliers_ = desired_object.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=10000)

[a_,b_,c_,d_] = plane_model_

floor = desired_object.select_by_index(inliers_)
floor.paint_uniform_color((0,0,1))
truck_wall = desired_object.select_by_index(inliers_, invert = True)
truck_wall.paint_uniform_color((0,1,0))



o3d.visualization.draw_geometries([truck_wall])
desired_object_pts = np.asarray(desired_object.points)


total_points_desirec_object = np.asarray(desired_object.points)
print("total_points_desirec_object",total_points_desirec_object.shape)

#STEP2 further clean the floor. >>>>>>>>>>>>>>>>>>>>>>>>  RADIUS OUTLIER
floor_clean, _ = floor.remove_radius_outlier(nb_points=100, radius=0.05)  #radii depends on sensor sensitivity #earlier it was 0.07
floor_clean.paint_uniform_color((0, 0, 1)) # bear + 

o3d.visualization.draw_geometries([floor_clean])

#further clean the floor 


#STEP3  ULTRA CLEAN FLOOR >>>>>>>>>>>>>>>>>>>>>>>>>>>> STATISTICAL OUTLIER METHOD 

print("Statistical oulier removal...begins")
cl, ind = floor_clean.remove_statistical_outlier(nb_neighbors=50,
                                                    std_ratio=2.0)

floor_clean = floor_clean.select_by_index(ind)
#outlier_cloud = cloud.select_by_index(ind, invert=True)


print("remove_statistical_outlier")
obb_clean_floor = floor_clean.get_oriented_bounding_box()
obb_clean_floor.color = (0,1,0)
o3d.visualization.draw_geometries([obb_clean_floor,floor_clean])

obb_min_bound = obb_clean_floor.get_min_bound()
obb_max_bound = obb_clean_floor.get_max_bound()
tmp_distance = np.sqrt(np.sum((obb_clean_floor.get_max_bound() - obb_clean_floor.get_min_bound())**2))

gravity_center = floor_clean.get_center()

floor_clean_pts = np.array(floor_clean.points)

distance = np.array([get_distance(i - gravity_center) for i in floor_clean_pts])
threshold = np.mean(distance)
add_offset = threshold*2

inliers__ = np.where(distance <= add_offset)[0]
super_clean_floor = floor_clean.select_by_index(inliers__) 

# # desired_object =desired_object.voxel_down_sample(0.009)
axes = o3d.geometry.TriangleMesh.create_coordinate_frame() 

print("super clean")
o3d.visualization.draw_geometries([super_clean_floor])





removed_pcd_pts = np.asarray(super_clean_floor.points)

#get threshold for x and y [x_min, x_max, y_min, y_max]

x_min = np.min(removed_pcd_pts[:,0])
x_max = np.max(removed_pcd_pts[:,1])

y_min = np.min(removed_pcd_pts[:,1])
y_max = np.max(removed_pcd_pts[:,1])

z_min = np.min(removed_pcd_pts[:,2])
z_max = np.max(removed_pcd_pts[:,2])

print(type(z_max))



# total_points_desirec_object
#crop desired object basis above threshold.

# mask = (removed_pcd_pts[:, 0] >= x_min) & (removed_pcd_pts[:, 0] <= x_max) & \
#        (removed_pcd_pts[:, 1] >= y_min) & (removed_pcd_pts[:, 1] <= y_max) & \
#        (removed_pcd_pts[:, 2] >= z_min) & (removed_pcd_pts[:, 2] <= z_max)

mask = (total_points_desirec_object[:, 0] >= x_min) & (total_points_desirec_object[:, 0] <= x_max) & \
       (total_points_desirec_object[:, 1] >= y_min) & (total_points_desirec_object[:, 1] <= y_max) & \
       (total_points_desirec_object[:, 2] >= z_min) & (total_points_desirec_object[:, 2] <= z_max)


cropped_pcd = desired_object.select_by_index(np.where(mask)[0])

cropped_pcd = np.asarray(cropped_pcd.points )

pcd_ = o3d.geometry.PointCloud()
pcd_.points = o3d.utility.Vector3dVector(cropped_pcd)


o3d.visualization.draw_geometries([pcd_,super_clean_floor])

#STEp 4 Clean further to remove outliers from cropped area.

####################### STATISTICAL OUTLIERS REMOVAL 
print("Statistical oulier removal...begins")
cl, ind = pcd_.remove_statistical_outlier(nb_neighbors=20,
                                                    std_ratio=2.0)

pcd_clean = pcd_.select_by_index(ind)


o3d.visualization.draw_geometries([pcd_clean])


#
downpdc = pcd_clean.voxel_down_sample(voxel_size=0.05)
xyz = np.asarray(downpdc.points)

xy_catalog = []
for point in xyz:
    xy_catalog.append([point[0], point[1]])
tri = Delaunay(np.array(xy_catalog))

surface = o3d.geometry.TriangleMesh()
surface.vertices = o3d.utility.Vector3dVector(xyz)
surface.triangles = o3d.utility.Vector3iVector(tri.simplices)
o3d.visualization.draw_geometries([surface], mesh_show_wireframe=True)

def get_triangles_vertices(triangles, vertices):
    triangles_vertices = []
    for triangle in triangles:
        new_triangles_vertices = [vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]]
        triangles_vertices.append(new_triangles_vertices)
    return np.array(triangles_vertices)

def volume_under_triangle(triangle):
    p1, p2, p3 = triangle
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    x3, y3, z3 = p3
    return abs((z1+z2+z3)*(x1*y2-x2*y1+x2*y3-x3*y2+x3*y1-x1*y3)/6)


volume = reduce(lambda a, b:  a + volume_under_triangle(b), get_triangles_vertices(surface.triangles, surface.vertices), 0)
print(f"The volume of the stockpile is: {round(volume, 4)} m3")


#METHOD 2 CONVEX HULL

# Compute convex hull
surface.compute_vertex_normals()
pcl = surface.sample_points_poisson_disk(number_of_points=2000)
hull, _ = pcl.compute_convex_hull()
hull.orient_triangles()
o3d.visualization.draw_geometries([hull])


# Calculate volume
volume = o3d.geometry.convex_hull_volume(hull)
print("Convex hull volume:", volume)
                                 








































# o3d.visualization.draw_geometries([removed_pcd])
# axes = o3d.geometry.TriangleMesh.create_coordinate_frame() #CARTESAN

# obb = desired_object.get_axis_aligned_bounding_box()
# obb.color = (0,1,0)
# obb= obb.translate((0,0,d/c))
# desired_object = desired_object.translate((0,0,d/c))

# plane_model_1, inliers_1 = desired_object.segment_plane(distance_threshold=0.01,
#                                          ransac_n=3,
#                                          num_iterations=10000)

# [a,b,c,d] = plane_model_1

# #METHOD STATISTICAL OUTLIER REMOVAL


# removed_pcd = desired_object.select_by_index(inliers_1)
# removed_pcd.paint_uniform_color((0,1,0))                 #This is floor === GREEN COLOR

# cl, inside = removed_pcd.remove_statistical_outlier(nb_neighbors= 100,std_ratio=2.0)

# removed_pcd = removed_pcd.select_by_index(inside)

# pcd, _ = removed_pcd.remove_radius_outlier(nb_points=50, radius=0.07)  #radii depends on sensor sensitivity #earlier it was 0.07
# pcd.paint_uniform_color((0, 0, 1)) #

# desired_object = desired_object.select_by_index(inliers_1, invert=True)
# desired_object.paint_uniform_color((1, 0, 0)) 
# o3d.visualization.draw_geometries([pcd])

# #finding z mean for truck deck

# deck_array = np.asarray(pcd.points)
# # print(deck_array[:,2])
# print(deck_array.shape)

# std_dev = np.std(deck_array)
# MEan = np.mean(deck_array)

# print(std_dev,MEan)


# height = z_pcd_array - MEan
# print("height",height)

# obb_des2 = pcd.get_oriented_bounding_box()
# obb_des2.color = (1,0,0)

# [a,b,c,d] = plane_model

# obb= obb_des2.translate((0,0,-d/c))

# o3d.visualization.draw_geometries([pcd_,obb,removed_pcd])


