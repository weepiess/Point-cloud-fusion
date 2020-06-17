#!/usr/bin/env python
# -*- coding:utf-8 -*-
import sys
import os
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2 as cv
import pcl
import os.path
import numpy as np
from sklearn.neighbors import NearestNeighbors
import math
import dlib
import point_cloud_utils as pcu
import time
import open3d as o3d

from utils.io import *

###################################################################################################
# for source point, get its closest point on ground truth
# input: s, shape = [N,3], source 3D vertices
#		 gt, shape = [M,3], ground truth 3D vertices
#	     nbrs, a kd tree structure for ground truth vertices
# output: gt_closest_P, shape = [N,3], closest point on ground truth for each vertex on s
#		  idx, shape = [N], index of closest point
# 		  dist, shape = [N], distance of each closest point pair
###################################################################################################
def get_closest_point(s,gt,nbrs):
	# land, vert, uv
	dist,idx = nbrs.kneighbors(s)
	idx = np.reshape(idx,[s.shape[0]])
	gt_closest_p = gt[idx,:]

	return gt_closest_p,idx,dist

###################################################################################################
# align source mesh to target mesh
# min ||target - (s*source*R + t)||^2
# input: source, shape = [N,3], source mesh to be aligned
# 		 target, shape = [N,3], target mesh
#		 scale, if True, consider scaling factor for alignment; if False, scale will be set to 1
# output: R, shape = [3,3], rotation matrix
#		  t, shape = [1,3], translation vector
# 		  s, scaling factor
###################################################################################################
def align_source_to_target(source,target,scale = False):

	tar = target.copy()
	sou = source.copy()
	center_tar = tar - np.mean(tar,0) # centralized target mesh
	center_sou = sou - np.mean(sou,0) # centralized source mesh
	# print('center_tar:',center_tar.size)
	# print('center_sou:',center_sou.size)
	W = np.matmul(center_tar.transpose(),center_sou)
	U,S,V = np.linalg.svd(W)
	R = np.matmul(np.matmul(V.transpose(),np.diag([1,1,np.linalg.det(np.matmul(V.transpose(),U.transpose()))])),U.transpose()) # calculate rotation matrix (exclude mirror symmetry)

	if scale:
		R_sou = np.matmul(center_sou,R)
		s = np.sum(R_sou*center_tar)/np.sum(R_sou*R_sou)
	else:
		s = 1

	t = np.mean(tar,0) - s*np.matmul(np.expand_dims(np.mean(sou,0),0),R)

	return R,t,s

###################################################################################################
# iterative closest point
# input: source, shape = [N,3], source vertex to be aligned
# 		 target, shape = [N,3], corresponding target vertex
#		 landmark_s, shape = [K,3], facial landmarks of source mesh
#		 landmark_t, shape = [K,3], facial landmakrs of target mesh
# 		 nbrs, a kd tree structure for target mesh
#		 max_iter, iterations for icp
# output: sou, shape = [N,3], aligned source mesh
# 		  RR, shape = [3,3], rotation matrix
# 		  tt, shape = [1,3], translation
#		  ss, scaling factor
###################################################################################################
def icp(source,target,landmark_s,landmark_t,nbrs,max_iter = 1000):

	sou = source.copy()
	tar = target
	i = 0

	# initialize rotation, translation and scale
	tt = np.zeros([1,3]).astype(np.float32)
	RR = np.eye(3)
	ss = 1

	while i < max_iter:
		# using landmarks for alignment in the first step
		if i == 0:
			s_match = landmark_s
			t_match = landmark_t
		# using closest point pairs for alignment from the second step
		else:
			s_match = sou
			t_match,_,_ = get_closest_point(sou,tar,nbrs)

		R,t,s = align_source_to_target(s_match,t_match,scale = True) # calculate scale and rigid transformation

		# accumulate rotation, translation and scaling factor
		RR = np.matmul(RR,R)
		tt = t + s*np.matmul(tt,R)
		ss = ss*s
		sou = s*np.matmul(sou,R) + t

		i += 1

	return sou,RR,tt,ss

###################################################################################################
# rotate img for 90 degree(clockwise)
# input: img =  [N,3], source image
#
# output: new_img = [N,3],rotated img
###################################################################################################
def RotateClockWise90(img):
    trans_img = cv.transpose( img )
    new_img = cv.flip(trans_img, 1)
    return new_img

###################################################################################################
# match texture uv with 2D key points to get 3D landmarks
# input: uv_2d =  [N,2], 2D key points
#        uv_3d =  [M,2], texture uv
#
# output: index = [N],3D landmarks
###################################################################################################
def match_uv(uv_2d,uv_3d):
    index = []
    for i in range(uv_2d.shape[0]):
        index_ = []

        min_val = 0x3f3f3f3f
        for j in range(uv_3d.shape[0]):
            dist = np.sqrt(np.sum(np.square(uv_2d[i]-uv_3d[j])))
            if dist<min_val:
                #print('dist: ',dist)
                min_val = dist
                index_.append(j)
        index.append(index_[-1])
    return index

###################################################################################################
# rotate img for 90 degree(clockwise)
# input: img =  [N,3], source image
#
# output: new_img = [N,3],rotated img
###################################################################################################
def getuv2d(height,width,image):
    image = RotateClockWise90(image)
    dlib_landmark_model = './Data/net-data/shape_predictor_68_face_landmarks.dat'
    face_regressor = dlib.shape_predictor(dlib_landmark_model)
    face_detector = dlib.get_frontal_face_detector()
    rects = face_detector(image, 1)
    for rect in rects:
        pts = face_regressor(image, rect).parts()
        pts = np.array([[pt.x, pt.y] for pt in pts]).T
        uv_ = pts.T.astype(float)
        x = pts.T.astype(float)

    temp = np.zeros([68,2],float)
    temp[:,0] = x[:,1]/float(width)
    temp[:,1] = x[:,0]/float(height)

    return temp




###################################################################################################
# choose several 3D landmarks for ICP
# input: landmark_3d =  [N,1], 3D landmark
#
# output: left_3d/right_3d = [M,1],choosed 3D landmark
###################################################################################################
def get_left_3d(landmark_3d):
    print('size: ',len(landmark_3d))
    #left_3d = [landmark_3d[17]]+[landmark_3d[21]]+[landmark_3d[36]]+[landmark_3d[39]]+[landmark_3d[30]]+[landmark_3d[31]]+[landmark_3d[48]]+[landmark_3d[50]]+[landmark_3d[58]]
    temp1 = landmark_3d[17:22]
    temp2 = landmark_3d[36:42]
    temp3 = landmark_3d[27:31]
    temp4 = landmark_3d[48:60:2]
    temp5 = landmark_3d[31:36]
    left_3d = temp1+temp2+temp3+temp4+temp5
    print('left_3d',left_3d)
    print('size: ',len(left_3d))
    return left_3d

def get_right_3d(landmark_3d):
    print('size: ',len(landmark_3d))
    #right_3d = [landmark_3d[17]]+[landmark_3d[21]]+[landmark_3d[36]]+[landmark_3d[39]]+[landmark_3d[30]]+[landmark_3d[31]]+[landmark_3d[48]]+[landmark_3d[50]]+[landmark_3d[58]]
    temp1 = landmark_3d[22:27]
    temp2 = landmark_3d[42:48]
    temp3 = landmark_3d[27:31]
    temp4 = landmark_3d[48:60:2]
    temp5 = landmark_3d[31:36]
    right_3d = temp1+temp2+temp3+temp4+temp5
    print('right_3d',right_3d)
    print('size: ',len(right_3d))
    return right_3d

###################################################################################################
# remove Possion external points
# output: vertices_to_remove = [N,1],Points id need to be removed
###################################################################################################
def remove(mesh,vtx_middle,reconstruct_shape_l,reconstruct_shape_r):
    vertices = np.asarray(mesh.vertices)
    v_f = np.array(vertices,dtype=np.float32)
    dists_middle, corrs_middle = pcu.point_cloud_distance(v_f, vtx_middle)
    dists_left, corrs_left = pcu.point_cloud_distance(v_f, reconstruct_shape_l)
    dists_right, corrs_right = pcu.point_cloud_distance(v_f, reconstruct_shape_r)
    dist_full = np.vstack((dists_middle,dists_left,dists_right))
    dists = np.min(dist_full,0)

    vertices_to_remove = dists >= 25

    return vertices_to_remove

if __name__ == "__main__":
    time_start=time.time()

    left_wrl = meshwrl()
    middle_wrl = meshwrl()
    right_wrl = meshwrl()

    #分别读入三个点云
    # get_uv2d：获取2d关键点
    # match_uv: 利用2d关键点与uv映射关系获得3D关键点

    #left
    left_wrl.load_wrl('./data/test1/0/t')
    left_wrl.uv2d = getuv2d(left_wrl.h,left_wrl.w,left_wrl.image)
    left_landmark3d = match_uv(left_wrl.uv2d,left_wrl.uv3d)
    l3d = get_left_3d(left_landmark3d)
    #print('3d: ',left_landmark3d)

    #middle
    middle_wrl.load_wrl('./data/test1/1/t')
    middle_wrl.uv2d = getuv2d(middle_wrl.h,middle_wrl.w,middle_wrl.image)
    middle_landmark3d = match_uv(middle_wrl.uv2d,middle_wrl.uv3d)
    m_l3d = get_left_3d(middle_landmark3d)
    m_r3d = get_right_3d(middle_landmark3d)

    #right
    right_wrl.load_wrl('./data/test1/2/t')
    right_wrl.uv2d = getuv2d(right_wrl.h,right_wrl.w,right_wrl.image)
    right_landmark3d = match_uv(right_wrl.uv2d,right_wrl.uv3d)
    r3d = get_right_3d(right_landmark3d)

    middle_color = []
    for i in range(len(middle_wrl.uv3d)):
        pos_x = middle_wrl.w*middle_wrl.uv3d[i,0]
        pos_y = middle_wrl.h - middle_wrl.h*middle_wrl.uv3d[i,1]
        value = middle_wrl.image[int(pos_y),int(pos_x)]
        middle_color.append(value)
    # 显示正脸关键点
    # for i in range(len(m_l3d)):
    #     middle_color[m_l3d[i]] = [0,0,0]
    middle_rgb = np.array(middle_color)

    left_color = []
    for i in range(len(left_wrl.uv3d)):
        pos_x = left_wrl.w*left_wrl.uv3d[i,0]
        pos_y = left_wrl.h - left_wrl.h*left_wrl.uv3d[i,1]
        value = left_wrl.image[int(pos_y),int(pos_x)]
        left_color.append(value)
    # 显示左脸关键点
    # for i in range(len(l3d)):
    #     left_color[l3d[i]] = [0,0,0]
    left_rgb = np.array(left_color)


    right_color = []
    for i in range(len(right_wrl.uv3d)):
        pos_x = right_wrl.w*right_wrl.uv3d[i,0]
        pos_y = right_wrl.h - right_wrl.h*right_wrl.uv3d[i,1]
        value = right_wrl.image[int(pos_y),int(pos_x)]
        right_color.append(value)
    right_rgb = np.array(right_color)
    print('end')


    # ICP
    print('processing ICP now')
    landmark_s_left = left_wrl.vertices[l3d]
    landmark_t_left = middle_wrl.vertices[m_l3d]
    nbrsT_4crop_left = NearestNeighbors(n_neighbors = 1, algorithm = 'kd_tree').fit(middle_wrl.vertices)
    reconstruct_shape_l,r_l ,t_l ,s_l = icp(left_wrl.vertices,middle_wrl.vertices,landmark_s_left,landmark_t_left,nbrsT_4crop_left,max_iter = 10)

    landmark_s_right = right_wrl.vertices[r3d]
    landmark_t_right = middle_wrl.vertices[m_r3d]
    nbrsT_4crop_right = NearestNeighbors(n_neighbors = 1, algorithm = 'kd_tree').fit(middle_wrl.vertices)
    reconstruct_shape_r,r_r ,t_r ,s_r = icp(right_wrl.vertices,middle_wrl.vertices,landmark_s_right,landmark_t_right,nbrsT_4crop_right,max_iter = 10)
    print('ICP complete')


    # 左中右点云合并
    vtx_middle = np.array(middle_wrl.vertices,dtype=np.float32)
    reconstruct_shape_r = np.array(reconstruct_shape_r,dtype=np.float32)
    reconstruct_shape_l = np.array(reconstruct_shape_l,dtype=np.float32)

    vtx_fusion = np.vstack((reconstruct_shape_l,reconstruct_shape_r,vtx_middle))
    print('vstack complete')

    # voxel filter
    #sample_vtx = voxel_filter(vtx_fusion,1.75)
    
    # use pcl function
    p_full = pcl.PointCloud(vtx_fusion)
    voxelFilter = p_full.make_voxel_grid_filter()
    voxelFilter.set_leaf_size(1.75,1.75,1.75)
    voxel_f = voxelFilter.filter()

    print('voxel filter complete')

    # 移除离群点 （效果有待考量）
    p = voxel_f
    fil = p.make_statistical_outlier_filter()
    fil.set_mean_k (50)
    fil.set_std_dev_mul_thresh (1.0)
    smooth_vtx = fil.filter()

    # MLS 平滑
    tree = smooth_vtx.make_kdtree()
    mls = smooth_vtx.make_moving_least_squares()
    mls.set_Compute_Normals(True)
    mls.set_polynomial_fit(True)
    mls.set_Search_Method(tree)
    mls.set_search_radius(10.0)
    mls_point = mls.process()
    s_vtx_n = np.asarray(mls_point)

    print('filter complete')

    # 计算法向量
    n = pcu.estimate_normals(s_vtx_n, k=16)



    # 三角面片重建
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(s_vtx_n)
    pcd.normals = o3d.utility.Vector3dVector(n)
    
    # depth越大细节越多
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=7)

    # 移除Possion重建后额外的平面点
    vertices_to_remove = remove(mesh,vtx_middle,reconstruct_shape_l,reconstruct_shape_r)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # 找最大联通区，去除浮空噪声
    triangle_clusters, cluster_n_triangles, cluster_area = (
        mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)

    largest_cluster_idx = cluster_n_triangles.argmax()
    triangles_to_remove = triangle_clusters != largest_cluster_idx
    mesh.remove_triangles_by_mask(triangles_to_remove)

    # 表面平滑
    mesh_out = mesh.filter_smooth_simple(number_of_iterations=10)

    vertices = np.asarray(mesh_out.vertices)
    faces = np.asarray(mesh_out.triangles)
    v_f = np.array(vertices,dtype=np.float32)

    dists_middle, corrs_middle = pcu.point_cloud_distance(v_f, vtx_middle)
    dists_left, corrs_left = pcu.point_cloud_distance(v_f, reconstruct_shape_l)
    dists_right, corrs_right = pcu.point_cloud_distance(v_f, reconstruct_shape_r)
    
    print('surface reconstruction complete')

    # 纹理融合
    vertex_with_rgb = []
    min_vtx = min(vertices[:,1])
    max_vtx = max(vertices[:,1])
    #计算点云宽度
    width = (max_vtx-min_vtx)*0.5
    
    min_y_vtx =min(vertices[:,0])
    max_y_vtx =max(vertices[:,0])
    #计算点云高度
    height = (max_y_vtx-min_y_vtx)*0.5

    # 根据点云所处的位置进行划分
    # 左侧点云区域使用左纹理/ 中间使用中间纹理/ 右边使用右边纹理
    # 交界处 进行了3次权重划分（可进一步细化）
    for i in range(len(dists_middle)):
        if dists_middle[i]<=25 and abs(vertices[i,1]/width) <=0.5:
            vertex_with_rgb.append([vertices[i,0],vertices[i,1],vertices[i,2],
                middle_rgb[corrs_middle[i],0],middle_rgb[corrs_middle[i],1],middle_rgb[corrs_middle[i],2]])

        elif dists_left[i]<=25 and vertices[i,1]/width <-0.5:
            if abs(vertices[i,1]/width) <=0.56 and dists_middle[i]<=25:
                r = (int(middle_rgb[corrs_middle[i],0])*0.7+int(left_rgb[corrs_left[i],0])*0.3)
                g = (int(middle_rgb[corrs_middle[i],1])*0.7+int(left_rgb[corrs_left[i],1])*0.3)
                b = (int(middle_rgb[corrs_middle[i],2])*0.7+int(left_rgb[corrs_left[i],2])*0.3)
                vertex_with_rgb.append([vertices[i,0],vertices[i,1],vertices[i,2],
                    r,g,b])
            elif abs(vertices[i,1]/width) <=0.57 and dists_middle[i]<=25:
                r = (int(middle_rgb[corrs_middle[i],0])*0.6+int(left_rgb[corrs_left[i],0])*0.4)
                g = (int(middle_rgb[corrs_middle[i],1])*0.6+int(left_rgb[corrs_left[i],1])*0.4)
                b = (int(middle_rgb[corrs_middle[i],2])*0.6+int(left_rgb[corrs_left[i],2])*0.4)
                vertex_with_rgb.append([vertices[i,0],vertices[i,1],vertices[i,2],
                    r,g,b])
            elif abs(vertices[i,1]/width) <=0.58 and dists_middle[i]<=25:
                r = (int(middle_rgb[corrs_middle[i],0])+int(left_rgb[corrs_left[i],0]))*0.5
                g = (int(middle_rgb[corrs_middle[i],1])+int(left_rgb[corrs_left[i],1]))*0.5
                b = (int(middle_rgb[corrs_middle[i],2])+int(left_rgb[corrs_left[i],2]))*0.5
                vertex_with_rgb.append([vertices[i,0],vertices[i,1],vertices[i,2],
                    r,g,b])
            else:
                vertex_with_rgb.append([vertices[i,0],vertices[i,1],vertices[i,2],
                    left_rgb[corrs_left[i],0],left_rgb[corrs_left[i],1],left_rgb[corrs_left[i],2]])
        elif dists_right[i]<=25 and vertices[i,1]/width >0.5:
            if abs(vertices[i,1]/width) <=0.56 and dists_middle[i]<=25:
                r = (int(middle_rgb[corrs_middle[i],0])*0.7+int(right_rgb[corrs_right[i],0])*0.3)
                g = (int(middle_rgb[corrs_middle[i],1])*0.7+int(right_rgb[corrs_right[i],1])*0.3)
                b = (int(middle_rgb[corrs_middle[i],2])*0.7+int(right_rgb[corrs_right[i],2])*0.3)
                vertex_with_rgb.append([vertices[i,0],vertices[i,1],vertices[i,2],
                    r,g,b])
            elif abs(vertices[i,1]/width) <=0.57 and dists_middle[i]<=25:
                r = (int(middle_rgb[corrs_middle[i],0])*0.6+int(right_rgb[corrs_right[i],0])*0.4)
                g = (int(middle_rgb[corrs_middle[i],1])*0.6+int(right_rgb[corrs_right[i],1])*0.4)
                b = (int(middle_rgb[corrs_middle[i],2])*0.6+int(right_rgb[corrs_right[i],2])*0.4)
                vertex_with_rgb.append([vertices[i,0],vertices[i,1],vertices[i,2],
                    r,g,b])
            elif abs(vertices[i,1]/width) <=0.58 and dists_middle[i]<=25:
                r = (int(middle_rgb[corrs_middle[i],0])+int(right_rgb[corrs_right[i],0]))*0.5
                g = (int(middle_rgb[corrs_middle[i],1])+int(right_rgb[corrs_right[i],1]))*0.5
                b = (int(middle_rgb[corrs_middle[i],2])+int(right_rgb[corrs_right[i],2]))*0.5
                vertex_with_rgb.append([vertices[i,0],vertices[i,1],vertices[i,2],
                    r,g,b])
            else :
                vertex_with_rgb.append([vertices[i,0],vertices[i,1],vertices[i,2],
                    right_rgb[corrs_right[i],0],right_rgb[corrs_right[i],1],right_rgb[corrs_right[i],2]])
        else:
            vertex_with_rgb.append([vertices[i,0],vertices[i,1],vertices[i,2],
                255,255,255])
    
    
    cloud = pcl.PointCloud()
    v_rgb = vertex_with_rgb.copy()
    vtx_temp = np.array(v_rgb,dtype=np.float32)
    vt = vtx_temp[:,0:3]
    cloud.from_array(vt)

    # 搜索每个点的20个最近点
    kdtree = cloud.make_kdtree_flann()
    # 用于左中右交界处融合
    [ind, sqdist] = kdtree.nearest_k_search_for_cloud(cloud, 20)
    # 用于白色区域填补
    [ind2, sqdist2] = kdtree.nearest_k_search_for_cloud(cloud, 20)

    void_list = []
    void_key = []
    # 记录下所有白色区域，及其20个最近点的20个最近点，进行填补
    for i in range(len(vertex_with_rgb)):
        if vertex_with_rgb[i][3]>=255 and vertex_with_rgb[i][4]>=255 and vertex_with_rgb[i][5]>=255:
            void_key.append(i)
            for  j in range(len(ind2[i])):
                void_list.append(vertex_with_rgb[ind2[i][j]][3:])
                # for k in range(len())
                void_key.append(ind2[i][j])
                for k in range(len(ind2[ind2[i][j]])):
                    void_list.append(vertex_with_rgb[ind2[ind2[i][j]][k]][3:])

    # 开始进行白色区域填补（耗时较多）
    print('white hole filling')
    step=0
    end_step = 10
    while True:
        print('step: [%d/%d]' %(step,end_step))
        #count_color = 0
        step += 1
        for i in range(len(void_key)):
            color_sum=[]
            for j in range(len(ind2[void_key[i]])):
                if vertex_with_rgb[ind2[void_key[i]][j]][3] != 255 and vertex_with_rgb[ind2[void_key[i]][j]][4] != 255 and vertex_with_rgb[ind2[void_key[i]][j]][5] != 255:
                    color_sum.append(vertex_with_rgb[ind2[void_key[i]][j]][3:])
            if len(color_sum)>=1:
                np_color = np.array(color_sum)
                color_m = np.mean(color_sum,0)
                vertex_with_rgb[void_key[i]][3:] = color_m
                void_list[i] = color_m
        # 迭代轮数
        if step>=end_step:
            break

    #进行两次左中右纹理交界融合
    print('texture fusion processing')
    for i in range(len(vertex_with_rgb)):
        if abs(vertices[i,1]/width) <= 0.65 and abs(vertices[i,1]/width)>= 0.49 and abs(vertices[i,0]/height) <= 0.4:
            color_sum = []
            for  j in range(len(ind[i])):
                color_sum.append(vertex_with_rgb[ind[i][j]][3:])
            if len(color_sum)>=1:
                np_color = np.array(color_sum)
                color_m = np.mean(color_sum,0)
                vertex_with_rgb[i][3:] = color_m

    for i in range(len(vertex_with_rgb)):
        if abs(vertices[i,1]/width) <= 0.7 and abs(vertices[i,1]/width)>= 0.49 and abs(vertices[i,0]/height) <= 0.4:
            color_sum = []
            for  j in range(len(ind[i])):
                if vertex_with_rgb[ind[i][j]][3] != 255 and vertex_with_rgb[ind[i][j]][4] != 255 and vertex_with_rgb[ind[i][j]][5] != 255:
                    color_sum.append(vertex_with_rgb[ind[i][j]][3:])
            if len(color_sum)>=1:
                np_color = np.array(color_sum)
                color_m = np.mean(color_sum,0)
                vertex_with_rgb[i][3:] = color_m


    vertex_with_color = np.array(vertex_with_rgb)

    time_end=time.time()
    print('time cost',time_end-time_start,'s')

    # 写入obj
    write_obj_with_ccolors('./output/left.obj',reconstruct_shape_l,left_wrl.faces,left_rgb)
    write_obj_with_ccolors('./output/middle.obj',middle_wrl.vertices,middle_wrl.faces,middle_rgb)
    write_obj_with_ccolors('./output/right.obj',reconstruct_shape_r,right_wrl.faces,right_rgb)
    # 最后结果
    write_obj_with_colors('./output/fusion.obj',vertex_with_color,faces)
