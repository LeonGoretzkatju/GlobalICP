import open3d as o3d
import numpy as np
import copy
import time
import os
import sys

sys.path.append('..')

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp,target_temp])
def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def preprocess_global_map(pcd,voxel_size):
    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd, pcd_fpfh

def prepare_datasetglobal56(voxel_size):
    source = o3d.io.read_point_cloud("/home/xiangchenliu/SLAMDatasets/123with4good.pcd")
    target = o3d.io.read_point_cloud("/home/xiangchenliu/SLAMDatasets/5with6.pcd")
    source_copy = copy.deepcopy(source)
    target_copy = copy.deepcopy(target)
    # trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
    #                          [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    trans_init = np.asarray([[1.0, 0.0, 0.0, 50.0], [0.0, 1.0, 0.0, 1580.0],
                             [0.0, 0.0, 1.0, 40.0], [0.0, 0.0, 0.0, 1.0]])
    source_copy.transform(trans_init)
    # draw_registration_result(source_copy, target_copy, np.identity(4))
    o3d.visualization.draw_geometries([source_copy, target_copy])
    source_down, source_fpfh = preprocess_point_cloud(source_copy, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target_copy, voxel_size)
    return source, target, source_copy, target_copy, source_down, target_down, source_fpfh, target_fpfh

def execute_point2point_registration(source, target, threshold, trans_init, voxel_size):
    distance_threshold = voxel_size * 0.4
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5000))
    return reg_p2p

def execute_point2plane_registration(source, target, trans_init, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5000))
    return result

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

voxel_size = 10.0 # means 5cm for this dataset
threshold = 10.0

source_global, target, source_global_copy, target_copy, source_global_down, target_down, source_global_fpfh, target_fpfh = prepare_datasetglobal56(
    voxel_size)
result_p2p = execute_point2point_registration(source_global_down,target_down,threshold,np.identity(4),voxel_size)
result_p2plane = execute_point2plane_registration(source_global_down,target_down,result_p2p.transformation,voxel_size)
draw_registration_result(source_global_down,target_down,result_p2plane.transformation)
source_global_copy.transform(result_p2plane.transformation)
global_point_map = source_global_copy + target_copy
o3d.io.write_point_cloud("/home/xiangchenliu/SLAMDatasets/1234with56.pcd", global_point_map)
