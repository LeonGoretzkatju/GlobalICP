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

def prepare_dataset23(voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    source = o3d.io.read_point_cloud("/home/xiangchenliu/SLAMDatasets/steel2.pcd")
    target = o3d.io.read_point_cloud("/home/xiangchenliu/SLAMDatasets/steel3.pcd")

    source_copy = copy.deepcopy(source)
    target_copy = copy.deepcopy(target)
    # trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
    #                          [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0],
                             [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source_copy.transform(trans_init)
    # draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source_copy, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target_copy, voxel_size)
    return source, target, source_copy, target_copy, source_down, target_down, source_fpfh, target_fpfh

def prepare_dataset12(voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    source = o3d.io.read_point_cloud("/home/xiangchenliu/SLAMDatasets/steel1.pcd")
    target = o3d.io.read_point_cloud("/home/xiangchenliu/SLAMDatasets/steel2.pcd")
    # trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
    #                          [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source_copy = copy.deepcopy(source)
    target_copy = copy.deepcopy(target)

    trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1000.0],
                             [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source_copy.transform(trans_init)
    # draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source_copy, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target_copy, voxel_size)
    return source, target, source_copy, target_copy, source_down, target_down, source_fpfh, target_fpfh

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

def execute_point2point_registration(source, target, threshold, trans_init, voxel_size):
    distance_threshold = voxel_size * 0.4
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return reg_p2p

def refine_registration12(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac12.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

def refine_registration23(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac23.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

voxel_size = 10.0 # means 5cm for this dataset
threshold = 10.0

source1, target2, source1_copy, target2_copy, source_down1, target_down2, source_fpfh1, target_fpfh2 = prepare_dataset12(
    voxel_size)

result_ransac12 = execute_global_registration(source_down1, target_down2,
                                            source_fpfh1, target_fpfh2,
                                            voxel_size)

result_icp12 = refine_registration12(source_down1, target_down2, source_fpfh1, target_fpfh2,
                                voxel_size)

source_down1.transform(result_icp12.transformation)
source1_copy.transform(result_icp12.transformation)
# draw_registration_result(source_down1, target_down2, result_icp12.transformation)
source2, target3, source2_copy, target3_copy, source_down2, target_down3, source_fpfh2, target_fpfh3 = prepare_dataset23(
    voxel_size)

result_ransac23 = execute_global_registration(source_down2, target_down3,
                                            source_fpfh2, target_fpfh3,
                                            voxel_size)
result_icp23 = refine_registration23(source_down2, target_down3, source_fpfh2, target_fpfh3,
                                voxel_size)
source_down1.transform(result_icp23.transformation)
source1_copy.transform(result_icp23.transformation)
source_down2.transform(result_icp23.transformation)
source2_copy.transform(result_icp23.transformation)
global_point_map = source1_copy+source2_copy+target3_copy
o3d.io.write_point_cloud("/home/xiangchenliu/SLAMDatasets/1with2with3.pcd", global_point_map)
o3d.visualization.draw_geometries([global_point_map])