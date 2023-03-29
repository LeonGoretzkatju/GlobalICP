import open3d as o3d
import numpy as np
import copy
import sys
import os
import time

def preprocess_point_cloud(pcd, voxel_size):
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    # pcd_down = pcd.voxel_down_sample(voxel_size)
    #
    # radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    # pcd_down.estimate_normals(
    #     o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd, pcd_fpfh

def load_point_clouds(voxel_size):
    pcds = []
    for i in range(4):
        pcd = o3d.io.read_point_cloud("/home/xiangchenliu/SLAMDatasets/steel%d.pcd" %
                                      (i+1))
        print(":: Downsample with a voxel size %.3f." % voxel_size)
        pcd_down = pcd.voxel_down_sample(voxel_size)

        radius_normal = voxel_size * 2
        print(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        if i == 0 and i == 1 and i ==2:
            trans_init = np.identity(4)
        else:
            trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 450.0],
                                     [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        pcd_down.transform(trans_init)
        pcds.append(pcd_down)
    return pcds

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

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

def execute_point2point_registration(source, target, trans_init, voxel_size):
    distance_threshold = voxel_size * 0.4
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return reg_p2p

def execute_point2plane_registration(source, target, trans_init, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    # result = o3d.pipelines.registration.registration_icp(
    #     source, target, distance_threshold, trans_init,
    #     o3d.pipelines.registration.TransformationEstimationPointToPlane())
    loss = o3d.pipelines.registration.TukeyLoss(k=10.0)
    print("Using robust loss:", loss)
    p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
    result = o3d.pipelines.registration.registration_icp(source, target,
                                                         distance_threshold, trans_init,
                                                         p2l)
    return result

def pairwise_registration(source, target, voxel_size, source_id, target_id):
    print("Apply point-to-plane ICP")
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source_down, target_down, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source_down, target_down, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    # icp_coarse = execute_point2point_registration(source_down, target_down, trans_init, voxel_size)
    # icp_fine = execute_point2plane_registration(source_down, target_down, icp_coarse.transformation, voxel_size)
    source_down.transform(icp_fine.transformation)
    result_ransac = execute_global_registration(source_down,target_down,source_fpfh,target_fpfh,voxel_size)
    result_icp = refine_registration(source_down,target_down,source_fpfh,target_fpfh,voxel_size,result_ransac)
    transformation_icp = result_icp.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source_down, target_down, max_correspondence_distance_fine,
        result_icp.transformation)
    return transformation_icp, information_icp

def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine, voxel_size):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id], voxel_size, source_id, target_id)
            print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
    return pose_graph

voxel_size = 10.0
pcds_down = load_point_clouds(voxel_size)
o3d.visualization.draw_geometries(pcds_down)
print("Full registration ...")
max_correspondence_distance_coarse = voxel_size * 1.5
max_correspondence_distance_fine = voxel_size * 0.4
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Error) as cm:
    pose_graph = full_registration(pcds_down,
                                   max_correspondence_distance_coarse,
                                   max_correspondence_distance_fine,voxel_size)
print("Optimizing PoseGraph ...")
option = o3d.pipelines.registration.GlobalOptimizationOption(
    max_correspondence_distance=max_correspondence_distance_fine,
    edge_prune_threshold=0.25,
    reference_node=0)
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Error) as cm:
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option)
print("Transform points and display")
pcd_combined = o3d.geometry.PointCloud()
for point_id in range(len(pcds_down)):
    print(pose_graph.nodes[point_id].pose)
    pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)
    pcd_combined += pcds_down[point_id]
o3d.visualization.draw_geometries(pcds_down)
o3d.io.write_point_cloud("multiway_registration.pcd", pcd_combined)
o3d.visualization.draw_geometries([pcd_combined])