import open3d as o3d
import numpy as np
import copy
import time
import os
import sys

# monkey patches visualization and provides helpers to load geometries
sys.path.append('..')


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    # o3d.visualization.draw_geometries([source_temp, target_temp],
    #                                   zoom=0.4559,
    #                                   front=[0.6452, -0.3036, -0.7011],
    #                                   lookat=[1.9892, 2.0208, 1.8945],
    #                                   up=[-0.2779, -0.9482, 0.1556])
    o3d.visualization.draw_geometries([source_temp, target_temp])


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


def preprocess_global_map(pcd, voxel_size):
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


def prepare_dataset56(voxel_size, global_map_update_update):
    print(":: Load two point clouds and disturb initial pose.")
    # source = o3d.io.read_point_cloud("/home/xiangchenliu/SLAMDatasets/steel3.pcd")
    source = global_map_update_update
    target = o3d.io.read_point_cloud("/home/xiangchenliu/SLAMDatasets/steel6.pcd")
    # trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
    #                          [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    trans_init = np.asarray([[1.0, 0.0, 0.0, -100.0], [0.0, 1.0, 0.0, 700.0],
                             [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    draw_registration_result(source, target, np.identity(4))

    # source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    source_down, source_fpfh = preprocess_global_map(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def prepare_dataset45(voxel_size, global_map_update):
    print(":: Load two point clouds and disturb initial pose.")
    # source = o3d.io.read_point_cloud("/home/xiangchenliu/SLAMDatasets/steel3.pcd")
    source = global_map_update
    target = o3d.io.read_point_cloud("/home/xiangchenliu/SLAMDatasets/steel5.pcd")
    # trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
    #                          [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    trans_init = np.asarray([[1.0, 0.0, 0.0, 200.0], [0.0, 1.0, 0.0, 700.0],
                             [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    draw_registration_result(source, target, np.identity(4))

    # source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    source_down, source_fpfh = preprocess_global_map(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def prepare_dataset34(voxel_size, global_map):
    print(":: Load two point clouds and disturb initial pose.")
    # source = o3d.io.read_point_cloud("/home/xiangchenliu/SLAMDatasets/steel3.pcd")
    source = global_map
    target = o3d.io.read_point_cloud("/home/xiangchenliu/SLAMDatasets/steel4.pcd")
    # trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
    #                          [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 300.0],
                             [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    # draw_registration_result(source, target, np.identity(4))

    # source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    source_down, source_fpfh = preprocess_global_map(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def prepare_dataset23(voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    source = o3d.io.read_point_cloud("/home/xiangchenliu/SLAMDatasets/plane3.pcd")
    # target = o3d.io.read_point_cloud("/home/xiangchenliu/SLAMDatasets/plane2.pcd")
    target = global_point_cloud
    # trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
    #                          [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 2300.0],
                             [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    # draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_global_map(target, voxel_size)
    draw_registration_result(source_down, target_down, np.identity(4))
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def prepare_dataset12(voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    source = o3d.io.read_point_cloud("/home/xiangchenliu/SLAMDatasets/plane2.pcd")  # source is colored in yellow
    target = o3d.io.read_point_cloud("/home/xiangchenliu/SLAMDatasets/plane1.pcd")  # target is colored in blue
    # trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
    #                          [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1200.0],
                             [0.0, 0.0, 1.0, 100.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    # draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    draw_registration_result(source_down, target_down, np.identity(4))
    return source, target, source_down, target_down, source_fpfh, target_fpfh


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


def refine_registration12(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    # result = o3d.pipelines.registration.registration_icp(
    #     source, target, distance_threshold, result_ransac12.transformation,
    #     o3d.pipelines.registration.TransformationEstimationPointToPlane())
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, np.identity(4),
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


def refine_registration34(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    trans_init_refine = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 300.0],
                                    [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, trans_init_refine,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result


def refine_registration45(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    # trans_init_refine = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 300.0],
    #                                 [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    # refine_trans_init = np.asarray([[1.0, 0.0, 0.0, 200.0], [0.0, 1.0, 0.0, 200.0],
    #                          [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result


def refine_registration56(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    # trans_init_refine = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 300.0],
    #                                 [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    # refine_trans_init = np.asarray([[1.0, 0.0, 0.0, 200.0], [0.0, 1.0, 0.0, 200.0],
    #                          [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    # trans_init_56 = np.asarray([[1.0, 0.0, 0.0, -100.0], [0.0, 1.0, 0.0, 700.0],
    #                          [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result


voxel_size = 10.0  # means 5cm for this dataset

source1, target2, source_down1, target_down2, source_fpfh1, target_fpfh2 = prepare_dataset12(
    voxel_size)

# trans_rough = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1000.0],
#                              [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
# draw_registration_result(source_down1, target_down2, trans_rough)


# result_ransac12 = execute_global_registration(source_down1, target_down2,
#                                             source_fpfh1, target_fpfh2,
#                                             voxel_size)

result_icp12 = refine_registration12(source_down1, target_down2, source_fpfh1, target_fpfh2,
                                     voxel_size)
print(result_icp12.transformation)

source_down1.transform(result_icp12.transformation)
# draw_registration_result(source_down1, target_down2, np.identity(4))
global_point_cloud = source_down1 + target_down2
global_point_cloud.paint_uniform_color([1, 0.706, 0])
o3d.visualization.draw_geometries([global_point_cloud])

source2, target3, source_down2, target_down3, source_fpfh2, target_fpfh3 = prepare_dataset23(
    voxel_size)

result_ransac23 = execute_global_registration(source_down2, target_down3,
                                            source_fpfh2, target_fpfh3,
                                            voxel_size)
#
result_icp23 = refine_registration23(source_down2, target_down3, source_fpfh2, target_fpfh3,
                                voxel_size)
# draw_registration_result(source_down2, target_down3, result_icp23.transformation)
source_down2.transform(result_icp23.transformation)
global_point_cloud = source_down2 + target_down3
global_point_cloud.paint_uniform_color([1, 0.706, 0])
o3d.visualization.draw_geometries([global_point_cloud])
# source_down1.transform(result_icp23.transformation)
# source_down2.transform(result_icp23.transformation)
# source_down2.transform(result_icp12.transformation)
# target_down3.transform(result_icp12.transformation)
# source_down2.paint_uniform_color([0, 0.651, 0.929])
# target_down3.paint_uniform_color([1, 0.651, 0.929])
# target_down2.paint_uniform_color([1, 0.706, 0])
# o3d.visualization.draw_geometries([source_down2,target_down3,target_down2])
#
# source_temp_down1 = copy.deepcopy(source_down1)
# target2_temp_down2 = copy.deepcopy(source_down2)
# target3_temp_down2 = copy.deepcopy(target_down3)
# source_temp_down1.paint_uniform_color([1, 0.706, 0])
# target2_temp_down2.paint_uniform_color([0, 0.651, 0.929])
# target3_temp_down2.paint_uniform_color([1, 0.651, 0.929])
# o3d.visualization.draw_geometries([source_temp_down1, target2_temp_down2,
#                                     target3_temp_down2])

# trans_rough = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1000.0],
#                              [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
# draw_registration_result(source_down2, target_down3, trans_rough)

# source3, target4, source_down3, target_down4, source_fpfh3, target_fpfh4 = prepare_dataset34(
#     voxel_size)

# result_ransac34 = execute_global_registration(source_down3, target_down4,
#                                             source_fpfh3, target_fpfh4,
#                                             voxel_size)
#
# result_icp34 = refine_registration34(source_down3, target_down4, source_fpfh3, target_fpfh4,
#                                 voxel_size)

# draw_registration_result(source_down3, target_down4, result_icp34.transformation)
#
# source_down1.transform(final_transformation)
# source_down2.transform(final_transformation)
# source_down3.transform(final_transformation)

# source_temp_down1 = copy.deepcopy(source_down1)
# target2_temp_down2 = copy.deepcopy(source_down2)
# target3_temp_down2 = copy.deepcopy(source_down3)
# target3_temp_down2 = copy.deepcopy(target_down3)
# final_temp = copy.deepcopy(target_down4)
# source_temp_down1.paint_uniform_color([1, 0.706, 0])
# target2_temp_down2.paint_uniform_color([0, 0.651, 0.929])
# target3_temp_down2.paint_uniform_color([1, 0.651, 0.929])
# final_temp.paint_uniform_color([0, 0.929, 0.651])
# o3d.visualization.draw_geometries([source_temp_down1, target2_temp_down2,
#                                    target3_temp_down2],
#                                     zoom=0.4559,
#                                     front=[0.6452, -0.3036, -0.7011],
#                                     lookat=[1.9892, 2.0208, 1.8945],
#                                     up=[-0.2779, -0.9482, 0.1556])

# o3d.visualization.draw_geometries([source_temp_down1, target2_temp_down2,
#                                     target3_temp_down2])
# global_map = source_down1+source_down2+target_down3
# o3d.visualization.draw_geometries([global_map])
# source3, target4, source_down3, target_down4, source_fpfh3, target_fpfh4 = prepare_dataset34(
#     voxel_size,global_map)

# result_ransac34 = execute_global_registration(source_down3, target_down4,
#                                             source_fpfh3, target_fpfh4,
#                                             voxel_size)

# result_icp34 = refine_registration34(source_down3, target_down4, source_fpfh3, target_fpfh4,
#                                 voxel_size)
#
# print(result_icp34.transformation)
# print(result_icp34.transformation[1, 3])
# refined_icp34transformation = copy.deepcopy(result_icp34.transformation)
# refined_icp34transformation[1,3] += 354

# draw_registration_result(source_down3, target_down4, result_icp34.transformation)
# draw_registration_result(source_down3, target_down4, refined_icp34transformation)
# source_down3.transform(refined_icp34transformation)
# global_map_update = source_down3 + target_down4
# # o3d.visualization.draw_geometries([global_map_update])
# source4, target5, source_down4, target_down5, source_fpfh4, target_fpfh5 = prepare_dataset45(
#     voxel_size, global_map_update)

# result_ransac45 = execute_global_registration(source_down4, target_down5,
#                                             source_fpfh4, target_fpfh5,
#                                             voxel_size)
# result_icp45 = refine_registration45(source_down4,target_down5,source_fpfh4,target_fpfh5,voxel_size)
# print(result_icp45.transformation)
# print(result_icp45.transformation[1, 3])
# draw_registration_result(source_down4,target_down5,result_ransac45.transformation)
# refined_icp45transformation = copy.deepcopy(result_icp45.transformation)
# refined_icp45transformation[1,3] += 200
# draw_registration_result(source_down4,target_down5,refined_icp45transformation)
# source_down4.transform(refined_icp45transformation)
# global_map_update_update = source_down4+target_down5
# o3d.visualization.draw_geometries([global_map_update_update])
# source5, target6, source_down5, target_down6, source_fpfh5, target_fpfh6 = prepare_dataset56(
#     voxel_size, global_map_update_update)
# result_ransac56 = execute_global_registration(source_down5,target_down6,source_fpfh5,target_fpfh6,voxel_size)
# draw_registration_result(source_down5,target_down6,result_ransac56.transformation)
# result_icp56 = refine_registration56(source_down5,target_down6,source_fpfh5,target_fpfh6,voxel_size)
# draw_registration_result(source_down5,target_down6,result_icp56.transformation)
