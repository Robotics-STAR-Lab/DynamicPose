import numpy as np
import open3d as o3d
import cv2
from scipy.spatial.transform import Rotation as R
from tf.transformations import euler_from_matrix, euler_from_quaternion, quaternion_from_matrix, quaternion_from_euler, quaternion_inverse, quaternion_multiply, quaternion_matrix
import torch
import trimesh
import itertools
from collections import defaultdict, Counter
from sklearn.cluster import DBSCAN


def get_rotMat_from_axis(samAxis, fpAxis):
        """
        Get the rotation matrix from 2 axis .
        """
        # 1. 确保输入向量归一化
        fpAxis = fpAxis / np.linalg.norm(fpAxis)
        samAxis = samAxis / np.linalg.norm(samAxis)
        # 2. 计算旋转轴（叉积）和旋转角度（点积）
        rotation_axis = np.cross(fpAxis, samAxis)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        # 如果叉积接近 0，说明两个向量平行或反平行
        if rotation_axis_norm < 1e-6:
            # 两个向量平行（无旋转）或反平行（180°旋转）
            if np.dot(fpAxis, samAxis) > 0:
                return np.eye(3)  # 单位矩阵，无旋转
            else:
                # 反平行，返回一个 180° 旋转矩阵
                # 找一个与 fpAxis 正交的任意轴作为旋转轴
                arbitrary_axis = np.array([1, 0, 0]) if abs(fpAxis[0]) < 0.9 else np.array([0, 1, 0])
                rotation_axis = np.cross(fpAxis, arbitrary_axis)
                rotation_axis /= np.linalg.norm(rotation_axis)
                return rodrigues_rotation_matrix(rotation_axis, np.pi)
            
        rotation_axis /= rotation_axis_norm  # 归一化旋转轴
        rotation_angle = np.arccos(np.clip(np.dot(fpAxis, samAxis), -1.0, 1.0))  # 夹角
        # 3. 使用 Rodrigues 公式构造旋转矩阵
        return rodrigues_rotation_matrix(rotation_axis, rotation_angle)
    
def rodrigues_rotation_matrix(axis, angle):
    """通过 Rodrigues 公式计算旋转矩阵"""
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])  # 反对称矩阵
    I = np.eye(3)  # 单位矩阵
    return I + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

def partialPcd2mainAxis(mask, depth, K, depth_min=0.05, depth_max=5.0, outlier_thresh=1.0, variance_threshold=1):
        """
        Get the main axis of the object from the partial point cloud.
        """
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        depth_masked = np.where(mask > 0, depth, 0)
        depth_masked = np.where((depth_masked > depth_min) & (depth_masked < depth_max), depth_masked, 0)
        # 使用中值滤波平滑深度图（减少噪声）
        depth_masked = cv2.medianBlur(depth_masked.astype(np.float32), 5)
        
        # rows, cols = np.where(depth_masked>0)
        # points = []
        # for u, v in zip(cols, rows):
        #     z = depth_masked[v, u]
        #     x = (u - cx) * z / fx
        #     y = (v - cy) * z / fy
        #     points.append([x, y, z])
        # points = np.array(points)
        
        # 获取深度图中非零元素的坐标
        rows, cols = np.where(depth_masked > 0)
        # 获取对应的深度值
        z = depth_masked[rows, cols]
        # 矢量化计算 x, y, z 点
        x = (cols - cx) * z / fx
        y = (rows - cy) * z / fy
        # 将 x, y, z 合并为点云数组
        points = np.vstack((x, y, z)).T
        
        
        point_center = np.mean(points, axis=0)
        # 离群点剔除（基于欧几里得距离）
        distances = np.linalg.norm(points - point_center, axis=1)
        inlier_mask = distances < (np.mean(distances) + outlier_thresh * np.std(distances))
        points_filtered = points[inlier_mask]

        # 如果过滤后点云过少，则返回默认值
        if len(points_filtered) < 3:
            return None,None
        
        points_centered = points_filtered - np.mean(points_filtered, axis=0)
        u, s, vh = np.linalg.svd(points_centered, full_matrices=False)
        mainAxis = vh[0]
        return mainAxis, np.mean(points_filtered, axis=0)
        
        # ## visualization
        # ## 可视化：绘制方向向量
        # ## 1. 创建方向向量的线段
        # line_points = []  # 存储线段的起点和终点
        # line_indices = []  # 存储线段的连接关系
        # colors = []  # 存储线段颜色
        # line_points.append(point_center)
        # line_points.append(point_center + mainAxis * 0.1)
        # line_indices.append([0, 1])
        # colors.append([1, 0, 0])
        
        # line_set = o3d.geometry.LineSet()
        # line_set.points = o3d.utility.Vector3dVector(np.array(line_points))  # 线段端点
        # line_set.lines = o3d.utility.Vector2iVector(np.array(line_indices))  # 线段连接关系
        # line_set.colors = o3d.utility.Vector3dVector(np.array(colors))  # 线段颜色
        
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points)
        # o3d.visualization.draw_geometries(
        #     [pcd, line_set],
        #     window_name="Partial Point Cloud with Main Directions",
        #     point_show_normal=False,
        # )
    
def align_z_axis_to_target(current_rotation, z_target):
    """
    Adjust the current rotation matrix to align its z-axis with the target direction.

    Parameters:
        current_rotation (np.array): Current 3x3 rotation matrix.
        z_target (np.array): Target z-axis direction (3D vector).

    Returns:
        np.array: New 3x3 rotation matrix with z-axis aligned to z_target.
    """
    # 1. Normalize the target z-axis
    z_target = z_target / np.linalg.norm(z_target)

    # 2. Extract the current z-axis from the rotation matrix
    z_current = current_rotation[:, 2]

    # 3. Compute the rotation axis (cross product) and angle (dot product)
    rotation_axis = np.cross(z_current, z_target)
    rotation_axis_norm = np.linalg.norm(rotation_axis)

    # Handle special cases
    if rotation_axis_norm < 1e-6:  # If the vectors are parallel or anti-parallel
        if np.dot(z_current, z_target) > 0:
            # z_current and z_target are parallel, no rotation needed
            return current_rotation
        else:
            # z_current and z_target are anti-parallel, rotate 180° around any orthogonal axis
            # Choose an arbitrary axis orthogonal to z_current
            arbitrary_axis = np.array([1, 0, 0]) if abs(z_current[0]) < 0.9 else np.array([0, 1, 0])
            rotation_axis = np.cross(z_current, arbitrary_axis)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)  # Normalize
            return rodrigues_rotation_matrix(rotation_axis, np.pi) @ current_rotation

    # Normalize the rotation axis
    rotation_axis = rotation_axis / rotation_axis_norm

    # Compute the rotation angle
    rotation_angle = np.arccos(np.clip(np.dot(z_current, z_target), -1.0, 1.0))

    # 4. Compute the alignment rotation matrix using Rodrigues' formula
    R_align = rodrigues_rotation_matrix(rotation_axis, rotation_angle)

    # 5. Compute the new rotation matrix
    R_new = R_align @ current_rotation
    return R_new
    
def align_z_axes(source_z, target_z):
    """
    Align the z-axis of a source coordinate system with the z-axis of a target coordinate system.
    
    Parameters:
    - source_z: np.ndarray, shape (3,), the z-axis direction vector of the source coordinate system.
    - target_z: np.ndarray, shape (3,), the z-axis direction vector of the target coordinate system.
    
    Returns:
    - rotation: scipy.spatial.transform.Rotation, the rotation that aligns source_z with target_z.
    """
    # Normalize the vectors
    source_z = source_z / np.linalg.norm(source_z)
    target_z = target_z / np.linalg.norm(target_z)
    
    # Calculate the rotation axis (cross product of source_z and target_z)
    rotation_axis = np.cross(source_z, target_z)
    
    # Calculate the cosine of the rotation angle (dot product of source_z and target_z)
    cos_theta = np.dot(source_z, target_z)
    
    # Calculate the sine of the rotation angle (needed for the rotation matrix)
    sin_theta = np.linalg.norm(rotation_axis)
    
    # Handle the case where source_z and target_z are (anti-)parallel
    if sin_theta == 0:
        if cos_theta > 0:
            # They are parallel, no rotation needed
            return R.from_euler('xyz', [0, 0, 0])
        else:
            # They are anti-parallel, rotate 180 degrees about any perpendicular axis
            # Here we choose the x-axis arbitrarily
            return R.from_euler('xyz', [np.pi, 0, 0])
    
    # Normalize the rotation axis
    rotation_axis = rotation_axis / sin_theta
    
    # Calculate the rotation angle
    theta = np.arccos(cos_theta)
    
    # Create the rotation matrix using scipy's Rotation class
    rotation = R.from_rotvec(theta * rotation_axis)
    
    return rotation

def Axis2RotMat_(fpAxis, samAxis, samPosition=None):#fpAxis:上一帧的主轴，sanAxis：目标轴
    ## v2
    mat1 = get_rotMats_from_axis_v3(fpAxis, samAxis)
    if samPosition is None:
        return mat1
    samAxis2, samAxis3 = GuessTopBottom(samAxis, samPosition)
    mat2 = get_rotMats_from_axis_v3(fpAxis, samAxis2)
    mat3 = get_rotMats_from_axis_v3(fpAxis, samAxis3)
    mats = np.concatenate((mat1, mat2, mat3), axis=0)
    return mats
def GuessTopBottom(samAxis, samPosition):
    """假设当前PCA主轴samAxism是拍到底/顶部, 需要旋转得到正确的主轴"""
    assert samPosition is not None
    assert len(samAxis) == len(samPosition) and len(samAxis)==3
    ### 计算平面法向量
    plane_p1 = np.array([samPosition[0], samPosition[1], 0])
    plane_p2 = np.array(samPosition+100*samAxis)
    plane_p3 = np.array(samPosition-100*samAxis)
    
    v1 = plane_p2 - plane_p1
    v2 = plane_p3 - plane_p1
    plane_norm = np.cross(v1, v2)
    
    ### 给定直线的方向向量 
    line_vector = plane_p3 - plane_p2
    
    ### 得到正确的主轴
    perpendicular_vector = np.cross(plane_norm, line_vector)
    
    return perpendicular_vector, plane_norm

def get_rotMats_from_axis_v3(fpAxis, samAxis):
        """
        Get the rotation matrix from 2 axis .
        """
        angles = [0.0, np.pi] # z轴对齐，正反两个方向
        # 1. 确保输入向量归一化
        fpAxis = fpAxis / np.linalg.norm(fpAxis)
        samAxis = samAxis / np.linalg.norm(samAxis)
        # 2. 计算旋转轴（叉积）和旋转角度（点积）
        rotation_axis = np.cross(fpAxis, samAxis)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        # 如果叉积接近 0，说明两个向量平行或反平行
        if rotation_axis_norm < 1e-2: ## TODO::fix the thresold
            arbitrary_axis = np.array([1, 0, 0]) if abs(fpAxis[0]) < 0.9 else np.array([0, 1, 0])
            rotation_axis = np.cross(fpAxis, arbitrary_axis)
            rotation_axis /= np.linalg.norm(rotation_axis)
            results = []
            for angle in angles:
                results.append(rodrigues_rotation_matrix(samAxis, angle))
            return results
            
        rotation_axis /= rotation_axis_norm  # 归一化旋转轴
        rotation_angle = np.arccos(np.clip(np.dot(fpAxis, samAxis), -1.0, 1.0))  # 夹角
        # 3. 使用 Rodrigues 公式构造旋转矩阵
        results = []
        for angle in angles:
            results.append(rodrigues_rotation_matrix(rotation_axis, angle+rotation_angle))
        return results
    
def rotate_vector_x(vector, angle):
    """
    让向量绕x轴旋转一定角度
    :param vector: 原始方向向量 [v_x, v_y, v_z]
    :param angle: 旋转角度，单位为弧度
    :return: 旋转后的向量
    """
    # 定义绕x轴的旋转矩阵
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)]
    ])
    
    # 将向量转换为列向量并进行旋转
    vector = np.array(vector)
    rotated_vector = np.dot(rotation_matrix, vector)
    
    return rotated_vector

def rotate_vector_y(vector, angle):
    """
    让向量绕y轴旋转一定角度
    :param vector: 原始方向向量 [v_x, v_y, v_z]
    :param angle: 旋转角度，单位为弧度
    :return: 旋转后的向量
    """
    # 定义绕x轴的旋转矩阵
    rotation_matrix = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])
    
    # 将向量转换为列向量并进行旋转
    vector = np.array(vector)
    rotated_vector = np.dot(rotation_matrix, vector)
    
    return rotated_vector

def Rotation_Similarity(a, b):
        #v1
        R = np.dot(a.T, b)
        theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
        return theta
        #v2:比较两个z轴的夹角
        # z_a = a[:3,2]
        # z_b = b[:3,2]
        # score = np.dot(z_a,z_b)
        # return score
        
def homogeneous_to_xyz_euler(matrix):
    """
    将一个4x4齐次变换矩阵转换为包含XYZ坐标和XYZ欧拉角的六元np.array数组。
    
    参数:
    matrix (numpy.ndarray): 一个4x4的齐次变换矩阵。
    
    返回:
    np.array: 一个六元数组，前三个元素为XYZ坐标，后三个元素为XYZ欧拉角。
    """
    # 提取平移部分（XYZ坐标）
    xyz = matrix[:3, 3]
    
    # 提取旋转部分并计算欧拉角
    # euler_from_matrix返回的是(roll, pitch, yaw)，这里我们将其视为XYZ欧拉角
    roll, pitch, yaw = euler_from_matrix(matrix[:3, :3], 'sxyz')
    
    # 将XYZ坐标和欧拉角组合成一个六元数组
    result = np.concatenate((xyz, [roll, pitch, yaw]))
    
    return result

def homogeneous_to_quaternion(matrix):
    qua = quaternion_from_matrix(matrix)
    pos = matrix[:3, 3]
    result = np.concatenate((pos,qua))
    return result

def find_min_angle_vector(A, B, C):
    """
    找出与向量A夹角最小的向量B或C。
    
    参数:
    A, B, C -- numpy数组表示的三维向量
    
    返回:
    与A夹角最小的向量（B或C）
    """
    # 计算A与B的点积
    dot_product_AB = np.dot(A, B)
    
    # 计算A与C的点积
    dot_product_AC = np.dot(A, C)
    
    # 计算A的模长（用于归一化点积）
    norm_A = np.linalg.norm(A)
    
    # 计算B和C的模长（也用于归一化点积）
    norm_B = np.linalg.norm(B)
    norm_C = np.linalg.norm(C)
    
    # 计算A与B夹角的余弦值（通过点积除以模长的乘积）
    cos_theta_AB = dot_product_AB / (norm_A * norm_B)
    
    # 计算A与C夹角的余弦值
    cos_theta_AC = dot_product_AC / (norm_A * norm_C)
    
    # 比较余弦值，选择夹角最小的向量
    if cos_theta_AB > cos_theta_AC:
        return B
    else:
        return C
    
def rgb_crop_with_mask(img, samMask, samCenter, crop_ratio=2.0):
    """根据掩码裁剪RGB图像"""
    image = img.copy()
    # 获取掩码的边界框
    # mask = samMask.astype(bool)
    # image = np.ones_like(img)
    # image = image*255
    # image[mask] = img[mask]
    
    x, y = samCenter
    contours, _ = cv2.findContours(samMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    cnt = contours[0]
    _, _, w, h = cv2.boundingRect(cnt)
    w = max(w, h)*crop_ratio  # 保证宽高相等
    h = w
    l = int(max(x-w//2, 0))
    r = int(min(x+w//2, image.shape[1]))
    t = int(max(y-h//2, 0))
    b = int(min(y+h//2, image.shape[0]))
    # 裁剪RGB图像
    cropped_image = image[t:b, l:r]
    return cropped_image

def rgb_crop_with_bbox(img, bbox, crop_ratio=2.0):
    """根据边界框裁剪RGB图像"""
    image = img.copy()
    x, y, w, h = bbox
    yc = int(y + h//2)
    xc = int(x + w//2)
    # tmp = image[y:y+h, x:x+w] ### FIXME
    # cv2.imshow('tmp', tmp)
    # cv2.waitKey(0)
    
    wid = max(w, h)*crop_ratio  # 保证宽高相等
    # wid = max(w, h)  # 保证宽高相等
    wid_2 = int(wid//2)
    # 裁剪RGB图像
    cropped_image = image[yc-wid_2:yc+wid_2, xc-wid_2:xc+wid_2]
    # cv2.imshow('cropped_image', cropped_image)
    # cv2.waitKey(0)
    return cropped_image


def make_mesh_tensors(mesh, device='cuda', max_tex_size=None):
  mesh_tensors = {}
  if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
    img = np.array(mesh.visual.material.image.convert('RGB'))
    img = img[...,:3]
    if max_tex_size is not None:
      max_size = max(img.shape[0], img.shape[1])
      if max_size>max_tex_size:
        scale = 1/max_size * max_tex_size
        img = cv2.resize(img, fx=scale, fy=scale, dsize=None)
    mesh_tensors['tex'] = torch.as_tensor(img, device=device, dtype=torch.float)[None]/255.0
    mesh_tensors['uv_idx']  = torch.as_tensor(mesh.faces, device=device, dtype=torch.int)
    uv = torch.as_tensor(mesh.visual.uv, device=device, dtype=torch.float)
    uv[:,1] = 1 - uv[:,1]
    mesh_tensors['uv']  = uv
  else:
    if mesh.visual.vertex_colors is None:
      mesh.visual.vertex_colors = np.tile(np.array([128,128,128]).reshape(1,3), (len(mesh.vertices), 1))
    mesh_tensors['vertex_color'] = torch.as_tensor(mesh.visual.vertex_colors[...,:3], device=device, dtype=torch.float)/255.0

  mesh_tensors.update({
    'pos': torch.tensor(mesh.vertices, device=device, dtype=torch.float),
    'faces': torch.tensor(mesh.faces, device=device, dtype=torch.int),
    'vnormals': torch.tensor(mesh.vertex_normals, device=device, dtype=torch.float),
  })
  return mesh_tensors

def projection_matrix_from_intrinsics(K, height, width, znear, zfar, window_coords='y_down'):
  """Conversion of Hartley-Zisserman intrinsic matrix to OpenGL proj. matrix.

  Ref:
  1) https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL
  2) https://github.com/strawlab/opengl-hz/blob/master/src/calib_test_utils.py

  :param K: 3x3 ndarray with the intrinsic camera matrix.
  :param x0 The X coordinate of the camera image origin (typically 0).
  :param y0: The Y coordinate of the camera image origin (typically 0).
  :param w: Image width.
  :param h: Image height.
  :param nc: Near clipping plane.
  :param fc: Far clipping plane.
  :param window_coords: 'y_up' or 'y_down'.
  :return: 4x4 ndarray with the OpenGL projection matrix.
  """
  x0 = 0
  y0 = 0
  w = width
  h = height
  nc = znear
  fc = zfar

  depth = float(fc - nc)
  q = -(fc + nc) / depth
  qn = -2 * (fc * nc) / depth

  # Draw our images upside down, so that all the pixel-based coordinate
  # systems are the same.
  if window_coords == 'y_up':
    proj = np.array([
      [2 * K[0, 0] / w, -2 * K[0, 1] / w, (-2 * K[0, 2] + w + 2 * x0) / w, 0],
      [0, -2 * K[1, 1] / h, (-2 * K[1, 2] + h + 2 * y0) / h, 0],
      [0, 0, q, qn],  # Sets near and far planes (glPerspective).
      [0, 0, -1, 0]
      ])

  # Draw the images upright and modify the projection matrix so that OpenGL
  # will generate window coords that compensate for the flipped image coords.
  elif window_coords == 'y_down':
    proj = np.array([
      [2 * K[0, 0] / w, -2 * K[0, 1] / w, (-2 * K[0, 2] + w + 2 * x0) / w, 0],
      [0, 2 * K[1, 1] / h, (2 * K[1, 2] - h + 2 * y0) / h, 0],
      [0, 0, q, qn],  # Sets near and far planes (glPerspective).
      [0, 0, -1, 0]
      ])
  else:
    raise NotImplementedError

  return proj

def transform_dirs(dirs,tf):
  """
  @dirs: (...,3)
  @tf: (...,4,4)
  """
  if len(tf.shape)>=3 and tf.shape[-3]!=dirs.shape[-2]:
    tf = tf[...,None,:,:]
  return (tf[...,:3,:3]@dirs[...,None])[...,0]

def transform_pts(pts,tf):
  """Transform 2d or 3d points
  @pts: (...,N_pts,3)
  @tf: (...,4,4)
  """
  if len(tf.shape)>=3 and tf.shape[-3]!=pts.shape[-2]:
    tf = tf[...,None,:,:]
  return (tf[...,:-1,:-1]@pts[...,None] + tf[...,:-1,-1:])[...,0]
def to_homo_torch(pts):
  '''
  @pts: shape can be (...,N,3 or 2) or (N,3) will homogeneliaze the last dimension
  '''
  ones = torch.ones((*pts.shape[:-1],1), dtype=torch.float, device=pts.device)
  homo = torch.cat((pts, ones),dim=-1)
  return homo

def bbox2mask(bbox:list, width, height, rate=1.0):
    """
    bbox: x, y, w, h
    """
    x, y, w, h = bbox
    res = np.zeros((height, width), dtype=np.uint8)
    # res[y:y+h, x:x+w] = 255
    y_c = y+h/2
    rate_h_2 = rate*h/2
    y_min = max(int(y_c - rate_h_2), 0)
    y_max = min(int(y_c + rate_h_2), height)
    x_c = x+w/2
    rate_w_2 = rate*w/2
    x_min = max(int(x_c - rate_w_2), 0)
    x_max = min(int(x_c + rate_w_2), width)
    res[y_min:y_max, x_min:x_max] = 255
    
    return res
    
# def generate_samples(pose_bar, P):
#     svd_x = np.sqrt(P[3][3])
#     svd_y = np.sqrt(P[4][4])
#     svd_z = np.sqrt(P[5][5])

#     # 生成每个欧拉角的采样点
#     samples_x = [pose_bar[0] - svd_x, pose_bar[0], pose_bar[0] + svd_x]
#     samples_y = [pose_bar[1] - svd_y, pose_bar[1], pose_bar[1] + svd_y]
#     samples_z = [pose_bar[2] - svd_z, pose_bar[2], pose_bar[2] + svd_z]
#     # samples_x =[pose_bar[0]]
#     # samples_y=[pose_bar[1]]
#     # samples_z=[pose_bar[2]]
#     # 生成所有组合
#     all_samples = list(itertools.product(samples_x, samples_y, samples_z))

#     return all_samples

def generate_independent_samples(mean, cov, low=0, max=0.5, var_threshold=0.01):
    """
    生成独立三轴采样点（每个轴3个值，共27个组合）

    参数:
        mean (np.array): 均值向量 [mu_x, mu_y, mu_z]
        cov (np.array): 协方差矩阵（必须为对角矩阵）
        low (float): 缩放标准差的最小值
        max (float): 缩放标准差的最大值
        var_threshold (float): 方差阈值，低于此值的轴只取均值

    返回:
        samples (list): 采样点的列表，每个元素为 (x, y, z)
    """
    # 检查协方差矩阵是否为对角矩阵
    assert np.allclose(cov, np.diag(np.diag(cov))), "协方差矩阵必须为对角矩阵"

    # 提取各轴标准差
    stds = np.sqrt(np.diag(cov))

    # 缩放标准差到0.1-0.5的区间
    # if stds.max() > max:
    if stds.max() - stds.min() < 1e-3:
        stds = np.ones_like(stds) * max
    else:
        stds = low + (stds - stds.min()) / (stds.max() - low) * (max - low)
    

    # 为每个轴生成采样点
    axis_samples = []
    max_var = stds.max() ** 2
    for mu, std in zip(mean, stds):
        var = std ** 2
        if var  < var_threshold:
            axis_samples.append([mu])
        else:
            axis_samples.append([mu - std, mu, mu + std])
            
        ## only mu:
        # axis_samples.append([mu])

    # 生成所有组合
    return list(itertools.product(*axis_samples))

# def generate_sigma_points(mean, cov, alpha=1e-3, beta=2, kappa=0):
#     n = len(mean)
#     lambda_ = alpha**2 * (n + kappa) - n
#     sigma_points = np.zeros((2*n + 1, n))
#     sigma_points[0] = mean
#     cov_reg = cov + 1e-6 * np.eye(n)  # 正则化确保正定
#     U = np.linalg.cholesky((n + lambda_) * cov_reg)
#     for i in range(n):
#         sigma_points[i+1] = mean + U[i, :]
#         sigma_points[n+i+1] = mean - U[i, :]
#     return sigma_points

def euler_to_rotation_matrix(euler_angles,t):
    """
    将 XYZ 顺序的欧拉角转换为旋转矩阵。
 
    参数:
    euler_angles (tuple or list): 包含三个元素的元组或列表，表示绕 X、Y、Z 轴的旋转角度（以弧度为单位）。
 
    返回:
    numpy.ndarray: 3x3 的旋转矩阵。
    """
    pose_matrix = np.eye(4)
    # 创建 Rotation 对象，指定使用 XYZ 顺序的欧拉角
    rot = R.from_euler('xyz', euler_angles)
    
    # 获取旋转矩阵
    rotation_matrix = rot.as_matrix()
    
    pose_matrix[:3,:3] = rotation_matrix
    pose_matrix[:3,3] = t
    return pose_matrix

def euler2quaternion(euler):
    r = R.from_euler('xyz', euler, degrees=True)
    quaternion = r.as_quat() ### xyzw，已经完成了归一化
    return quaternion

def rotation_matrix_to_quaternion(matrix):
    """
    将 3x3 旋转矩阵转换为四元数。
 
    参数:
    matrix (np.ndarray): 3x3 的旋转矩阵。
 
    返回:
    np.ndarray: 对应的四元数 [x, y, z, w]。
    """
    # 使用 scipy 的 Rotation 对象
    r = R.from_matrix(matrix)
    # 转换为四元数
    quaternion = r.as_quat()  # 返回 [x, y, z, w]
    return quaternion

# def getTrackPosition(mask, depth, K, eps=0.08, min_samples=20):
#     object_indices = np.where(mask > 0)
#     # 获取物体区域对应的深度值
#     object_depth_values = depth[object_indices]
#     object_depth_values = object_depth_values.reshape(-1, 1)
#     dbscan = DBSCAN(eps=eps, min_samples=min_samples)  # eps 是一个超参数，控制聚类的紧密度
#     labels = dbscan.fit_predict(object_depth_values)
#     label_counts = Counter(labels)
#     print(f'[getTrackPosition]label_counts: {label_counts}')
#     valid_labels = [label for label in label_counts if label != -1]
#     Z_vals = []
#     ### 输出所有有效类
#     for label in valid_labels:
#         class_depth_values = object_depth_values[labels == label]
#         # z_val = np.mean(class_depth_values)
#         z_val = np.mean(class_depth_values)
#         print(f'\033[35m[getTrackPosition] label={label}, depth={z_val} \033[0m')
#         if z_val < 0.1:
#             continue
#         Z_vals.append(z_val)
    
#     Z = min(Z_vals)
#     u_center = np.mean(object_indices[1])  # x 坐标（列）
#     v_center = np.mean(object_indices[0])  # y 坐标（行）
#     fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
#     X = (u_center - cx) * Z / fx
#     Y = (v_center - cy) * Z / fy
#     print(f'[getTrackPosition] Z: {Z}')
#     return np.array([X, Y, Z])

### TODO 参数eps和min_samples需要调整
def getTrackPosition(mask, depth, K, eps=0.01, min_samples=50, USE_CENTER=True, USE_CENTER_DEPTH=False):
    object_indices = np.where(mask > 0)
    ## test
    if USE_CENTER:
        u_center = np.mean(object_indices[1])  # x 坐标（列）
        v_center = np.mean(object_indices[0])  # y 坐标（行）
        Z = depth[int(v_center), int(u_center)]
        # Zs = depth[object_indices]
        # Z = np.mean(Zs)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        X = (u_center - cx) * Z / fx
        Y = (v_center - cy) * Z / fy
        print(f'[getTrackPosition] Z: {Z}')
        return np.array([X, Y, Z])
    
    # 获取物体区域对应的深度值
    object_depth_values = depth[object_indices]
    object_depth_values = object_depth_values.reshape(-1, 1)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)  # eps 是一个超参数，控制聚类的紧密度
    labels = dbscan.fit_predict(object_depth_values)
    label_counts = Counter(labels)
    print(f'[getTrackPosition]label_counts: {label_counts}')
    valid_labels = [label for label in label_counts if label != -1]
    Z_vals = []
    uvs = []
    ### 输出所有有效类
    for label in valid_labels:
        class_depth_values = object_depth_values[labels == label]
        # z_val = np.min(class_depth_values)
        z_val = np.mean(class_depth_values)
        print(f'\033[35m[getTrackPosition] label={label}, depth={z_val} \033[0m')
        if z_val < 0.1:
            continue
        u_ = np.mean(object_indices[1][labels == label])  # x 坐标（列）
        v_ = np.mean(object_indices[0][labels == label])  # y 坐标（行）
        uvs.append([u_, v_])
        Z_vals.append(z_val)
    
    # Z = min(Z_vals)
    if len(Z_vals) == 0:
        return None
    Z_arg = np.argmin(Z_vals)
    Z = Z_vals[Z_arg]
    u_center, v_center = uvs[Z_arg]
    if USE_CENTER:
        u_center = np.mean(object_indices[1])  # x 坐标（列）
        v_center = np.mean(object_indices[0])  # y 坐标（行）
    if USE_CENTER_DEPTH:
        Z = depth[int(v_center), int(u_center)]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    X = (u_center - cx) * Z / fx
    Y = (v_center - cy) * Z / fy
    print(f'[getTrackPosition] Z: {Z}')
    return np.array([X, Y, Z])


def get_6d_pose_arr_from_mat(pose):
    if torch.is_tensor(pose):
        is_batched = pose.ndim == 3
        if is_batched:
            pose_np = pose[0].cpu().numpy()
        else:
            pose_np = pose.cpu().numpy()
    else:
        pose_np = pose

    xyz = pose_np[:3, 3]
    rotation_matrix = pose_np[:3, :3]
    euler_angles = R.from_matrix(rotation_matrix).as_euler('xyz', degrees=False)
    return np.r_[xyz, euler_angles]

def get_mat_from_6d_pose_arr(pose_arr):
    # 提取位移 (xyz)
    xyz = pose_arr[:3]
    
    # 提取欧拉角
    euler_angles = pose_arr[3:]
    
    # 从欧拉角生成旋转矩阵
    rotation = R.from_euler('xyz', euler_angles, degrees=False)
    rotation_matrix = rotation.as_matrix()
    
    # 创建 4x4 变换矩阵
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = xyz
    
    return transformation_matrix