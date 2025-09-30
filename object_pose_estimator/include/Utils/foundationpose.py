import enum
import sys

import rospy
import numpy as np
import os
from cv_bridge import CvBridge
import sensor_msgs.msg
from scipy.spatial.transform import Rotation as R
import imageio
import cv2
from estimater import *
import time
import trimesh

class _foundationpose_:
    def __init__(self, config, _mesh_file_):
        # _mesh_file_ = config['FoundationPose']['mesh_file']
        self.est_refine_iter = config['FoundationPose']['est_refine_iter']
        self.track_refine_iter = config['FoundationPose']['track_refine_iter']
        self.K = np.loadtxt(config["FoundationPose"]["K_path"]).reshape(3,3)
        self.debug=config['debug']
        self.debug_dir=config['debug_dir'] + '/FoundationPose'
        self.init_rot_num = config['FoundationPose']['init_rot_num']
        self.crop_size = config['FoundationPose']['crop_size']
        self.img_with_pose_pub_ = rospy.Publisher(config['FoundationPose']['img_with_pose_topic'], sensor_msgs.msg.Image, queue_size=10)
        self.bridge = CvBridge()
        os.system(f'rm -rf {self.debug_dir}/* && mkdir -p {self.debug_dir}/track_vis {self.debug_dir}/ob_in_cam')
        
        self.is_changing_mesh = False
        self.mesh_reset(_mesh_file_)
        self.rot_grid = self.est.rot_grid.data.cpu().numpy()
        if self.debug>=2:
            os.makedirs(f'{self.debug_dir}/track_vis', exist_ok=True)
            crop_image_dir = os.path.join(self.debug_dir, 'crop_image')
            os.makedirs(crop_image_dir, exist_ok=True)
        rospy.loginfo("FoundationPose initialization done")
        
        self.id = 0
        self.i = 1
        
    # def init(self, rgb, mask, depth): #不需要用该函数，因此没有更改
    #     self.H, self.W = rgb.shape[:2]
    #     color = self.get_color(rgb)
    #     depth = self.get_depth(depth)
    #     mask = self.get_mask(mask)
    #     pose = None
    #     try:
    #         pose = self.est.register(
    #             K=self.K, 
    #             rgb=color, 
    #             depth=depth, 
    #             ob_mask=mask, 
    #             iteration=self.est_refine_iter)
    #     except Exception as e:
    #         print(e)
        
    #     if pose is None:
    #         return None
    #     if self.debug>=1:
    #         center_pose = pose@np.linalg.inv(self.to_origin)
    #         vis = draw_posed_3d_box(self.K, img=color, ob_in_cam=center_pose, bbox=self.bbox)
    #         vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=self.K, thickness=1, transparency=0, is_input_rgb=True)
    #         ros_img = self.bridge.cv2_to_imgmsg(vis, encoding='rgb8')
    #         self.img_with_pose_pub_.publish(ros_img)
            
    #         if self.debug>=2:
    #             imageio.imwrite(f'{self.debug_dir}/track_vis/{self.id:04d}.png', vis)
    #         self.id += 1
        
    #     return pose  
    
    # def init_with_guess_pose(self, color, mask, depth, cam2world, preset_grip_pose_):
    #     self.H, self.W = color.shape[:2]
    #     # color = self.get_color(rgb)
    #     # depth = self.get_depth(depth)
    #     mask = self.get_mask(mask).astype(bool)
        
    #     # obj2world = np.eye(4)
    #     # obj2world[:3, 3] = preset_grip_pose_[0]
    #     # obj2world[:3, :3] = R.from_quat(preset_grip_pose_[1]).as_matrix()

    #     # obj2cam = np.linalg.inv(cam2world) @ obj2world
    #     obj2cams = []
    #     for i in range(self.init_rot_num):
    #         obj2world = np.eye(4)
    #         obj2world[:3, 3] = preset_grip_pose_[0]
    #         obj2world[:3, :3] = R.from_quat(preset_grip_pose_[1]).as_matrix()
    #         obj2world[:3, :3] = obj2world[:3, :3] @ R.from_euler('xyz', [0, 0, 2*np.pi*i/self.init_rot_num]).as_matrix()
    #         obj2cam = np.linalg.inv(cam2world) @ obj2world
    #         obj2cams.append(obj2cam)
           
    #     try: 
    #         pose = self.est.my_register_with_init_pose(
    #             K=self.K, 
    #             rgb=color, 
    #             depth=depth, 
    #             ob_mask=mask, 
    #             iteration=self.est_refine_iter,
    #             init_pose=np.array(obj2cams))
    #     except Exception as e:
    #         print(e)
        
    #     if self.debug>=1:
    #         color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    #         center_pose = pose@np.linalg.inv(self.to_origin)
    #         vis = draw_posed_3d_box(self.K, img=color, ob_in_cam=center_pose, bbox=self.bbox)
    #         vis = draw_xyz_axis(color, ob_in_cam=pose, scale=0.1, K=self.K, thickness=1, transparency=0, is_input_rgb=True)
            
    #         ros_img = self.bridge.cv2_to_imgmsg(vis, encoding='rgb8')
    #         self.img_with_pose_pub_.publish(ros_img)
    #         if self.debug >= 2:
    #             imageio.imwrite(f'{self.debug_dir}/track_vis/{self.id:04d}.png', vis)
    #         self.id += 1
            
    #     return pose
    
    # def init_with_last_obj2world(self, color, mask, depth, cam2world, last_obj2world):
    #     self.H, self.W = color.shape[:2]
    #     # color = self.get_color(rgb)
    #     # depth = self.get_depth(depth)
    #     mask = self.get_mask(mask).astype(bool)
        
    #     # obj2world = np.eye(4)
    #     # obj2world[:3, 3] = preset_grip_pose_[0]
    #     # obj2world[:3, :3] = R.from_quat(preset_grip_pose_[1]).as_matrix()

    #     # obj2cam = np.linalg.inv(cam2world) @ obj2world
    #     obj2cams = []
    #     for i in range(self.init_rot_num):
    #         obj2world = np.eye(4)
    #         obj2world[:3, 3] = last_obj2world[:3, 3]
    #         obj2world[:3, :3] = last_obj2world[:3, :3] @ R.from_euler('xyz', [0, 0, 2*np.pi*i/self.init_rot_num]).as_matrix()
    #         obj2cam = np.linalg.inv(cam2world) @ obj2world
    #         obj2cams.append(obj2cam)
    #     try:    
    #         pose = self.est.my_register_with_init_pose(
    #             K=self.K, 
    #             rgb=color, 
    #             depth=depth, 
    #             ob_mask=mask, 
    #             iteration=self.est_refine_iter,
    #             init_pose=np.array(obj2cams))
    #     except Exception as e:
    #         print(e)
        
    #     if self.debug>=1:
    #         color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    #         center_pose = pose@np.linalg.inv(self.to_origin)
    #         vis = draw_posed_3d_box(self.K, img=color, ob_in_cam=center_pose, bbox=self.bbox)
    #         vis = draw_xyz_axis(color, ob_in_cam=pose, scale=0.1, K=self.K, thickness=1, transparency=0, is_input_rgb=True)
            
    #         ros_img = self.bridge.cv2_to_imgmsg(vis, encoding='rgb8')
    #         self.img_with_pose_pub_.publish(ros_img)
    #         if self.debug >= 2:
    #             imageio.imwrite(f'{self.debug_dir}/track_vis/{self.id:04d}.png', vis)
    #         self.id += 1
            
    #     return pose
    
    def track(self, color, depth):
        # color = self.get_color(rgb)
        # depth = self.get_depth(depth)
        try:
            pose = self.est.track_one(
                rgb=color, 
                depth=depth, 
                K=self.K, 
                iteration=self.track_refine_iter)
        except Exception as e:
            print(e)
        
        if self.debug>=1:
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            center_pose = pose@np.linalg.inv(self.to_origin)
            vis = draw_posed_3d_box(self.K, img=color, ob_in_cam=center_pose, bbox=self.bbox)
            vis = draw_xyz_axis(color, ob_in_cam=pose, scale=0.1, K=self.K, thickness=1, transparency=0, is_input_rgb=True)
            ros_img = self.bridge.cv2_to_imgmsg(vis, encoding='rgb8')
            self.img_with_pose_pub_.publish(ros_img)
            
            if self.debug >= 2:
                imageio.imwrite(f'{self.debug_dir}/track_vis/{self.id:04d}.png', vis)
            self.id += 1
        return pose
    
    def track_with_curr_guess(self, color, depth, samCenter, curr_guess_obj2cam, last_obj2cam):
        t = time.time()
        self.H, self.W = color.shape[:2]
         # rgb, depth, K_new = self.crop_image(color, depth, samCenter) 
        rgb, K_new = color, self.K
        
        # # ### translation correction
        # guess_rot = curr_guess_obj2cam[:3, :3]
        # transformed_vertices = np.dot(self.mesh_vertices, guess_rot.T)
        # min_z = np.min(transformed_vertices[:, 2])
        # curr_guess_obj2cam[2, 3] += np.abs(min_z)
        
        pose, vis = self.est.my_track_one_with_last_pose(
            rgb=rgb, 
            depth=depth, 
            K=K_new, 
            iteration=self.track_refine_iter,
            last_o2c=curr_guess_obj2cam)
        
        if self.debug>=2 and vis is not None:
            imageio.imwrite(f'{self.debug_dir}/vis_refine_{self.i:04d}.png', vis)
            
        if self.debug>=1:
            vis = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            
            ### 上一帧的pose可视化
            # center_pose_last = last_obj2cam@np.linalg.inv(self.to_origin)
            # vis = draw_posed_3d_box(self.K, img=vis, ob_in_cam=center_pose_last, bbox=self.bbox, line_color = (255,0,255), alpha=0.5)
            # vis = draw_xyz_axis(color, ob_in_cam=last_o2c, scale=0.1, K=self.K, thickness=1, transparency=0, is_input_rgb=True)   
            
            ### 当前一帧的pose可视化
            center_pose = pose@np.linalg.inv(self.to_origin)
            vis = draw_posed_3d_box(self.K, img=vis, ob_in_cam=center_pose, bbox=self.bbox)
            vis = draw_xyz_axis(vis, ob_in_cam=pose, scale=0.1, K=self.K, thickness=1, transparency=0, is_input_rgb=True)
            
            ros_img = self.bridge.cv2_to_imgmsg(vis, encoding='rgb8')
            self.img_with_pose_pub_.publish(ros_img)
            
            if self.debug >= 2:
                imageio.imwrite(f'{self.debug_dir}/track_vis/{self.id:04d}.png', vis)
            self.id += 1
        # if self.debug>=2 and vis is not None:
        #     imageio.imwrite(f'{self.debug_dir}/vis_refine_{self.i:04d}.png', vis)
            
        print(f"[FoundationPose] Pose time: {time.time() - t:.3f}s")
        return pose
    
    def track_with_last_multiRotation(self, color, depth, samCenter, last_o2c_s):
        self.H, self.W = color.shape[:2]
        # rgb, depth, K_new = self.crop_image(color, depth, samCenter) 
        rgb, K_new = color, self.K
        try:
            pose = self.est.my_track_one_with_last_multiPose(
                rgb=rgb, 
                depth=depth, 
                K=K_new, 
                iteration=self.track_refine_iter,
                last_o2c_s=last_o2c_s)
        except Exception as e:
            print(e)
            return None
        
        if self.debug>=1:
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            center_pose = pose@np.linalg.inv(self.to_origin)
            vis = draw_posed_3d_box(self.K, img=color, ob_in_cam=center_pose, bbox=self.bbox)
            vis = draw_xyz_axis(color, ob_in_cam=pose, scale=0.1, K=self.K, thickness=1, transparency=0, is_input_rgb=True)
            
            
            ## test
            # T_trans_to_origin = np.eye(4)
            # T_trans_to_origin[:3, 3] = - pose[:3, 3]
            # T_rot_z = np.eye(4)
            # T_rot_z[:3, :3] = R.from_euler('z', np.pi/2).as_matrix()
            # T_trans_back = np.eye(4)
            # T_trans_back[:3, 3] = pose[:3, 3]
            # tmp_pose = T_trans_back @ T_rot_z @ T_trans_to_origin @ pose
            
            # pixel_hull = self.bbox_project_pixel_hull(tmp_pose) # 
            # mask = self.pixel_hull_2_mask(pixel_hull)
            # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # vis = cv2.addWeighted(vis, 0.5, mask[...,None], 0.5, 0) # bug
            
            ros_img = self.bridge.cv2_to_imgmsg(vis, encoding='rgb8')
            self.img_with_pose_pub_.publish(ros_img)
            
            if self.debug >= 2:
                # debug_dir = '/home/niloiv/pose_estimate_new/src/Dynamic_6D/object_pose_estimator/debug/FoundationPose'
                # crop_image_dir = os.path.join(self.debug_dir, 'crop_image')
                # os.makedirs(crop_image_dir, exist_ok=True)
                imageio.imwrite(f'{self.debug_dir}/track_vis/{self.id:04d}.png', vis)
                imageio.imwrite(f'{self.debug_dir}/crop_image/{self.id:04d}.png', rgb)
            self.id += 1
        return pose
    
    def get_color(self, color):
        H, W = color.shape[:2]
        color = cv2.resize(color, (W,H), interpolation=cv2.INTER_NEAREST)
        # color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        return color
    
    def get_depth(self, depth, zfar=np.inf):
        H, W = depth.shape[:2]
        depth = np.array(depth, copy=True)  # 创建一个深拷贝，确保数组是可写的
        # depth = erode_depth(depth, radius=2, device='cuda')
        # depth = bilateral_filter_depth(depth, radius=2, device='cuda')
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
        depth[(depth<0.001) | (depth>=zfar)] = 0
        return depth
    
    def get_mask(self, mask):
        # if len(mask.shape)==3:
        #     for c in range(3):
        #         if mask[...,c].sum()>0:
        #             mask = mask[...,c]
        #             break
        mask = cv2.resize(mask, (self.W,self.H), interpolation=cv2.INTER_NEAREST).astype(bool)
        # mask = mask.astype(bool)
        return mask
    
    def mesh_reset(self, _mesh_file_):
        if self.is_changing_mesh:
            return
        self.is_changing_mesh = True
        mesh = trimesh.load_mesh(_mesh_file_)
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        self.to_origin = to_origin
        self.extents = extents
        self.bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
        
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        self.est = FoundationPose(
            model_pts=mesh.vertices, 
            model_normals=mesh.vertex_normals, 
            mesh=mesh, 
            scorer=scorer, 
            refiner=refiner, 
            debug_dir=self.debug_dir, 
            debug=self.debug,  
            glctx=glctx,
        )
        self.is_changing_mesh = False
        self.mesh_vertices = mesh.vertices
        rospy.logwarn(f"Reset FoundationPose with mesh file: {_mesh_file_}")
    
    def _reset_(self, _mesh_file_):
        rospy.logwarn("Reset FoundationPose")
        self.mesh_reset(_mesh_file_)
        self.id = 0
        self.est.last_pose = None
        return
    
    def bbox_project_pixel_hull(self, obj2cam):
        min_xyz = self.bbox.min(axis=0)
        max_xyz = self.bbox.max(axis=0)
        p_local = np.array([
            [min_xyz[0], min_xyz[1], min_xyz[2]],
            [min_xyz[0], min_xyz[1], max_xyz[2]],
            [min_xyz[0], max_xyz[1], min_xyz[2]],
            [min_xyz[0], max_xyz[1], max_xyz[2]],
            [max_xyz[0], min_xyz[1], min_xyz[2]],
            [max_xyz[0], min_xyz[1], max_xyz[2]],
            [max_xyz[0], max_xyz[1], min_xyz[2]],
            [max_xyz[0], max_xyz[1], max_xyz[2]],
        ])
        p_local_h = np.hstack([p_local, np.ones((8,1))])
        # the vertex in camera coordinate
        p_camera = obj2cam @ p_local_h.T
        P_pixel = (self.K @ p_camera[:3, :]).T
        P_pixel /= P_pixel[:, 2:]
        pixel_points = P_pixel[:, :2].astype(np.int32)
        
        return cv2.convexHull(pixel_points)
        
    def pixel_hull_2_mask(self, pixel_hull):
        mask = np.zeros((self.H, self.W), dtype=np.uint8)
        cv2.fillPoly(mask, [pixel_hull.astype(np.int32)], 255)
        # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return mask
    
    def bbox_project_to_mask(self, obj2cam):
        pixel_hull = self.bbox_project_pixel_hull(obj2cam)
        return self.pixel_hull_2_mask(pixel_hull)
    
    def crop_image(self, rgb, depth, samCenter):
        center_x = samCenter[0]
        center_y = samCenter[1]

        half_crop_size = self.crop_size // 2   
        # 计算裁剪区域的边界
        height, width = rgb.shape[:2]
        left = int(max(0, center_x - half_crop_size))
        top = int(max(0, center_y - half_crop_size))
        right = int(min(width, center_x + half_crop_size))
        bottom = int(min(height, center_y + half_crop_size))
        cropped_rgb = rgb[top:bottom, left:right]
        cropped_depth = depth[top:bottom, left:right]
        new_cx = self.K[0, 2] - center_x + half_crop_size
        new_cy = self.K[1, 2] - center_y + half_crop_size
        scale_x = self.crop_size / width
        scale_y = self.crop_size / height
        K_new = np.array([[self.K[0,0], 0, new_cx],
                          [0, self.K[1,1], new_cy],
                          [0, 0, 1]])
        return cropped_rgb, cropped_depth, K_new
    
    def get_zaxis_pose(self):
       pass
   
    def coarse_pose(self, color, depth, currPoses):
        # t = time.time()
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        rgb, K_new = color, self.K
        t = time.time()
        poses, vis = self.est.coarse(
            K=K_new,
            rgb=rgb,
            depth=depth,
            last_o2c_s=currPoses,
        )
        if self.debug>=2 and vis is not None:
            imageio.imwrite(f'{self.debug_dir}/vis_score_{self.i:04d}.png', vis)
            self.i += 1
        print(f'coarse_pose_i = {self.i}')
        print(f"[FoundationPose] Coarse time: {time.time() - t:.3f}s")
        return poses
    
    def coarse_refine(self, color, depth, track_center, center=None):
        self.H, self.W = color.shape[:2]
        # rgb, depth, K_new = self.crop_image(color, depth, samCenter) 
        rgb, K_new = color, self.K
        
        sample_poses = self.rot_grid.copy()
        sample_poses[:, :3, 3] = track_center
        pose = self.est.coarse_refine_register(
            K=K_new, 
            rgb=rgb, 
            depth=depth, 
            poses=sample_poses,
            iteration=self.est_refine_iter,
        )
        
        if self.debug>=1:
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            center_pose = pose@np.linalg.inv(self.to_origin)
            vis = draw_posed_3d_box(self.K, img=color, ob_in_cam=center_pose, bbox=self.bbox)
            vis = draw_xyz_axis(color, ob_in_cam=pose, scale=0.1, K=self.K, thickness=1, transparency=0, is_input_rgb=True)
            
            if center is not None:
                show_rgb = cv2.rectangle(
                    rgb, 
                    (int(center[0]-5), int(center[1]-5)), 
                    (int(center[0]+5), int(center[1]+5)), 
                    (0, 255, 0), 2
                )
                imageio.imwrite(f'{self.debug_dir}/center.png', show_rgb)
            ## test
            # T_trans_to_origin = np.eye(4)
            # T_trans_to_origin[:3, 3] = - pose[:3, 3]
            # T_rot_z = np.eye(4)
            # T_rot_z[:3, :3] = R.from_euler('z', np.pi/2).as_matrix()
            # T_trans_back = np.eye(4)
            # T_trans_back[:3, 3] = pose[:3, 3]
            # tmp_pose = T_trans_back @ T_rot_z @ T_trans_to_origin @ pose
            
            # pixel_hull = self.bbox_project_pixel_hull(tmp_pose) # 
            # mask = self.pixel_hull_2_mask(pixel_hull)
            # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # vis = cv2.addWeighted(vis, 0.5, mask[...,None], 0.5, 0) # bug
            
            ros_img = self.bridge.cv2_to_imgmsg(vis, encoding='rgb8')
            self.img_with_pose_pub_.publish(ros_img)
            
            if self.debug >= 2:
                # debug_dir = '/home/niloiv/pose_estimate_new/src/Dynamic_6D/object_pose_estimator/debug/FoundationPose'
                # crop_image_dir = os.path.join(self.debug_dir, 'crop_image')
                # os.makedirs(crop_image_dir, exist_ok=True)
                imageio.imwrite(f'{self.debug_dir}/track_vis/{self.id:04d}.png', vis)
                imageio.imwrite(f'{self.debug_dir}/crop_image/{self.id:04d}.png', rgb)
            self.id += 1
            
        return pose
        
    def refine_pyramid(self, color, depth, currPoses, last_obj2cam):
        self.H, self.W = color.shape[:2]
        # rgb, depth, K_new = self.crop_image(color, depth, samCenter) 
        rgb, K_new = color, self.K
        pose, vis = self.est.track_pyramid(
            K=K_new,
            rgb=rgb,
            depth=depth,
            last_o2c_s=currPoses,
        )
        if self.debug>=2 and vis is not None:
            for idx, vi in enumerate(vis):
                imageio.imwrite(f'{self.debug_dir}/vis_pyramid_{self.i:04d}_{idx}.png', vi)
            self.i += 1
        if self.debug>=1:
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            
            ### 上一帧的pose可视化
            center_pose_last = last_obj2cam@np.linalg.inv(self.to_origin)
            last_color = (255,0,255)
            vis = draw_posed_3d_box(self.K, img=color, ob_in_cam=center_pose_last, bbox=self.bbox, line_color=last_color, alpha=0.0)
            # vis = draw_posed_3d_box(self.K, img=color, ob_in_cam=center_pose_last, bbox=self.bbox, line_color=last_color)
            # vis = draw_xyz_axis(color, ob_in_cam=last_obj2cam, scale=0.1, K=self.K, thickness=1, transparency=0, is_input_rgb=True)   
            
            ### 当前一帧的pose可视化
            center_pose = pose@np.linalg.inv(self.to_origin)
            vis = draw_posed_3d_box(self.K, img=vis, ob_in_cam=center_pose, bbox=self.bbox)
            vis = draw_xyz_axis(vis, ob_in_cam=pose, scale=0.1, K=self.K, thickness=1, transparency=0, is_input_rgb=True)
            
            ros_img = self.bridge.cv2_to_imgmsg(vis, encoding='rgb8')
            self.img_with_pose_pub_.publish(ros_img)
            
            if self.debug >= 2:
                imageio.imwrite(f'{self.debug_dir}/track_vis/{self.id:04d}.png', vis)
            self.id += 1
        print(f'refine_pyramid = {self.i}')
        print(f'currPoses.shape = {currPoses.shape}')
        # print(f'refine_pyramid res = {pose}')
        return pose