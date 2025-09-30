import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../include/FoundationPose'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../include/Utils'))

import rospy
import torch
import enum
import numpy as np
import time
from cv_bridge import CvBridge
import message_filters
import sensor_msgs.msg, geometry_msgs.msg, visualization_msgs.msg, std_msgs.msg
from std_msgs.msg import Int32, Bool
from nav_msgs.msg import Odometry
from collections import deque
import cv2
import PIL
import imageio
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
# from pykalman import KalmanFilter
from tf.transformations import euler_from_matrix, euler_from_quaternion, quaternion_from_matrix, quaternion_from_euler, quaternion_inverse, quaternion_multiply, quaternion_matrix
from geometry_msgs.msg import Quaternion
from Utils import draw_posed_3d_box, draw_xyz_axis, project_3d_box_to_2d_box
from matplotlib.widgets import RectangleSelector

from foundationpose import _foundationpose_
from groundingDINO import _grounding_dino_
# from SAM2 import _sam2_
# from yolo import _yolo_
from utils import *
# from coarse_model import PosePredictor
from mixformer_tracker import _mixformer_tracker_
from kalman_filter_6d import KalmanFilter6D

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autocast(device_type="cuda", dtype=torch.float32).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
class STATES(enum.Enum):
        INIT = 0
        GD_Setting = 1
        WAITING_FP = 2
        TRACKING = 3
        
USE_PCA = False
USE_DINO = False

class poseEstimator:
    def __init__(self, config):
        self.bridge = CvBridge()
        self.STATE = STATES.INIT
        
        self.crop_ratio = config['PoseEstimator']['crop_ratio']
        self.mesh_files = config['mesh_files']
        self.taskTexts = config['taskTexts']
        assert len(self.mesh_files) == len(self.taskTexts), 'The number of mesh files and task texts should be the same'
        
        self.currTaskText = self.taskTexts[0]
        self.currMeshFile = self.mesh_files[0]
        
        # needed module
        self.foundationpose = _foundationpose_(config, self.currMeshFile)
        # self.sam2 = _sam2_(config)
        self.grounding_dino = _grounding_dino_(config, device)
        # self.yolo = _yolo_(config)
        # self.rot_matcher = rotationMatcher.rotMatcher(config, self.foundationpose.K)
        # self.pose_predictor = PosePredictor(config, self.foundationpose.K)
        self.mixformer_tracker = _mixformer_tracker_()
        
        ## module setting
        self.box_threshold = config['GroundingDINO']['box_threshold']
        self.text_threshold = config['GroundingDINO']['text_threshold']
        self.avoid_detect = config['aviod_detect']
        
        ## pose estimator setting
        self.init_window_length = config['PoseEstimator']['init_window_length']
        self.window_length = config['PoseEstimator']['window_length']
        self.occluded_threshold = config['PoseEstimator']['occluded_threshold']
        self.img_crop_scale = config['PoseEstimator']['img_crop_scale']
        self.last_obj2cam = np.eye(4)

        # 临时debug话题，发布KF滤波结果
        self.kf_predict_pose_pub = rospy.Publisher("/KFImage", sensor_msgs.msg.Image, queue_size=10)

        ###KF for pose
        self.R = 0.1 * np.eye(6)
        self.R_limit = 4
        self.Q = 0.01*np.eye(6)
        self.P_upd  = 0.01*np.eye(6)
        self.pose_obv = np.zeros(6)
        self.pose_pre = np.zeros(6)
        self.P_pre = 0.01*np.eye(6)
        self.first_frame = True#判断KF是否需要初始化
        self.i = 0
        self.odom_msg = None
        self.pose_matrix = None
        self.angular_vel_last = None
        self.odom_file_path = "odom_data.txt"  # 定义文件路径

        ###KF for z axis
        # self.R_z = 0.05 * np.eye(3)
        # self.Q_z = 0.6*np.eye(3)
        self.R_z = 0.5 * np.eye(3)
        self.Q_z = 0.01*np.eye(3)
        self.P_upd_z  = 1*np.eye(3)
        self.pose_obv_z = np.zeros(3)
        self.pose_pre_z = np.zeros(3)
        self.P_pre_z = 0.3*np.eye(3)
        
        ### TODO::FIXME, maybe you need to combine IMU data here
        self.state_pub_= rospy.Publisher('state_machine_state', Int32, queue_size=10)
        rgb_sub_ = message_filters.Subscriber(config['topics']['rgb'], sensor_msgs.msg.Image)
        depth_sub_ = message_filters.Subscriber(config['topics']['depth'], sensor_msgs.msg.Image)
        odometry_sub_ = message_filters.Subscriber(config['topics']['odometry'], Odometry)
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub_, depth_sub_, odometry_sub_], 1, config['PoseEstimator']['sync_time']) ## TODO::FIX the queue size
        
        ts.registerCallback(self.rgb_depth_vins_callback_)
        # self.timer = rospy.Timer(rospy.Duration(0.1), self.publish_pose_world)
        self.odom_sub_ = rospy.Subscriber(config['topics']['odometry'], Odometry, self.set_odom_callback)
        ### camera info
        self._cam_info_sub_ = rospy.Subscriber(config['topics']['cam_info'], sensor_msgs.msg.CameraInfo, self.cam_info_callback_)
        self.traking_pub_ = rospy.Publisher(config['topics']['traking_result'], geometry_msgs.msg.PoseStamped, queue_size=10)
        self.toggle_play_pub_ = rospy.Publisher("/toggle_play", Bool, queue_size=10)
        
        ### for debug
        self.debug = config['debug']
        self.debug_dir = config['debug_dir'] + '/poseEstimator'
        if self.debug>=1:
            os.makedirs(f'{self.debug_dir}', exist_ok=True)
        os.system(f'rm -rf {self.debug_dir}/*')
        self.gd_idx = 0
        self.sam2_idx = 0
        self.closest_img_idx  = 0
        self.global_first_frame = True
        # self.last_timestamp = None
        
        
        ### other setting
        self.missing_frame = 0
        self.start_time = time.time()
        self._rgb_depth_que_ = deque(maxlen=self.init_window_length)
        self.pose_world_que_ = deque(maxlen=2)
        self.acc_world_que_ = deque(maxlen=5)
        self.pose_world = None
        self.last_timestamp = None
        self.cam_info_done = False
        # self.pose_predictor.warmup()
        rospy.sleep(0.5)  # 等待 subscriber 连接
        rospy.loginfo(" Pose Estimator initialization done")
        print(f'\033[32m############ CONFIG ##############\033[0m')
        print(f'\033[32m####### NO_VINS = {NO_VINS} ######\033[0m')
        print(f'\033[32m# NO_2D_TRACKER = {NO_2D_TRACKER} #########\033[0m')
        print(f'\033[32m######### NO_KF = {NO_KF} ########\033[0m')
        self.toggle_play_pub_.publish(Bool(data=True)) # start playing
        # self.toggle_play_pub_.publish(Bool(data=True)) # start playing
        # self.toggle_play_pub_.publish(Bool(data=True)) # start playing
        # self.toggle_play_pub_.publish(Bool(data=True)) # start playing
        rospy.loginfo("##### Pose Estimator Ready #####")
    
    def rgb_depth_vins_callback_(self, rgb, depth_msg, cam2world_pose):
        # rospy.loginfo("Pose Estimator: Data received")
        if not self.cam_info_done:
            print("\033[31mCam_Info is not Done\033[0m")
            return
        timestamp = rgb.header.stamp
        # timestamp = depth_msg.header.stamp
        
        ### no vins
        if NO_VINS:
            tmp = geometry_msgs.msg.Pose()
            tmp.position.x = 0
            tmp.position.y = 0
            tmp.position.z = 0
            tmp.orientation.x = 0
            tmp.orientation.y = 0
            tmp.orientation.z = 0
            tmp.orientation.w = 1
            cam2world_pose.pose.pose = tmp
        ###
        rgb = self.bridge.imgmsg_to_cv2(rgb, 'bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough') # in Sim： 32FC1, in Real: 16UC1
        if depth_msg.encoding == '16UC1':
            depth = depth.astype(np.float32) / 1000.0
        rgb = self.foundationpose.get_color(rgb)
        depth = self.foundationpose.get_depth(depth)
        position = np.array([cam2world_pose.pose.pose.position.x, cam2world_pose.pose.pose.position.y, cam2world_pose.pose.pose.position.z])
        orientation = np.array([cam2world_pose.pose.pose.orientation.x, cam2world_pose.pose.pose.orientation.y, cam2world_pose.pose.pose.orientation.z, cam2world_pose.pose.pose.orientation.w])
        ### TODO  return here to jump the following code
        
        if self.last_timestamp is not None and timestamp - self.last_timestamp < rospy.Duration(0.01):
            return
        
        ### TODO::FIXME maybe you need to combine IMU data here
        self._rgb_depth_que_.append((
            self.edge_mask_with_scale(rgb), 
            self.edge_mask_with_scale(depth), 
            position,
            orientation,
            timestamp))
        # self.last_timestamp = timestamp
        
    def run_dynamic_object(self):
        rospy.loginfo("### Pose Estimator is running ###")
        kf = KalmanFilter6D(0.05)
        is_tracking = False
        kf_mean, kf_covariance = None, None
        
        while not rospy.is_shutdown():
            self.start_time = time.time()
            current_time = rospy.Time.now()
            # rospy.loginfo(f'Current State: {self.STATE}')
            rgb, depth, position, orientation, timestamp = None, None, None,None, None
            try:
                rgb, depth, position, orientation, timestamp = self._rgb_depth_que_.popleft() ## TODO::FIXME, maybe you need to combine IMU data here
                time_diff = current_time - timestamp
                # if self.last_timestamp is not None:
                #     if (timestamp - self.last_timestamp).to_sec() < 1e-5:
                #         rospy.loginfo("Pose Estimator: Duplicate timestamp, discarding frame")
                #         continue
                time_diff_seconds = time_diff.to_sec()
                ### TODO::FIXME get a more proper parameter here
                # if(time_diff_seconds > 0.5): ##删除0.5s前的帧
                #     rgb, depth, position, orientation, timestamp = self._rgb_depth_que_.popleft()
                self.missing_frame = 0
                # rospy.loginfo("Pose Estimator: No data received")
            except: ## necessary
                self.missing_frame += 1
                if self.missing_frame > 10:
                    # rospy.loginfo("Pose Estimator: No data received")
                    time.sleep(0.01)
                    continue

            if rgb is None or depth is None:
                # rospy.loginfo("Pose Estimator: No data received")
                continue
            width, height = rgb.shape[:2][::-1]
            pose = None
            
            T = self.quaternion2T(orientation, position)
            print(f'\033[32m[PoseEstimator] T = \n{T}\033[0m')
            
            self.state_pub_.publish(self.STATE.value) #发布状态机状态
            ### FSM
            if self.STATE == STATES.INIT:
                self.STATE = STATES.GD_Setting
                self.toggle_play_pub_.publish(Bool(data=False)) # stop playing
                gdRes = None
                if USE_DINO:
                    gdRes = self.getGDINO(rgb)
                else:
                    gdRes = self.getByClick(rgb)
                    print(f'[getByClick] gdRes = {gdRes}')
                ## miss
                if gdRes is None:
                    self.STATE = STATES.INIT
                    self.toggle_play_pub_.publish(Bool(data=True)) # start playing
                    continue
                ## get gdRes
                bbox = gdRes.tolist()
                # bbox, conf = self.yolo.predict(rgb, self.currTaskText)
                
                if bbox is None or len(bbox) == 0:
                    rospy.logwarn("No object detected")
                    self.STATE = STATES.INIT
                    self.toggle_play_pub_.publish(Bool(data=True)) # start playing
                else:
                    rospy.logdebug("Object detected")
                    if self.debug >= 1:
                        tmp = rgb.copy()
                        pt1 = (int(bbox[0]), int(bbox[1]))
                        pt2 = (int(bbox[2]), int(bbox[3]))
                        cv2.rectangle(tmp, pt1, pt2, (0, 255, 0), 2)
                        cv2.imwrite(f'{self.debug_dir}/gd_{self.gd_idx}.jpg', tmp)
                        self.gd_idx += 1
                    bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]
                    # bbox = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]], dtype=np.float32)
                    self.mixformer_tracker.init(rgb, bbox)
                    # self.sam2.load_first_frame(rgb)
                    # self.sam2.add_new_prompt(
                    #     frame_idx=0, 
                    #     obj_id=0, 
                    #     bbox=bbox)
                    
                    self.STATE = STATES.WAITING_FP
                    self.toggle_play_pub_.publish(Bool(data=True)) # start playing
                    self._rgb_depth_can2world_que = deque(maxlen=self.window_length)
            
            elif self.STATE == STATES.WAITING_FP:
                self.toggle_play_pub_.publish(Bool(data=False)) # stop playing
                # start = time.time()
                state = self.mixformer_tracker.track(rgb, depth) # xywh
                end = time.time()

                # rospy.logwarn(f"track;{end-start}")
                if state is None:
                    rospy.logdebug('No object detected')
                    self.STATE = STATES.INIT
                    self.toggle_play_pub_.publish(Bool(data=True)) # start playing
                    continue
                bbox_mask = bbox2mask(state, width, height)
                
                # out_obj_ids, out_mask_logits = self.sam2.track(rgb)
                # bbox_mask = np.zeros((height, width, 1), dtype=np.uint8)
                # for i in range(0, len(out_obj_ids)):
                #     out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(
                #         np.uint8
                #     ) * 255
                #     bbox_mask = cv2.bitwise_or(bbox_mask, out_mask)

                
                # _, track_Position = partialPcd2mainAxis(bbox_mask, depth, self.foundationpose.K)
                # bbox_center = np.array([state[0] + state[2] / 2, state[1] + state[3] / 2])
                # track_Position_uv = self.project_2d_to_3d(depth, self.foundationpose.K, bbox_center)
                # # track_Position[:2] = track_Position_uv[:2]
                # track_Position[:] = track_Position_uv[:]
                track_Position = getTrackPosition(bbox_mask, depth, self.foundationpose.K)
                print(f'[PoseEstimator] track_Position = {track_Position}')
                track_center = self.project_3d_to_2d(track_Position)
                if track_Position is None or track_center is None:
                    rospy.logdebug('No object detected')
                    self.STATE = STATES.INIT
                    self.toggle_play_pub_.publish(Bool(data=True)) # start playing
                    continue
                # crop_rgb = rgb_crop_with_bbox(rgb, state, self.crop_ratio)
                # obj2cams = self.pose_predictor.init_query(crop_rgb)
                # obj2cams[:, :3, 3] = track_Position
                # print(obj2cams)
                # pose = self.foundationpose.track_with_last_multiRotation(rgb, depth, track_center, np.array(obj2cams))

                pose = self.foundationpose.coarse_refine(rgb, depth, track_Position)
                
                print(f"INIT Pose={pose}")
                # self.pose_world = T @ pose
                # # self.pose_obv = homogeneous_to_xyz_euler(self.pose_world)
                # self.pose_obv = homogeneous_to_quaternion(self.pose_world)
                self.STATE = STATES.TRACKING
                self.toggle_play_pub_.publish(Bool(data=True)) # start playing
                
            elif self.STATE == STATES.TRACKING:
                last_obj2cam = self.get_curr_guess_obj2cam_pose(self.last_obj2world, T)
                
                start = time.time()
                if not NO_2D_TRACKER:
                    #########
                    state = self.mixformer_tracker.track(rgb, depth) # xywh
                    end = time.time()

                    rospy.logwarn(f"track : {end-start}")
                    if state is None:
                        rospy.logdebug('No object detected')
                        self.STATE = STATES.INIT
                    bbox_mask = bbox2mask(state, width, height)
                    
                    #############
                    # out_obj_ids, out_mask_logits = self.sam2.track(rgb)
                    # bbox_mask = np.zeros((height, width, 1), dtype=np.uint8)
                    # for i in range(0, len(out_obj_ids)):
                    #     out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(
                    #         np.uint8
                    #     ) * 255
                    # bbox_mask = cv2.bitwise_or(bbox_mask, out_mask)
                    
                    print(f'[PoseEstimator] bbox2mask Time = {time.time() - start}')
                    # _, track_Position = partialPcd2mainAxis(bbox_mask, depth, self.foundationpose.K)
                    # bbox_center = np.array([state[0] + state[2] / 2, state[1] + state[3] / 2])
                    # track_Position_uv = self.project_2d_to_3d(depth, self.foundationpose.K, bbox_center)
                    # # track_Position[:2] = track_Position_uv[:2]
                    # track_Position[:] = track_Position_uv[:]
                    track_Position = getTrackPosition(bbox_mask, depth, self.foundationpose.K)
                    print(f'[PoseEstimator] getTrackPosition Time = {time.time() - start}')
                    print(f'[PoseEstimator] track_Position = {track_Position}')
                    # track_Position_world = (T @ np.append(track_Position, 1))[:3]
                    track_center = self.project_3d_to_2d(track_Position)
                    if track_Position is None or track_center is None:
                        rospy.logdebug('No object detected')
                        # self.STATE = STATES.INIT
                        continue
                    # crop_rgb = rgb_crop_with_bbox(rgb, state, self.crop_ratio)
                else:
                    track_Position = last_obj2cam[:3, 3]
                print(f'[PoseEstimator] 2D_TRACKER Time = {time.time() - start}')
                if not NO_KF:
                    dt = (timestamp - self.last_timestamp).to_sec()
                    print(f'dt = {(timestamp - self.last_timestamp).to_sec()}')
                    kf.update_dt(dt)
                    
                    kf_mean, kf_covariance = kf.predict(kf_mean, kf_covariance)
                    kf_mean, kf_covariance = kf.update(kf_mean, kf_covariance, get_6d_pose_arr_from_mat(self.last_obj2world))
                    guess_pose_world = get_mat_from_6d_pose_arr(kf_mean[:6])
                    guess_pose = self.get_curr_guess_obj2cam_pose(guess_pose_world, T)
                    guess_pose[:3, 3] = track_Position
                else:
                    guess_pose = last_obj2cam.copy()
                    guess_pose[:3, 3] = track_Position
                pose = self.foundationpose.track_with_curr_guess(rgb, depth, track_center, guess_pose, last_obj2cam)
            
            else: ## wrong case
                pass
            
            if pose is not None:
                # self.publish_traking_result(pose, timestamp)
                self.last_obj2world = T @ pose

                if not is_tracking:
                    kf_mean, kf_covariance = kf.initiate(get_6d_pose_arr_from_mat(self.last_obj2world))
                # else:
                #     kf_mean, kf_covariance = kf.predict(kf_mean, kf_covariance)
                
                if not NO_2D_TRACKER:
                    center_pose = pose@np.linalg.inv(self.foundationpose.to_origin)
                    color = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                    bbox_2d = project_3d_box_to_2d_box(self.foundationpose.K, img=color, ob_in_cam=center_pose, bbox=self.foundationpose.bbox, line_color = (255,0,0))
                    self.mixformer_tracker.update_template(bbox = bbox_2d, K=self.foundationpose.K, img=rgb)
                
            self.last_timestamp = timestamp
            self.show_curr_state()          

    def zAxis_KF(self, fpAxis, samAxis, omega, dt):
        R = rodrigues_rotation_matrix(omega, omega*dt)
        # R = np.eye(3)
        axis_pre = np.dot(R, fpAxis)
        axis_pre = axis_pre/np.linalg.norm(axis_pre) 
        # print(f"axis_pre:{axis_pre}")     
        self.P_pre_z = R @ self.P_upd_z @ R.T + self.R_z #预测方差
        self.K_gain_z = self.P_pre_z @ np.linalg.inv(self.P_pre_z + self.Q_z)
        axis_filter = axis_pre + self.K_gain_z @ (samAxis - axis_pre)
        self.P_upd_z = (np.eye(3) - self.K_gain_z) @ self.P_pre_z
        return axis_filter

    def show_curr_state(self):
        # rospy.loginfo('###########################')
        # rospy.loginfo(f'Current State: {self.STATE}')
        rospy.loginfo(f'Time Cost: {time.time() - self.start_time} s')
        rospy.loginfo('###############################################')
        return
    
    def get_curr_guess_obj2cam_poses(self, last_obj2world_poses, T):
        """
        输入上一帧的物体到相机的位姿，返回当前帧的物体到相机的位姿的猜测值。
        """
        ### TODO::FIXME with vio
        # poses_in_cam = np.zeros_like(last_obj2world_poses)
        rotations_in_cam = np.zeros((last_obj2world_poses.shape[0], 3, 3))
        T_inv = np.linalg.inv(T) 
        for i in range(last_obj2world_poses.shape[0]):
            pose_in_cam = T_inv @ last_obj2world_poses[i]
            rotations_in_cam[i] = pose_in_cam[:3, :3]
        return rotations_in_cam
    
    def calOdom(self):
        if len(self.pose_world_que_) == 2:
            if self.pose_world_que_[0] is not None and self.pose_world_que_[1] is not None:
                dt = self.pose_world_que_[1][0].to_sec() - self.pose_world_que_[0][0].to_sec()
                print(f'[calOdom {self.i}] dt = {dt}')
                # 创建Odometry消息
                self.odom_msg = Odometry()
                
                # 提取当前和上一次的位置和方向
                R = self.pose_world_que_[1][1][:3, :3]
                t = self.pose_world_que_[1][1][:3, 3]
                R_last = self.pose_world_que_[0][1][:3, :3]
                t_last = self.pose_world_que_[0][1][:3, 3]
                
                # 填充位置信息
                self.odom_msg.pose.pose.position.x = t[0]
                self.odom_msg.pose.pose.position.y = t[1]
                self.odom_msg.pose.pose.position.z = t[2]
                
                # 将旋转矩阵转换为四元数
                quat_current = quaternion_from_matrix(self.pose_world_que_[1][1]) #xyzw
                quat_last = quaternion_from_matrix(self.pose_world_que_[0][1])
                
                # Normalize the quaternions
                quat_current /= np.linalg.norm(quat_current)
                quat_last /= np.linalg.norm(quat_last)

                # 处理四元数符号一致性
                if np.dot(quat_current, quat_last) < 0:
                    print(f'\033[31m [calOdom] dot result wrong\033[0m')
                    quat_current = -quat_current  # 确保取最短路径

                # if quat_current[3] < 0:
                #     quat_current = -quat_current
                # if quat_last[3] < 0:
                #     quat_last = -quat_last
                if self.debug >= 3:
                    with open('pose.txt', 'a') as file:
                            file.write(f"pose_{self.i} :{quat_current}\n")
                # 填充方向信息
                self.odom_msg.pose.pose.orientation = Quaternion(*quat_current)
                
                # 计算四元数差分（旋转）
                ang_1 = euler_from_quaternion(quat_current)
                ang_2 = euler_from_quaternion(quat_last)
                print(f'ang_1 = {ang_1}')
                print(f'ang_2 = {ang_2}')
                quat_inv_last = quaternion_inverse(quat_last)
                q_diff = quaternion_multiply(quat_current, quat_inv_last)
                q_diff = self.normalize_quaternion(q_diff)  # 确保四元数归一化
                # 提取旋转轴和旋转角（这里假设旋转是小角度的）
                axis, angle = self.quaternion_to_axis_angle(q_diff)
                if (angle > np.pi):
                    angle = 2 * np.pi - angle
                    axis = -axis
                print(f"angle_{self.i}={angle}")
                
                # 检查时间差是否有效
                if dt > 1e-5:
                    # 计算角速度（假设旋转是均匀发生的）
                    angular_velocity = axis * (angle / dt)
                else:
                    # 如果时间差为零，则角速度也为零
                    angular_velocity = np.zeros(3)
                
                 # 检查时间差是否有效
                if dt > 1e-5:
                    # 计算线速度
                    linear_velocity = (t - t_last) / dt
                else:
                    # 如果时间差为零，则速度也为零
                    linear_velocity = np.zeros(3)
                
                low_thre = np.pi * 30/180
                high_thre = np.pi * 270/180
                for i, item in enumerate(angular_velocity):
                    if item < low_thre or item > high_thre:
                        angular_velocity[i] = 0
                print(f'[calOdom {self.i}] angular_velocity = {angular_velocity}')
                #计算角加速度
                if self.angular_vel_last is None:
                    # self.angular_vel_last = angular_velocity
                    angular_acc = np.zeros(3)
                else:
                    
                    # 检查时间差是否有效
                    if dt > 1e-5:
                        angular_acc = (angular_velocity-self.angular_vel_last)/dt
                    else:
                        # 如果时间差为零，则速度也为零
                        angular_acc = np.zeros(3)
                self.angular_vel_last = angular_velocity
                # self.acc_world_que_.append(angular_acc)
                acc_sum = np.zeros(3)
                # print(f"self.acc_world_que_ length: {len(self.acc_world_que_)}")
                if len(self.acc_world_que_) != 0:
                    for acc in self.acc_world_que_:
                        acc_sum += acc
                    avg_acc = (acc_sum+angular_acc) / (len(self.acc_world_que_))
                else:
                    avg_acc = angular_acc
                self.acc_world_que_.append(avg_acc)
                print(f'[calOdom {self.i}] avg_acc = {avg_acc}')
                # 填充角速度信息
                self.odom_msg.twist.twist.angular.x = angular_velocity[0]
                self.odom_msg.twist.twist.angular.y = angular_velocity[1]
                self.odom_msg.twist.twist.angular.z = angular_velocity[2]
            
                self.odom_msg.twist.twist.linear.x = linear_velocity[0]
                self.odom_msg.twist.twist.linear.y = linear_velocity[1]
                self.odom_msg.twist.twist.linear.z = linear_velocity[2]

                self.odom_msg.header.stamp = self.pose_world_que_[1][0]

                # 将odom_msg保存到txt文件
                if self.debug >= 3:
                    with open(self.odom_file_path, "a") as file:
                        file.write(f"self.odom_msg_{self.i}\n :"
                                f"{self.odom_msg.header.stamp.to_sec()} "
                                f"{self.odom_msg.pose.pose.position.x} "
                                f"{self.odom_msg.pose.pose.position.y} "
                                f"{self.odom_msg.pose.pose.position.z} "
                                f"{self.odom_msg.pose.pose.orientation.x} "
                                f"{self.odom_msg.pose.pose.orientation.y} "
                                f"{self.odom_msg.pose.pose.orientation.z} "
                                f"{self.odom_msg.pose.pose.orientation.w} "
                                f"{self.odom_msg.twist.twist.linear.x} "
                                f"{self.odom_msg.twist.twist.linear.y} "
                                f"{self.odom_msg.twist.twist.linear.z} "
                                f"{self.odom_msg.twist.twist.angular.x} "
                                f"{self.odom_msg.twist.twist.angular.y} "
                                f"{self.odom_msg.twist.twist.angular.z}\n")
                return linear_velocity, angular_velocity, avg_acc
            else:
                return None,None,None
        else:
            return None,None,None

    def set_odom_callback(self, imu_data):
        self.odom_timestamp = imu_data.header.stamp #当前imu时间戳
        self.position = np.array([imu_data.pose.pose.position.x, imu_data.pose.pose.position.y, imu_data.pose.pose.position.z])
        self.orientation = np.array([imu_data.pose.pose.orientation.x, imu_data.pose.pose.orientation.y, imu_data.pose.pose.orientation.z, imu_data.pose.pose.orientation.w])
        norm = np.linalg.norm(self.orientation)
        self.orientation = self.orientation/norm
        
    def KF_predict(self, timestamp):
        if(self.odom_msg is not None): 
            # 发布消息
            rospy.logwarn(f"[KF_predict]timestamp={timestamp}")
            rospy.logwarn(f"[KF_predict]self.odom_msg.header.stamp={self.odom_msg.header.stamp}")
            dt = (timestamp - self.odom_msg.header.stamp).to_sec()
            rospy.logwarn(f"[KF_predict]dt={dt}")
            t_predict = np.zeros(3)
            if dt > 0:
                # rospy.loginfo("IN!!!")
                if self.first_frame:
                    object_position = np.array([self.odom_msg.pose.pose.position.x, self.odom_msg.pose.pose.position.y, self.odom_msg.pose.pose.position.z])
                    orientation = self.odom_msg.pose.pose.orientation
                    self.first_frame = False
                    rospy.logwarn(f"self.first_frame:{self.first_frame}")
                else:
                    object_position = np.zeros(3)
                    object_position = self.pose_matrix[:3,3]
                    # r = self.pose_matrix[:3,:3]
                    orientation_np = quaternion_from_matrix(self.pose_matrix) 
                    orientation = Quaternion()
                    orientation.x = orientation_np[0]
                    orientation.y = orientation_np[1]
                    orientation.z = orientation_np[2]
                    orientation.w = orientation_np[3]
                # object_position = np.array([self.odom_msg.pose.pose.position.x, self.odom_msg.pose.pose.position.y, self.odom_msg.pose.pose.position.z])
                # orientation = self.odom_msg.pose.pose.orientation
                object_vel_linear = np.array([self.odom_msg.twist.twist.linear.x, self.odom_msg.twist.twist.linear.y, self.odom_msg.twist.twist.linear.z])

                t_predict[:3] = object_position[:3] + object_vel_linear[:3]*dt
                # t_predict[:3] = object_position[:3] + object_vel_linear[:3]*0.1
                # print(f"t_predict={t_predict}")
                
                # angular_velocity_quaternion = quaternion_from_euler(self.odom_msg.twist.twist.linear.x * dt / 2, 
                #                                                     self.odom_msg.twist.twist.linear.y * dt / 2,
                #                                                     self.odom_msg.twist.twist.linear.z * dt / 2)
                # angular_velocity_quaternion = (self.odom_msg.twist.twist.angular.x, 
                #                                 self.odom_msg.twist.twist.angular.y,
                #                                 self.odom_msg.twist.twist.angular.z,
                #                                 0)
                # q0 = (orientation.x, orientation.y, orientation.z, orientation.w)
                # angular_velocity_quaternion = 0.5*quaternion_multiply(q0, angular_velocity_quaternion)
                # #欧拉法
                # predicted_orientation = q0 + angular_velocity_quaternion * dt
                # predicted_orientation = quaternion_multiply(angular_velocity_quaternion, q0)
                # predicted_orientation = quaternion_multiply(predicted_orientation, np.conj(angular_velocity_quaternion))
                # predicted_orientation = predicted_orientation / np.linalg.norm(predicted_orientation)
                
                # # 计算预测的四元数方向
                # q0 = np.array([orientation.x, orientation.y, orientation.z, orientation.w])
                # angular_velocity = np.array([self.odom_msg.twist.twist.angular.x,
                #                             self.odom_msg.twist.twist.angular.y,
                #                             self.odom_msg.twist.twist.angular.z])
                # omega_norm = np.linalg.norm(angular_velocity)
                
                # if omega_norm > 0:
                #     theta = omega_norm * dt / 2.0
                #     sin_theta = np.sin(theta)
                #     cos_theta = np.cos(theta)
                #     delta_q = np.array([sin_theta * angular_velocity[0] / omega_norm,
                #                         sin_theta * angular_velocity[1] / omega_norm,
                #                         sin_theta * angular_velocity[2] / omega_norm,
                #                         cos_theta])
                # else:
                #     delta_q = np.array([0, 0, 0, 1])
                
                # # 使用四元数乘法更新方向
                # current_rotation = R.from_quat(q0)
                # delta_rotation = R.from_quat(delta_q)
                # predicted_rotation = current_rotation * delta_rotation
                # predicted_orientation = predicted_rotation.as_quat()

                # # 将旋转矩阵转换为四元数
                # quat_current = quaternion_from_matrix(self.pose_world_que_[1][1])
                # quat_last = quaternion_from_matrix(self.pose_world_que_[0][1])
                
                # # Normalize the quaternions
                # quat_current /= np.linalg.norm(quat_current)
                # quat_last /= np.linalg.norm(quat_last)
                # # 计算四元数的点积
                # dot = np.dot(quat_last, quat_current)
                
                # # 处理四元数符号一致性
                # if dot < 0:
                #     quat_current = -quat_current  # 确保取最短路径
                
                # # 计算插值系数
                # theta = np.arccos(np.clip(dot, -1, 1))
                # sin_theta = np.sin(theta)
                # if sin_theta < 1e-6:
                #     predicted_rotation = np.array(quat_current)
                # else:
                #     t0 = np.sin((1 - dt) * theta) / sin_theta
                #     t1 = np.sin(dt * theta) / sin_theta
                #     # 执行插值
                #     predicted_rotation = t0 * np.array(quat_last) + t1 * np.array(quat_current)
                # 使用角速度预测旋转
                angular_velocity = np.array([self.odom_msg.twist.twist.angular.x,
                                             self.odom_msg.twist.twist.angular.y,
                                             self.odom_msg.twist.twist.angular.z])
                angle = np.linalg.norm(angular_velocity) * dt
                if angle > 0:
                    axis = angular_velocity / np.linalg.norm(angular_velocity)
                    delta_rotation = R.from_rotvec(axis * angle)
                else:
                    delta_rotation = R.from_quat([0, 0, 0, 1])

                current_rotation = R.from_quat([orientation.x, orientation.y, orientation.z, orientation.w])
                predicted_rotation = current_rotation * delta_rotation
                predicted_rotation = predicted_rotation.as_quat()
                
                # 将四元数转换为欧拉角
                predicted_rotation = self.normalize_quaternion(predicted_rotation)
                predicted_orientation = R.from_quat(predicted_rotation)
                predicted_Eulur = predicted_orientation.as_euler('xyz')
                print(f"dt={dt}")
                
                # print(f"predicted_orientation:{predicted_orientation}")
                # rospy.logwarn(f"predicted_orientation={predicted_orientation}")
                # rotation_matrix = quaternion_matrix(predicted_orientation)[:3, :3]
                rotation_matrix = predicted_orientation.as_matrix()
                rotation_matrix = rotation_matrix[:3, :3]
                self.pose_matrix = np.eye(4)
                self.pose_matrix[:3, :3] = rotation_matrix
                self.pose_matrix[:3, 3] = t_predict

                self.pose_pre[:3] = t_predict#预测的平移
                # self.pose_pre[3:] = predicted_orientation #预测旋转
                self.pose_pre[3:] = predicted_Eulur#欧拉角  
                self.P_pre = self.P_upd + self.R #预测方差

                if self.debug >= 3:
                    with open('P_pre.txt', 'a') as file:
                        file.write(f"P_pre matrix_{self.i}:\n")
                    np.savetxt(file, self.P_pre, fmt='%.6f')
                print(f"predicted_orientation={predicted_orientation}")
                print(f"predicted_Eulur={predicted_Eulur}")
                cov_euler = self.P_pre[3:6, 3:6]
                samples = generate_independent_samples(predicted_Eulur, cov_euler)
                # samples = generate_sigma_points(predicted_Eulur, cov_euler)
                # print(f"samples={samples}")
                poses_guess = [euler_to_rotation_matrix(sample, t_predict) for sample in samples]
                # poses_guess = np.expand_dims(self.last_obj2world,axis=0)

                return  poses_guess
                

    def KF_update(self,track_Position_world):
        self.K_gain = self.P_pre@np.linalg.inv(self.P_pre+self.Q)
        # print(f"self.pose_pre={self.pose_pre}")
        vector_after_filter = self.pose_pre + self.K_gain @ (self.pose_obv - self.pose_pre)
        I = np.eye(6)
        self.P_upd = (I - self.K_gain) @ self.P_pre
        self.pose_matrix[:3,3] = track_Position_world
        euler_angles = vector_after_filter[3:]
        rotation = R.from_euler('xyz', euler_angles)
        # quaternion = vector_after_filter[3:]
        # rotation = R.from_quat(quaternion)
        self.pose_matrix[:3,:3] =  rotation.as_matrix()
        if self.debug >= 3:
            try:
                with open('P_upd.txt', 'a') as file:
                    file.write(f"P_upd matrix_{self.i}:\n")
                    np.savetxt(file, self.P_upd, fmt='%.6f')
                with open('K_gain.txt', 'a') as file:
                    file.write(f"K_gain matrix_{self.i}:\n")
                    np.savetxt(file, self.K_gain, fmt='%.6f')
            except Exception as e:
                print(f"An error occurred: {e}")
        # print(f"self.pose_pre:{self.pose_pre}")
        # print(f"vector_after_filter={vector_after_filter}")
        # print(f"self.pose_obv={self.pose_obv}")
        # print(f"self.K_gain={self.K_gain}")
        # print(f"pose_matrix:{self.pose_matrix}")
        # print(f"pose_world_obv:{self.pose_world}")

        # self.position = np.array([data.pose.pose.position.x, data.pose.pose.position.y, data.pose.pose.position.z])
        # self.orientation = np.array([data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w])
        # rotation_matrix_vins = quaternion_matrix(self.orientation)[:3, :3]
        
        # world2cam_matrix = np.eye(4)
        # world2cam_matrix = self.quaternion2T(self.orientation, self.position)


        # pose = np.linalg.inv(world2cam_matrix) @ self.pose_matrix
       

        # self.i+=1
        # if(self.i==10):
        #     exit()

    def KF(self):
        if(self.odom_msg is not None): 
            # 发布消息
            dt = (self.odom_timestamp - self.odom_msg.header.stamp).to_sec()
            # rospy.logwarn(f"dt={dt}")
            t_predict = np.zeros(3)
            if dt > 0:
                # rospy.loginfo("IN!!!")
                if self.first_frame:
                    object_position = np.array([self.odom_msg.pose.pose.position.x, self.odom_msg.pose.pose.position.y, self.odom_msg.pose.pose.position.z])
                    orientation = self.odom_msg.pose.pose.orientation
                    self.first_frame = False
                    rospy.logwarn(f"self.first_frame:{self.first_frame}")
                else:
                    object_position = np.zeros(3)
                    object_position = self.pose_matrix[:3,3]
                    # r = self.pose_matrix[:3,:3]
                    orientation_np = quaternion_from_matrix(self.pose_matrix) 
                    orientation = Quaternion()
                    orientation.x = orientation_np[0]
                    orientation.y = orientation_np[1]
                    orientation.z = orientation_np[2]
                    orientation.w = orientation_np[3]
                # object_position = np.array([self.odom_msg.pose.pose.position.x, self.odom_msg.pose.pose.position.y, self.odom_msg.pose.pose.position.z])
                # orientation = self.odom_msg.pose.pose.orientation
                object_vel_linear = np.array([self.odom_msg.twist.twist.linear.x, self.odom_msg.twist.twist.linear.y, self.odom_msg.twist.twist.linear.z])

                t_predict[:3] = object_position[:3] + object_vel_linear[:3]*dt
                # t_predict[:3] = object_position[:3] + object_vel_linear[:3]*0.1
                # print(f"t_predict={t_predict}")
                
                # angular_velocity_quaternion = quaternion_from_euler(self.odom_msg.twist.twist.linear.x * dt / 2, 
                #                                                     self.odom_msg.twist.twist.linear.y * dt / 2,
                #                                                     self.odom_msg.twist.twist.linear.z * dt / 2)
                # angular_velocity_quaternion = (self.odom_msg.twist.twist.angular.x, 
                #                                 self.odom_msg.twist.twist.angular.y,
                #                                 self.odom_msg.twist.twist.angular.z,
                #                                 0)
                # q0 = (orientation.x, orientation.y, orientation.z, orientation.w)
                # angular_velocity_quaternion = 0.5*quaternion_multiply(q0, angular_velocity_quaternion)
                # #欧拉法
                # predicted_orientation = q0 + angular_velocity_quaternion * dt
                # predicted_orientation = quaternion_multiply(angular_velocity_quaternion, q0)
                # predicted_orientation = quaternion_multiply(predicted_orientation, np.conj(angular_velocity_quaternion))
                # predicted_orientation = predicted_orientation / np.linalg.norm(predicted_orientation)
                
                # 计算预测的四元数方向
                q0 = np.array([orientation.x, orientation.y, orientation.z, orientation.w])
                angular_velocity = np.array([self.odom_msg.twist.twist.angular.x,
                                            self.odom_msg.twist.twist.angular.y,
                                            self.odom_msg.twist.twist.angular.z])
                omega_norm = np.linalg.norm(angular_velocity)
                
                if omega_norm > 0:
                    theta = omega_norm * dt / 2.0
                    sin_theta = np.sin(theta)
                    cos_theta = np.cos(theta)
                    delta_q = np.array([sin_theta * angular_velocity[0] / omega_norm,
                                        sin_theta * angular_velocity[1] / omega_norm,
                                        sin_theta * angular_velocity[2] / omega_norm,
                                        cos_theta])
                else:
                    delta_q = np.array([0, 0, 0, 1])
                
                # 使用四元数乘法更新方向
                current_rotation = R.from_quat(q0)
                delta_rotation = R.from_quat(delta_q)
                predicted_rotation = current_rotation * delta_rotation
                predicted_orientation = predicted_rotation.as_quat()
                print(f"predicted_orientation:{predicted_orientation}")
                rospy.logwarn(f"predicted_orientation={predicted_orientation}")
                rotation_matrix = quaternion_matrix(predicted_orientation)[:3, :3]

                self.pose_matrix = np.eye(4)
                # self.pose_matrix[:3, :3] = rotation_matrix
                # self.pose_matrix[:3, 3] = t_predict

                self.pose_pre[:3] = t_predict#预测的平移
                self.pose_pre[3:] = predicted_orientation #预测旋转
                self.P_pre = self.P_upd + self.R #预测方差
                self.K_gain = self.P_pre@np.linalg.inv(self.P_pre+self.Q)
                # print(f"self.pose_pre={self.pose_pre}")
                vector_after_filter = self.pose_pre + self.K_gain @ (self.pose_obv - self.pose_pre)
                I = np.eye(7)
                self.P_upd = (I - self.K_gain) @ self.P_pre
                self.pose_matrix[:3,3] = vector_after_filter[:3]
                # euler_angles = vector_after_filter[3:]
                # rotation = R.from_euler('xyz', euler_angles)
                quaternion = vector_after_filter[3:]
                rotation = R.from_quat(quaternion)
                self.pose_matrix[:3,:3] =  rotation.as_matrix()
                # print(f"self.pose_pre:{self.pose_pre}")
                # print(f"vector_after_filter={vector_after_filter}")
                # print(f"self.pose_obv={self.pose_obv}")
                # print(f"self.K_gain={self.K_gain}")
                # print(f"pose_matrix:{self.pose_matrix}")
                # print(f"pose_world_obv:{self.pose_world}")

                # self.position = np.array([data.pose.pose.position.x, data.pose.pose.position.y, data.pose.pose.position.z])
                # self.orientation = np.array([data.pose.pose.orientation.x, data.pose.pose.orientation.y, data.pose.pose.orientation.z, data.pose.pose.orientation.w])
                # rotation_matrix_vins = quaternion_matrix(self.orientation)[:3, :3]
                
                world2cam_matrix = np.eye(4)
                world2cam_matrix = self.quaternion2T(self.orientation, self.position)
                # world2cam_matrix[:3, :3] = rotation_matrix_vins
                # world2cam_matrix[:3, 3] = self.position
                # print(f"world2cam:{world2cam_matrix}")

                pose = np.linalg.inv(world2cam_matrix) @ self.pose_matrix
                # print(f"pose_cam:{pose}")
                # print(f"pose_cam_track:{self.pose_cam}")

                # print(self.pose_matrix)
                # if self.debug>=1:
                # color = cv2.cvtColor(self.color, cv2.COLOR_BGR2RGB)
                # center_pose = pose@np.linalg.inv(self.foundationpose.to_origin)
                # vis = draw_posed_3d_box(self.foundationpose.K, img=color, ob_in_cam=center_pose, bbox=self.foundationpose.bbox)
                # vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=self.foundationpose.K, thickness=3, transparency=0, is_input_rgb=True)
            
                # ros_img = self.bridge.cv2_to_imgmsg(vis, encoding='rgb8')
                # self.img_with_pose_pub_.publish(ros_img)
                # odom_track = Odometry()

                self.i+=1
                # if(self.i==10):
                #     exit()
            # self.pose_world_pub_.publish(self.odom_msg)
            # self.odom_msg_last = self.odom_msg

    # 辅助函数：将四元数转换为旋转轴和角度
    # def quaternion_to_axis_angle(self, q):
    #     # q应该是一个四元数[x, y, z, w]
    #     sin_half_angle = np.linalg.norm(q[:3])
    #     cos_half_angle = q[3]
    #     angle = 2 * np.arctan2(sin_half_angle, cos_half_angle)
    #     if sin_half_angle == 0:
    #         # 当角度为0时，旋转轴可以是任意单位向量，这里选择[1, 0, 0]
    #         return np.array([1, 0, 0]), 0
    #     axis = q[:3] / sin_half_angle
    #     return axis, angle
    def quaternion_to_axis_angle(self,q):
        q = q / np.linalg.norm(q)
        angle = 2 * np.arccos(q[3])  # 提取角度
        axis = q[:3] / np.sqrt(max(1 - q[3]**2, 1e-10))  # 避免除以零
        # 强制轴方向的一致性（例如确保第一个非零分量为正）
        if np.sum(axis) < 0:
            axis = -axis
        return axis, angle
    def normalize_quaternion(self, q):
        norm = np.linalg.norm(q)
        if norm == 0:
            return np.array([0, 0, 0, 1])  # 返回一个默认的单位四元数
        return q / norm
    
    def cam_info_callback_(self, data):
        """update camera info"""
        print(f'\033[31m Cam_Info Done! \033[0m')
        self.foundationpose.K = np.array(data.K).reshape(3, 3)
        self.cam_info_done = True
        self._cam_info_sub_.unregister()
    
    def get_mask_center(self, mask):
        non_zero_coords = np.argwhere(mask)
        centroid = np.mean(non_zero_coords, axis=0)
        centroid = np.array(centroid.astype(int))
        centroid=centroid[::-1]
        return centroid
    
    def getGDINO(self, rgb):
        rospy.logdebug('Getting grounding DINO')
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = PIL.Image.fromarray(rgb)
        text = ('').join(self.taskTexts + self.avoid_detect)
        inputs = self.grounding_dino.processor(rgb, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.grounding_dino.model(**inputs)
        results = self.grounding_dino.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[rgb.size[::-1]]
        )
        labels = results[0]['labels']
        bboxs = results[0]['boxes']
        scores = results[0]['scores']
        
        best_idx = -1
        best_score = -1
        for i in range(len(labels)):
            if labels[i] == self.currTaskText[:-1] and scores[i] > best_score:
                best_idx = i
                best_score = scores[i]
        if best_idx == -1:
            return None
        else:
            return bboxs[best_idx]   
    
    def edge_mask_with_scale(self, img):
        h, w = img.shape[:2]
        mask = np.ones((h, w), dtype=np.uint8) * 0 # all black
        target_h = int(h * self.img_crop_scale)
        target_w = int(w * self.img_crop_scale)
        x_start = (w - target_w) // 2
        y_start = (h - target_h) // 2
        mask[y_start:y_start+target_h, x_start:x_start+target_w] = 255
        return cv2.bitwise_and(img, img, mask=mask)
    
    def project_2d_to_3d(self, depth, K, xy):
        """
        Project a 2D point to 3D space using depth information.

        Args:
            depth: A 2D numpy array representing the depth image.
            K: A 3x3 numpy array representing the camera matrix.
            x: The x coordinate of the 2D point.
            y: The y coordinate of the 2D point.

        Returns:
            A 3D numpy array representing the 3D point.
        """
        ### TODO:: 为什么有时候这里的z会是0？？
        x, y = xy
        z = depth[int(y), int(x)]
        x = (x - K[0, 2]) * z / K[0, 0]
        y = (y - K[1, 2]) * z / K[1, 1]
        return np.array([x, y, z])

    def project_2d_to_3d_with_mask(self, depth, K, xy, mask):
        depth_masked = np.where(mask > 0, depth, 0)
        # 使用中值滤波平滑深度图（减少噪声）
        depth_masked = cv2.medianBlur(depth_masked.astype(np.float32), 5)
        
        # 获取深度图中非零元素的坐标
        rows, cols = np.where(depth_masked > 0)
        # 获取对应的深度值
        z = depth_masked[rows, cols]
        z = np.min(z)
        
        x, y = xy
        x = (x - K[0, 2]) * z / K[0, 0]
        y = (y - K[1, 2]) * z / K[1, 1]
        return np.array([x, y, z])
    
    def project_2d_to_3d_with_z(self, z, K, xy):
        x, y = xy
        x = (x - K[0, 2]) * z / K[0, 0]
        y = (y - K[1, 2]) * z / K[1, 1]
        return np.array([x, y, z])
    
    def get_curr_guess_obj2cam_pose(self, last_obj2world, T):
        """
        输入上一帧的物体到相机的位姿，返回当前帧的物体到相机的位姿的猜测值。
        """
        ### TODO::FIXME with vio
        return np.linalg.inv(T) @ last_obj2world 
    
    def quaternion2T(self, quaternion, translation):
        r = R.from_quat(quaternion)
        rot = r.as_matrix()
        T = np.eye(4)
        T[:3,:3] = rot
        T[:3,3] = translation
        return T

    def project_3d_to_2d(self, pos_3d)->list:
        if pos_3d is None:
            return None
        x = pos_3d[0] * self.foundationpose.K[0, 0] / pos_3d[2] + self.foundationpose.K[0, 2]
        y = pos_3d[1] * self.foundationpose.K[1, 1] / pos_3d[2] + self.foundationpose.K[1, 2]
        return np.array([x, y])
    
    def getByClick(self, rgb):
        coords = [None, None, None, None]
        def on_select(eclick, erelease):
            # 获取框选区域的坐标 (左下角和右上角的坐标)
            coords[0], coords[1], coords[2], coords[3] = int(eclick.xdata), int(eclick.ydata), int(erelease.xdata), int(erelease.ydata)
            print(f"Selected box: ({eclick.xdata}, {eclick.ydata}) to ({erelease.xdata}, {erelease.ydata})")
            print(f'coords = {coords}')
            rectangle_selector.set_active(False)
            plt.close()
        fig, ax = plt.subplots()
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        rectangle_selector = RectangleSelector(ax, on_select,
                                       useblit=True,     # 加速绘图
                                       button=[1],       # 鼠标左键拖动
                                       minspanx=5,       # 最小拖动跨度
                                       minspany=5)
        plt.show()
        if coords[0] is None or coords[1] is None or coords[2] is None or coords[3] is None:
            print('Invalid selection')
            return None
        return np.array(coords)
        
    def publish_traking_result(self, pose, timestamp):
        pose_msg = geometry_msgs.msg.PoseStamped()
        pose_msg.header.stamp = timestamp
        pose_msg.header.frame_id = 'World' ## TODO
        pose_msg.pose.position.x = pose[0, 3]
        pose_msg.pose.position.y = pose[1, 3]
        pose_msg.pose.position.z = pose[2, 3]
        quaternion = quaternion_from_matrix(pose)
        pose_msg.pose.orientation.x = quaternion[0]
        pose_msg.pose.orientation.y = quaternion[1]
        pose_msg.pose.orientation.z = quaternion[2]
        pose_msg.pose.orientation.w = quaternion[3]
        self.traking_pub_.publish(pose_msg)

if __name__ == "__main__":
    import yaml
    with open('../config/pose_estimator.yaml', 'r') as file:
        config = yaml.safe_load(file)
    log_level = rospy.DEBUG if config['debug'] >= 2 else rospy.INFO
    # log_level = rospy.INFO
    
    ## ABLATION_EXPERIMENT CONFIG SETTING
    NO_VINS         = config['ABLATION_EXPERIMENT']['NO_VINS']
    NO_2D_TRACKER   = config['ABLATION_EXPERIMENT']['NO_2D_TRACKER']
    NO_KF           = config['ABLATION_EXPERIMENT']['NO_KF']

    ## START NODE
    rospy.init_node('poseEstimator', log_level=log_level)
    pose_estimator = poseEstimator(config)
    pose_estimator.run_dynamic_object()