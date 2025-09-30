import numpy as np
import rospy
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R
from transformations import quaternion_from_matrix, quaternion_matrix

class KalmanFilter:
    def __init__(self, initial_covariance, process_noise, measurement_noise):
        # 初始化状态变量
        self.pose_matrix = None
        self.P_pre = initial_covariance  # 预测的协方差矩阵
        self.P_upd = initial_covariance  # 更新的协方差矩阵
        self.Q = measurement_noise  # 过程噪声
        self.R = process_noise  # 测量噪声
        self.K_gain = np.zeros((7, 7))  # 卡尔曼增益
        
        self.odom_msg = Odometry()
        self.odom_timestamp = None
        self.pose_obv = np.zeros(7)  # 观测到的位姿
        self.pose_pre = np.zeros(7)
        self.i = 0
        self.first_frame = True

    def quaternion2T(self, orientation, position):
        rotation_matrix = quaternion_matrix(orientation)[:3, :3]
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = position
        return transform_matrix

    def KF_predict(self, odom_msg, odom_timestamp):#odom_timestamp当前时刻
        self.odom_msg = odom_msg
        self.odom_timestamp = odom_timestamp
        
        dt = (self.odom_timestamp - self.odom_msg.header.stamp).to_sec()
        if dt > 0:
            rospy.loginfo("Prediction Step")
            if self.first_frame:
                object_position = np.array([self.odom_msg.pose.pose.position.x, 
                                           self.odom_msg.pose.pose.position.y, 
                                           self.odom_msg.pose.pose.position.z])
                orientation = self.odom_msg.pose.pose.orientation
                self.first_frame = False
            else:
                object_position = self.pose_matrix[:3, 3]
                orientation_np = quaternion_from_matrix(self.pose_matrix)
                orientation = Quaternion()
                orientation.x = orientation_np[0]
                orientation.y = orientation_np[1]
                orientation.z = orientation_np[2]
                orientation.w = orientation_np[3]

            object_vel_linear = np.array([self.odom_msg.twist.twist.linear.x, 
                                          self.odom_msg.twist.twist.linear.y, 
                                          self.odom_msg.twist.twist.linear.z])

            t_predict = object_position + object_vel_linear * dt

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
                
            current_rotation = R.from_quat(q0)
            delta_rotation = R.from_quat(delta_q)
            predicted_rotation = current_rotation * delta_rotation
            predicted_orientation = predicted_rotation.as_quat()
            rotation_matrix = quaternion_matrix(predicted_orientation)[:3, :3]

            self.pose_matrix = np.eye(4)
            self.pose_matrix[:3, :3] = rotation_matrix
            self.pose_matrix[:3, 3] = t_predict

            self.pose_pre[:3] = t_predict  # 预测的平移
            self.pose_pre[3:] = predicted_orientation  # 预测旋转
            self.P_pre = self.P_upd + self.R  # 预测方差
            return self.pose_matrix

        else:
            rospy.logwarn("Prediction Step fail")
            return self.pose_matrix
        

    def KF_update(self, pose_obv):
        self.pose_obv = pose_obv
        
        self.K_gain = self.P_pre @ np.linalg.inv(self.P_pre + self.Q)
        vector_after_filter = self.pose_pre + self.K_gain @ (self.pose_obv - self.pose_pre)
        I = np.eye(7)
        self.P_upd = (I - self.K_gain) @ self.P_pre
        
        self.pose_matrix[:3, 3] = vector_after_filter[:3]
        quaternion = vector_after_filter[3:]
        rotation = R.from_quat(quaternion)
        self.pose_matrix[:3, :3] = rotation.as_matrix()

        # world2cam_matrix = self.quaternion2T(self.orientation, self.position)
        # pose = np.linalg.inv(world2cam_matrix) @ self.pose_matrix
        
        self.i += 1

        return self.pose_matrix
