#!/usr/bin/env python3

import rospy
import numpy as np
from message_filters import Subscriber, ApproximateTimeSynchronizer
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf.transformations import quaternion_matrix
from cv_bridge import CvBridge
import sensor_msgs.msg
import cv2
import scipy.stats as stats

class MatrixSynchronizer:
    def __init__(self):
        # 初始化节点并设置日志级别
        rospy.init_node('matrix_synchronizer', log_level=rospy.DEBUG)
        
        # 配置参数
        self.target_frame = "mustard_bottle"
        self.source_frame = "World"
        self.sync_slop = 0.02  # 时间同步窗口(s)
        self.i = 0
        
        # 发布筛选后的TF变换
        self.tf_pub = rospy.Publisher(
            '/tf_world_to_mustard', 
            TransformStamped, 
            queue_size=10
        )
        
        # 订阅原始TF数据
        rospy.Subscriber(
            '/tf', 
            TFMessage, 
            self.tf_callback,
            queue_size=20  # 增大缓冲队列
        )
        
        # 初始化同步器
        self._init_synchronizer()
        
        # 文件初始化
        self._init_files()
        
        rospy.loginfo("Node initialized successfully")

    def _init_synchronizer(self):
        """初始化时间同步器"""
        gt_sub = Subscriber('/object_pose', Odometry)
        pose_sub = Subscriber('/tracking_result', PoseStamped)
        depth_sub = Subscriber('/camera/depth/image_raw', sensor_msgs.msg.Image)

        self.ts = ApproximateTimeSynchronizer(
            [gt_sub, pose_sub, depth_sub],
            queue_size=2,  # 增大同步队列
            slop=self.sync_slop,
            allow_headerless=False  # 强制要求header
        )
        self.ts.registerCallback(self.sync_callback)

    def _init_files(self):
        """初始化输出文件并写入头信息"""
        header = "# Timestamp, R00, R01, R02, Tx, R10, R11, R12, Ty, R20, R21, R22, Tz\n"
        
        for filename in ['pose_matrix_gt.txt', 'pose_matrix.txt']:
            try:
                with open(filename, 'w') as f:
                    f.write(header)
                rospy.loginfo(f"Created {filename} successfully")
            except IOError as e:
                rospy.logfatal(f"Failed to create {filename}: {str(e)}")
                rospy.signal_shutdown("File creation failed")

    def tf_callback(self, msg):
        """处理原始TF数据"""
        start_time = rospy.Time.now()
        found = False
        
        for transform in msg.transforms:
            if (transform.header.frame_id == self.source_frame and 
                transform.child_frame_id == self.target_frame):
                
                # 添加时间戳校验
                if transform.header.stamp.to_sec() < 1e-6:
                    rospy.logwarn_throttle(5, "Received invalid zero timestamp")
                    continue
                
                self.tf_pub.publish(transform)
                found = True
                rospy.logdebug_once("Published first transform")
        
        # 性能监控
        if rospy.get_param("~enable_perf_log", False):
            duration = (rospy.Time.now() - start_time).to_sec()
            rospy.loginfo(f"TF processing time: {duration*1000:.2f}ms")
        
        if not found:
            rospy.logdebug_throttle(10, f"No transform from {self.source_frame} to {self.target_frame}")

    @staticmethod
    def _convert_to_matrix(translation, rotation):
        """通用转换函数"""
        matrix = quaternion_matrix([
            rotation.x,
            rotation.y,
            rotation.z,
            rotation.w
        ])
        matrix[:3, 3] = [translation.x, translation.y, translation.z]
        return matrix

    def transform_to_matrix(self, transform):
        """转换TF变换"""
        return self._convert_to_matrix(
            transform.position,
            transform.orientation
        )

    def pose_to_matrix(self, pose):
        """转换位姿信息"""
        return self._convert_to_matrix(
            pose.position,
            pose.orientation
        )

    def _save_matrix(self, matrix, timestamp, filename):
        """优化后的保存函数"""
        try:
            # 格式化为CSV行
            flat_data = [
                timestamp,
                *matrix[0, :3], matrix[0, 3],
                *matrix[1, :3], matrix[1, 3],
                *matrix[2, :3], matrix[2, 3]
            ]
            line = ",".join(f"{x:.6f}" for x in flat_data) + "\n"
            
            with open(filename, 'a') as f:
                f.write(line)
                
        except Exception as e:
            rospy.logerr(f"Save failed for {filename}: {str(e)}")

    def sync_callback(self, gt_msg, tracking_msg, depth_msg):
        """同步数据处理"""
        start_time = rospy.Time.now()
        
        try:
            # # 时间差分析
            time_diff = abs((gt_msg.header.stamp - tracking_msg.header.stamp).to_sec())
            if time_diff > self.sync_slop * 1.5:
                rospy.logwarn(f"Excessive sync delay: {time_diff:.3f}s")
            
            # 数据转换
            gt_matrix = self.transform_to_matrix(gt_msg.pose.pose)
            tracking_matrix = self.pose_to_matrix(tracking_msg.pose)
            
            print(gt_matrix)
            # 数据保存
            self._save_matrix(gt_matrix, gt_msg.header.stamp.to_sec(), '/home/niloiv/pose_estimate_new/demo/mustard_bottle/pose_matrix_gt.txt')
            self._save_matrix(tracking_matrix, tracking_msg.header.stamp.to_sec(), '/home/niloiv/pose_estimate_new/demo/mustard_bottle/pose_matrix.txt')
            
            # 深度图像转换
            bridge = CvBridge()
            depth = bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
            depth_image_filename = f"/home/niloiv/pose_estimate_new/demo/mustard_bottle/depth/depth_image_{self.i}.png"
            cv2.imwrite(depth_image_filename, depth.astype(np.uint16))
            # rospy.loginfo(f"Saved depth image: {depth_image_filename}")
            self.i+=1
            
            # 性能日志
            proc_time = (rospy.Time.now() - start_time).to_sec()
            rospy.loginfo(
                f"Saved matrices and depth image | Δt={time_diff*1000:.1f}ms | "
                f"Proc={proc_time*1000:.1f}ms"
            )
            
        except Exception as e:
            rospy.logerr(f"Sync processing failed: {str(e)}")
            rospy.logdebug("Error details:", exc_info=True)

if __name__ == '__main__':
    try:
        MatrixSynchronizer()
        rospy.spin()
    except rospy.ROSException as e:
        rospy.logerr(f"ROS error occurred: {str(e)}")
    except Exception as e:
        rospy.logfatal(f"Unexpected error: {str(e)}")