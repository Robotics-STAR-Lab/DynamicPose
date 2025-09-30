import rospy
import cv2
import numpy as np
from sam2.build_sam import build_sam2_camera_predictor
import torch
import time

class _sam2_:
    def __init__(self, config):
        sam2_checkpoint = config['Sam2']['sam2_checkpoint']
        model_cfg = config['Sam2']['model_cfg']
        self.predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)
        rospy.loginfo("SAM2 initialization done")
        
    def load_first_frame(self, rgb):
        self.predictor.load_first_frame(rgb)
    
    def add_new_prompt(self, frame_idx, obj_id, bbox):
        frame_idx, out_obj_ids, out_mask_logits = self.predictor.add_new_prompt(
            frame_idx=frame_idx, obj_id=obj_id, bbox=bbox
        )
        return frame_idx, out_obj_ids, out_mask_logits

    def track(self, rgb):
        t = time.time()
        # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            out_obj_ids, out_mask_logits = self.predictor.track(rgb)
            print(f'SAM2 Inference time: {time.time() - t:.3f}s')
            return out_obj_ids, out_mask_logits
        
    def _reset_(self):
        rospy.logwarn("Reset SAM2")
        return

    def get_center_line_(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            rospy.logerr("No object detected")
            return None
        all_points = np.vstack(contours).squeeze()
        moments = cv2.moments(mask)
        if moments["m00"] == 0:
            rospy.logerr("No object detected")
            return None
        center = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])
        centered_points = all_points - center
        cov_matrix = np.cov(centered_points, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        main_axis = eigenvectors[:, np.argmax(eigenvalues)]
        main_axis = main_axis / np.linalg.norm(main_axis)  # 单位化
        return center, main_axis, all_points
        