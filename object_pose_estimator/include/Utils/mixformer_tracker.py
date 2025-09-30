import sys
import rospy
import numpy as np
import os
import cv2
from cv_bridge import CvBridge
import sensor_msgs.msg

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../MixFormerV2"))
print(project_root)
sys.path.append(project_root)
from lib.test.evaluation import Tracker
import time

class _mixformer_tracker_:
    def __init__(self):
        tracker_name = "mixformer2_vit_online"
        tracker_param = "224_depth4_mlp1_score"
        tracker_params ={
            "model": "models/mixformerv2_small.pth.tar",
            "search_area_scale": 4.5,
            "debug": 0,
            "update_interval": 25,
            "online_size": 1,
            "save_results": False
        }
        # print(tracker_params)
        self.mixformer_tracker = Tracker(tracker_name, tracker_param, "video", tracker_params=tracker_params)
        ## multiobj_mode == "default"
        params = self.mixformer_tracker.params
        self.tracker = self.mixformer_tracker.create_tracker(params)
        
        ### ros
        self.debug = True
        self.bridge = CvBridge()
        self.img_with_trackBBOX_pub = rospy.Publisher("/img_with_trackBBOX", sensor_msgs.msg.Image, queue_size=1)
        self.depth_pub = rospy.Publisher("/img_with_trackBBOX_depth", sensor_msgs.msg.Image, queue_size=1)
        print(f'_mixformer_tracker_ init done')
        
    def init(self, image, optional_box):
        assert optional_box is not None and image is not None
        assert isinstance(optional_box, (list, tuple))
        assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
        
        self.tracker.initialize(image, _build_init_info(optional_box))
        
    def track(self, image, depth=None):
        t = time.time()
        out = self.tracker.track(image)
        state = [int(s) for s in out['target_bbox']] # x,y,w,h
        
        if self.debug:
            frame_disp = image.copy()
            cv2.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                            (0, 255, 0), 2)
            
            ros_img = self.bridge.cv2_to_imgmsg(frame_disp, "bgr8")
            self.img_with_trackBBOX_pub.publish(ros_img)
            
            if depth is not None:
                depth_msg = self.bridge.cv2_to_imgmsg(depth, "passthrough")
                self.depth_pub.publish(depth_msg)
            # cv2.imshow("tracking", frame_disp)
            # key = cv2.waitKey(1)
            # if key == ord('q'):
            #     return
        print(f'[mixformer_tracker] Track time: {time.time() - t}')
        return state
    
        
    def update_template(self, bbox, K, img):
        """使用该函数来update self.tracker中的online_template"""
        self.tracker.update_template(image=img, bbox=bbox)
        
    
def _build_init_info(box):
    return {'init_bbox': box}

if __name__ == "__main__":
    rospy.init_node('MixFormer_tracker')
    tracker = _mixformer_tracker_()
    optional_box = [443, 204, 22, 51]
    video_path  = "/home/linbei/workspace/d6d/src/Dynamic_6D/TEST/1.mp4"
    cap = cv2.VideoCapture(video_path)
    i = 0
    while True:
        ret, frame = cap.read()
        if frame is None or not ret:
                break
        if i == 0:
            tracker.init(frame, optional_box)
        else:
            tracker.track(frame)
        i += 1
        
    cap.release()
    cv2.destroyAllWindows()