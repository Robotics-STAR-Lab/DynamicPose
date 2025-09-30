import os
import rospy
from ultralytics import YOLO
import time

class _yolo_:
    def __init__(self, config):
        model_id = config['Yolo']['model_id']
        if not os.path.exists(model_id):
            assert False, f"Model file {model_id} does not exist"
        
        self.model = YOLO(model_id)
        model_names = list(self.model.names.values())
        taskClasses = config['taskTexts']
        for taskClass in taskClasses:
            assert taskClass in model_names, f"{taskClass} not in model names"
            
        rospy.loginfo("YOLO initialization done")  
        
    def predict(self, img, target_class):
        results = self.model.predict(img, save=False, verbose=False)
        names = results[0].names
        target_conf = 0.0
        target_box = None
        for box in results[0].boxes:
            cls = int(box.cls)
            conf = float(box.conf)  # 置信度
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            # print(names[cls], x1, y1, x2, y2)
            
            ## get the target class with the highest confidence
            if names[cls] == target_class:   
                if conf > target_conf:
                    target_conf = conf
                    target_box = [x1, y1, x2, y2]
        return target_box, target_conf