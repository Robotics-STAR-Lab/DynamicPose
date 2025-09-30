import rospy
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 

class _grounding_dino_:
    def __init__(self, config, device):
        model_id = config['GroundingDINO']['model_id']
        self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, local_files_only=True).to(device)
        rospy.loginfo("Grounding DINO  initialization done")  