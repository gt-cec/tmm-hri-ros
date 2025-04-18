import rospy
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection, SamProcessor, SamModel
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import time
import numpy as np
from std_msgs.msg import Int32MultiArray
import cec_pose
import mental_model

class WebcamObjectSegmentor:
    def __init__(self):
        rospy.init_node('webcam_object_segmentor', anonymous=True)
        self.image_sub = rospy.Subscriber('webcam_image', Image, self.image_callback)
        self.image_pub = rospy.Publisher('webcam_segmented_objects', Image, queue_size=10)
        self.box_pub = rospy.Publisher('/human_boxes', Int32MultiArray, queue_size=10)
        self.mask_pub = rospy.Publisher('/human_masks', Image, queue_size=10)

        self.bridge = CvBridge()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        pose_detector = cec_pose.PoseDetector()  # share this across mental models, it has no state so no data leakage

        # create the mental models
        self.robot_mm = mental_model.MentalModel(pose_detector=pose_detector)  # initialize the robot's mental model
        self.pred_human_mm = mental_model.MentalModel(pose_detector=pose_detector)  # initialize the predicted human's mental model

        # initialize the mental models
        self.robot_mm.initialize(objects=[], verbose=False)  # set the initial environment state
        self.pred_human_mm.initialize(objects=[], verbose=False)

        self.last_saved_time = time.time()

    def image_callback(self, data):
        try:
            # Convert ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            agent_pose = (1, 0)   # placeholder
            classes = ["apple", "cup"]

            robot_detected_objects, robot_human_detections = self.robot_mm.update_from_rgbd_and_pose(rgb_image, depth_1channel, agent_pose, classes, class_to_class_id=class_to_class_id, depth_classes=depth_classes, detect_threshold=0.4, seg_save_name=f"{episode_dir}/{agent_id}/Action_{str(frame_id).zfill(4)}")

            depth_map = np.zeros((cv_image.shape[0], cv_image.shape[1]))
            robot_pose = [np.array([0, 0]), np.array([1, 0])]

            detected_humans = []
            for box, mask in zip(boxes, masks):
                box_coords = [[box[0], box[1]], [box[2], box[3]]]
                detected_humans.append({
                "seg mask": mask,
                "box": box_coords
                })

            pred_person_loc, predicted_heading, other_data = self.pose_detector.get_heading_of_person(
                rgb=cv_image,
                depth_map=depth_map,
                detected_humans=detected_humans,
                robot_pose=robot_pose
            )

        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

if __name__ == '__main__':
    try:
        WebcamObjectSegmentor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass