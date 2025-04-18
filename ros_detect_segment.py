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

        self.mental_model = mental_model.MentalModel()

        self.last_saved_time = time.time()

    def image_callback(self, data):
        try:
            # Convert ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # Detect objects using OWL-ViT
            text_queries = ["human"]
            inputs = self.owl_processor(text=text_queries, images=rgb_image, return_tensors="pt").to(self.device)

            # Process the image with the model
            with torch.no_grad():
                outputs = self.owl_model(**inputs)

            # Post-process the outputs to get bounding boxes
            target_sizes = torch.tensor([rgb_image.shape[:2]]).to(self.device)
            results = self.owl_processor.post_process_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=0.05  # Lowered threshold to detect more objects
            )[0]

            if len(results["scores"]) > 0:  # Check if any objects were detected
                boxes = []
                for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                    box = [int(i) for i in box]  # Convert to integer coordinates
                    boxes.append(box)

                masks = []
                for box in boxes:
                    sam_inputs = self.sam_processor(images=rgb_image, input_boxes=[box], return_tensors="pt").to(self.device)
                    
                    # Perform segmentation
                    with torch.no_grad():
                        sam_outputs = self.sam_model(**sam_inputs)
                        mask = sam_outputs.pred_masks.squeeze(0).cpu().numpy()
                        masks.append(mask)

            #     # Overlay the masks on the original image
            #     segmented_image = cv_image.copy()
            #     for mask in masks:
            #         segmented_image[mask > 0] = [0, 255, 0]  # Apply green overlay for segmentation

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