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

import cec_detect
import cec_pose
import cec_dsg

class WebcamObjectSegmentor:
    def __init__(self):
        rospy.init_node('webcam_object_segmentor', anonymous=True)
        self.image_sub = rospy.Subscriber('webcam_image', Image, self.image_callback)
        self.image_pub = rospy.Publisher('webcam_segmented_objects', Image, queue_size=10)
        self.box_pub = rospy.Publisher('/human_boxes', Int32MultiArray, queue_size=10)
        self.mask_pub = rospy.Publisher('/human_masks', Image, queue_size=10)

        self.bridge = CvBridge()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load Owlv2
        self.owl_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.owl_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(self.device)

        # Load SAMv2
        self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
        self.sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(self.device)

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

            pose_detector = cec_pose.PoseDetector()
            depth_map = np.zeros((cv_image.shape[0], cv_image.shape[1]))
            robot_pose = [np.array([0, 0]), np.array([1, 0])]

            detected_humans = []
            for box, mask in zip(boxes, masks):
                box_coords = [[box[0], box[1]], [box[2], box[3]]]
                detected_humans.append({
                "seg mask": mask,
                "box": box_coords
                })


            pred_person_loc, predicted_heading, other_data = pose_detector.get_heading_of_person(
            rgb=cv_image,
            depth_map=depth_map,
            detected_humans=detected_humans,
            robot_pose=robot_pose
        )

            # for box, mask in zip(boxes, masks):
            #     # Publish bounding box
            #     bbox_msg = Int32MultiArray(data=box)  # box = [x_min, y_min, x_max, y_max]
            #     self.box_pub.publish(bbox_msg)

            #     # Publish segmentation mask
            #     mask = (mask > 0).astype(np.uint8) * 255  # Binary mask
            #     mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
            #     self.mask_pub.publish(mask_msg)

            

            # for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            #     box_int = [int(i) for i in box]

            #     sam_inputs = self.sam_processor(images=rgb_image, input_boxes=[box_int], return_tensors="pt").to(self.device)
            #     with torch.no_grad():
            #         sam_outputs = self.sam_model(**sam_inputs)
            #         mask = sam_outputs.pred_masks.squeeze(0).cpu().numpy().astype(np.uint8) * 255

            #     mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
                
                # # Check if 5 seconds have passed since the last saved image
                # current_time = time.time()
                # if current_time - self.last_saved_time >= 5:
                #     # Save the segmented image to disk
                #     output_dir = "output_images"
                #     if not os.path.exists(output_dir):
                #         os.makedirs(output_dir)  # Create directory if it doesn't exist

                #     image_filename = f"{output_dir}/segmented_image_{int(current_time)}.jpg"
                #     cv2.imwrite(image_filename, segmented_image)
                #     rospy.loginfo(f"Image saved to: {image_filename}")

                #     # Update the last saved time
                #     self.last_saved_time = current_time

                # # Convert the segmented image back to ROS Image message
                # segmented_image_msg = self.bridge.cv2_to_imgmsg(segmented_image, encoding="bgr8")

                # # Publish the segmented image with detections
                # self.image_pub.publish(segmented_image_msg)

        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

if __name__ == '__main__':
    try:
        WebcamObjectSegmentor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass