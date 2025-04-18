import rospy
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection, SamProcessor, SamForImageSegmentation
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import time
import numpy as np

class WebcamObjectSegmentor:
    def __init__(self):
        rospy.init_node('webcam_object_segmentor', anonymous=True)
        self.image_sub = rospy.Subscriber('webcam_image', Image, self.image_callback)
        self.image_pub = rospy.Publisher('webcam_segmented_objects', Image, queue_size=10)

        self.bridge = CvBridge()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load Owlv2
        self.owl_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.owl_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(self.device)

        # Load SAMv2
        self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
        self.sam_model = SamForImageSegmentation.from_pretrained("facebook/sam-vit-huge").to(self.device)

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

                # Overlay the masks on the original image
                segmented_image = cv_image.copy()
                for mask in masks:
                    segmented_image[mask > 0] = [0, 255, 0]  # Apply green overlay for segmentation

                # Check if 5 seconds have passed since the last saved image
                current_time = time.time()
                if current_time - self.last_saved_time >= 5:
                    # Save the segmented image to disk
                    output_dir = "output_images"
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)  # Create directory if it doesn't exist

                    image_filename = f"{output_dir}/segmented_image_{int(current_time)}.jpg"
                    cv2.imwrite(image_filename, segmented_image)
                    rospy.loginfo(f"Image saved to: {image_filename}")

                    # Update the last saved time
                    self.last_saved_time = current_time

                # Convert the segmented image back to ROS Image message
                segmented_image_msg = self.bridge.cv2_to_imgmsg(segmented_image, encoding="bgr8")

                # Publish the segmented image with detections
                self.image_pub.publish(segmented_image_msg)

        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

if __name__ == '__main__':
    try:
        WebcamObjectSegmentor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass