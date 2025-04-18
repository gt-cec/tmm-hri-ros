import rospy
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection, SamProcessor, SamModel
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time
import numpy as np
from std_msgs.msg import Int32MultiArray
import cec_pose
import mental_model
import predict

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
            robot_pose = [np.array([0, 0]), np.array([1, 0])]   # placeholder
            classes = ["apple", "cup"]
            class_to_class_id = {o : i for i, o in enumerate(classes)}
            depth_classes = ["human", "person", "human standing", "person standing", "silhouette of a person", "silhouette of a human", "silhouette of a person from the side", "silhouette of a human from the side", "silhouette of a person"]  # not used for ground truth sim data

            robot_detected_objects, robot_human_detections = self.robot_mm.update_from_rgbd_and_pose(rgb_image, depth_1channel, robot_pose, classes, class_to_class_id=class_to_class_id, depth_classes=depth_classes, detect_threshold=0.4, seg_save_name=f"./seg.png")

            print("detected objects: ", robot_detected_objects)
            print("detected humans: ", robot_human_detections)

            depth_map = np.zeros((cv_image.shape[0], cv_image.shape[1]))

            pred_person_loc, predicted_heading, other_data = self.pose_detector.get_heading_of_person(
                rgb=cv_image,
                depth_map=depth_map,
                detected_humans=robot_human_detections,
                robot_pose=robot_pose
            )

            # if the human is visible to the robot, run the trajectory prediction
            objects_visible_to_human = []  # objects that the robot thinks the human can see
            human_trajectory_debug = None
            # update the human pose 
            if len(robot_human_detections[0]) > 0:  # if a human was seen, use the first one (can place this in a loop to support multiple humans, but we only have one human mental model in play)
                print("  Human was observed, so updating the predicted human mental model")
                human_pose = robot_human_detections[0][0]["pose"]  # get the human's pose
                human_location = [human_pose[0], human_pose[1], human_pose[2]] # pose[0] is the base joint, using [east, north, vertical]
                robot_human_detections[0][0]["visible objects"] = objects_visible_to_human  # update the human's visible objects in the detections
                
                # if human has not been seen since before the last frame, predict where the human went since the last view
                if last_saw_human[0] is not None:
                    # if using ground truth human trajectory, get the poses
                    gt_human_poses = []
                    # get objects along the path that the human took
                    objects_visible_to_human, human_trajectory_debug = predict.get_objects_visible_from_last_seen(last_saw_human[1][:2], human_location[:2], np.ones((512, 512)), self.robot_mm.dsg, human_fov=120, end_direction=robot_human_detections[0][0]["direction"][:2], use_gt_human_trajectory=False, gt_human_poses=gt_human_poses, infer_human_trajectory=True, debug_tag=f"{int(time.time()) % 1000}")
                    self.pred_human_mm.update_from_detected_objects(objects_visible_to_human)  # update the predicted human's mental model
                last_saw_human = (time.time(), human_location)

        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

if __name__ == '__main__':
    try:
        WebcamObjectSegmentor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass