import rospy
import torch
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time
import numpy as np
from std_msgs.msg import Int32MultiArray
import cec_pose
import mental_model
import predict

class PerceptionNode:
    latest_image = None
    latest_depth = None

    def __init__(self):
        rospy.init_node('perception', anonymous=True)

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
        self.previous_human_location = (0, None)

        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        self.image_sub = rospy.Subscriber('/camera/depth/image_rect_raw', Image, self.depth_callback)

    def image_callback(self, data):
        cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        depth_image = self.latest_depth if self.latest_depth is not None else np.ones((cv_image.shape[0], cv_image.shape[1]))  # placeholder for depth image

        robot_pose = [np.array([0, 0]), np.array([1, 0])]   # placeholder
        classes = ["human", "apple", "cup"]  # placeholder
        class_to_class_id = {o : i for i, o in enumerate(classes)}
        depth_classes = ["human", "person", "human standing", "person standing", "silhouette of a person", "silhouette of a human", "silhouette of a person from the side", "silhouette of a human from the side", "silhouette of a person"]  # not used for ground truth sim data

        robot_detected_objects, robot_human_detections = self.robot_mm.update_from_rgbd_and_pose(rgb_image, depth_image, robot_pose, classes, class_to_class_id=class_to_class_id, depth_classes=depth_classes, detect_threshold=0.4, seg_save_name=f"./seg.png")

        print("detected objects: ", robot_detected_objects)
        print("detected humans: ", robot_human_detections)

        # pred_person_loc, predicted_heading, other_data = self.pose_detector.get_heading_of_person(
        #     rgb=cv_image,
        #     depth_map=depth_image,
        #     detected_humans=robot_human_detections,
        #     robot_pose=robot_pose
        # )

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
            # get objects along the path that the human took
            objects_visible_to_human, human_trajectory_debug = predict.get_objects_visible_from_last_seen(self.previous_human_location, human_location[:2], np.ones((512, 512)), self.robot_mm.dsg, human_fov=120, end_direction=robot_human_detections[0][0]["direction"][:2], use_gt_human_trajectory=False, gt_human_poses=None, infer_human_trajectory=True, debug_tag=f"{int(time.time()) % 1000}")
            self.pred_human_mm.update_from_detected_objects(objects_visible_to_human)  # update the predicted human's mental model
            self.previous_human_location = (time.time(), human_location)

        # visual or something here
        print("  Predicted human mental model: ", self.pred_human_mm.dsg.get_all_objects())

    def depth_callback(self, data):
        image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        self.latest_depth = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

if __name__ == '__main__':
    try:
        PerceptionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass