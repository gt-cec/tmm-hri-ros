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
import plot_full_tmm
import matplotlib

class PerceptionNode:
    latest_image = None
    latest_depth = None

    def __init__(self):
        rospy.init_node('perception', anonymous=True)

        self.bridge = CvBridge()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # set these classes to the classes you want to detect
        self.classes = ["human", "apple", "cup"]  # placeholder
        self.depth_classes = ["human", "person", "human standing", "person standing", "silhouette of a person", "silhouette of a human", "silhouette of a person from the side", "silhouette of a human from the side", "silhouette of a person"]  # not used for ground truth sim data

        self.class_to_class_id = {o : i for i, o in enumerate(self.classes)}
        class_id_to_color_map = matplotlib.cm.ScalarMappable(norm=matplotlib.pyplot.Normalize(vmin=1, vmax=len(self.classes)), cmap=matplotlib.cm.hsv).to_rgba([i for i, x in enumerate(self.classes)])  # color mapper

        pose_detector = cec_pose.PoseDetector()  # share this across mental models, it has no state so no data leakage

        # create the mental models
        self.robot_mm = mental_model.MentalModel(pose_detector=pose_detector)  # initialize the robot's mental model
        self.pred_human_mm = mental_model.MentalModel(pose_detector=pose_detector)  # initialize the predicted human's mental model

        # initialize the mental models
        initial_objects = [
            {"class": "apple", "x": 1, "y": 0, "z": 0},
            {"class": "apple", "x": 2, "y": 0, "z": 0},
            {"class": "cup", "x": 1, "y": 0, "z": 0},
            {"class": "cup", "x": 2, "y": 0, "z": 0},
            {"class": "cup", "x": 3, "y": 0, "z": 0},
        ]
        self.robot_mm.initialize(objects=initial_objects, verbose=False)  # set the initial environment state
        self.pred_human_mm.initialize(objects=initial_objects, verbose=False)

        self.last_saved_time = time.time()
        self.previous_human_location = (0, None)

        # visualization plot
        self.plot = plot_full_tmm.PlotFullTMM(self.classes, self.class_to_class_id, class_id_to_color_map, use_gt_semantics=False)  # initialize the plot

        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        self.image_sub = rospy.Subscriber('/camera/depth/image_rect_raw', Image, self.depth_callback)

    def image_callback(self, data):
        rgb_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        depth_image = self.latest_depth if self.latest_depth is not None else np.ones((rgb_image.shape[0], rgb_image.shape[1]))  # placeholder for depth image
        # resize the depth image to match the RGB image
        depth_image = cv2.resize(depth_image, (rgb_image.shape[1], rgb_image.shape[0]))
        # rotate the RGB images by 90 degrees to match robot orientation
        rgb_image = cv2.rotate(rgb_image, cv2.ROTATE_90_CLOCKWISE)
        depth_image = cv2.rotate(depth_image, cv2.ROTATE_90_CLOCKWISE)

        # save the rgb image to a file
        cv2.imwrite(f"rgb_image_{int(time.time())}.png", rgb_image)
        cv2.imwrite(f"depth_image_{int(time.time())}.png", depth_image)

        robot_pose = [np.array([0, 0, 0]), np.array([1, 0, 0])]   # placeholder
        
        robot_detected_objects, robot_human_detections = self.robot_mm.update_from_rgbd_and_pose(rgb_image, depth_image, robot_pose, self.classes, class_to_class_id=self.class_to_class_id, depth_classes=self.depth_classes, detect_threshold=0.4, seg_save_name=None)

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
        print("  Predicted human mental model: ", self.pred_human_mm.dsg.get_objects_by_id())
        self.plot.update(robot_mm=self.robot_mm, pred_human_mm=self.pred_human_mm, gt_human_mm=None, agent_pose=robot_pose, detected_objects=robot_detected_objects, human_detections=robot_human_detections, objects_visible_to_human=objects_visible_to_human, rgb_image=rgb_image, depth_image=depth_image, frame_num=int(time.time()) % 1000)

    def depth_callback(self, data):
        image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        self.latest_depth = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

if __name__ == '__main__':
    try:
        PerceptionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass