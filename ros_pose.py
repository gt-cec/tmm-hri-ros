import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
# import the package, this will cause some spam from mmengine
import cec_pose
import numpy as np
from std_msgs.msg import Int32MultiArray

latest_frame = None
latest_boxes = []
latest_masks = []


# Get the latest frame from the webcam feed
def image_callback(msg):
    global latest_frame
    latest_frame = msg

def box_callback(msg):
    global latest_boxes
    latest_boxes.append([msg.data[0], msg.data[1], msg.data[2], msg.data[3]])

def mask_callback(msg):
    global latest_masks
    mask = bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
    latest_masks.append(mask)


# Service handler for the pose detection service
def handle_pose_detection_request(req):
    global latest_frame
    bridge = CvBridge()

    if latest_frame is None:
        return PoseDetectionResponse(success=False, message="No image available", pose_image=None)

    try:
        # Convert ROS Image message to OpenCV image
        cv_image = bridge.imgmsg_to_cv2(latest_frame, desired_encoding='bgr8')
    except CvBridgeError as e:
        rospy.logerr(f"Failed to convert image: {e}")
        return PoseDetectionResponse(success=False, message="Failed to convert image", pose_image=None)

    try:
        # Initialize the PoseDetection object
        pose_detector = cec_pose.PoseDetector()


        bounding_boxes = np.array([[[0, 0], [cv_image.shape[0], cv_image.shape[1]]]])

        # Use the get_heading_of_person function from PoseDetection class
        
        # Placeholder: depth_map, detected_humans, robot_pose
        depth_map = np.zeros((cv_image.shape[0], cv_image.shape[1]))
        detected_humans = []
        for box, mask in zip(latest_boxes, latest_masks):
            box_coords = [[box[0], box[1]], [box[2], box[3]]]
            detected_humans.append({
            "seg mask": mask,
            "box": box_coords
            })
        robot_pose = [np.array([0, 0]), np.array([1, 0])]

        # Get heading of person using the `get_heading_of_person` method
        pred_person_loc, predicted_heading, other_data = pose_detector.get_heading_of_person(
            rgb=cv_image,
            depth_map=depth_map,
            detected_humans=detected_humans,
            robot_pose=robot_pose
        )

        # Here we are just passing the original frame
        pose_image_msg = bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")

        return PoseDetectionResponse(success=True, message="Pose detection successful", pose_image=pose_image_msg)

    except Exception as e:
        rospy.logerr(f"Pose detection failed: {e}")
        return PoseDetectionResponse(success=False, message="Pose detection failed", pose_image=None)

# Main function to set up the ROS service node
def pose_detection_service():
    print("Check1")
    rospy.init_node('pose_detection_service')
    print("Check2")
    rospy.Subscriber('/webcam/image_raw', Image, image_callback)
    rospy.Subscriber('/human_boxes', Int32MultiArray, box_callback)
    rospy.Subscriber('/human_masks', Image, mask_callback)

    service = rospy.Service('/get_pose', PoseDetection, handle_pose_detection_request)
    rospy.loginfo("Pose Detection Service is ready.")
    rospy.spin()

if __name__ == "__main__":
    try:
        pose_detection_service()
    except rospy.ROSInterruptException:
        pass
