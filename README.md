# TMM-HRI ROS

ROS demonstration of the [TMM-HRI](https://github.com/gt-cec/tmm-hri) project.

The project infers a person's belief state from a third-person view (i.e., a robot). Importantly, the system supports:
1. Partial Observability: the robot cannot see everything at once.
2. Object Permanence: the scene can have multiple objects of the same class.
3. Open-Vocabulary: the scene supports open vocabulary class descriptions.
4. Zero-Shot: all models are zero-shot to new environments.

The upstream project codebase used offline simulation recordings. This codebase integrates with ROS for a real-time live demonstration.

## Usage

Run `main.py` to start a ROS node that listens for RGB and Depth topics and maintains two belief states (robot and inferred human).

If you are using a RealSense camera and have the RealSense library installed, run `camera.py` to start a relay to the ROS nodes used by `main.py`.

If you are using a Stretch RE2 robot, or a rosbag from that robot, it will use those topics.

`main.py` will processes images as it can (will take a few seconds per run on most laptops) and save a visualization of the belief states as png files `visualization_(time)`. `(time)` is derived from the current `time.time()`.

We trialed this system using a 