# TMM-HRI ROS

ROS demonstration of the [TMM-HRI](https://github.com/gt-cec/tmm-hri) project.

The project infers a person's belief state from a third-person view (i.e., a robot). Importantly, the system supports:
1. Partial Observability: the robot cannot see everything at once.
2. Object Permanence: the scene can have multiple objects of the same class.
3. Open-Vocabulary: the scene supports open vocabulary class descriptions.
4. Zero-Shot: all models are zero-shot to new environments.

The upstream project codebase used offline simulation recordings. This codebase integrates with ROS for a real-time live demonstration.

## Usage

Run `main.py` to start a ROS node that listens for RGB and Depth topics and maintains two belief states (robot and inferred human). Upon recei