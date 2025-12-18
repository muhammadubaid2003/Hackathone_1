---
id: python-agents
title: Controlling Robots with ROS 2 and Python Agents
sidebar_label: Python Agents with ROS 2
sidebar_position: 2
---

# Controlling Robots with ROS 2 and Python Agents

## Using rclpy to Build ROS 2 Nodes

The `rclpy` package provides Python bindings for the ROS 2 client library (rcl). It enables Python developers to create ROS 2 nodes, publish and subscribe to topics, provide and use services, and work with actions.

### Setting Up Your Environment

Before creating your first ROS 2 node with Python, ensure you have:

1. **ROS 2 installed**: Follow the official installation guide for your platform (Ubuntu, Windows, or macOS)
2. **Python 3.6+**: ROS 2 requires Python 3.6 or higher
3. **Workspace created**: Set up a ROS 2 workspace for your project

### Basic Node Structure

Every ROS 2 Python node follows a similar structure:

```python
import rclpy
from rclpy.node import Node

class MyNode(Node):
    def __init__(self):
        super().__init__('node_name')
        # Initialize node components here
        # Publishers, subscribers, timers, services, etc.

def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Key Components of a Python Node

1. **Import Statements**: Import rclpy and any message types you'll use
2. **Node Class**: Inherit from `rclpy.node.Node` to create your custom node
3. **Initialization**: Call `super().__init__()` with a unique node name
4. **Main Function**: Initialize ROS, create node instance, spin, and cleanup

### Creating Your First Node

Let's create a simple node that demonstrates the basic structure:

```python
import rclpy
from rclpy.node import Node

class RobotControllerNode(Node):
    def __init__(self):
        super().__init__('robot_controller')
        self.get_logger().info('Robot Controller Node has been started')

def main(args=None):
    rclpy.init(args=args)
    node = RobotControllerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Node Parameters

ROS 2 allows nodes to accept parameters that can be configured at runtime:

```python
class RobotControllerNode(Node):
    def __init__(self):
        super().__init__('robot_controller')

        # Declare parameters with default values
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('control_frequency', 50)

        # Get parameter values
        self.max_velocity = self.get_parameter('max_velocity').value
        self.control_frequency = self.get_parameter('control_frequency').value

        self.get_logger().info(f'Max velocity: {self.max_velocity}, Frequency: {self.control_frequency}')
```

### Best Practices for Node Development

1. **Use descriptive node names**: Choose names that clearly indicate the node's function
2. **Log important events**: Use `self.get_logger().info()` for important messages
3. **Handle cleanup**: Ensure proper cleanup in `destroy_node()`
4. **Use try/except blocks**: Handle potential exceptions during execution
5. **Validate inputs**: Check parameter values and message contents when appropriate

## Publishing and Subscribing to Topics

The publish-subscribe pattern is fundamental to ROS 2 communication. Publishers send data to topics, and subscribers receive data from topics. This decoupled communication allows for flexible and robust robot systems.

### Creating a Publisher

To create a publisher in a ROS 2 Python node, use the `create_publisher()` method:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1
```

Key components of a publisher:
- **Message type**: The type of message being published (e.g., `String`, `Int32`, custom messages)
- **Topic name**: The name of the topic to publish to
- **Queue size**: The number of messages to queue if subscribers are slow to process

### Creating a Subscriber

To create a subscriber, use the `create_subscription()` method:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)
```

Key components of a subscriber:
- **Message type**: Must match the type published to the topic
- **Topic name**: The name of the topic to subscribe to
- **Callback function**: Called when a message is received
- **Queue size**: The number of messages to queue if the callback is slow

### Publisher-Subscriber Example for Robot Control

Here's a more practical example showing how publishers and subscribers can be used for robot control:

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')

        # Publisher for velocity commands
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscriber for laser scan data
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Timer for control loop
        self.timer = self.create_timer(0.1, self.control_loop)

        self.safety_distance = 0.5  # meters
        self.obstacle_detected = False

    def scan_callback(self, msg):
        # Check if there are obstacles in front of the robot
        if len(msg.ranges) > 0:
            front_range = msg.ranges[len(msg.ranges) // 2]  # Approximate front range
            self.obstacle_detected = front_range < self.safety_distance

    def control_loop(self):
        cmd_vel = Twist()

        if self.obstacle_detected:
            # Stop the robot if obstacle detected
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
        else:
            # Move forward if no obstacles
            cmd_vel.linear.x = 0.5  # m/s
            cmd_vel.angular.z = 0.0

        self.cmd_vel_publisher.publish(cmd_vel)

def main(args=None):
    rclpy.init(args=args)
    robot_controller = RobotController()

    try:
        rclpy.spin(robot_controller)
    except KeyboardInterrupt:
        pass
    finally:
        robot_controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Quality of Service (QoS) Settings

For more robust communication, especially in safety-critical applications like humanoid robots, you can configure Quality of Service settings:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

# Create a QoS profile for reliable communication
qos_profile = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,  # Ensure all messages are delivered
    durability=DurabilityPolicy.VOLATILE,    # Don't keep messages for late-joining subscribers
)

# Use the QoS profile when creating publisher/subscriber
self.publisher_ = self.create_publisher(String, 'topic', qos_profile)
self.subscription = self.create_subscription(
    String,
    'topic',
    self.listener_callback,
    qos_profile
)
```

## Bridging AI Logic with Robot Controllers

One of the most important aspects of humanoid robotics is connecting AI decision-making algorithms with physical robot controllers. This connection enables the robot to act on intelligent decisions and interact with the real world.

### Architecture of AI-Robot Integration

The integration typically follows this pattern:

1. **Perception Layer**: Sensors provide data about the environment and robot state
2. **AI Decision Layer**: AI algorithms process information and make decisions
3. **Action Layer**: Robot controllers execute physical actions based on AI decisions
4. **Feedback Loop**: Sensor data confirms action results and updates the AI

### Implementing an AI Agent Node

Here's an example of a simple AI agent that makes decisions and sends commands to robot controllers:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import numpy as np

class AIAgentNode(Node):
    def __init__(self):
        super().__init__('ai_agent')

        # Subscribers for sensor data
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        # Publisher for robot commands
        self.cmd_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # Timer for AI decision-making loop
        self.timer = self.create_timer(0.2, self.ai_decision_loop)

        # Internal state
        self.laser_data = None
        self.robot_state = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
        self.goal = {'x': 5.0, 'y': 5.0}

    def scan_callback(self, msg):
        self.laser_data = msg.ranges

    def ai_decision_loop(self):
        if self.laser_data is None:
            return

        # Simple AI decision: navigate to goal while avoiding obstacles
        cmd_vel = self.make_navigation_decision()
        self.cmd_publisher.publish(cmd_vel)

    def make_navigation_decision(self):
        cmd_vel = Twist()

        # Check for obstacles
        if self.detect_obstacles():
            # Emergency stop or obstacle avoidance
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.5  # Turn to avoid
        else:
            # Navigate toward goal
            cmd_vel.linear.x = 0.5
            cmd_vel.angular.z = self.calculate_turn_to_goal()

        return cmd_vel

    def detect_obstacles(self):
        if self.laser_data is None:
            return False

        # Check if there are obstacles within 0.8m in front
        front_ranges = self.laser_data[len(self.laser_data)//2-30:len(self.laser_data)//2+30]
        min_range = min([r for r in front_ranges if r > 0.01], default=float('inf'))

        return min_range < 0.8

    def calculate_turn_to_goal(self):
        # Simple proportional controller for turning toward goal
        # In a real system, you'd use proper path planning
        return 0.2  # Placeholder - in reality, calculate based on goal direction

def main(args=None):
    rclpy.init(args=args)
    ai_agent = AIAgentNode()

    try:
        rclpy.spin(ai_agent)
    except KeyboardInterrupt:
        pass
    finally:
        ai_agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Integration with Popular AI Libraries

ROS 2 nodes can easily integrate with popular AI libraries like TensorFlow, PyTorch, or scikit-learn:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
from cv_bridge import CvBridge
import tensorflow as tf

class VisionAIAgent(Node):
    def __init__(self):
        super().__init__('vision_ai_agent')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Load pre-trained model
        self.model = tf.keras.models.load_model('path/to/model.h5')

        # Subscribers and publishers
        self.image_subscriber = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.cmd_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

    def image_callback(self, msg):
        # Convert ROS image to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Preprocess image for the model
        input_image = cv2.resize(cv_image, (224, 224))
        input_image = input_image.astype('float32') / 255.0
        input_image = np.expand_dims(input_image, axis=0)

        # Run inference
        prediction = self.model.predict(input_image)

        # Convert prediction to robot command
        cmd_vel = self.prediction_to_command(prediction)
        self.cmd_publisher.publish(cmd_vel)

def main(args=None):
    rclpy.init(args=args)
    vision_agent = VisionAIAgent()

    try:
        rclpy.spin(vision_agent)
    except KeyboardInterrupt:
        pass
    finally:
        vision_agent.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Best Practices for AI-Robot Integration

1. **Modularity**: Keep AI logic separate from ROS 2 communication logic
2. **Real-time considerations**: Ensure AI algorithms can run within timing constraints
3. **Error handling**: Plan for AI failures and provide safe fallbacks
4. **State management**: Maintain consistent state between AI and robot systems
5. **Testing**: Test AI-robot integration thoroughly in simulation before real robots

## Conceptual Mapping Between AI Decisions and Motor Actions

Understanding how AI decisions translate to physical motor actions is crucial for effective humanoid robot control. This mapping forms the bridge between high-level cognitive functions and low-level motor control.

### Hierarchical Control Architecture

Humanoid robots typically employ a hierarchical control architecture:

```
High-Level AI (Reasoning, Planning)
    ↓ (Goals, Tasks)
Mid-Level Controllers (Path Planning, Trajectory Generation)
    ↓ (Trajectories, Waypoints)
Low-Level Controllers (Motor Control, Feedback Control)
    ↓ (Motor Commands)
Actuators (Physical Movement)
```

### Types of AI-to-Motor Mappings

#### 1. Direct Mapping
- **Simple behaviors**: AI outputs directly map to motor commands
- **Example**: AI decides "move forward" → Robot moves forward at fixed velocity
- **Use case**: Basic navigation, simple reactive behaviors

#### 2. Indirect Mapping through Trajectories
- **Complex movements**: AI generates high-level goals, trajectory planners create detailed paths
- **Example**: AI decides "pick up object" → Trajectory planner creates arm movement sequence
- **Use case**: Manipulation tasks, complex navigation

#### 3. Learned Mappings
- **Adaptive behaviors**: Neural networks or learned controllers map AI decisions to motor actions
- **Example**: Deep reinforcement learning policies that directly output motor commands
- **Use case**: Locomotion, complex adaptive behaviors

### Implementation Example: Mapping Decision to Action

Here's how you might implement the mapping in code:

```python
class DecisionToActionMapper(Node):
    def __init__(self):
        super().__init__('decision_to_action')

        # Publishers for different types of commands
        self.velocity_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.joint_publisher = self.create_publisher(JointState, '/joint_commands', 10)

        # Subscriber for AI decisions
        self.decision_subscriber = self.create_subscription(
            String,
            '/ai_decisions',
            self.decision_callback,
            10
        )

        # Robot-specific parameters
        self.robot_config = self.get_robot_configuration()

    def decision_callback(self, msg):
        ai_decision = msg.data

        # Map AI decision to appropriate motor action
        if ai_decision == 'explore_environment':
            self.execute_exploration()
        elif ai_decision == 'approach_object':
            self.execute_approach()
        elif ai_decision == 'grasp_object':
            self.execute_grasp()
        elif ai_decision == 'avoid_obstacle':
            self.execute_avoidance()
        else:
            self.get_logger().warn(f'Unknown AI decision: {ai_decision}')

    def execute_exploration(self):
        # Convert exploration decision to velocity commands
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.3  # Move forward slowly
        cmd_vel.angular.z = 0.1  # Slight turn to explore
        self.velocity_publisher.publish(cmd_vel)

    def execute_approach(self):
        # More complex mapping for approaching an object
        # This might involve path planning, inverse kinematics, etc.
        pass

    def execute_grasp(self):
        # Map to joint position or trajectory commands
        joint_cmd = JointState()
        # Set appropriate joint positions for grasping
        self.joint_publisher.publish(joint_cmd)

    def execute_avoidance(self):
        # Emergency behavior mapping
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0  # Stop
        cmd_vel.angular.z = 0.5  # Turn away
        self.velocity_publisher.publish(cmd_vel)
```

### Considerations for Effective Mapping

#### Safety Constraints
- **Joint limits**: Ensure AI decisions don't command movements beyond physical limits
- **Velocity limits**: Prevent dangerous speeds
- **Force limits**: Avoid excessive forces that could damage the robot or environment
- **Emergency stops**: Always have safety mechanisms to override AI decisions

#### Timing and Synchronization
- **Control frequency**: Match AI decision rate with motor control capabilities
- **Latency**: Account for communication delays between AI and motors
- **Synchronization**: Coordinate multiple actuators for complex movements

#### Feedback Integration
- **State estimation**: Use sensor feedback to update the AI's understanding of the robot's state
- **Adaptive control**: Adjust motor commands based on real-time feedback
- **Error correction**: Implement mechanisms to correct for discrepancies between planned and executed actions

### Real-World Example: Humanoid Walking

Consider how AI decisions map to motor actions in humanoid walking:

1. **AI Decision**: "Walk forward at 0.5 m/s"
2. **Step Planning**: Generate footstep plan based on terrain and goal
3. **Trajectory Generation**: Create center of mass and foot trajectories
4. **Inverse Kinematics**: Calculate joint angles for each trajectory point
5. **Motor Control**: Send joint commands to actuators with appropriate gains
6. **Feedback Control**: Adjust based on sensor readings (IMU, force sensors, encoders)

This chain of mappings transforms a high-level goal into coordinated motor actions across multiple joints.

## Practical Python Code Examples

Here's a complete example that demonstrates how to create a robot controller that bridges AI decisions with motor actions:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import math

class PracticalRobotController(Node):
    """
    A practical example of bridging AI decisions with robot controllers.
    This node demonstrates:
    1. Subscribing to sensor data
    2. Making simple AI decisions based on sensor data
    3. Converting decisions to motor commands
    4. Managing robot state
    """

    def __init__(self):
        super().__init__('practical_robot_controller')

        # Robot state
        self.position_x = 0.0
        self.position_y = 0.0
        self.orientation = 0.0
        self.laser_ranges = []
        self.goal_reached = False

        # Robot parameters
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.safety_distance = 0.6  # meters
        self.goal_tolerance = 0.2   # meters
        self.goal_x = 5.0
        self.goal_y = 5.0

        # Publishers
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscribers
        self.scan_subscriber = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.odom_subscriber = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)

        # Timer for control loop (10 Hz)
        self.control_timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info('Practical Robot Controller initialized')

    def scan_callback(self, msg):
        """Process laser scan data"""
        self.laser_ranges = msg.ranges

    def odom_callback(self, msg):
        """Process odometry data to track robot position"""
        self.position_x = msg.pose.pose.position.x
        self.position_y = msg.pose.pose.position.y

        # Extract orientation from quaternion (simplified)
        # In practice, use tf_transformations or similar
        orientation_q = msg.pose.pose.orientation
        self.orientation = math.atan2(2.0 * (orientation_q.w * orientation_q.z +
                                            orientation_q.x * orientation_q.y),
                                     1.0 - 2.0 * (orientation_q.y * orientation_q.y +
                                                 orientation_q.z * orientation_q.z))

    def control_loop(self):
        """Main control loop that makes decisions and sends commands"""
        if not self.laser_ranges:
            return  # Wait for laser data

        # Simple AI decision-making
        if self.is_goal_reached():
            self.linear_vel = 0.0
            self.angular_vel = 0.0
            if not self.goal_reached:
                self.get_logger().info('Goal reached!')
                self.goal_reached = True
        else:
            # Check for obstacles
            if self.is_obstacle_ahead():
                # Emergency behavior: stop and turn
                self.linear_vel = 0.0
                self.angular_vel = 0.5  # Turn right
            else:
                # Navigate toward goal
                self.navigate_to_goal()

        # Publish command
        cmd_msg = Twist()
        cmd_msg.linear.x = self.linear_vel
        cmd_msg.angular.z = self.angular_vel
        self.cmd_vel_publisher.publish(cmd_msg)

    def is_obstacle_ahead(self):
        """Check if there's an obstacle in front of the robot"""
        if not self.laser_ranges:
            return False

        # Check the front 30 degrees
        front_start = len(self.laser_ranges) // 2 - 15
        front_end = len(self.laser_ranges) // 2 + 15

        for i in range(front_start, front_end):
            idx = i % len(self.laser_ranges)
            if 0.01 < self.laser_ranges[idx] < self.safety_distance:
                return True
        return False

    def is_goal_reached(self):
        """Check if robot has reached the goal"""
        distance_to_goal = math.sqrt((self.position_x - self.goal_x)**2 +
                                    (self.position_y - self.goal_y)**2)
        return distance_to_goal < self.goal_tolerance

    def navigate_to_goal(self):
        """Navigate toward the goal using simple proportional control"""
        # Calculate desired direction
        dx = self.goal_x - self.position_x
        dy = self.goal_y - self.position_y
        desired_angle = math.atan2(dy, dx)

        # Calculate angle difference
        angle_diff = desired_angle - self.orientation
        # Normalize angle to [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        # Simple proportional controller for rotation
        self.angular_vel = max(-0.5, min(0.5, 2.0 * angle_diff))

        # Move forward if roughly aligned with goal
        if abs(angle_diff) < 0.5:  # 0.5 radians ~ 28 degrees
            self.linear_vel = 0.5
        else:
            self.linear_vel = 0.0

def main(args=None):
    rclpy.init(args=args)

    controller = PracticalRobotController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down Practical Robot Controller')
    finally:
        # Stop the robot before shutting down
        stop_msg = Twist()
        controller.cmd_vel_publisher.publish(stop_msg)
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

To run this example in a simulated environment:

1. Launch a ROS 2 robot simulation (e.g., TurtleBot3 in Gazebo)
2. Source your ROS 2 environment: `source /opt/ros/humble/setup.bash`
3. Run the controller: `python3 practical_robot_controller.py`

This practical example demonstrates the complete pipeline from sensor data processing to AI decision-making to motor command execution.

## Node Communication Patterns Diagrams

*Figure 3: Architecture showing how AI nodes communicate with robot controllers through various topics and services.*

*Figure 4: Detailed view of publisher-subscriber communication in robot control systems.*

These diagrams illustrate how AI decision-making nodes communicate with robot controllers through the ROS 2 publish-subscribe pattern, with proper Quality of Service settings for reliable communication in robotic systems.

## Summary and Cross-References

In this chapter, we've explored how to create ROS 2 nodes using Python and connect AI decision-making algorithms with robot controllers. We've covered:

- Using rclpy to build ROS 2 nodes with proper structure and best practices
- Implementing publisher-subscriber patterns for robot communication
- Bridging AI logic with robot controllers through various mapping techniques
- Converting high-level AI decisions to low-level motor actions

This builds upon the foundational concepts introduced in the [Introduction to ROS 2 as a Robotic Nervous System](./introduction.md) chapter, where you learned about:
- The core concepts of nodes, topics, services, and messages
- How data flows inside a humanoid robot system
- The role of ROS 2 as the nervous system of robots

In the next chapter, we'll explore how to model humanoid robot bodies using URDF, which will provide the physical structure these AI and control systems will operate on.

[Continue to Chapter 3: Modeling the Humanoid Body with URDF](./urdf-modeling.md)

## Additional Resources

- [Glossary of ROS 2 Terms](./glossary.md) - Definitions of key terminology used throughout this module
- [Module Summary](./summary.md) - Comprehensive overview of all concepts covered in this module

## Learning Objectives

By the end of this chapter, you will be able to:
- Create ROS 2 nodes using Python and the rclpy library
- Implement publisher and subscriber nodes to communicate with robot controllers
- Bridge AI decision-making algorithms with robot control systems
- Understand the conceptual mapping between AI decisions and motor actions

## Exercises for Hands-On Practice

To reinforce your understanding of Python agents with ROS 2, complete the following exercises:

### Exercise 1: Create a Simple Publisher Node
Create a ROS 2 Python node that publishes sensor-like data (e.g., temperature readings, distance measurements) to a topic. Include:
- A custom message type or use a standard message type
- A timer that publishes data at regular intervals
- Proper node initialization and cleanup
- Logging of published values

### Exercise 2: Create a Subscriber Node
Create a ROS 2 Python node that subscribes to the topic from Exercise 1 and processes the data:
- Subscribe to the topic created in Exercise 1
- Process the incoming data (e.g., calculate averages, detect thresholds)
- Log the processed results
- Handle potential message processing errors

### Exercise 3: Implement a Simple Robot Controller
Create a robot controller node that:
- Subscribes to sensor data (e.g., laser scan, camera image)
- Makes simple control decisions based on the sensor data
- Publishes velocity commands to control the robot
- Includes safety checks to prevent collisions

### Exercise 4: Add Parameters to Your Node
Enhance your robot controller from Exercise 3 by adding parameters:
- Add parameters for control gains, safety distances, or other configurable values
- Use `declare_parameter()` to define default values
- Access parameter values during runtime
- Test changing parameters without restarting the node

### Exercise 5: Implement a Service Client and Server
Create both a service server and client:
- Create a service definition file (.srv) for a simple robot action (e.g., move_to_position)
- Implement a service server that performs the action
- Implement a client that calls the service
- Handle service responses and potential errors

### Exercise 6: Quality of Service Configuration
Modify your nodes to use different QoS settings:
- Implement a publisher with reliable delivery
- Implement a subscriber with the corresponding QoS profile
- Compare performance between different QoS configurations
- Test behavior under simulated network conditions

## Troubleshooting Common Issues

When working with ROS 2 Python agents, you may encounter several common issues. Here are solutions to the most frequent problems:

### Import Errors
**Problem**: `ModuleNotFoundError: No module named 'rclpy'`
**Solution**:
- Ensure ROS 2 is properly installed and sourced: `source /opt/ros/<distro>/setup.bash`
- Install ROS 2 Python packages: `pip3 install ros-<distro>-rclpy`
- Check that your Python environment can access ROS 2 packages

### Node Connection Issues
**Problem**: Nodes cannot communicate with each other
**Solution**:
- Verify nodes are on the same ROS domain: `echo $ROS_DOMAIN_ID`
- Check that topic names match exactly (including case sensitivity)
- Ensure nodes are running in the same network namespace if using multi-robot systems
- Use `ros2 topic list` and `ros2 node list` to verify discovery

### Message Type Mismatches
**Problem**: Publisher and subscriber message types don't match
**Solution**:
- Verify that both publisher and subscriber use the same message type
- Check that custom message definitions are properly built and installed
- Use `ros2 interface show <msg_type>` to verify message structure
- Ensure all nodes are using the same version of message definitions

### Memory Leaks in Long-Running Nodes
**Problem**: Python nodes consume increasing memory over time
**Solution**:
- Properly destroy nodes in the main function: `node.destroy_node()`
- Use try/finally blocks to ensure cleanup happens even if exceptions occur
- Limit queue sizes for publishers and subscribers
- Consider using weak references for callbacks that capture large objects

### Threading and Concurrency Issues
**Problem**: Race conditions or unexpected behavior in multi-threaded nodes
**Solution**:
- ROS 2 Python nodes are generally single-threaded by default
- Use `rclpy.spin()` for single-threaded execution
- If multi-threading is required, use appropriate locks for shared data
- Consider using separate nodes instead of multi-threading within a single node

### Timer and Callback Issues
**Problem**: Timers not firing at expected intervals or callbacks not executing
**Solution**:
- Check timer period is set correctly (in seconds)
- Ensure the node is being spun with `rclpy.spin()` or similar
- Verify callback functions are defined correctly
- Monitor system load which may affect timer precision

### Parameter Issues
**Problem**: Parameters not being set or retrieved correctly
**Solution**:
- Always declare parameters before using them with `declare_parameter()`
- Provide default values for all parameters
- Use `get_parameter()` to retrieve parameter values
- Check parameter names for typos or case sensitivity

### Quality of Service Problems
**Problem**: Messages being dropped or not received reliably
**Solution**:
- Match QoS profiles between publishers and subscribers
- Use appropriate QoS settings for your use case (reliable vs. best-effort)
- Consider history depth for late-joining subscribers
- Test with different durability settings for transient data

### Resource Cleanup
**Problem**: Nodes not shutting down cleanly or leaving resources open
**Solution**:
- Always use try/except/finally blocks for proper cleanup
- Call `rclpy.shutdown()` when exiting
- Destroy nodes explicitly before shutting down
- Use signal handlers to catch Ctrl+C and perform cleanup

### Debugging Tips
- Use `self.get_logger().info()`, `warn()`, and `error()` for logging
- Use `rqt_console` to view logs from multiple nodes
- Use `rqt_graph` to visualize the node graph and connections
- Use `ros2 topic echo <topic_name>` to verify message content
- Use `ros2 doctor` to check ROS 2 system health