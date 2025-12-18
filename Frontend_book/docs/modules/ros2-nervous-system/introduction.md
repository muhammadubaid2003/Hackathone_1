---
id: introduction
title: Introduction to ROS 2 as a Robotic Nervous System
sidebar_label: Introduction to ROS 2
sidebar_position: 1
---

# Introduction to ROS 2 as a Robotic Nervous System

## What is ROS 2 and Why It Matters for Physical AI

Robot Operating System 2 (ROS 2) is not an operating system in the traditional sense, but rather a flexible framework for writing robot software. It provides a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms and environments.

For Physical AI and humanoid robotics, ROS 2 serves as the "nervous system" of the robot, enabling seamless communication between perception, decision-making, and action. Just as the human nervous system coordinates sensory input with motor responses, ROS 2 facilitates the flow of information between sensors, AI agents, and actuators in a humanoid robot.

ROS 2 is particularly important for Physical AI because it:
- Provides standardized interfaces for hardware abstraction
- Enables modular software development
- Offers tools for visualization, debugging, and simulation
- Supports real-time and safety-critical applications
- Facilitates integration with machine learning and AI frameworks

### The Evolution from ROS 1 to ROS 2

ROS 2 was developed to address the limitations of the original ROS framework, particularly in areas of:
- Real-time performance
- Multi-robot systems
- Commercial product development
- Deterministic behavior
- Security and authentication

These improvements make ROS 2 suitable for deployment in real-world applications, including humanoid robots that interact with humans in various environments.

### Physical AI and the Need for Robust Communication

Physical AI represents the convergence of artificial intelligence with physical systems. For AI to be truly effective, it must be embodied in physical agents that can interact with the real world. This requires:

- **Perception**: Understanding the environment through sensors
- **Cognition**: Processing information and making decisions
- **Action**: Executing physical movements and manipulations

ROS 2 provides the communication infrastructure that connects these three components, allowing AI agents to receive sensory input, process information, and control physical actuators in a coordinated manner.

## Core ROS 2 Concepts: Nodes, Topics, Services, and Messages

Understanding the fundamental building blocks of ROS 2 is crucial for developing effective robotic applications. These concepts form the backbone of the ROS 2 communication architecture.

### Nodes

A **node** is a process that performs computation in ROS 2. It's the basic unit of execution that can perform tasks such as:

- Reading data from sensors
- Processing sensor data
- Making decisions based on AI algorithms
- Controlling actuators
- Publishing or subscribing to data streams

In a humanoid robot, you might have nodes for:
- Camera processing
- Motor control
- Path planning
- Speech recognition
- Decision making

Nodes are designed to be modular and reusable, allowing complex robotic systems to be built from smaller, focused components.

### Topics and Publishing/Subscription Model

**Topics** enable asynchronous communication between nodes using a publish/subscribe pattern. Key characteristics include:

- **Publishers**: Nodes that send data to a topic
- **Subscribers**: Nodes that receive data from a topic
- **Messages**: The data structures exchanged on topics

For example, a camera node might publish image data to a `/camera/image_raw` topic, while multiple other nodes (object detection, SLAM, etc.) might subscribe to this topic to receive the image data simultaneously.

This decoupling allows for flexible system design where nodes can be added or removed without affecting others, as long as they adhere to the message types.

### Services

**Services** provide synchronous request/response communication between nodes. Unlike topics, services are:

- Request-response based (blocking until response received)
- Designed for tasks that require a specific response
- Useful for actions that should only happen once (e.g., calibration, activation)

A service consists of:
- A **service client** that sends a request
- A **service server** that processes the request and sends a response

For example, a humanoid robot might use a service to request the current position from a localization system.

### Messages

**Messages** are the data structures that are passed between nodes. They define:

- The format of data exchanged
- The fields and their data types
- How data is serialized and deserialized

ROS 2 provides many standard message types (sensors, geometry, navigation) and allows custom message definitions for specific applications.

### Communication Architecture Benefits

This architecture provides several advantages for humanoid robotics:

- **Decoupling**: Nodes don't need to know about each other directly
- **Scalability**: Multiple nodes can subscribe to the same topic
- **Robustness**: Failure of one node doesn't necessarily affect others
- **Flexibility**: Easy to add, remove, or replace nodes

## How Data Flows Inside a Humanoid Robot

Understanding data flow is crucial for designing effective humanoid robots. In a typical humanoid robot system, data flows through multiple layers and components, coordinated by ROS 2.

### Perception Layer

The perception layer is responsible for interpreting the robot's environment and internal state:

- **Sensors**: Cameras, LiDAR, IMU, force/torque sensors, joint encoders
- **Sensor processing nodes**: Convert raw sensor data into meaningful information
- **Perception algorithms**: Object detection, SLAM (Simultaneous Localization and Mapping), state estimation

For example, a humanoid robot might have:
- Stereo cameras publishing image streams to `/camera/left/image_raw` and `/camera/right/image_raw`
- An IMU publishing orientation data to `/imu/data`
- Joint encoders publishing position data to `/joint_states`

### Cognition Layer

The cognition layer processes information and makes decisions:

- **AI agents**: Path planning, behavior trees, decision-making algorithms
- **World modeling**: Maintaining an internal representation of the environment
- **Task planning**: Determining sequences of actions to achieve goals

For instance, a path planning node might:
- Subscribe to sensor data topics (`/scan`, `/map`, `/tf`)
- Process this information to determine safe paths
- Publish velocity commands to `/cmd_vel`

### Action Layer

The action layer executes physical movements:

- **Controller nodes**: Motor control, trajectory execution
- **Actuators**: Motors, servos, pneumatic/hydraulic systems
- **Feedback systems**: Monitoring actual vs. desired positions/forces

### Example Data Flow Scenario

Consider a humanoid robot navigating around an obstacle:

1. **Perception**: Camera nodes publish images to `/camera/image_raw`
2. **Processing**: Object detection node subscribes to images, detects obstacle, publishes to `/detected_objects`
3. **Decision**: Path planner subscribes to `/detected_objects` and `/map`, computes new path, publishes to `/move_base_simple/goal`
4. **Action**: Navigation stack processes goal, sends velocity commands to `/cmd_vel`
5. **Execution**: Motor control nodes receive velocity commands and actuate joints
6. **Feedback**: Joint encoders publish current positions to `/joint_states`, closing the loop

### ROS 2's Role as the Nervous System

Just as the human nervous system coordinates these functions, ROS 2 provides:

- **Real-time coordination**: Ensuring timely communication between perception, cognition, and action
- **Fault tolerance**: Allowing the system to continue operating even if some nodes fail
- **Modularity**: Enabling different teams to work on different components independently
- **Scalability**: Supporting systems from simple robots to complex multi-robot teams

## Basic ROS 2 Code Examples

To better understand how ROS 2 concepts work in practice, let's look at some basic code examples that demonstrate nodes, topics, and services.

### Simple Publisher Node Example

Here's a basic ROS 2 publisher node that publishes "Hello World" messages:

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

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Simple Subscriber Node Example

Here's a corresponding subscriber that receives the messages:

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

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Service Server Example

Here's a simple service server that adds two integers:

```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalService(Node):

    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info('Incoming request\na: %d b: %d' % (request.a, request.b))
        return response

def main(args=None):
    rclpy.init(args=args)
    minimal_service = MinimalService()
    rclpy.spin(minimal_service)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Service Client Example

Here's a client that calls the service:

```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalClientAsync(Node):

    def __init__(self):
        super().__init__('minimal_client_async')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main():
    rclpy.init()
    minimal_client = MinimalClientAsync()
    response = minimal_client.send_request(1, 2)
    minimal_client.get_logger().info(
        'Result of add_two_ints: for %d + %d = %d' %
        (1, 2, response.sum))
    minimal_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

These examples demonstrate the fundamental ROS 2 concepts in code. In a humanoid robot context, you would replace the simple "Hello World" messages with actual sensor data, control commands, or other relevant information specific to the robot's operation.

## ROS 2 Architecture Diagrams

*Figure 1: High-level overview of ROS 2 architecture showing nodes, topics, services, and the DDS communication layer.*

*Figure 2: Data flow in a humanoid robot system showing perception, cognition, and action layers.*

These diagrams illustrate the key concepts we've discussed:
- How nodes communicate through topics and services
- The layered architecture of a humanoid robot system
- The flow of information from sensors to actuators

## Learning Objectives

By the end of this chapter, you will be able to:
- Explain the core concepts of ROS 2: nodes, topics, services, and messages
- Describe how data flows between different components in a humanoid robot system
- Understand the role of ROS 2 in connecting AI agents to physical actuators and sensors

## Summary and Next Steps

In this chapter, we've covered the fundamental concepts of ROS 2 and its role as the nervous system of humanoid robots. We've explored:

- What ROS 2 is and why it matters for Physical AI
- The core concepts: nodes, topics, services, and messages
- How data flows inside a humanoid robot system
- Basic code examples demonstrating these concepts

In the next chapter, we'll dive deeper into how to connect AI agents to robot controllers using Python and the rclpy library, building on the foundational concepts we've learned here.

[Continue to Chapter 2: Controlling Robots with ROS 2 and Python Agents](./python-agents.md)

## Additional Resources

- [Glossary of ROS 2 Terms](./glossary.md) - Definitions of key terminology used throughout this module
- [Module Summary](./summary.md) - Comprehensive overview of all concepts covered in this module