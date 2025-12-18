---
id: urdf-modeling
title: Modeling the Humanoid Body with URDF
sidebar_label: URDF Modeling
sidebar_position: 3
---

# Modeling the Humanoid Body with URDF

## Purpose of URDF in Humanoid Robotics

Before diving into URDF modeling, it's important to understand the foundational concepts of ROS 2 and how AI agents interact with robot controllers. If you haven't already, review:

- [Introduction to ROS 2 as a Robotic Nervous System](./introduction.md) - for understanding the core concepts of ROS 2, nodes, topics, services, and messages
- [Controlling Robots with ROS 2 and Python Agents](./python-agents.md) - for understanding how to bridge AI logic with robot controllers using rclpy

Unified Robot Description Format (URDF) is an XML-based format used to describe robot models in ROS. In humanoid robotics, URDF serves as the digital blueprint that defines the physical structure, kinematic properties, and visual representation of the robot. It's essential for both simulation and real-world robot control.

### What is URDF?

URDF stands for Unified Robot Description Format. It's an XML-based format that describes a robot in terms of:

- **Links**: Rigid bodies that make up the robot (e.g., torso, arms, legs)
- **Joints**: Connections between links that allow relative motion
- **Visual and collision properties**: How the robot looks and interacts with its environment
- **Inertial properties**: Mass, center of mass, and moments of inertia for each link
- **Sensors and actuators**: Where sensors and motors are mounted on the robot

### Why URDF is Critical for Humanoid Robots

Humanoid robots present unique challenges that make URDF particularly important:

1. **Complex Kinematics**: Humanoid robots have many degrees of freedom (DOF) that must be precisely defined to enable proper motion planning and control.

2. **Dynamic Balance**: Unlike wheeled robots, humanoid robots must maintain balance during locomotion, requiring accurate inertial properties.

3. **Collision Avoidance**: With multiple limbs that can collide with each other and the environment, detailed collision models are essential.

4. **Simulation Accuracy**: Realistic simulation is crucial for humanoid robots since physical testing can be dangerous and expensive.

### Core Components of a URDF

A URDF file typically contains:

```xml
<?xml version="1.0"?>
<robot name="my_robot">
  <!-- Links define rigid bodies -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Joints define connections between links -->
  <joint name="base_to_wheel" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_link"/>
    <origin xyz="0 0.25 -0.05" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <link name="wheel_link">
    <visual>
      <geometry>
        <cylinder length="0.1" radius="0.1"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.1" radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>
</robot>
```

### URDF in the ROS Ecosystem

URDF integrates with various ROS tools and packages:

- **RViz**: Visualizes the robot model in 3D
- **Gazebo**: Provides physics simulation based on URDF models
- **MoveIt!**: Uses URDF for motion planning and inverse kinematics
- **Robot State Publisher**: Publishes TF transforms based on joint states and URDF
- **Controllers**: Use URDF to understand robot kinematics for control

### Best Practices for Humanoid URDF Design

1. **Start Simple**: Begin with a basic skeleton and add complexity gradually
2. **Use Standard Dimensions**: Base your model on actual robot measurements
3. **Include Proper Inertial Properties**: Essential for accurate simulation
4. **Group Related Parts**: Organize your URDF logically (e.g., all head parts in one section)
5. **Use Xacro for Complex Models**: Xacro (XML Macros) can simplify complex humanoid models

## Links, Joints, Sensors, and Actuators

URDF models are composed of several fundamental elements that define the robot's physical structure and capabilities. Understanding these elements is crucial for creating accurate and functional humanoid robot models.

### Links

Links represent the rigid bodies of the robot. In a humanoid robot, links correspond to physical parts such as:

- **Torso**: The main body segment
- **Head**: Contains cameras, IMUs, and other sensors
- **Arms**: Upper arm, lower arm, and hand segments
- **Legs**: Thigh, shank, and foot segments
- **Spine**: For robots with articulated backs

Each link has several properties:

#### Visual Properties
The visual element defines how the link appears in visualization tools like RViz and simulation environments:

```xml
<link name="upper_arm">
  <visual>
    <origin xyz="0 0 0.15" rpy="0 0 0"/>
    <geometry>
      <mesh filename="package://my_robot_description/meshes/upper_arm.dae"/>
    </geometry>
    <material name="gray">
      <color rgba="0.5 0.5 0.5 1.0"/>
    </material>
  </visual>
</link>
```

#### Collision Properties
The collision element defines the shape used for collision detection:

```xml
<link name="upper_arm">
  <collision>
    <origin xyz="0 0 0.15" rpy="0 0 0"/>
    <geometry>
      <cylinder length="0.3" radius="0.05"/>
    </geometry>
  </collision>
</link>
```

#### Inertial Properties
The inertial element is crucial for simulation and defines the physical properties of the link:

```xml
<link name="upper_arm">
  <inertial>
    <origin xyz="0 0 0.15" rpy="0 0 0"/>
    <mass value="2.0"/>
    <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.005"/>
  </inertial>
</link>
```

### Joints

Joints define the connections between links and specify the allowed motion. For humanoid robots, the most common joint types are:

#### Fixed Joints
Fixed joints connect two links without allowing relative motion:

```xml
<joint name="head_to_camera" type="fixed">
  <parent link="head"/>
  <child link="camera_link"/>
  <origin xyz="0.05 0 0.1" rpy="0 0 0"/>
</joint>
```

#### Revolute Joints
Revolute joints allow rotation around a single axis with position limits:

```xml
<joint name="shoulder_pitch" type="revolute">
  <parent link="torso"/>
  <child link="upper_arm"/>
  <origin xyz="0 0.15 0.3" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="-2.0" upper="1.5" effort="100" velocity="2.0"/>
  <dynamics damping="0.5" friction="0.1"/>
</joint>
```

#### Continuous Joints
Continuous joints allow unlimited rotation around an axis (like a wheel):

```xml
<joint name="head_pan" type="continuous">
  <parent link="head"/>
  <child link="camera_link"/>
  <origin xyz="0.05 0 0.1" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <dynamics damping="0.2" friction="0.05"/>
</joint>
```

#### Prismatic Joints
Prismatic joints allow linear motion along an axis:

```xml
<joint name="linear_slider" type="prismatic">
  <parent link="torso"/>
  <child link="slider_mount"/>
  <origin xyz="0 0 0.2" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="0" upper="0.5" effort="50" velocity="1.0"/>
</joint>
```

### Sensors in URDF

Sensors are represented as additional links connected to the robot body via fixed joints. They don't affect the robot's dynamics but provide reference frames for sensor data:

```xml
<!-- IMU Sensor -->
<joint name="imu_joint" type="fixed">
  <parent link="torso"/>
  <child link="imu_link"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
</joint>

<link name="imu_link">
  <inertial>
    <mass value="0.01"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
  </inertial>
</link>

<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
  </sensor>
</gazebo>
```

### Actuators and Transmission Elements

While URDF doesn't directly model actuators, it can include transmission elements that define how joints are controlled:

```xml
<transmission name="shoulder_pitch_trans">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="shoulder_pitch">
    <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
  </joint>
  <actuator name="shoulder_pitch_motor">
    <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

### Complete Example: Simple Humanoid Arm

Here's a complete example of a simple humanoid arm with links, joints, and basic properties:

```xml
<?xml version="1.0"?>
<robot name="simple_arm">
  <!-- Base of the arm -->
  <link name="arm_base">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
      <material name="dark_gray">
        <color rgba="0.3 0.3 0.3 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Shoulder joint -->
  <joint name="shoulder_pitch" type="revolute">
    <parent link="arm_base"/>
    <child link="upper_arm"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="2.0"/>
  </joint>

  <!-- Upper arm -->
  <link name="upper_arm">
    <visual>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
      <material name="light_gray">
        <color rgba="0.7 0.7 0.7 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.3" radius="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.005"/>
    </inertial>
  </link>

  <!-- Elbow joint -->
  <joint name="elbow_pitch" type="revolute">
    <parent link="upper_arm"/>
    <child link="lower_arm"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.5" upper="0.5" effort="80" velocity="2.5"/>
  </joint>

  <!-- Lower arm -->
  <link name="lower_arm">
    <visual>
      <geometry>
        <cylinder length="0.25" radius="0.04"/>
      </geometry>
      <material name="light_gray">
        <color rgba="0.7 0.7 0.7 1.0"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.25" radius="0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.008" ixy="0.0" ixz="0.0" iyy="0.008" iyz="0.0" izz="0.003"/>
    </inertial>
  </link>
</robot>
```

## How URDF Enables Simulation and Control Consistency

URDF serves as the single source of truth that ensures consistency between simulation and real-world robot control. This consistency is crucial for humanoid robots, where extensive testing in simulation is necessary before deploying on physical hardware.

### Kinematic Consistency

URDF defines the kinematic structure of the robot, which is used by both simulation and control systems:

1. **Forward Kinematics**: Given joint angles, calculate the position and orientation of end-effectors
2. **Inverse Kinematics**: Given desired end-effector position, calculate required joint angles
3. **Jacobian Calculations**: Determine how joint velocities affect end-effector velocities

Both simulation environments (like Gazebo) and real robot controllers use the same URDF model to perform these calculations, ensuring that movements planned in simulation will closely match those on the real robot.

### Dynamic Consistency

The inertial properties defined in URDF are critical for:

1. **Physics Simulation**: Accurate modeling of robot dynamics in simulation
2. **Controller Tuning**: Properly tuned controllers based on real robot dynamics
3. **Trajectory Planning**: Account for robot dynamics when planning movements

```xml
<link name="thigh">
  <inertial>
    <origin xyz="0 0 -0.15" rpy="0 0 0"/>
    <mass value="5.0"/>
    <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.05"/>
  </inertial>
</link>
```

### Sensor Consistency

URDF defines sensor positions and orientations, ensuring that:

1. **Sensor Data Interpretation**: Same coordinate frames used in simulation and reality
2. **Perception Algorithms**: Same reference frames for object detection, mapping, etc.
3. **Sensor Fusion**: Consistent data fusion across simulation and real robot

### Control Consistency

URDF provides the foundation for consistent control by defining:

1. **Joint Limits**: Same limits applied in simulation and on real robot
2. **Joint Types**: Ensuring the same motion capabilities
3. **Transmission Specifications**: Same control interfaces and hardware interfaces

### The Robot State Publisher

The `robot_state_publisher` package uses URDF to publish TF (Transform) frames based on joint states, providing consistent coordinate transformations across the system:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
from robot_state_publisher import RobotStatePublisher

class MyRobotStatePublisher(Node):
    def __init__(self):
        super().__init__('robot_state_publisher')

        # Subscribe to joint states
        self.subscription = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )

        # Initialize robot state publisher
        self.tf_broadcaster = TransformBroadcaster(self)
        # The robot state publisher will automatically use the URDF
        # from the parameter server to publish transforms

    def joint_state_callback(self, msg):
        # Process joint states and publish transforms
        pass
```

### Consistency Verification Techniques

To ensure consistency between simulation and reality:

1. **Model Validation**: Compare URDF model with actual robot measurements
2. **Parameter Identification**: Use system identification to refine inertial properties
3. **Sensor Calibration**: Ensure sensor positions in URDF match real positions
4. **Controller Testing**: Test controllers in simulation before deployment

### Common Consistency Issues and Solutions

#### Issue: Simulation vs. Reality Differences
**Solution**: Regularly update URDF with real robot measurements and calibrated parameters

#### Issue: Joint Limit Mismatches
**Solution**: Ensure URDF joint limits match physical robot capabilities

#### Issue: Inertial Property Errors
**Solution**: Use system identification tools to measure real inertial properties

#### Issue: Sensor Frame Misalignments
**Solution**: Calibrate sensor positions and orientations using calibration tools

## Preparing Robot Models for Gazebo and Isaac

To run your URDF models in simulation environments like Gazebo and Isaac, you need to add specific tags and configurations that these simulators understand. This section covers the essential elements for preparing your models for these environments.

### Gazebo-Specific Elements

Gazebo requires special tags within your URDF to properly simulate the robot. These tags are added using the `<gazebo>` element:

#### Gazebo Materials and Visual Properties

```xml
<link name="link_name">
  <visual>
    <geometry>
      <mesh filename="package://my_robot_description/meshes/link_name.dae"/>
    </geometry>
    <material name="Red">
      <color rgba="0.8 0.0 0.0 1.0"/>
    </material>
  </visual>
</link>

<gazebo reference="link_name">
  <material>Gazebo/Red</material>
  <turnGravityOff>false</turnGravityOff>
</gazebo>
```

#### Gazebo Sensors

To add sensors that work in Gazebo, you need to include sensor specifications within the gazebo tags:

```xml
<!-- Camera Sensor -->
<gazebo reference="camera_link">
  <sensor name="camera1" type="camera">
    <update_rate>30.0</update_rate>
    <camera name="head">
      <horizontal_fov>1.3962634</horizontal_fov>
      <image>
        <width>800</width>
        <height>600</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>camera_optical_frame</frame_name>
    </plugin>
  </sensor>
</gazebo>

<!-- IMU Sensor -->
<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
  </sensor>
</gazebo>
```

#### Gazebo Controllers

To control joints in Gazebo, you need to add transmission elements and Gazebo plugins:

```xml
<!-- Transmission for a joint -->
<transmission name="wheel_trans">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="wheel_joint">
    <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
  </joint>
  <actuator name="wheel_motor">
    <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>

<!-- Gazebo plugin for joint control -->
<gazebo>
  <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
    <robotNamespace>/my_robot</robotNamespace>
  </plugin>
</gazebo>
```

#### Gazebo Physics Properties

You can also specify physics properties for links in Gazebo:

```xml
<gazebo reference="link_name">
  <mu1>0.9</mu1>
  <mu2>0.9</mu2>
  <kp>1000000.0</kp>
  <kd>100.0</kd>
  <minDepth>0.001</minDepth>
  <maxVel>1.0</maxVel>
</gazebo>
```

### Isaac-Specific Elements

Isaac ROS (part of the Isaac ecosystem) uses different mechanisms to integrate with ROS 2. For Isaac, you'll need to prepare your robot model with specific configurations:

#### Isaac Sensor Configuration

Isaac ROS uses specific sensor bridge packages to connect Gazebo sensors to Isaac. The URDF itself doesn't change significantly, but you'll need additional launch files and configuration files:

```xml
<!-- In URDF, sensors are defined similarly to Gazebo -->
<link name="rgb_camera_link">
  <inertial>
    <mass value="0.01"/>
    <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
  </inertial>
</link>

<joint name="camera_joint" type="fixed">
  <parent link="head"/>
  <child link="rgb_camera_link"/>
  <origin xyz="0.1 0 0" rpy="0 0 0"/>
</joint>

<!-- Gazebo sensor definition (Isaac will use this through bridges) -->
<gazebo reference="rgb_camera_link">
  <sensor name="rgb_camera" type="camera">
    <update_rate>30</update_rate>
    <camera name="camera">
      <horizontal_fov>1.089</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>rgb_camera_link</frame_name>
      <min_depth>0.1</min_depth>
      <max_depth>100</max_depth>
    </plugin>
  </sensor>
</gazebo>
```

### Complete Example: URDF for Gazebo Simulation

Here's a complete example of a URDF that's properly prepared for Gazebo simulation:

```xml
<?xml version="1.0"?>
<robot name="gazebo_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.2" radius="0.15"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.2" radius="0.15"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.15"/>
    </inertial>
  </link>

  <!-- Gazebo plugin for the base link -->
  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
  </gazebo>

  <!-- Wheel joints and links -->
  <joint name="wheel_left_joint" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_left"/>
    <origin xyz="0 0.18 -0.05" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <link name="wheel_left">
    <visual>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 0.8"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.05" radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <!-- Gazebo plugins for wheels -->
  <gazebo reference="wheel_left">
    <material>Gazebo/Black</material>
    <mu1>1.0</mu1>
    <mu2>1.0</mu2>
    <kp>1000000.0</kp>
    <kd>100.0</kd>
  </gazebo>

  <!-- Transmission for wheel -->
  <transmission name="wheel_left_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="wheel_left_joint">
      <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
    </joint>
    <actuator name="wheel_left_motor">
      <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <!-- Gazebo plugin for ROS control -->
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/gazebo_robot</robotNamespace>
    </plugin>
  </gazebo>

  <!-- Gazebo plugin for ground truth odometry -->
  <gazebo>
    <plugin name="ground_truth_odom" filename="libgazebo_ros_p3d.so">
      <alwaysOn>true</alwaysOn>
      <updateRate>30</updateRate>
      <bodyName>base_link</bodyName>
      <topicName>odom_ground_truth</topicName>
      <gaussianNoise>0.01</gaussianNoise>
      <frameName>map</frameName>
    </plugin>
  </gazebo>

</robot>
```

### Launching in Gazebo

To launch your robot model in Gazebo, you'll typically need a launch file that:

1. Spawns the robot model in Gazebo
2. Loads robot controllers
3. Starts necessary nodes (robot state publisher, controllers, etc.)

```python
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    pkg_gazebo_ros = FindPackageShare('gazebo_ros').find('gazebo_ros')
    pkg_share = FindPackageShare('my_robot_description').find('my_robot_description')

    # Model path
    model_path = os.path.join(pkg_share, 'urdf', 'my_robot.urdf')

    # Launch configuration variables
    use_rviz = LaunchConfiguration('use_rviz')
    use_sim_time = LaunchConfiguration('use_sim_time')

    # Declare launch arguments
    declare_use_rviz_cmd = DeclareLaunchArgument(
        name='use_rviz',
        default_value='True',
        description='Whether to start RViz')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        name='use_sim_time',
        default_value='True',
        description='Use simulation (Gazebo) clock if true')

    # Start Gazebo with empty world
    start_gazebo_server_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')),
    )

    start_gazebo_client_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')),
    )

    # Robot State Publisher node
    robot_state_publisher_cmd = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time,
                    'robot_description': open(model_path).read()}]
    )

    # Spawn the robot in Gazebo
    spawn_entity_cmd = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-entity', 'my_robot',
                   '-file', model_path,
                   '-x', '0', '-y', '0', '-z', '0.5'],
        output='screen'
    )

    # RViz node
    rviz_cmd = Node(
        condition=IfCondition(use_rviz),
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen'
    )

    # Create the launch description and populate
    ld = LaunchDescription()

    # Declare the launch options
    ld.add_action(declare_use_rviz_cmd)
    ld.add_action(declare_use_sim_time_cmd)

    # Add any conditioned actions
    ld.add_action(start_gazebo_server_cmd)
    ld.add_action(start_gazebo_client_cmd)
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(spawn_entity_cmd)
    ld.add_action(rviz_cmd)

    return ld
```

## URDF Structure and Kinematics Diagrams

*Figure 5: Overview of URDF structure showing links, joints, and their hierarchical relationship.*

*Figure 6: Kinematic chain representation of a humanoid robot showing the connection between torso, arms, legs, and head.*

*Figure 7: The pipeline from URDF definition to simulation in Gazebo, showing how visual, collision, and inertial properties are used.*

These diagrams illustrate the key concepts of URDF structure and how it represents the kinematic properties of humanoid robots, from the basic link-joint relationships to the complete simulation pipeline.

## Learning Objectives

By the end of this chapter, you will be able to:
- Understand the purpose and structure of URDF in humanoid robotics
- Create URDF models with links, joints, sensors, and actuators
- Prepare robot models for simulation in environments like Gazebo and Isaac
- Understand how URDF enables simulation and control consistency

## Summary and Next Steps

In this chapter, we've covered the fundamentals of URDF modeling for humanoid robots, including how to define links, joints, sensors, and actuators. We've also explored how URDF enables consistency between simulation and real-world control, and how to prepare models for Gazebo and Isaac simulation environments.

This completes our module on ROS 2 as a robotic nervous system. You now have a comprehensive understanding of:

- The core concepts of ROS 2 and its role as a robotic nervous system ([Introduction](./introduction.md))
- How to control robots using Python agents and rclpy ([Python Agents](./python-agents.md))
- How to model humanoid robots using URDF ([URDF Modeling](./urdf-modeling.md))

These concepts work together to create the foundation for connecting AI agents to physical actuators and sensors in humanoid robots, enabling the embodiment of AI in physical systems.

## Additional Resources

- [Glossary of ROS 2 Terms](./glossary.md) - Definitions of key terminology used throughout this module
- [Module Summary](./summary.md) - Comprehensive overview of all concepts covered in this module

## Exercises for Hands-On Practice

To reinforce your understanding of URDF modeling, complete the following exercises:

### Exercise 1: Create a Simple Two-Wheeled Robot
Create a URDF file for a simple two-wheeled robot with a rectangular body and two wheels. Include:
- A base link for the robot body
- Two wheel links and joints
- Visual and collision properties
- Inertial properties for each link
- Transmission elements for the wheels

You can find a sample solution in [simple_robot.urdf](/files/simple_robot.urdf).

### Exercise 2: Model a Simple Robotic Arm
Create a URDF file for a 3-DOF robotic arm with:
- A fixed base
- An upper arm link connected by a revolute joint
- A lower arm link connected by another revolute joint
- A gripper or end-effector
- Proper joint limits and visual properties

You can find a sample solution in [simple_arm.urdf](/files/simple_arm.urdf).

### Exercise 3: Validate Your URDF Model
Use the following ROS 2 tools to validate your URDF models:

1. Check for XML syntax errors:
   ```bash
   xmllint --noout your_model.urdf
   ```

2. Check for URDF-specific errors:
   ```bash
   check_urdf your_model.urdf
   ```

3. Visualize your robot model in RViz:
   ```bash
   # Launch robot state publisher with your URDF
   ros2 launch your_robot_description display.launch.py
   ```

### Exercise 4: Add Sensors to Your Model
Enhance your robot model by adding sensor links and joints:
- Add an IMU sensor to the robot body
- Add a camera to the robot (position it appropriately)
- Include proper sensor frames for TF transformations

### Exercise 5: Prepare for Simulation
Add the necessary Gazebo-specific tags to your URDF:
- Add `<gazebo>` tags for visual materials
- Include sensor definitions with appropriate noise parameters
- Add transmission elements for joint control
- Include the gazebo_ros_control plugin

### Exercise 6: Create a Launch File for Your Robot
Create a launch file that:
- Loads your robot description parameter
- Starts the robot_state_publisher
- Optionally launches RViz for visualization

Example structure:
```python
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Load the URDF file
    urdf_file = os.path.join(
        get_package_share_directory('your_robot_description'),
        'urdf',
        'your_robot.urdf'
    )

    with open(urdf_file, 'r') as infp:
        robot_desc = infp.read()

    # Robot State Publisher node
    rsp_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_desc}]
    )

    return LaunchDescription([rsp_node])
```

These exercises will help you practice creating, validating, and preparing URDF models for real-world robotics applications. Each exercise builds on the concepts covered in this chapter and provides hands-on experience with the tools and techniques used in professional robotics development.