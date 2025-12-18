---
id: physics-simulation-gazebo
title: Physics Simulation with Gazebo
sidebar_label: Physics Simulation with Gazebo
sidebar_position: 1
---

# Physics Simulation with Gazebo

## Introduction to Gazebo Physics

Gazebo is a powerful physics-based simulation environment that provides accurate and realistic simulation capabilities for robotics applications. For digital twin implementations in humanoid robotics, Gazebo serves as the core physics engine that models real-world physical interactions, enabling the creation of faithful virtual replicas of physical robots and their environments.

### Why Physics Simulation Matters for Digital Twins

Physics simulation is fundamental to creating effective digital twins because it enables:

- **Accurate Robot Behavior Prediction**: Simulating how robots will behave in real environments before deployment
- **Sensor Data Generation**: Creating realistic sensor data streams that match physical world conditions
- **Control Algorithm Validation**: Testing control algorithms in a safe, virtual environment
- **Scenario Testing**: Evaluating robot performance across diverse environmental conditions

## Gravity Models and Configuration

Gravity is the fundamental force that affects all objects in a simulated environment. In Gazebo, gravity is configured globally for the entire world but can be overridden for specific models or regions.

### Global Gravity Configuration

The global gravity vector is typically set to Earth's gravity (9.81 m/s²) in the negative Z direction:

```xml
<sdf version='1.7'>
  <world name='default'>
    <!-- Global gravity setting -->
    <gravity>0 0 -9.8</gravity>
    <!-- Rest of the world configuration -->
  </world>
</sdf>
```

### Custom Gravity for Specialized Scenarios

For applications requiring different gravitational forces (e.g., lunar or Martian environments), you can configure custom gravity values:

```xml
<gravity>0 0 -3.7</gravity>  <!-- Mars gravity -->
<gravity>0 0 -1.6</gravity>  <!-- Moon gravity -->
<gravity>0 0 -0.0</gravity>  <!-- Zero gravity (space) -->
```

## Collision Detection and Algorithms

Collision detection is critical for realistic physics simulation. Gazebo employs sophisticated algorithms to detect when objects intersect and respond appropriately.

### Collision Detection Types

Gazebo supports multiple collision detection engines:

1. **ODE (Open Dynamics Engine)**: Default engine, good for most applications
2. **Bullet**: Provides more robust collision detection for complex scenarios
3. **Simbody**: Advanced engine for biomechanical simulations
4. **DART**: Dynamic Animation and Robotics Toolkit for articulated bodies

### Collision Properties Configuration

Each link in a robot model can have specific collision properties defined:

```xml
<link name="link_name">
  <collision name="collision">
    <geometry>
      <box>
        <size>0.1 0.1 0.1</size>
      </box>
    </geometry>
    <surface>
      <friction>
        <ode>
          <mu>1.0</mu>
          <mu2>1.0</mu2>
        </ode>
      </friction>
      <bounce>
        <restitution_coefficient>0.1</restitution_coefficient>
        <threshold>100000</threshold>
      </bounce>
      <contact>
        <ode>
          <soft_cfm>0</soft_cfm>
          <soft_erp>0.2</soft_erp>
          <kp>1e+13</kp>
          <kd>1</kd>
          <max_vel>0.01</max_vel>
          <min_depth>0</min_depth>
        </ode>
      </contact>
    </surface>
  </collision>
</link>
```

### Collision Meshes vs Primitive Shapes

For complex geometries, you can use mesh collision shapes:

```xml
<collision name="mesh_collision">
  <geometry>
    <mesh>
      <uri>model://your_robot/meshes/part.dae</uri>
      <scale>1.0 1.0 1.0</scale>
    </mesh>
  </geometry>
</collision>
```

## Joint Types and Dynamics Simulation

Joints define how different parts of a robot connect and move relative to each other. Gazebo supports various joint types with different dynamic properties.

### Joint Types in Gazebo

1. **Revolute Joint**: Rotational movement around a single axis
2. **Prismatic Joint**: Linear movement along a single axis
3. **Spherical Joint**: Rotational movement around multiple axes
4. **Universal Joint**: Two degrees of rotational freedom
5. **Fixed Joint**: No movement between connected links
6. **Continuous Joint**: Unlimited rotational movement
7. **Floating Joint**: Six degrees of freedom (not commonly used)

### Joint Dynamics Configuration

Each joint can have detailed dynamic properties:

```xml
<joint name="joint_name" type="revolute">
  <parent>parent_link</parent>
  <child>child_link</child>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <axis xyz="0 0 1">
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0"/>
    <dynamics damping="0.1" friction="0.0"/>
  </axis>
</joint>
```

### Joint Actuation and Control

Joints can be actuated using various control methods:

```xml
<!-- Transmission for joint control -->
<transmission name="transmission_joint_1">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="joint_name">
    <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
  </joint>
  <actuator name="actuator_name">
    <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

## Dynamics Simulation Parameters

Proper dynamics simulation requires careful configuration of various parameters to ensure realistic behavior.

### Material Properties

Different materials have different physical properties that affect simulation:

```xml
<gazebo reference="link_name">
  <material>Gazebo/Blue</material>
  <!-- Material properties -->
  <mu1>1.0</mu1>  <!-- Primary friction coefficient -->
  <mu2>1.0</mu2>  <!-- Secondary friction coefficient -->
  <kp>1000000.0</kp>  <!-- Contact stiffness -->
  <kd>100.0</kd>      <!-- Contact damping -->
  <maxVel>1.0</maxVel> <!-- Maximum contact penetration velocity -->
  <minDepth>0.001</minDepth> <!-- Minimum contact penetration depth -->
</gazebo>
```

### Simulation Performance vs Accuracy

Balance between simulation accuracy and performance:

- **Real-time factor**: Adjust the real-time update rate for optimal performance
- **ODE parameters**: Tune ODE parameters for stability and performance
- **Contact resolution**: Configure contact resolution parameters for accuracy

## Running Humanoid Robots in Simulated Worlds

Creating realistic humanoid robot simulations requires careful attention to world setup, robot configuration, and environmental factors.

### World Configuration

A typical world file for humanoid simulation includes:

```xml
<sdf version='1.7'>
  <world name='humanoid_world'>
    <!-- Gravity -->
    <gravity>0 0 -9.8</gravity>

    <!-- Physics engine configuration -->
    <physics name='default_physics' type='ode'>
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- Include models -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Your humanoid robot model -->
    <include>
      <uri>model://humanoid_robot</uri>
      <pose>0 0 1 0 0 0</pose>
    </include>

    <!-- Additional environment objects -->
    <model name='table'>
      <pose>2 0 0 0 0 0</pose>
      <include>
        <uri>model://table</uri>
      </include>
    </model>
  </world>
</sdf>
```

### Launching Simulations

To launch a humanoid robot simulation in Gazebo:

```bash
# Launch Gazebo with a specific world file
gzserver --verbose /path/to/your/world_file.world

# Or use ROS 2 launch system
ros2 launch your_robot_gazebo your_robot_world.launch.py
```

### Realistic Humanoid Simulation Considerations

For realistic humanoid robot simulation, consider:

1. **Proper mass distribution**: Accurate inertial properties for each link
2. **Realistic joint limits**: Match physical robot capabilities
3. **Appropriate control gains**: Tuned for stable control
4. **Environmental complexity**: Realistic terrain and objects
5. **Sensor simulation**: Accurate sensor models with realistic noise

## Practical Example: Simple Humanoid Model in Gazebo

Here's a complete example of a simple humanoid model configuration for Gazebo:

```xml
<?xml version="1.0" ?>
<robot name="simple_humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Base link -->
  <link name="base_link">
    <inertial>
      <mass value="10.0"/>
      <origin xyz="0 0 0.25" rpy="0 0 0"/>
      <inertia ixx="0.2" ixy="0.0" ixz="0.0" iyy="0.2" iyz="0.0" izz="0.2"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.25" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.2 0.5"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0.25" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.2 0.5"/>
      </geometry>
    </collision>
  </link>

  <!-- Hip joint -->
  <joint name="hip_joint" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.25" rpy="0 0 0"/>
  </joint>

  <!-- Torso -->
  <link name="torso">
    <inertial>
      <mass value="5.0"/>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.2 0.6"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0.3" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.2 0.6"/>
      </geometry>
    </collision>
  </link>

  <!-- Head -->
  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.6" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="10" velocity="1"/>
  </joint>

  <link name="head">
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
    </collision>
  </link>
</robot>
```

## Launching and Managing Simulations

### Basic Simulation Launch

```bash
# Launch Gazebo with empty world
gzserver --verbose

# Launch with specific world
gzserver --verbose /path/to/world_file.world

# Launch with GUI
gzclient &
```

### Using ROS 2 Integration

```python
# Example ROS 2 launch file for Gazebo simulation
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    pkg_gazebo_ros = FindPackageShare('gazebo_ros').find('gazebo_ros')
    pkg_share = FindPackageShare('your_robot_description').find('your_robot_description')

    # World file path
    world_file = os.path.join(pkg_share, 'worlds', 'humanoid_world.world')

    # Launch Gazebo server
    start_gazebo_server_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')),
        launch_arguments={'world': world_file}.items()
    )

    # Launch Gazebo client
    start_gazebo_client_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py'))
    )

    return LaunchDescription([
        start_gazebo_server_cmd,
        start_gazebo_client_cmd
    ])
```

## Summary and Next Steps

Physics simulation with Gazebo provides the foundation for creating realistic digital twins of humanoid robots. By properly configuring gravity, collision detection, and joint dynamics, you can create simulations that accurately reflect real-world robot behavior.

In the next chapter, we'll explore how to enhance these physics simulations with high-fidelity visualization using Unity, creating even more realistic digital twin experiences.

## Learning Objectives

By the end of this chapter, you should be able to:
- Configure gravity models for different simulation environments
- Set up collision detection with appropriate parameters
- Define joint types and dynamics for humanoid robots
- Launch and manage physics-based simulations in Gazebo
- Understand the relationship between physics parameters and realistic robot behavior

## Summary and Next Steps

Physics simulation with Gazebo provides the foundation for creating realistic digital twins of humanoid robots. By properly configuring gravity, collision detection, and joint dynamics, you can create simulations that accurately reflect real-world robot behavior.

[Continue to Chapter 2: High-Fidelity Interaction with Unity](./high-fidelity-unity.md)

## Additional Resources

- [Glossary of Digital Twin Terms](./glossary.md) - Definitions of key terminology used throughout this module
- [Module Summary](./summary.md) - Comprehensive overview of all concepts covered in this module

## Exercises for Hands-On Practice

To reinforce your understanding of physics simulation with Gazebo, complete the following exercises:

### Exercise 1: Configure a Custom Gravity Environment
Create a Gazebo world file with custom gravity settings for a different planet (e.g., Mars with gravity of -3.7 m/s²). Test how this affects robot movement and object interactions.

### Exercise 2: Implement Collision Detection Parameters
Create a robot model with custom collision properties for different materials (e.g., rubber, metal, wood). Adjust friction coefficients and restitution values to see how they affect interactions.

### Exercise 3: Design Joint Constraints
Create a simple robot arm with different joint types (revolute, prismatic, continuous) and configure their limits and dynamics to achieve specific movement patterns.

### Exercise 4: Simulate a Humanoid Robot in Various Environments
Create multiple world files with different terrains (flat ground, stairs, uneven terrain) and test how your humanoid robot model behaves in each environment.

### Exercise 5: Optimize Simulation Performance
Experiment with different physics engine parameters (ODE, Bullet) and adjust real-time update rates to find the optimal balance between accuracy and performance.

### Exercise 6: Implement Sensor Integration
Add sensors to your robot model (IMU, camera, LiDAR) and verify that they function correctly within the physics simulation environment.

## Troubleshooting Common Issues

When working with Gazebo physics simulation, you may encounter several common issues. Here are solutions to the most frequent problems:

### Simulation Instability
**Problem**: Robot joints are jittery or unstable during simulation.
**Solution**:
- Check joint limits and ensure they're properly configured
- Adjust physics engine parameters (increase solver iterations)
- Verify mass and inertia properties are realistic
- Reduce simulation time step if necessary

### Collision Issues
**Problem**: Objects pass through each other or unexpected collisions occur.
**Solution**:
- Verify collision geometries match visual geometries
- Check that all links have proper collision elements defined
- Adjust surface contact parameters (kp, kd values)
- Ensure proper mesh scaling for collision models

### Performance Problems
**Problem**: Simulation runs slowly or drops frames.
**Solution**:
- Reduce the number of complex collision meshes
- Adjust real-time update rate and max step size
- Use simpler collision shapes where possible
- Consider using faster physics engines like Bullet

### Robot Falling Through Ground
**Problem**: Robot falls through the ground plane.
**Solution**:
- Verify the ground plane model is properly loaded
- Check that the robot's initial position is above the ground
- Adjust contact parameters for the ground plane
- Ensure proper mass and inertia values for the robot