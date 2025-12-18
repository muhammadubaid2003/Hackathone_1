---
id: sensor-simulation
title: Sensor Simulation for Perception
sidebar_label: Sensor Simulation for Perception
sidebar_position: 3
---

# Sensor Simulation for Perception

## Introduction to Sensor Simulation in Digital Twins

Sensor simulation is a critical component of digital twin technology for robotics, providing realistic sensor data that mirrors the physical world. In humanoid robotics applications, accurate sensor simulation enables the development and testing of perception algorithms in a safe, controlled environment before deployment on physical robots.

### Why Sensor Simulation Matters

Sensor simulation in digital twins serves several crucial purposes:

- **Algorithm Development**: Test perception algorithms without physical hardware
- **Safety**: Validate sensor-based behaviors in virtual environments
- **Cost Reduction**: Minimize the need for expensive physical testing
- **Scenario Testing**: Create diverse environmental conditions for robustness testing
- **Data Generation**: Generate large datasets for training machine learning models
- **Hardware Abstraction**: Develop algorithms that work across different sensor configurations

## LiDAR Simulation

Light Detection and Ranging (LiDAR) sensors are essential for 3D mapping, localization, and obstacle detection in robotics. Simulating LiDAR data accurately is crucial for developing robust perception systems.

### LiDAR Physics and Characteristics

Realistic LiDAR simulation must account for:

- **Beam Propagation**: How laser beams travel through space
- **Reflection Properties**: How different materials reflect laser light
- **Range Limitations**: Maximum and minimum detection distances
- **Angular Resolution**: The precision of angle measurements
- **Noise Characteristics**: Realistic noise patterns in measurements

### Gazebo LiDAR Simulation

Gazebo provides built-in LiDAR sensor simulation with realistic properties:

```xml
<!-- Example LiDAR sensor configuration for Gazebo -->
<gazebo reference="lidar_link">
  <sensor name="lidar_sensor" type="ray">
    <pose>0 0 0 0 0 0</pose>
    <visualize>true</visualize>
    <update_rate>10</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>720</samples>
          <resolution>1</resolution>
          <min_angle>-3.14159</min_angle>
          <max_angle>3.14159</max_angle>
        </horizontal>
      </scan>
      <range>
        <min>0.1</min>
        <max>30.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="lidar_controller" filename="libgazebo_ros_ray_sensor.so">
      <ros>
        <namespace>/lidar</namespace>
        <remapping>~/out:=scan</remapping>
      </ros>
      <output_type>sensor_msgs/LaserScan</output_type>
      <frame_name>lidar_link</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

### Unity LiDAR Simulation

Unity can simulate LiDAR sensors using raycasting techniques:

```csharp
using UnityEngine;
using System.Collections.Generic;

public class UnityLidarSimulation : MonoBehaviour
{
    [Header("Lidar Configuration")]
    public int horizontalSamples = 720;
    public int verticalSamples = 1;
    public float minAngle = -Mathf.PI;
    public float maxAngle = Mathf.PI;
    public float minRange = 0.1f;
    public float maxRange = 30.0f;
    public float updateRate = 10f; // Hz
    public LayerMask detectionLayers = -1;

    [Header("Noise Parameters")]
    public float rangeNoiseStdDev = 0.01f;
    public float angularNoiseStdDev = 0.001f;

    [Header("Output Settings")]
    public string topicName = "/scan";

    private float nextUpdateTime;
    private List<float> ranges;
    private List<float> intensities;

    void Start()
    {
        ranges = new List<float>(new float[horizontalSamples]);
        intensities = new List<float>(new float[horizontalSamples]);
        nextUpdateTime = 0f;
    }

    void Update()
    {
        if (Time.time >= nextUpdateTime)
        {
            SimulateLidarScan();
            nextUpdateTime = Time.time + (1f / updateRate);
        }
    }

    void SimulateLidarScan()
    {
        float angleStep = (maxAngle - minAngle) / horizontalSamples;

        for (int i = 0; i < horizontalSamples; i++)
        {
            float angle = minAngle + (i * angleStep);

            // Add angular noise
            float noisyAngle = angle + RandomGaussian(0, angularNoiseStdDev);

            // Calculate ray direction
            Vector3 rayDirection = new Vector3(
                Mathf.Cos(noisyAngle),
                0f,
                Mathf.Sin(noisyAngle)
            );

            // Perform raycast
            RaycastHit hit;
            float range = maxRange;

            if (Physics.Raycast(transform.position, rayDirection, out hit, maxRange, detectionLayers))
            {
                range = hit.distance;

                // Add range noise
                range += RandomGaussian(0, rangeNoiseStdDev);

                // Clamp to valid range
                range = Mathf.Clamp(range, minRange, maxRange);

                // Calculate intensity based on surface properties
                intensities[i] = CalculateIntensity(hit);
            }
            else
            {
                // No hit - return maximum range
                intensities[i] = 0f; // Low intensity for no return
            }

            ranges[i] = range;
        }

        // Publish the simulated scan data
        PublishLidarData();
    }

    float CalculateIntensity(RaycastHit hit)
    {
        // Calculate intensity based on surface properties
        // This is a simplified model - real LiDAR intensity depends on
        // material reflectance, surface angle, distance, etc.
        float baseIntensity = 1000f;
        float distanceFactor = Mathf.Clamp01(1f - (hit.distance / maxRange));
        float surfaceFactor = 1f; // Could be modified based on material

        return baseIntensity * distanceFactor * surfaceFactor;
    }

    void PublishLidarData()
    {
        // This would typically publish to a ROS topic or other communication system
        Debug.Log($"Published LiDAR scan with {horizontalSamples} samples to {topicName}");

        // In a real implementation, this would serialize and publish the data
        // using ROS# or other Unity-ROS bridge
    }

    // Generate Gaussian-distributed random numbers
    float RandomGaussian(float mean, float stdDev)
    {
        // Box-Muller transform for generating Gaussian random numbers
        float u1 = Random.value;
        float u2 = Random.value;

        if (u1 < float.Epsilon) u1 = float.Epsilon;

        float normal = Mathf.Sqrt(-2.0f * Mathf.Log(u1)) * Mathf.Cos(2.0f * Mathf.PI * u2);
        return mean + stdDev * normal;
    }
}
```

### LiDAR Noise Modeling

Realistic LiDAR simulation includes various noise sources:

```csharp
using UnityEngine;

public class LidarNoiseModel : MonoBehaviour
{
    [Header("Systematic Errors")]
    public float rangeBias = 0.0f;
    public float angularBias = 0.0f;

    [Header("Random Noise")]
    public float rangeNoiseStdDev = 0.01f;
    public float angularNoiseStdDev = 0.001f;

    [Header("Environmental Effects")]
    public float weatherAttenuation = 0.0f; // Additional noise in poor weather
    public float multiPathFactor = 0.0f;    // Effects of multi-path reflections

    public float ApplyNoise(float range, float angle, float distance)
    {
        // Apply systematic errors
        float noisyRange = range + rangeBias;
        float noisyAngle = angle + angularBias;

        // Apply random noise
        noisyRange += RandomGaussian(0, rangeNoiseStdDev);
        noisyAngle += RandomGaussian(0, angularNoiseStdDev);

        // Apply environmental effects
        float environmentalNoise = weatherAttenuation * distance * Random.value;
        noisyRange += environmentalNoise;

        // Apply multi-path effects (simplified)
        if (Random.value < multiPathFactor)
        {
            // Add occasional multi-path artifacts
            noisyRange += Random.Range(-0.1f, 0.1f);
        }

        return noisyRange;
    }

    float RandomGaussian(float mean, float stdDev)
    {
        float u1 = Random.value;
        float u2 = Random.value;

        if (u1 < float.Epsilon) u1 = float.Epsilon;

        float normal = Mathf.Sqrt(-2.0f * Mathf.Log(u1)) * Mathf.Cos(2.0f * Mathf.PI * u2);
        return mean + stdDev * normal;
    }
}
```

## Depth Camera Simulation

Depth cameras provide 3D point cloud data essential for navigation, mapping, and object recognition. Simulating depth cameras requires consideration of optical properties and noise characteristics.

### Depth Camera Configuration in Gazebo

```xml
<!-- Example depth camera configuration for Gazebo -->
<gazebo reference="camera_link">
  <sensor name="depth_camera" type="depth">
    <pose>0 0 0 0 0 0</pose>
    <visualize>true</visualize>
    <update_rate>30</update_rate>
    <camera name="depth_cam">
      <horizontal_fov>1.047</horizontal_fov> <!-- 60 degrees -->
      <image>
        <width>640</width>
        <height>480</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.1</near>
        <far>10.0</far>
      </clip>
      <noise>
        <type>gaussian</type>
        <mean>0.0</mean>
        <stddev>0.007</stddev>
      </noise>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_openni_kinect.so">
      <cameraName>depth_camera</cameraName>
      <imageTopicName>/rgb/image_raw</imageTopicName>
      <depthImageTopicName>/depth/image_raw</depthImageTopicName>
      <pointCloudTopicName>/depth/points</pointCloudTopicName>
      <cameraInfoTopicName>/rgb/camera_info</cameraInfoTopicName>
      <depthImageCameraInfoTopicName>/depth/camera_info</depthImageCameraInfoTopicName>
      <frameName>camera_depth_optical_frame</frameName>
      <baseline>0.1</baseline>
      <distortion_k1>0.0</distortion_k1>
      <distortion_k2>0.0</distortion_k2>
      <distortion_k3>0.0</distortion_k3>
      <distortion_t1>0.0</distortion_t1>
      <distortion_t2>0.0</distortion_t2>
      <pointCloudCutoff>0.1</pointCloudCutoff>
      <pointCloudCutoffMax>3.0</pointCloudCutoffMax>
    </plugin>
  </sensor>
</gazebo>
```

### Unity Depth Camera Simulation

```csharp
using UnityEngine;
using System.Collections;

public class UnityDepthCamera : MonoBehaviour
{
    [Header("Camera Configuration")]
    public int width = 640;
    public int height = 480;
    public float fieldOfView = 60f;
    public float nearClip = 0.1f;
    public float farClip = 10.0f;

    [Header("Noise Parameters")]
    public float depthNoiseStdDev = 0.01f;
    public float depthBias = 0.0f;

    [Header("Output Settings")]
    public string depthTopic = "/depth/image_raw";
    public string pointCloudTopic = "/depth/points";

    private Camera depthCamera;
    private RenderTexture depthTexture;
    private float[,] depthBuffer;
    private Color32[] colorBuffer;

    void Start()
    {
        InitializeDepthCamera();
        depthBuffer = new float[height, width];
        colorBuffer = new Color32[width * height];
    }

    void InitializeDepthCamera()
    {
        // Create the depth camera
        depthCamera = GetComponent<Camera>();
        if (depthCamera == null)
        {
            depthCamera = gameObject.AddComponent<Camera>();
        }

        depthCamera.fieldOfView = fieldOfView;
        depthCamera.nearClipPlane = nearClip;
        depthCamera.farClipPlane = farClip;
        depthCamera.depth = -1; // Render after other cameras
        depthCamera.enabled = false; // Don't render normally

        // Create render texture for depth data
        depthTexture = new RenderTexture(width, height, 24, RenderTextureFormat.Depth);
        depthCamera.targetTexture = depthTexture;
    }

    void Update()
    {
        // Capture depth data at desired rate
        if (Time.frameCount % 3 == 0) // ~10 Hz at 30 FPS
        {
            CaptureDepthData();
        }
    }

    void CaptureDepthData()
    {
        // Render the depth camera
        RenderTexture.active = depthTexture;
        depthCamera.Render();

        // Read depth values
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                // Sample depth value at pixel location
                float depthValue = SampleDepthAtPixel(x, y);

                // Apply noise and bias
                depthValue = ApplyDepthNoise(depthValue, x, y);

                depthBuffer[y, x] = depthValue;
            }
        }

        // Publish the depth data
        PublishDepthData();
        PublishPointCloud();
    }

    float SampleDepthAtPixel(int x, int y)
    {
        // This is a simplified approach - in practice, you'd read from the depth buffer
        // For a more accurate approach, use Unity's Compute Shader or custom rendering pipeline
        Ray ray = depthCamera.ViewportPointToRay(new Vector3((float)x / width, (float)y / height, 0));

        RaycastHit hit;
        if (Physics.Raycast(ray, out hit, farClip))
        {
            return hit.distance;
        }
        else
        {
            return float.MaxValue; // No hit
        }
    }

    float ApplyDepthNoise(float depth, int x, int y)
    {
        if (depth >= farClip) return depth; // No noise for max range

        // Apply systematic bias
        float noisyDepth = depth + depthBias;

        // Apply random noise
        float noise = RandomGaussian(0, depthNoiseStdDev);
        noisyDepth += noise;

        // Apply edge effects (cameras often have less accurate depth at edges)
        float edgeFactor = CalculateEdgeFactor(x, y);
        noisyDepth += RandomGaussian(0, depthNoiseStdDev * edgeFactor);

        return Mathf.Clamp(noisyDepth, nearClip, farClip);
    }

    float CalculateEdgeFactor(int x, int y)
    {
        // Calculate how close to image edge this pixel is (0=center, 1=edge)
        float normX = (float)x / width;
        float normY = (float)y / height;

        float centerX = Mathf.Abs(normX - 0.5f) * 2f;
        float centerY = Mathf.Abs(normY - 0.5f) * 2f;

        return Mathf.Max(centerX, centerY); // Maximum of X and Y distance from center
    }

    void PublishDepthData()
    {
        // Serialize and publish depth data
        Debug.Log($"Published depth image {width}x{height} to {depthTopic}");
    }

    void PublishPointCloud()
    {
        // Convert depth image to point cloud
        Vector3[] points = new Vector3[width * height];
        int pointCount = 0;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float depth = depthBuffer[y, x];

                if (depth < farClip && depth > nearClip)
                {
                    // Convert pixel coordinates to 3D world coordinates
                    Vector3 point = PixelTo3D(x, y, depth);
                    points[pointCount] = point;
                    pointCount++;
                }
            }
        }

        // Publish point cloud data
        Debug.Log($"Published point cloud with {pointCount} points to {pointCloudTopic}");
    }

    Vector3 PixelTo3D(int x, int y, float depth)
    {
        // Convert pixel coordinates to normalized device coordinates
        float normX = (float)x / width;
        float normY = (float)y / height;

        // Convert to camera space
        float cameraX = (normX - 0.5f) * 2f * Mathf.Tan(Mathf.Deg2Rad * fieldOfView / 2f) * depth;
        float cameraY = (normY - 0.5f) * 2f * Mathf.Tan(Mathf.Deg2Rad * fieldOfView / 2f) * depth;

        // Transform to world space
        Vector3 cameraPoint = new Vector3(cameraX, -cameraY, depth);
        return transform.TransformPoint(cameraPoint);
    }

    float RandomGaussian(float mean, float stdDev)
    {
        float u1 = Random.value;
        float u2 = Random.value;

        if (u1 < float.Epsilon) u1 = float.Epsilon;

        float normal = Mathf.Sqrt(-2.0f * Mathf.Log(u1)) * Mathf.Cos(2.0f * Mathf.PI * u2);
        return mean + stdDev * normal;
    }
}
```

## IMU Simulation

Inertial Measurement Units (IMUs) provide crucial data about robot orientation, acceleration, and angular velocity. Accurate IMU simulation is essential for navigation and control systems.

### IMU Configuration in Gazebo

```xml
<!-- Example IMU sensor configuration for Gazebo -->
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
            <bias_mean>0.0000075</bias_mean>
            <bias_stddev>0.0000008</bias_stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
            <bias_mean>0.0000075</bias_mean>
            <bias_stddev>0.0000008</bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
            <bias_mean>0.0000075</bias_mean>
            <bias_stddev>0.0000008</bias_stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
            <bias_mean>0.1</bias_mean>
            <bias_stddev>0.001</bias_stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
            <bias_mean>0.1</bias_mean>
            <bias_stddev>0.001</bias_stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
            <bias_mean>0.1</bias_mean>
            <bias_stddev>0.001</bias_stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
    <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
      <ros>
        <namespace>/imu</namespace>
        <remapping>~/out:=data</remapping>
      </ros>
      <frame_name>imu_link</frame_name>
      <body_name>imu_link</body_name>
      <update_rate>100</update_rate>
    </plugin>
  </sensor>
</gazebo>
```

### Unity IMU Simulation

```csharp
using UnityEngine;
using System.Collections.Generic;

public class UnityIMUSimulation : MonoBehaviour
{
    [Header("IMU Configuration")]
    public float updateRate = 100f; // Hz
    public string topicName = "/imu/data";

    [Header("Gyroscope Noise")]
    public float gyroNoiseStdDev = 0.0002f; // rad/s
    public float gyroBiasStdDev = 0.0000008f; // rad/s
    public float gyroBiasDrift = 0.0000075f; // rad/s

    [Header("Accelerometer Noise")]
    public float accelNoiseStdDev = 0.017f; // m/s²
    public float accelBiasStdDev = 0.001f; // m/s²
    public float accelBiasDrift = 0.1f; // m/s²

    [Header("Gravity")]
    public Vector3 gravity = new Vector3(0, 0, -9.81f);

    private float nextUpdateTime;
    private Vector3 trueAngularVelocity;
    private Vector3 trueLinearAcceleration;
    private Vector3 gyroBias;
    private Vector3 accelBias;

    void Start()
    {
        nextUpdateTime = 0f;

        // Initialize biases
        gyroBias = new Vector3(
            RandomGaussian(0, gyroBiasStdDev),
            RandomGaussian(0, gyroBiasStdDev),
            RandomGaussian(0, gyroBiasStdDev)
        );

        accelBias = new Vector3(
            RandomGaussian(0, accelBiasStdDev),
            RandomGaussian(0, accelBiasStdDev),
            RandomGaussian(0, accelBiasStdDev)
        );
    }

    void Update()
    {
        if (Time.time >= nextUpdateTime)
        {
            SimulateIMU();
            nextUpdateTime = Time.time + (1f / updateRate);
        }
    }

    void SimulateIMU()
    {
        // Get true values from Unity's physics system
        Rigidbody rb = GetComponent<Rigidbody>();
        if (rb != null)
        {
            // True angular velocity from rigidbody
            trueAngularVelocity = rb.angularVelocity;

            // True linear acceleration (remove gravity)
            trueLinearAcceleration = rb.velocity / Time.fixedDeltaTime;
            trueLinearAcceleration -= gravity;
        }
        else
        {
            // If no rigidbody, estimate from transform changes
            trueAngularVelocity = EstimateAngularVelocity();
            trueLinearAcceleration = EstimateLinearAcceleration();
        }

        // Apply noise and biases to measurements
        Vector3 measuredAngularVelocity = ApplyGyroNoise(trueAngularVelocity);
        Vector3 measuredLinearAcceleration = ApplyAccelNoise(trueLinearAcceleration);

        // Publish IMU data
        PublishIMUData(measuredAngularVelocity, measuredLinearAcceleration);
    }

    Vector3 ApplyGyroNoise(Vector3 trueValue)
    {
        Vector3 noise = new Vector3(
            RandomGaussian(0, gyroNoiseStdDev),
            RandomGaussian(0, gyroNoiseStdDev),
            RandomGaussian(0, gyroNoiseStdDev)
        );

        // Update bias with drift
        gyroBias += new Vector3(
            RandomGaussian(0, gyroBiasDrift * Time.deltaTime),
            RandomGaussian(0, gyroBiasDrift * Time.deltaTime),
            RandomGaussian(0, gyroBiasDrift * Time.deltaTime)
        );

        return trueValue + noise + gyroBias;
    }

    Vector3 ApplyAccelNoise(Vector3 trueValue)
    {
        Vector3 noise = new Vector3(
            RandomGaussian(0, accelNoiseStdDev),
            RandomGaussian(0, accelNoiseStdDev),
            RandomGaussian(0, accelNoiseStdDev)
        );

        // Update bias with drift
        accelBias += new Vector3(
            RandomGaussian(0, accelBiasDrift * Time.deltaTime),
            RandomGaussian(0, accelBiasDrift * Time.deltaTime),
            RandomGaussian(0, accelBiasDrift * Time.deltaTime)
        );

        return trueValue + noise + accelBias;
    }

    Vector3 EstimateAngularVelocity()
    {
        // Estimate angular velocity from rotation changes
        // This is a simplified approach - in practice, you'd track rotation over time
        static Vector3 lastAngularVelocity = Vector3.zero;
        return lastAngularVelocity;
    }

    Vector3 EstimateLinearAcceleration()
    {
        // Estimate linear acceleration from position changes
        // This is a simplified approach - in practice, you'd track velocity changes
        static Vector3 lastLinearAcceleration = Vector3.zero;
        return lastLinearAcceleration;
    }

    void PublishIMUData(Vector3 angularVelocity, Vector3 linearAcceleration)
    {
        // This would typically publish to a ROS topic or other communication system
        Debug.Log($"IMU Data - Angular Vel: {angularVelocity}, Linear Accel: {linearAcceleration}");

        // In a real implementation, this would serialize and publish the data
        // using ROS# or other Unity-ROS bridge
    }

    float RandomGaussian(float mean, float stdDev)
    {
        float u1 = Random.value;
        float u2 = Random.value;

        if (u1 < float.Epsilon) u1 = float.Epsilon;

        float normal = Mathf.Sqrt(-2.0f * Mathf.Log(u1)) * Mathf.Cos(2.0f * Mathf.PI * u2);
        return mean + stdDev * normal;
    }
}
```

## Sensor Data Pipelines for AI Systems

Creating efficient sensor data pipelines is crucial for feeding realistic sensor data to AI systems for training and testing.

### ROS-Based Sensor Pipeline

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image, PointCloud2, Imu
from nav_msgs.msg import Odometry
import numpy as np
from scipy.spatial.transform import Rotation as R

class SensorPipelineNode(Node):
    """
    A ROS 2 node that processes and combines sensor data for AI systems.
    """

    def __init__(self):
        super().__init__('sensor_pipeline')

        # Publishers for processed data
        self.perception_pub = self.create_publisher(
            PointCloud2,
            '/sensor_fusion/pointcloud',
            10
        )

        self.fused_imu_pub = self.create_publisher(
            Imu,
            '/sensor_fusion/imu',
            10
        )

        # Subscribers for raw sensor data
        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )

        self.depth_sub = self.create_subscription(
            Image,
            '/depth/image_raw',
            self.depth_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        # Timer for processing loop
        self.process_timer = self.create_timer(0.1, self.process_sensors)

        # Storage for sensor data
        self.lidar_data = None
        self.depth_data = None
        self.imu_data = None
        self.odom_data = None

        self.get_logger().info('Sensor Pipeline Node initialized')

    def lidar_callback(self, msg):
        """Process LiDAR data"""
        self.lidar_data = msg
        self.get_logger().debug(f'Received LiDAR data with {len(msg.ranges)} points')

    def depth_callback(self, msg):
        """Process depth camera data"""
        self.depth_data = msg
        self.get_logger().debug(f'Received depth image: {msg.width}x{msg.height}')

    def imu_callback(self, msg):
        """Process IMU data"""
        self.imu_data = msg
        self.get_logger().debug(f'Received IMU data')

    def process_sensors(self):
        """Process and fuse sensor data for AI systems"""
        if not all([self.lidar_data, self.depth_data, self.imu_data]):
            return  # Wait for all sensors to have data

        # Create point cloud from LiDAR and depth data
        pointcloud = self.create_fused_pointcloud()

        # Fuse IMU data with other sensors
        fused_imu = self.fuse_imu_data()

        # Publish processed data
        self.perception_pub.publish(pointcloud)
        self.fused_imu_pub.publish(fused_imu)

        # Send data to AI system
        self.send_to_ai_system(pointcloud, fused_imu)

    def create_fused_pointcloud(self):
        """Create a combined point cloud from LiDAR and depth camera"""
        # Convert LiDAR scan to point cloud
        lidar_points = self.lidar_to_pointcloud(self.lidar_data)

        # Convert depth image to point cloud
        depth_points = self.depth_to_pointcloud(self.depth_data)

        # Combine both point clouds in a common coordinate frame
        combined_points = np.concatenate([lidar_points, depth_points], axis=0)

        # Create PointCloud2 message
        pointcloud_msg = self.create_pointcloud2_msg(combined_points)

        return pointcloud_msg

    def lidar_to_pointcloud(self, scan_msg):
        """Convert LaserScan to numpy array of 3D points"""
        ranges = np.array(scan_msg.ranges)
        angles = np.linspace(
            scan_msg.angle_min,
            scan_msg.angle_max,
            len(scan_msg.ranges)
        )

        # Filter out invalid ranges
        valid_indices = (ranges >= scan_msg.range_min) & (ranges <= scan_msg.range_max)
        valid_ranges = ranges[valid_indices]
        valid_angles = angles[valid_indices]

        # Convert to Cartesian coordinates (assuming 2D scan for simplicity)
        x = valid_ranges * np.cos(valid_angles)
        y = valid_ranges * np.sin(valid_angles)
        z = np.zeros_like(x)  # 2D scan, so z=0

        return np.column_stack([x, y, z])

    def depth_to_pointcloud(self, depth_msg):
        """Convert depth image to numpy array of 3D points"""
        # This is a simplified example - in practice, you'd use camera intrinsics
        # and convert depth pixels to 3D points

        width = depth_msg.width
        height = depth_msg.height

        # For simplicity, assume we have depth values as float32
        # In practice, you'd decode the image data properly
        depth_array = np.frombuffer(depth_msg.data, dtype=np.float32)
        depth_array = depth_array.reshape((height, width))

        # Create grid of pixel coordinates
        u, v = np.meshgrid(np.arange(width), np.arange(height))

        # Convert to 3D points using camera intrinsics
        # This is a simplified pinhole camera model
        fx, fy = 525.0, 525.0  # Focal lengths
        cx, cy = width/2, height/2  # Principal point

        # Convert pixel coordinates to 3D
        x = (u - cx) * depth_array / fx
        y = (v - cy) * depth_array / fy
        z = depth_array

        # Flatten and combine valid points
        valid_mask = (depth_array > 0) & (depth_array < 10.0)  # Valid depth range
        points = np.column_stack([
            x[valid_mask],
            y[valid_mask],
            z[valid_mask]
        ])

        return points

    def fuse_imu_data(self):
        """Fuse IMU data with other sensor information"""
        # Create a fused IMU message with enhanced accuracy
        fused_imu = Imu()

        # Copy original IMU data
        fused_imu.orientation = self.imu_data.orientation
        fused_imu.angular_velocity = self.imu_data.angular_velocity
        fused_imu.linear_acceleration = self.imu_data.linear_acceleration

        # Add covariance information based on sensor characteristics
        self.add_sensor_covariance(fused_imu)

        return fused_imu

    def add_sensor_covariance(self, imu_msg):
        """Add realistic covariance values to IMU message"""
        # Typical covariance values for a good IMU
        # These values should be tuned based on your specific sensors
        covariance_orientation = [0.01] * 9  # 3x3 covariance matrix
        covariance_angular_velocity = [0.01] * 9
        covariance_linear_acceleration = [0.05] * 9

        imu_msg.orientation_covariance = covariance_orientation
        imu_msg.angular_velocity_covariance = covariance_angular_velocity
        imu_msg.linear_acceleration_covariance = covariance_linear_acceleration

    def create_pointcloud2_msg(self, points):
        """Create a PointCloud2 message from numpy array of points"""
        from sensor_msgs.msg import PointCloud2, PointField
        import struct

        # Create PointCloud2 message
        cloud_msg = PointCloud2()
        cloud_msg.header.stamp = self.get_clock().now().to_msg()
        cloud_msg.header.frame_id = 'sensor_fusion_frame'

        # Define fields
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        cloud_msg.fields = fields
        cloud_msg.point_step = 12  # 3 * 4 bytes per float
        cloud_msg.row_step = cloud_msg.point_step * len(points)
        cloud_msg.is_dense = True

        # Pack point data
        data = []
        for point in points:
            data.append(struct.pack('fff', point[0], point[1], point[2]))

        cloud_msg.data = b''.join(data)
        cloud_msg.height = 1
        cloud_msg.width = len(points)

        return cloud_msg

    def send_to_ai_system(self, pointcloud, fused_imu):
        """Send processed sensor data to AI system"""
        # In a real implementation, this would send data to your AI system
        # This could be via ROS topics, shared memory, network, etc.

        # Example: Send to a perception AI node
        self.get_logger().info(
            f'Sent sensor data to AI system: '
            f'{pointcloud.width} points, IMU updated'
        )

def main(args=None):
    rclpy.init(args=args)

    sensor_pipeline = SensorPipelineNode()

    try:
        rclpy.spin(sensor_pipeline)
    except KeyboardInterrupt:
        pass
    finally:
        sensor_pipeline.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Unity-ROS Sensor Pipeline Integration

```csharp
using UnityEngine;
using System.Collections.Generic;
using RosSharp.RosBridgeClient;
using RosSharp.Messages.Sensor;
using RosSharp.Messages.Nav;

public class UnitySensorPipeline : MonoBehaviour
{
    [Header("ROS Connection")]
    public string rosBridgeUrl = "ws://127.0.0.1:9090";

    [Header("Sensor Topics")]
    public string lidarTopic = "/scan";
    public string depthTopic = "/depth/image_raw";
    public string imuTopic = "/imu/data";
    public string fusedTopic = "/sensor_fusion/data";

    [Header("Pipeline Settings")]
    public float pipelineUpdateRate = 10f; // Hz

    private RosSocket rosSocket;
    private string lidarPublisherId;
    private string depthPublisherId;
    private string imuPublisherId;
    private string fusedPublisherId;

    private UnityLidarSimulation lidarSim;
    private UnityDepthCamera depthSim;
    private UnityIMUSimulation imuSim;

    private float nextPipelineUpdate;

    void Start()
    {
        // Find sensor simulation components
        lidarSim = GetComponent<UnityLidarSimulation>();
        depthSim = GetComponent<UnityDepthCamera>();
        imuSim = GetComponent<UnityIMUSimulation>();

        ConnectToROS();
        nextPipelineUpdate = 0f;
    }

    void ConnectToROS()
    {
        RosBridgeClient.WebSocketNative.ClientWebSocket webSocket =
            new RosBridgeClient.WebSocketNative.ClientWebSocket();

        rosSocket = new RosSocket(WebSocketProtocol.WebSocketSharp, webSocket);
        rosSocket.OnConnected += OnROSConnected;
        rosSocket.OnError += OnROSError;

        rosSocket.Connect(rosBridgeUrl);
    }

    void OnROSConnected()
    {
        Debug.Log("Connected to ROS Bridge for sensor pipeline");

        // Advertise sensor topics
        lidarPublisherId = rosSocket.Advertise<LaserScan>(lidarTopic);
        depthPublisherId = rosSocket.Advertise<Image>(depthTopic);
        imuPublisherId = rosSocket.Advertise<Imu>(imuTopic);
        fusedPublisherId = rosSocket.Advertise<PointCloud2>(fusedTopic);
    }

    void OnROSError(string error)
    {
        Debug.LogError($"ROS Bridge Error: {error}");
    }

    void Update()
    {
        if (Time.time >= nextPipelineUpdate)
        {
            ProcessSensorPipeline();
            nextPipelineUpdate = Time.time + (1f / pipelineUpdateRate);
        }
    }

    void ProcessSensorPipeline()
    {
        // Collect data from all sensors
        LaserScan lidarData = GetLidarData();
        Image depthData = GetDepthData();
        Imu imuData = GetImuData();

        if (lidarData != null && depthData != null && imuData != null)
        {
            // Publish individual sensor data
            rosSocket.Publish(lidarPublisherId, lidarData);
            rosSocket.Publish(depthPublisherId, depthData);
            rosSocket.Publish(imuPublisherId, imuData);

            // Create fused sensor data
            PointCloud2 fusedData = CreateFusedPointCloud(lidarData, depthData);
            rosSocket.Publish(fusedPublisherId, fusedData);

            // Send to AI system
            SendToAISystem(fusedData, imuData);
        }
    }

    LaserScan GetLidarData()
    {
        // In a real implementation, this would get data from the lidar simulation
        // For now, we'll create a mock LaserScan message
        LaserScan scan = new LaserScan();
        scan.angle_min = -Mathf.PI;
        scan.angle_max = Mathf.PI;
        scan.angle_increment = (2 * Mathf.PI) / 720;
        scan.time_increment = 0.0f;
        scan.scan_time = 0.1f;
        scan.range_min = 0.1f;
        scan.range_max = 30.0f;

        // Fill with mock data
        scan.ranges = new double[720];
        for (int i = 0; i < 720; i++)
        {
            scan.ranges[i] = Random.Range(0.1f, 30.0f);
        }

        return scan;
    }

    Image GetDepthData()
    {
        // In a real implementation, this would get data from the depth camera simulation
        Image depth = new Image();
        depth.width = 640;
        depth.height = 480;
        depth.encoding = "32FC1"; // 32-bit float, 1 channel
        depth.is_bigendian = 0;
        depth.step = 640 * 4; // 4 bytes per float

        // Mock data
        depth.data = new byte[640 * 480 * 4]; // 4 bytes per float

        return depth;
    }

    Imu GetImuData()
    {
        // In a real implementation, this would get data from the IMU simulation
        Imu imu = new Imu();

        // Mock orientation (identity)
        imu.orientation.x = 0.0;
        imu.orientation.y = 0.0;
        imu.orientation.z = 0.0;
        imu.orientation.w = 1.0;

        // Mock angular velocity
        imu.angular_velocity.x = Random.Range(-0.1f, 0.1f);
        imu.angular_velocity.y = Random.Range(-0.1f, 0.1f);
        imu.angular_velocity.z = Random.Range(-0.1f, 0.1f);

        // Mock linear acceleration
        imu.linear_acceleration.x = Random.Range(-1.0f, 1.0f);
        imu.linear_acceleration.y = Random.Range(-1.0f, 1.0f);
        imu.linear_acceleration.z = Random.Range(-11.0f, -8.0f); // Gravity

        return imu;
    }

    PointCloud2 CreateFusedPointCloud(LaserScan lidarData, Image depthData)
    {
        // Create a fused point cloud combining LiDAR and depth data
        PointCloud2 pointCloud = new PointCloud2();

        pointCloud.header.frame_id = "sensor_fusion_frame";
        pointCloud.header.stamp = new TimeStamp();

        // Define point fields (x, y, z)
        pointCloud.fields = new RosSharp.Messages.Sensor.PointField[3];
        pointCloud.fields[0] = new RosSharp.Messages.Sensor.PointField("x", 0, RosSharp.Messages.Sensor.PointField.FLOAT32, 1);
        pointCloud.fields[1] = new RosSharp.Messages.Sensor.PointField("y", 4, RosSharp.Messages.Sensor.PointField.FLOAT32, 1);
        pointCloud.fields[2] = new RosSharp.Messages.Sensor.PointField("z", 8, RosSharp.Messages.Sensor.PointField.FLOAT32, 1);

        pointCloud.point_step = 12; // 3 * 4 bytes
        pointCloud.height = 1;
        pointCloud.width = (uint)(lidarData.ranges.Length + (depthData.width * depthData.height) / 100); // Simplified
        pointCloud.row_step = pointCloud.point_step * pointCloud.width;
        pointCloud.is_dense = true;

        // In a real implementation, this would properly combine the point clouds
        // For now, we'll create mock data
        int pointCount = (int)pointCloud.width;
        pointCloud.data = new byte[pointCount * 12]; // 12 bytes per point (x,y,z as float32)

        return pointCloud;
    }

    void SendToAISystem(PointCloud2 fusedData, Imu imuData)
    {
        // Send sensor data to AI system for processing
        Debug.Log($"Sent fused sensor data to AI system: {fusedData.width} points");
    }

    void OnDestroy()
    {
        if (rosSocket != null)
        {
            rosSocket.Close();
        }
    }
}
```

## Practical Example: Complete Sensor Simulation System

Here's a complete example of a sensor simulation system that integrates all the sensor types:

```csharp
using UnityEngine;
using System.Collections.Generic;

public class CompleteSensorSystem : MonoBehaviour
{
    [Header("System Configuration")]
    public float simulationUpdateRate = 30f; // Hz
    public string robotNamespace = "/robot1";

    [Header("Sensor Configuration")]
    public UnityLidarSimulation lidarSim;
    public UnityDepthCamera depthCam;
    public UnityIMUSimulation imuSim;

    [Header("Sensor Fusion")]
    public bool enableFusion = true;
    public float fusionUpdateRate = 10f; // Hz

    [Header("AI Integration")]
    public bool publishToAI = true;
    public string aiTopicPrefix = "/ai_input";

    private float nextSimulationUpdate;
    private float nextFusionUpdate;

    void Start()
    {
        InitializeSensors();
        nextSimulationUpdate = 0f;
        nextFusionUpdate = 0f;
    }

    void InitializeSensors()
    {
        // Find sensor components if not assigned
        if (lidarSim == null) lidarSim = GetComponent<UnityLidarSimulation>();
        if (depthCam == null) depthCam = GetComponent<UnityDepthCamera>();
        if (imuSim == null) imuSim = GetComponent<UnityIMUSimulation>();

        // Configure sensors with common settings
        ConfigureSensors();
    }

    void ConfigureSensors()
    {
        if (lidarSim != null)
        {
            lidarSim.topicName = robotNamespace + "/scan";
        }

        if (depthCam != null)
        {
            depthCam.depthTopic = robotNamespace + "/depth/image_raw";
            depthCam.pointCloudTopic = robotNamespace + "/depth/points";
        }

        if (imuSim != null)
        {
            imuSim.topicName = robotNamespace + "/imu/data";
        }
    }

    void Update()
    {
        if (Time.time >= nextSimulationUpdate)
        {
            UpdateSensors();
            nextSimulationUpdate = Time.time + (1f / simulationUpdateRate);
        }

        if (enableFusion && Time.time >= nextFusionUpdate)
        {
            PerformSensorFusion();
            nextFusionUpdate = Time.time + (1f / fusionUpdateRate);
        }
    }

    void UpdateSensors()
    {
        // Sensors update automatically based on their own timing
        // This method can be used for coordination or additional processing
        Debug.Log("Sensor system update");
    }

    void PerformSensorFusion()
    {
        if (!enableFusion) return;

        // Collect data from all sensors
        var sensorData = CollectSensorData();

        // Perform sensor fusion
        var fusedData = FuseSensorData(sensorData);

        // Publish fused data
        PublishFusedData(fusedData);

        // Send to AI system
        if (publishToAI)
        {
            SendToFusionAI(fusedData);
        }
    }

    SensorData CollectSensorData()
    {
        // Collect current sensor readings
        var data = new SensorData
        {
            timestamp = Time.time,
            lidarRanges = GetLidarRanges(),
            depthImage = GetDepthImage(),
            imuData = GetImuData()
        };

        return data;
    }

    float[] GetLidarRanges()
    {
        // This would get actual ranges from the lidar simulation
        // For now, return mock data
        return new float[720];
    }

    Texture2D GetDepthImage()
    {
        // This would get actual depth image from the camera simulation
        // For now, return null
        return null;
    }

    ImuData GetImuData()
    {
        // This would get actual IMU data from the simulation
        // For now, return mock data
        return new ImuData();
    }

    FusedData FuseSensorData(SensorData sensorData)
    {
        // Perform sensor fusion algorithm
        // This is a simplified example - real fusion would use Kalman filters,
        // particle filters, or other advanced techniques
        var fused = new FusedData
        {
            timestamp = sensorData.timestamp,
            positionEstimate = EstimatePosition(sensorData),
            orientationEstimate = EstimateOrientation(sensorData),
            obstacleMap = CreateObstacleMap(sensorData)
        };

        return fused;
    }

    Vector3 EstimatePosition(SensorData sensorData)
    {
        // Estimate position using sensor fusion
        // This would typically use a Kalman filter or other estimation algorithm
        return transform.position;
    }

    Quaternion EstimateOrientation(SensorData sensorData)
    {
        // Estimate orientation using sensor fusion
        return transform.rotation;
    }

    ObstacleMap CreateObstacleMap(SensorData sensorData)
    {
        // Create obstacle map from sensor data
        // This would combine LiDAR and depth camera data into a unified map
        return new ObstacleMap();
    }

    void PublishFusedData(FusedData fusedData)
    {
        // Publish fused sensor data to ROS or other systems
        Debug.Log($"Published fused sensor data at {fusedData.timestamp}");
    }

    void SendToFusionAI(FusedData fusedData)
    {
        // Send fused data to AI system for perception and decision making
        Debug.Log($"Sent fused data to AI system");
    }

    // Data structures for sensor fusion
    [System.Serializable]
    public class SensorData
    {
        public float timestamp;
        public float[] lidarRanges;
        public Texture2D depthImage;
        public ImuData imuData;
    }

    [System.Serializable]
    public class ImuData
    {
        public Vector3 angularVelocity;
        public Vector3 linearAcceleration;
        public Quaternion orientation;
    }

    [System.Serializable]
    public class FusedData
    {
        public float timestamp;
        public Vector3 positionEstimate;
        public Quaternion orientationEstimate;
        public ObstacleMap obstacleMap;
    }

    [System.Serializable]
    public class ObstacleMap
    {
        // Representation of obstacles in the environment
        // Could be a grid, point cloud, or other format
    }
}
```

## Quality Assurance for Sensor Simulation

### Validation Techniques

To ensure realistic sensor simulation, implement validation techniques:

1. **Cross-validation**: Compare simulation results with real sensor data
2. **Statistical analysis**: Verify that noise characteristics match real sensors
3. **Edge case testing**: Test simulation under extreme conditions
4. **Consistency checks**: Ensure sensor data is temporally and spatially consistent

### Performance Considerations

Sensor simulation can be computationally expensive. Consider:

- **Parallel processing**: Use multi-threading for sensor simulation
- **Level of detail**: Adjust simulation quality based on performance needs
- **Batch processing**: Process sensor data in batches when possible
- **Optimized algorithms**: Use efficient algorithms for raycasting and other operations

## Summary and Next Steps

Sensor simulation is a critical component of digital twin technology, providing realistic perception data for AI systems. By accurately modeling LiDAR, depth cameras, and IMUs with appropriate noise characteristics and environmental effects, you can create compelling digital twin experiences that closely mirror real-world sensor behavior.

This completes the Digital Twin module, covering physics simulation with Gazebo, high-fidelity visualization with Unity, and realistic sensor simulation for perception systems.

## Learning Objectives

By the end of this chapter, you should be able to:
- Simulate LiDAR sensors with realistic noise and environmental effects
- Create accurate depth camera simulations with proper optical properties
- Model IMU sensors with drift, bias, and noise characteristics
- Build sensor data pipelines for AI system integration
- Implement sensor fusion techniques for enhanced perception
- Validate sensor simulation quality and performance

## Summary and Cross-References

Sensor simulation is a critical component of digital twin technology, providing realistic perception data for AI systems. By accurately modeling LiDAR, depth cameras, and IMUs with appropriate noise characteristics and environmental effects, you can create compelling digital twin experiences that closely mirror real-world sensor behavior.

This chapter builds upon the concepts from:
- [Physics Simulation with Gazebo](./physics-simulation-gazebo.md) - for understanding the physical world that sensors perceive
- [High-Fidelity Interaction with Unity](./high-fidelity-unity.md) - for visualizing sensor data and creating realistic environments

This completes the Digital Twin module, covering physics simulation with Gazebo, high-fidelity visualization with Unity, and realistic sensor simulation for perception systems.

## Additional Resources

- [Glossary of Digital Twin Terms](./glossary.md) - Definitions of key terminology used throughout this module
- [Module Summary](./summary.md) - Comprehensive overview of all concepts covered in this module

## Exercises for Hands-On Practice

To reinforce your understanding of sensor simulation for perception, complete the following exercises:

### Exercise 1: Configure LiDAR with Realistic Noise
Create a LiDAR sensor configuration in Gazebo with realistic noise parameters based on a real LiDAR model (e.g., Hokuyo UTM-30LX or Velodyne VLP-16). Test the sensor in various environments and analyze the noise characteristics.

### Exercise 2: Implement Depth Camera Simulation
Create a depth camera simulation in Gazebo with realistic optical properties and noise models. Verify that the point cloud generated matches the expected field of view and resolution parameters.

### Exercise 3: Model IMU with Drift and Bias
Implement an IMU sensor model with realistic drift, bias, and noise characteristics. Test how these errors accumulate over time and implement basic calibration techniques.

### Exercise 4: Create Sensor Fusion Pipeline
Build a ROS node that subscribes to multiple sensor streams (LiDAR, camera, IMU) and implements a basic sensor fusion algorithm to create a unified perception output.

### Exercise 5: Validate Sensor Simulation Quality
Compare simulated sensor data with real sensor data from the same environment to validate the realism of your simulation. Analyze statistical properties of the noise and accuracy.

### Exercise 6: Implement Dynamic Obstacle Simulation
Create a simulation environment with moving obstacles and verify that all sensors properly detect and track these dynamic objects with appropriate temporal consistency.

## Troubleshooting Common Issues

When working with sensor simulation in digital twin applications, you may encounter several common issues. Here are solutions to the most frequent problems:

### LiDAR Simulation Issues
**Problem**: LiDAR returns show unexpected patterns or missing data.
**Solution**:
- Check ray count and resolution settings match your desired specifications
- Verify that the update rate is appropriate for your application
- Ensure proper coordinate frame alignment between sensor and robot
- Check for interference between multiple sensors

### Depth Camera Problems
**Problem**: Depth images show artifacts, incorrect scaling, or missing data.
**Solution**:
- Verify camera intrinsics are properly configured
- Check that the depth format is correctly specified (16-bit, 32-bit float)
- Ensure proper clipping distances (near/far planes) are set
- Verify that the point cloud generation parameters are correct

### IMU Simulation Errors
**Problem**: IMU data shows unrealistic values or excessive drift.
**Solution**:
- Verify that the noise parameters match the specifications of your target IMU
- Check that the coordinate frames are properly aligned
- Ensure proper integration of angular velocity to orientation
- Validate that gravity is properly accounted for in linear acceleration

### Sensor Integration Issues
**Problem**: Sensor data is not properly synchronized or integrated with the physics simulation.
**Solution**:
- Verify that all sensors are properly attached to the robot model
- Check that the update rates are appropriate for your application
- Ensure proper timestamping and message synchronization
- Validate that the sensor frames are properly defined in the URDF