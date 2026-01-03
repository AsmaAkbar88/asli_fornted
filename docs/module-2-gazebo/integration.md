---
sidebar_label: 'Gazebo-ROS Integration'
---

# Gazebo-ROS Integration

This section covers the integration between Gazebo simulation environment and ROS 2, enabling seamless development and testing of robotic applications.

## Gazebo-ROS Architecture

The integration between Gazebo and ROS 2 is facilitated by the Gazebo ROS packages, which provide:
- Bridge between Gazebo's transport system and ROS 2 middleware
- ROS 2 interfaces for Gazebo simulation services
- Plugin system for custom ROS 2 communication
- TF tree publishing for coordinate transformations

## Core Gazebo-ROS Packages

### Essential Packages:
```bash
# Core Gazebo-ROS packages
ros-humble-gazebo-ros
ros-humble-gazebo-plugins
ros-humble-gazebo-ros-pkgs
```

### Key Components:
- **gazebo_ros**: Core ROS 2 interface to Gazebo
- **libgazebo_ros**: Library for creating custom ROS 2 plugins
- **gazebo_ros_factory**: Dynamic model spawning capabilities

## Launching Robots in Gazebo

### Basic Launch File Example:
```python
# launch/robot_world.launch.py
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    pkg_gazebo_ros = FindPackageShare('gazebo_ros')
    pkg_robot_description = FindPackageShare('my_robot_description')

    # Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': PathJoinSubstitution([
                FindPackageShare('my_robot_gazebo'),
                'worlds',
                'my_world.sdf'
            ])
        }.items()
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'robot_description': PathJoinSubstitution([
                FindPackageShare('my_robot_description'),
                'urdf',
                'my_robot.urdf'
            ])
        }]
    )

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'my_robot',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.5'
        ],
        output='screen'
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_entity
    ])
```

## Gazebo Plugins for ROS 2

### Common Gazebo Plugins:

#### Joint State Publisher:
```xml
<gazebo>
  <plugin name="joint_state_publisher" filename="libgazebo_ros_joint_state_publisher.so">
    <ros>
      <namespace>/my_robot</namespace>
      <remapping>~/out:=joint_states</remapping>
    </ros>
    <update_rate>30</update_rate>
    <joint_name>joint1</joint_name>
    <joint_name>joint2</joint_name>
  </plugin>
</gazebo>
```

#### Diff Drive Controller:
```xml
<gazebo>
  <plugin name="diff_drive" filename="libgazebo_ros_diff_drive.so">
    <ros>
      <namespace>/my_robot</namespace>
    </ros>
    <left_joint>left_wheel_joint</left_joint>
    <right_joint>right_wheel_joint</right_joint>
    <wheel_separation>0.4</wheel_separation>
    <wheel_diameter>0.2</wheel_diameter>
    <max_wheel_torque>20</max_wheel_torque>
    <max_wheel_acceleration>1.0</max_wheel_acceleration>
    <publish_odom>true</publish_odom>
    <publish_odom_tf>true</publish_odom_tf>
    <publish_wheel_tf>true</publish_wheel_tf>
    <odometry_frame>odom</odometry_frame>
    <robot_base_frame>base_link</robot_base_frame>
  </plugin>
</gazebo>
```

#### Camera Sensor:
```xml
<gazebo reference="camera_link">
  <sensor name="camera" type="camera">
    <update_rate>30</update_rate>
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
      <ros>
        <namespace>/my_robot</namespace>
        <remapping>~/image_raw:=/camera/image_raw</remapping>
        <remapping>~/camera_info:=/camera/camera_info</remapping>
      </ros>
      <camera_name>camera</camera_name>
      <frame_name>camera_optical_frame</frame_name>
      <hack_baseline>0.07</hack_baseline>
      <distortion_k1>0.0</distortion_k1>
      <distortion_k2>0.0</distortion_k2>
      <distortion_k3>0.0</distortion_k3>
      <distortion_t1>0.0</distortion_t1>
      <distortion_t2>0.0</distortion_t2>
    </plugin>
  </sensor>
</gazebo>
```

## Custom Plugin Development

### Basic ROS 2 Plugin Template:
```cpp
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/common/Time.hh>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64.hpp>

namespace gazebo
{
  class ROS2Plugin : public ModelPlugin
  {
    public: void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
    {
      // Store the model pointer for convenience
      this->model = _model;

      // Initialize ROS 2 node
      if (!rclcpp::ok()) {
        rclcpp::init(0, nullptr);
      }

      this->node = std::make_shared<rclcpp::Node>("gazebo_custom_plugin");

      // Create ROS 2 publisher
      this->pub = this->node->create_publisher<std_msgs::msg::Float64>(
        "/custom_topic", 10);

      // Listen to the update event (gazebo calls this at every simulation iteration)
      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
          std::bind(&ROS2Plugin::OnUpdate, this));
    }

    public: void OnUpdate()
    {
      // Custom update logic
      auto msg = std_msgs::msg::Float64();
      msg.data = this->model->GetWorldPose().Pos().X();

      this->pub->publish(msg);
    }

    private: physics::ModelPtr model;
    private: rclcpp::Node::SharedPtr node;
    private: rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub;
    private: event::ConnectionPtr updateConnection;
  };

  GZ_REGISTER_MODEL_PLUGIN(ROS2Plugin)
}
```

## Simulation Control and Monitoring

### Common Gazebo-ROS Services:
```bash
# Reset simulation
ros2 service call /reset_simulation std_srvs/srv/Empty

# Reset world
ros2 service call /reset_world std_srvs/srv/Empty

# Pause/unpause simulation
ros2 service call /pause_physics std_srvs/srv/Empty
ros2 service call /unpause_physics std_srvs/srv/Empty

# Spawn model
ros2 run gazebo_ros spawn_entity.py -entity my_robot -file /path/to/model.sdf -x 0 -y 0 -z 1

# Delete model
ros2 service call /delete_entity gazebo_msgs/srv/DeleteEntity "{name: 'my_robot'}"
```

### Simulation Parameters:
```bash
# Get/set simulation parameters
ros2 param list
ros2 param get /gazebo use_sim_time
```

## Hardware-in-the-Loop (HIL) Testing

### HIL Architecture:
Hardware-in-the-loop testing allows connecting real hardware components to simulated environments:

```
Real Controller ←→ ROS 2 ←→ Gazebo Simulator ←→ Simulated Sensors
                    ↑
            Real Sensors/Actuators
```

### Implementation Considerations:
- Network latency compensation
- Real-time performance requirements
- Safety measures for physical systems
- Synchronization between real and simulated components

## Unity-ROS Integration

### Unity Robotics Setup:
Unity can also be integrated with ROS for advanced simulation scenarios:

```bash
# Install Unity ROS TCP Connector
git clone https://github.com/Unity-Technologies/ROS-TCP-Connector.git
```

### Unity-ROS Communication:
```csharp
using ROS2;
using UnityEngine;

public class UnityROSBridge : MonoBehaviour
{
    private ROS2UnityComponent ros2_unity;
    private ROS2Node node;
    private Publisher<geometry_msgs.msg.Twist> cmdVelPub;
    private Subscriber<sensor_msgs.msg.LaserScan> laserSub;

    void Start()
    {
        ros2_unity = GetComponent<ROS2UnityComponent>();
        ros2_unity.Initialize();
        node = ros2_unity.CreateNode("unity_robot");

        cmdVelPub = node.CreatePublisher<geometry_msgs.msg.Twist>("/cmd_vel");
        laserSub = node.CreateSubscriber<sensor_msgs.msg.LaserScan>("/scan");
    }

    void Update()
    {
        // Publish commands to ROS
        var twist = new geometry_msgs.msg.Twist();
        twist.linear.x = 1.0f; // Move forward
        cmdVelPub.Publish(twist);
    }

    void OnLaserScanMessage(sensor_msgs.msg.LaserScan msg)
    {
        // Process laser scan data from ROS
        Debug.Log("Received laser scan with " + msg.ranges.Count + " points");
    }
}
```

## Best Practices for Integration

### Performance Optimization:
- Use appropriate update rates for different sensors
- Minimize unnecessary ROS 2 topics and services
- Optimize collision geometries for simulation performance
- Use efficient data structures for sensor data

### Debugging Strategies:
- Monitor simulation timing and real-time factor
- Check TF tree for transformation issues
- Validate sensor data quality and ranges
- Use Gazebo's built-in visualization tools

### Development Workflow:
- Test components individually before integration
- Use simulation checkpoints for debugging
- Implement proper error handling for connection failures
- Document all ROS 2 interfaces and dependencies

### Testing and Validation:
- Compare simulation and real robot behavior
- Validate sensor models against real hardware
- Test edge cases and failure scenarios
- Monitor computational performance requirements

The integration between Gazebo/Unity and ROS 2 provides a powerful platform for robotics development, enabling rapid prototyping, testing, and validation of complex robotic systems before deployment on real hardware.