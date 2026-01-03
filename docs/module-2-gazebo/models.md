---
sidebar_label: 'Robot and Environment Modeling'
---

# Robot and Environment Modeling

This section covers the creation and configuration of robot models and environment objects for use in simulation environments.

## Robot Modeling in SDF/URDF

Robot models in Gazebo are typically defined using either SDF (Simulation Description Format) or URDF (Unified Robot Description Format). Both formats describe the physical and visual properties of robots.

### SDF Robot Model Example:
```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="simple_robot">
    <pose>0 0 0.5 0 0 0</pose>

    <!-- Base link -->
    <link name="chassis">
      <pose>0 0 0 0 0 0</pose>
      <inertial>
        <mass>5.0</mass>
        <inertia>
          <ixx>0.1</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.1</iyy>
          <iyz>0.0</iyz>
          <izz>0.1</izz>
        </inertia>
      </inertial>

      <visual name="chassis_visual">
        <geometry>
          <box>
            <size>1.0 0.5 0.3</size>
          </box>
        </geometry>
        <material>
          <ambient>0.8 0.8 0.8 1</ambient>
          <diffuse>0.8 0.8 0.8 1</diffuse>
        </material>
      </visual>

      <collision name="chassis_collision">
        <geometry>
          <box>
            <size>1.0 0.5 0.3</size>
          </box>
        </geometry>
      </collision>
    </link>

    <!-- Wheel joints and links -->
    <joint name="front_left_wheel_joint" type="revolute">
      <parent>chassis</parent>
      <child>front_left_wheel</child>
      <axis>
        <xyz>0 1 0</xyz>
        <limit>
          <lower>-1.75</lower>
          <upper>1.75</upper>
          <effort>100</effort>
          <velocity>10</velocity>
        </limit>
      </axis>
      <pose>0.3 0.3 0 0 0 0</pose>
    </joint>

    <link name="front_left_wheel">
      <inertial>
        <mass>0.5</mass>
        <inertia>
          <ixx>0.01</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.01</iyy>
          <iyz>0.0</iyz>
          <izz>0.01</izz>
        </inertia>
      </inertial>

      <visual name="wheel_visual">
        <geometry>
          <cylinder>
            <radius>0.15</radius>
            <length>0.05</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.2 0.2 0.2 1</ambient>
          <diffuse>0.2 0.2 0.2 1</diffuse>
        </material>
      </visual>

      <collision name="wheel_collision">
        <geometry>
          <cylinder>
            <radius>0.15</radius>
            <length>0.05</length>
          </cylinder>
        </geometry>
      </collision>
    </link>
  </model>
</sdf>
```

### URDF to SDF Conversion:
URDF models can be converted to SDF for use in Gazebo:

```xml
<!-- URDF file example -->
<robot name="my_robot">
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>

    <collision>
      <geometry>
        <box size="0.5 0.5 0.2"/>
      </geometry>
    </collision>

    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>
</robot>
```

## Model Database and Custom Models

### Using Gazebo Model Database:
Gazebo provides a large database of pre-built models that can be used in simulations:

```bash
# List available models
gz model --list

# Insert a model into the simulation
gz model --model-name "unit_cylinder" --model-file `gz model --info unit_cylinder | grep -o "file://.*"` --spawn-position 0,0,1,0,0,0
```

### Creating Custom Models:
Custom models should be organized in the following structure:
```
models/
├── my_robot/
│   ├── model.sdf
│   ├── model.config
│   └── meshes/
│       ├── visual/
│       └── collision/
└── my_environment/
    ├── model.sdf
    └── model.config
```

### Model Configuration File (model.config):
```xml
<?xml version="1.0"?>
<model>
  <name>My Robot</name>
  <version>1.0</version>
  <sdf version="1.7">model.sdf</sdf>

  <author>
    <name>Your Name</name>
    <email>your.email@example.com</email>
  </author>

  <description>
    A custom robot model for simulation.
  </description>
</model>
```

## Environment Modeling

### Creating Custom Environments:
Environment models can be created as static objects or complex interactive worlds:

```xml
<!-- Simple room environment -->
<sdf version="1.7">
  <world name="simple_room">
    <!-- Room walls -->
    <model name="wall_1">
      <static>true</static>
      <link name="link">
        <visual name="visual">
          <geometry>
            <box><size>10 0.2 3</size></box>
          </geometry>
          <material>
            <ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
          </material>
        </visual>
        <collision name="collision">
          <geometry>
            <box><size>10 0.2 3</size></box>
          </geometry>
        </collision>
      </link>
      <pose>0 5 1.5 0 0 0</pose>
    </model>

    <!-- Furniture and obstacles -->
    <model name="table">
      <static>true</static>
      <link name="table_top">
        <visual name="visual">
          <geometry>
            <box><size>1.5 0.8 0.02</size></box>
          </geometry>
        </visual>
        <collision name="collision">
          <geometry>
            <box><size>1.5 0.8 0.02</size></box>
          </geometry>
        </collision>
      </link>
      <link name="leg_1">
        <visual name="visual">
          <geometry>
            <box><size>0.05 0.05 0.7</size></box>
          </geometry>
        </visual>
        <collision name="collision">
          <geometry>
            <box><size>0.05 0.05 0.7</size></box>
          </geometry>
        </collision>
      </link>
      <!-- Additional legs would be connected with joints -->
    </model>
  </world>
</sdf>
```

## Unity Robot and Environment Modeling

### Unity Asset Structure:
When using Unity for robotics simulation, the asset structure typically includes:
- **Prefabs**: Reusable robot and object models
- **Materials**: Surface properties and visual appearance
- **Scenes**: Complete simulation environments
- **Scripts**: Custom behaviors and ROS communication

### Unity Robot Model Example:
Unity robot models typically include:
- Colliders for physics simulation
- Joints for articulated movement
- Sensors (cameras, LIDAR, etc.)
- ROS communication components

```csharp
// Example Unity script for ROS communication
using ROS2;
using UnityEngine;

public class UnityRobotController : MonoBehaviour
{
    private ROS2UnityComponent ros2Unity;
    private ROS2Node node;
    private Publisher<std_msgs.msg.Float64MultiArray> jointPub;

    void Start()
    {
        ros2Unity = GetComponent<ROS2UnityComponent>();
        ros2Unity.Initialize();
        node = ros2Unity.CreateNode("unity_robot_controller");
        jointPub = node.CreatePublisher<std_msgs.msg.Float64MultiArray>("/joint_commands");
    }

    void Update()
    {
        // Publish joint states or other sensor data
        var jointMsg = new std_msgs.msg.Float64MultiArray();
        // ... populate message with joint data
        jointPub.Publish(jointMsg);
    }
}
```

## Best Practices for Modeling

### Physics Considerations:
- Balance visual fidelity with computational efficiency
- Use appropriate collision geometries (simpler than visual meshes)
- Set realistic inertial properties for stable simulation
- Consider the trade-off between accuracy and performance

### Model Organization:
- Use consistent naming conventions
- Group related models in logical directories
- Document model parameters and assumptions
- Version control complex models separately if needed

### Validation:
- Test models in simple environments before complex scenarios
- Verify mass, center of mass, and inertial properties
- Check joint limits and ranges of motion
- Validate sensor placements and fields of view

## Model Import and Export

### Converting Existing Models:
- Use tools like `collada_urdf` for converting COLLADA files to URDF
- Use `xacro` for parameterized URDF models
- Validate models using `check_urdf` command

```bash
# Check URDF validity
check_urdf my_robot.urdf

# Convert XACRO to URDF
xacro my_robot.xacro > my_robot.urdf
```

Proper modeling is essential for realistic simulation, as it directly affects how robots interact with their environment and how accurately the simulation represents real-world physics.