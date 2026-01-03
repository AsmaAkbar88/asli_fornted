---
sidebar_label: 'ROS 2 Nodes'
---

# ROS 2 Nodes

Nodes are fundamental building blocks in ROS 2 that represent individual processes performing computation. This section covers the creation, management, and communication patterns of ROS 2 nodes.

## Understanding Nodes

A node is a single executable that uses ROS 2 to communicate with other nodes. Nodes can:
- Publish messages to topics
- Subscribe to topics to receive messages
- Provide services
- Call services
- Execute actions
- Manage parameters

## Creating Nodes

### C++ Node Example:
```cpp
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

class MinimalPublisher : public rclcpp::Node
{
public:
  MinimalPublisher()
  : Node("minimal_publisher"), count_(0)
  {
    publisher_ = this->create_publisher<std_msgs::msg::String>("topic", 10);
    timer_ = this->create_wall_timer(
      500ms, std::bind(&MinimalPublisher::timer_callback, this));
  }

private:
  void timer_callback()
  {
    auto message = std_msgs::msg::String();
    message.data = "Hello, world! " + std::to_string(count_++);
    RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
    publisher_->publish(message);
  }
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
  size_t count_;
};
```

### Python Node Example:
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

## Node Lifecycle

ROS 2 nodes follow a specific lifecycle state machine:
- **Unconfigured**: Node is created but not configured
- **Inactive**: Node is configured but not activated
- **Active**: Node is fully operational
- **Finalized**: Node is shut down

### Lifecycle Node Example:
```cpp
#include "rclcpp_lifecycle/lifecycle_node.hpp"

class LifecycleNodeExample : public rclcpp_lifecycle::LifecycleNode
{
public:
  LifecycleNodeExample() : rclcpp_lifecycle::LifecycleNode("lifecycle_node")
  {
  }

  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_configure(const rclcpp_lifecycle::State &)
  {
    RCLCPP_INFO(get_logger(), "Configuring");
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
  }

  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_activate(const rclcpp_lifecycle::State &)
  {
    RCLCPP_INFO(get_logger(), "Activating");
    return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
  }
};
```

## Node Management

### Launching Nodes:
```bash
# Run a node directly
ros2 run package_name executable_name

# Launch with arguments
ros2 run package_name executable_name --ros-args --remap __node:=new_node_name
```

### Launch Files:
```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='package_name',
            executable='node_name',
            name='node_name',
            parameters=[
                {'param_name': 'param_value'}
            ]
        )
    ])
```

## Best Practices

- Keep nodes focused on a single responsibility
- Use meaningful node names
- Implement proper error handling
- Consider using lifecycle nodes for complex systems
- Use launch files for managing multiple nodes
- Follow ROS 2 naming conventions

Nodes form the foundation of ROS 2 applications, and understanding their creation and management is essential for building robust robotic systems.