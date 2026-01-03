---
sidebar_label: 'ROS 2 Topics'
---

# ROS 2 Topics

Topics in ROS 2 enable asynchronous communication between nodes using a publish-subscribe pattern. This section covers the fundamentals of topic-based communication in ROS 2.

## Understanding Topics

Topics are named buses over which nodes exchange messages. The communication is:
- **Asynchronous**: Publishers and subscribers don't need to be active simultaneously
- **Loosely coupled**: Publishers don't know who subscribes, and subscribers don't know who publishes
- **Many-to-many**: Multiple publishers can publish to a topic, and multiple subscribers can listen to the same topic

## Message Types

Messages are the data structures exchanged over topics. ROS 2 provides standard message types and allows custom message definitions.

### Common Message Types:
- `std_msgs`: Basic data types (Int32, Float64, String, etc.)
- `geometry_msgs`: Geometric primitives (Point, Pose, Twist, etc.)
- `sensor_msgs`: Sensor data (LaserScan, Image, JointState, etc.)
- `nav_msgs`: Navigation messages (Odometry, Path, OccupancyGrid, etc.)

## Creating Publishers and Subscribers

### C++ Publisher Example:
```cpp
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

class MinimalPublisher : public rclcpp::Node
{
public:
  MinimalPublisher()
  : Node("minimal_publisher")
  {
    publisher_ = this->create_publisher<std_msgs::msg::String>("topic", 10);
    timer_ = this->create_wall_timer(
      500ms, std::bind(&MinimalPublisher::timer_callback, this));
  }

private:
  void timer_callback()
  {
    auto message = std_msgs::msg::String();
    message.data = "Hello, world!";
    RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
    publisher_->publish(message);
  }
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
};
```

### C++ Subscriber Example:
```cpp
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

class MinimalSubscriber : public rclcpp::Node
{
public:
  MinimalSubscriber()
  : Node("minimal_subscriber")
  {
    subscription_ = this->create_subscription<std_msgs::msg::String>(
      "topic", 10,
      std::bind(&MinimalSubscriber::topic_callback, this, std::placeholders::_1));
  }

private:
  void topic_callback(const std_msgs::msg::String::SharedPtr msg) const
  {
    RCLCPP_INFO(this->get_logger(), "I heard: '%s'", msg->data.c_str());
  }
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
};
```

### Python Publisher Example:
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

### Python Subscriber Example:
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

## Quality of Service (QoS)

QoS settings allow fine-tuning communication behavior:

```cpp
// Reliable delivery
rclcpp::QoS reliable_qos(10);
reliable_qos.reliable();

// Best effort delivery
rclcpp::QoS best_effort_qos(10);
best_effort_qos.best_effort();

// Keep last N messages
rclcpp::QoS keep_last_qos(5);
keep_last_qos.keep_last(5);
```

## Topic Commands

### Command Line Tools:
```bash
# List all topics
ros2 topic list

# Echo messages on a topic
ros2 topic echo /topic_name

# Publish a message to a topic
ros2 topic pub /topic_name std_msgs/String "data: 'Hello'"

# Get info about a topic
ros2 topic info /topic_name

# Show topic type
ros2 topic type /topic_name
```

## Advanced Topic Concepts

### Message Filters:
For processing messages that arrive at different times but need to be processed together:

```cpp
#include "message_filters/subscriber.h"
#include "message_filters/time_synchronizer.h"

// Example for synchronizing multiple topics
auto sub1 = std::make_shared<message_filters::Subscriber<MsgType1>>(node, "topic1");
auto sub2 = std::make_shared<message_filters::Subscriber<MsgType2>>(node, "topic2");

auto sync = std::make_shared<message_filters::TimeSynchronizer<MsgType1, MsgType2>>(*sub1, *sub2, 10);
sync->registerCallback(std::bind(&callback, std::placeholders::_1, std::placeholders::_2));
```

## Best Practices

- Use meaningful topic names following ROS naming conventions
- Choose appropriate QoS settings based on application requirements
- Implement proper error handling for communication failures
- Use latching for static data that new subscribers should receive immediately
- Consider message frequency and bandwidth when designing topic architecture
- Use appropriate message types or create custom messages when needed

Topics form the backbone of ROS 2 communication, enabling flexible and scalable robotic systems.