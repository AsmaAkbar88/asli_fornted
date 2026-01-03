---
sidebar_label: 'ROS 2 Services'
---

# ROS 2 Services

Services in ROS 2 provide synchronous request-response communication between nodes. This section covers the fundamentals of service-based communication in ROS 2.

## Understanding Services

Services enable synchronous communication where:
- A client sends a request to a server
- The server processes the request and sends back a response
- The client waits for the response before continuing
- Communication is one-to-one (one client to one server)

## Service Types

Services use service definition files (.srv) that define the request and response message structure:

```
# Request (before the --- separator)
string name
int32 age
---
# Response (after the --- separator)
bool success
string message
```

## Creating Services and Clients

### C++ Service Server Example:
```cpp
#include "rclcpp/rclcpp.hpp"
#include "example_interfaces/srv/add_two_ints.hpp"

class MinimalService : public rclcpp::Node
{
public:
  MinimalService()
  : Node("minimal_service")
  {
    // Create service server
    service_ = create_service<example_interfaces::srv::AddTwoInts>(
      "add_two_ints",
      [this](
        const example_interfaces::srv::AddTwoInts::Request::SharedPtr request,
        example_interfaces::srv::AddTwoInts::Response::SharedPtr response)
      {
        response->sum = request->a + request->b;
        RCLCPP_INFO(
          this->get_logger(), "Incoming request\na: %ld, b: %ld",
          request->a, request->b);
        RCLCPP_INFO(this->get_logger(), "Sending response: [%ld]", response->sum);
      });
  }

private:
  rclcpp::Service<example_interfaces::srv::AddTwoInts>::SharedPtr service_;
};
```

### C++ Service Client Example:
```cpp
#include "rclcpp/rclcpp.hpp"
#include "example_interfaces/srv/add_two_ints.hpp"

class MinimalClient : public rclcpp::Node
{
public:
  MinimalClient()
  : Node("minimal_client")
  {
    client_ = create_client<example_interfaces::srv::AddTwoInts>("add_two_ints");
  }

  void send_request()
  {
    // Wait for service to be available
    while (!client_->wait_for_service(1s)) {
      if (!rclcpp::ok()) {
        RCLCPP_ERROR(this->get_logger(), "Interrupted while waiting for the service. Exiting.");
        return;
      }
      RCLCPP_INFO(this->get_logger(), "Service not available, waiting again...");
    }

    // Create request
    auto request = std::make_shared<example_interfaces::srv::AddTwoInts::Request>();
    request->a = 2;
    request->b = 3;

    // Send async request
    auto result_future = client_->async_send_request(request);

    // Wait for result
    if (rclcpp::spin_until_future_complete(this->shared_from_this(), result_future) ==
        rclcpp::FutureReturnCode::SUCCESS)
    {
      RCLCPP_INFO(this->get_logger(), "Result of add_two_ints: %ld", result_future.get()->sum);
    } else {
      RCLCPP_ERROR(this->get_logger(), "Failed to call service add_two_ints");
    }
  }

private:
  rclcpp::Client<example_interfaces::srv::AddTwoInts>::SharedPtr client_;
};
```

### Python Service Server Example:
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
        self.get_logger().info('Incoming request\na: %d, b: %d' % (request.a, request.b))
        return response
```

### Python Service Client Example:
```python
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class MinimalClient(Node):

    def __init__(self):
        super().__init__('minimal_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')

    def send_request(self):
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        request = AddTwoInts.Request()
        request.a = 2
        request.b = 3

        self.future = self.cli.call_async(request)
        rclpy.spin_until_future_complete(self, self.future)

        return self.future.result()
```

## Service Commands

### Command Line Tools:
```bash
# List all services
ros2 service list

# Get info about a service
ros2 service info /service_name

# Show service type
ros2 service type /service_name

# Call a service from command line
ros2 service call /add_two_ints example_interfaces/AddTwoInts "{a: 1, b: 2}"
```

## Defining Custom Services

### Creating a Custom Service Definition:
Create a file named `CustomService.srv`:

```
# Request
string name
int32 id
---
# Response
bool success
string message
float64 result
```

### Using Custom Services in Packages:
In your package.xml, add:
```xml
<build_depend>rosidl_default_generators</build_depend>
<exec_depend>rosidl_default_runtime</exec_depend>
<member_of_group>rosidl_interface_packages</member_of_group>
```

In your CMakeLists.txt:
```cmake
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/CustomService.srv"
)
```

## Advanced Service Concepts

### Service Quality of Service (QoS):
Services have their own QoS settings that can be configured:

```cpp
rmw_qos_profile_t qos_profile = rmw_qos_profile_services_default;
qos_profile.reliability = RMW_QOS_POLICY_RELIABILITY_RELIABLE;
qos_profile.history = RMW_QOS_POLICY_HISTORY_KEEP_ALL;

auto service = create_service<ServiceType>(
  "service_name",
  callback,
  qos_profile
);
```

### Exception Handling in Services:
```cpp
try {
  auto result = client_->async_send_request(request);
  // Handle response
} catch (const std::exception& e) {
  RCLCPP_ERROR(node->get_logger(), "Service call failed: %s", e.what());
}
```

## Best Practices

- Use services for operations that have a clear request-response pattern
- Choose appropriate timeouts for service calls to avoid indefinite blocking
- Implement proper error handling for service unavailability
- Use meaningful service names following ROS naming conventions
- Design service interfaces to be intuitive and easy to use
- Consider using actions instead of services for long-running operations
- Document service interfaces clearly with examples

Services provide a reliable way to implement request-response communication patterns in ROS 2, making them ideal for operations that require a response before the client can proceed.