---
sidebar_label: 'ROS 2 Basics'
---

# ROS 2 Basics

This section covers the fundamental concepts of ROS 2 that form the foundation for developing robotic applications.

## What is ROS 2?

ROS 2 (Robot Operating System 2) is not an operating system but rather a flexible framework for writing robot software. It provides a collection of tools, libraries, and conventions that aim to simplify the development of complex robotic applications.

Key features of ROS 2 include:
- Distributed computing framework
- Language independence (C++, Python, and others)
- Hardware abstraction
- Device drivers
- Libraries for common robot functionality
- Tools for visualization, debugging, and simulation

## ROS 2 Architecture

ROS 2 uses a distributed system architecture based on the DDS (Data Distribution Service) standard for communication. Unlike ROS 1 which relied on a centralized master, ROS 2 employs a peer-to-peer discovery mechanism.

### Key Components:
- **Nodes**: Individual processes that perform computation
- **Topics**: Named buses over which nodes exchange messages
- **Services**: Synchronous request/response communication
- **Actions**: Asynchronous request/response with feedback
- **Parameters**: Configuration values that can be changed at runtime

## Installation and Setup

ROS 2 supports multiple platforms including Ubuntu, Windows, and macOS. The most common installation is on Ubuntu LTS versions.

### Installing ROS 2:
```bash
# Add ROS 2 apt repository
sudo apt update && sudo apt install curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install ros-humble-desktop
```

### Environment Setup:
```bash
source /opt/ros/humble/setup.bash
```

## Workspaces and Packages

ROS 2 organizes code into packages, which are grouped into workspaces.

### Creating a Workspace:
```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
colcon build
source install/setup.bash
```

### Creating a Package:
```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_cmake my_robot_package
```

## Core Concepts Summary

- **Nodes** are individual programs that communicate with each other
- **Topics** enable asynchronous, publish-subscribe communication
- **Services** provide synchronous request-response communication
- **Actions** offer asynchronous communication with feedback and goal preemption
- **Parameters** allow configuration values to be changed at runtime

Understanding these basics is crucial for building more complex robotic systems in subsequent modules.