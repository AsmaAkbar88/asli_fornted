---
sidebar_label: 'Navigation Integration'
---

# Navigation Integration

This section covers the integration of navigation capabilities into the Physical AI system, enabling autonomous movement and path planning for the integrated robotic platform. Navigation bridges the gap between planning and physical movement, allowing the robot to traverse environments safely and efficiently.

## Navigation Architecture Overview
 
The navigation system in our integrated Physical AI system follows a layered approach:

```
Global Planner (Path Planning) → Local Planner (Obstacle Avoidance) → Controller (Motion Execution)
```

Each layer operates at different frequencies and spatial scales, working together to achieve safe and efficient navigation.

## Global Path Planning

### Costmap Management:
```python
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

class CostmapManager:
    def __init__(self, resolution=0.05, size_x=100, size_y=100):
        self.resolution = resolution
        self.size_x = size_x
        self.size_y = size_y
        self.origin_x = 0.0
        self.origin_y = 0.0

        # Initialize costmap
        self.costmap = np.zeros((size_y, size_x), dtype=np.uint8)
        self.footprint_radius = 0.3  # Robot footprint in meters

    def update_from_perception(self, detected_objects):
        """
        Update costmap based on perception data
        """
        # Clear current costmap
        self.costmap.fill(0)

        for obj in detected_objects:
            if obj['type'] in ['obstacle', 'furniture', 'wall']:
                self.add_obstacle_to_costmap(obj['pose'], obj['size'])

        # Apply inflation to account for robot footprint
        self.inflate_costmap()

    def add_obstacle_to_costmap(self, pose, size):
        """
        Add an obstacle to the costmap
        """
        # Convert world coordinates to grid coordinates
        grid_x = int((pose.position.x - self.origin_x) / self.resolution)
        grid_y = int((pose.position.y - self.origin_y) / self.resolution)

        # Calculate grid size based on object dimensions
        grid_size_x = max(1, int(size[0] / self.resolution))
        grid_size_y = max(1, int(size[1] / self.resolution))

        # Add obstacle to costmap
        start_x = max(0, grid_x - grid_size_x // 2)
        end_x = min(self.size_x, grid_x + grid_size_x // 2 + 1)
        start_y = max(0, grid_y - grid_size_y // 2)
        end_y = min(self.size_y, grid_y + grid_size_y // 2 + 1)

        self.costmap[start_y:end_y, start_x:end_x] = 254  # Lethal obstacle

    def inflate_costmap(self):
        """
        Inflate costmap to account for robot footprint
        """
        # Convert lethal obstacles to boolean mask
        obstacle_mask = self.costmap >= 254

        # Calculate inflation radius in grid cells
        inflation_cells = int(self.footprint_radius / self.resolution)

        # Dilate obstacles by inflation radius
        inflated_mask = binary_dilation(obstacle_mask, iterations=inflation_cells)

        # Update costmap with inflated obstacles
        self.costmap[inflated_mask] = 254

        # Add gradual cost gradient around obstacles
        distance_map = distance_transform_edt(~inflated_mask)
        cost_gradient = np.clip(distance_map / inflation_cells, 0, 1) * 253
        self.costmap = np.maximum(self.costmap, cost_gradient.astype(np.uint8))

    def is_valid_cell(self, x, y):
        """
        Check if a grid cell is valid for navigation
        """
        if x < 0 or x >= self.size_x or y < 0 or y >= self.size_y:
            return False
        return self.costmap[y, x] < 254  # Not a lethal obstacle

    def get_cost(self, x, y):
        """
        Get cost of a grid cell
        """
        if x < 0 or x >= self.size_x or y < 0 or y >= self.size_y:
            return 255  # Invalid/outside bounds
        return self.costmap[y, x]
```

### A* Path Planner:
```python
import heapq

class AStarPlanner:
    def __init__(self, costmap_manager):
        self.costmap = costmap_manager

    def plan_path(self, start_pose, goal_pose):
        """
        Plan path using A* algorithm
        """
        # Convert poses to grid coordinates
        start_grid = self.pose_to_grid(start_pose)
        goal_grid = self.pose_to_grid(goal_pose)

        # Check if start and goal are valid
        if not self.costmap.is_valid_cell(start_grid[0], start_grid[1]):
            print(f"Start position {start_grid} is not valid")
            return None
        if not self.costmap.is_valid_cell(goal_grid[0], goal_grid[1]):
            print(f"Goal position {goal_grid} is not valid")
            return None

        # A* algorithm
        open_set = [(0, start_grid)]  # (f_score, position)
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self.heuristic(start_grid, goal_grid)}

        # Convert to set for O(1) lookup
        open_set_hash = {start_grid}

        while open_set:
            current = heapq.heappop(open_set)
            current_pos = current[1]

            # Remove from hash set
            open_set_hash.remove(current_pos)

            if current_pos == goal_grid:
                return self.reconstruct_path(came_from, current_pos)

            # Check 8-connected neighbors
            for neighbor in self.get_neighbors(current_pos):
                tentative_g_score = g_score[current_pos] + self.get_move_cost(current_pos, neighbor)

                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current_pos
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal_grid)

                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)

        # No path found
        return None

    def heuristic(self, pos1, pos2):
        """
        Heuristic function for A* (Euclidean distance)
        """
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5

    def get_neighbors(self, pos):
        """
        Get valid 8-connected neighbors
        """
        neighbors = []
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

        for dx, dy in directions:
            neighbor = (pos[0] + dx, pos[1] + dy)
            if self.costmap.is_valid_cell(neighbor[0], neighbor[1]):
                neighbors.append(neighbor)

        return neighbors

    def get_move_cost(self, pos1, pos2):
        """
        Get cost of moving from pos1 to pos2
        """
        # Diagonal move cost is higher
        if abs(pos1[0] - pos2[0]) == 1 and abs(pos1[1] - pos2[1]) == 1:
            base_cost = 1.414  # sqrt(2)
        else:
            base_cost = 1.0

        # Add costmap cost
        costmap_cost = self.costmap.get_cost(pos2[0], pos2[1]) / 254.0
        return base_cost * (1 + costmap_cost)

    def pose_to_grid(self, pose):
        """
        Convert world pose to grid coordinates
        """
        grid_x = int((pose.position.x - self.costmap.origin_x) / self.costmap.resolution)
        grid_y = int((pose.position.y - self.costmap.origin_y) / self.costmap.resolution)
        return (grid_x, grid_y)

    def reconstruct_path(self, came_from, current):
        """
        Reconstruct path from A* result
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)

        path.reverse()

        # Convert grid path to world coordinates
        world_path = []
        for grid_pos in path:
            world_x = grid_pos[0] * self.costmap.resolution + self.costmap.origin_x
            world_y = grid_pos[1] * self.costmap.resolution + self.costmap.origin_y
            world_path.append((world_x, world_y))

        return world_path
```

## Local Path Planning and Obstacle Avoidance

### Dynamic Window Approach:
```python
import math

class LocalPlanner:
    def __init__(self, robot_params):
        self.max_vel_x = robot_params.get('max_vel_x', 0.5)
        self.max_vel_theta = robot_params.get('max_vel_theta', 1.0)
        self.min_vel_x = robot_params.get('min_vel_x', 0.1)
        self.acc_lim_x = robot_params.get('acc_lim_x', 2.5)
        self.acc_lim_theta = robot_params.get('acc_lim_theta', 3.2)
        self.sim_time = robot_params.get('sim_time', 1.7)
        self.sim_granularity = robot_params.get('sim_granularity', 0.025)

        self.costmap = None
        self.robot_pose = None
        self.robot_vel = None

    def set_costmap(self, costmap):
        self.costmap = costmap

    def set_robot_state(self, pose, velocity):
        self.robot_pose = pose
        self.robot_vel = velocity

    def compute_velocity_commands(self, global_plan):
        """
        Compute velocity commands using Dynamic Window Approach
        """
        if not global_plan or len(global_plan) < 2:
            return (0.0, 0.0)  # Stop

        # Get local goal from global plan
        local_goal = self.get_local_goal(global_plan)

        # Calculate dynamic window
        vs = self.calculate_velocity_space()
        vd = self.calculate_dynamic_window()

        # Score trajectories
        best_score = float('-inf')
        best_vel = (0.0, 0.0)

        for vel_x in np.arange(vd[0], vd[1], 0.1):  # Linear velocity
            for vel_theta in np.arange(vd[2], vd[3], 0.1):  # Angular velocity
                score = self.score_trajectory((vel_x, vel_theta), local_goal)

                if score > best_score:
                    best_score = score
                    best_vel = (vel_x, vel_theta)

        return best_vel

    def calculate_velocity_space(self):
        """
        Calculate the complete velocity space
        """
        return (-self.max_vel_x, self.max_vel_x, -self.max_vel_theta, self.max_vel_theta)

    def calculate_dynamic_window(self):
        """
        Calculate the dynamic window based on robot constraints
        """
        # Calculate velocity limits based on acceleration limits
        max_acc_x = self.acc_lim_x * self.sim_time
        max_acc_theta = self.acc_lim_theta * self.sim_time

        min_vel_x = max(self.robot_vel.linear.x - max_acc_x, -self.max_vel_x)
        max_vel_x = min(self.robot_vel.linear.x + max_acc_x, self.max_vel_x)
        min_vel_theta = max(self.robot_vel.angular.z - max_acc_theta, -self.max_vel_theta)
        max_vel_theta = min(self.robot_vel.angular.z + max_acc_theta, self.max_vel_theta)

        return (min_vel_x, max_vel_x, min_vel_theta, max_vel_theta)

    def score_trajectory(self, velocity, local_goal):
        """
        Score a trajectory based on multiple criteria
        """
        # Simulate trajectory
        sim_time = 0.0
        pose = self.robot_pose

        while sim_time < self.sim_time:
            # Update pose based on velocity
            dt = self.sim_granularity
            pose = self.update_pose(pose, velocity, dt)
            sim_time += dt

            # Check for collisions
            if self.is_in_collision(pose):
                return float('-inf')  # Invalid trajectory

        # Calculate scores for different criteria
        goal_dist_score = self.calculate_goal_distance_score(pose, local_goal)
        heading_score = self.calculate_heading_score(pose, local_goal)
        velocity_score = self.calculate_velocity_score(velocity)

        # Weighted combination of scores
        total_score = (0.8 * goal_dist_score +
                      0.1 * heading_score +
                      0.1 * velocity_score)

        return total_score

    def calculate_goal_distance_score(self, pose, goal):
        """
        Calculate score based on distance to goal
        """
        dist = math.sqrt((pose.position.x - goal[0])**2 + (pose.position.y - goal[1])**2)
        return 1.0 / (1.0 + dist)  # Higher score for closer distance

    def calculate_heading_score(self, pose, goal):
        """
        Calculate score based on heading toward goal
        """
        desired_theta = math.atan2(goal[1] - pose.position.y, goal[0] - pose.position.x)
        current_theta = self.get_yaw_from_quaternion(pose.orientation)

        heading_diff = abs(desired_theta - current_theta)
        return 1.0 / (1.0 + heading_diff)

    def calculate_velocity_score(self, velocity):
        """
        Calculate score based on velocity (prefer higher velocities)
        """
        return math.sqrt(velocity[0]**2 + velocity[1]**2)

    def update_pose(self, pose, velocity, dt):
        """
        Update robot pose based on velocity and time step
        """
        new_pose = copy.deepcopy(pose)

        # Update position
        yaw = self.get_yaw_from_quaternion(pose.orientation)
        new_pose.position.x += velocity[0] * math.cos(yaw) * dt
        new_pose.position.y += velocity[0] * math.sin(yaw) * dt

        # Update orientation
        new_yaw = yaw + velocity[1] * dt
        new_pose.orientation = self.yaw_to_quaternion(new_yaw)

        return new_pose

    def is_in_collision(self, pose):
        """
        Check if pose is in collision with obstacles
        """
        grid_x = int((pose.position.x - self.costmap.origin_x) / self.costmap.resolution)
        grid_y = int((pose.position.y - self.costmap.origin_y) / self.costmap.resolution)

        return not self.costmap.is_valid_cell(grid_x, grid_y)

    def get_yaw_from_quaternion(self, quat):
        """
        Extract yaw from quaternion
        """
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def yaw_to_quaternion(self, yaw):
        """
        Convert yaw to quaternion
        """
        from geometry_msgs.msg import Quaternion
        quat = Quaternion()
        quat.z = math.sin(yaw / 2.0)
        quat.w = math.cos(yaw / 2.0)
        return quat

    def get_local_goal(self, global_plan):
        """
        Get the local goal from the global plan
        """
        # For simplicity, use the last point in the global plan
        # In practice, this would be the point closest to the robot within a certain distance
        if global_plan:
            return global_plan[-1]
        return (0.0, 0.0)
```

## Navigation Integration with ROS 2

### Navigation Action Server:
```python
import rclpy
from rclpy.action import ActionServer, CancelResponse
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import LaserScan
import math

class NavigationActionServer(Node):
    def __init__(self):
        super().__init__('navigation_action_server')

        # Action server
        self._action_server = ActionServer(
            self,
            NavigateToPose,
            'navigate_to_pose',
            self.execute_callback,
            cancel_callback=self.cancel_callback
        )

        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.laser_sub = self.create_subscription(
            LaserScan, 'scan', self.laser_callback, 10
        )

        # Navigation components
        self.costmap_manager = CostmapManager()
        self.global_planner = AStarPlanner(self.costmap_manager)
        self.local_planner = LocalPlanner({
            'max_vel_x': 0.5,
            'max_vel_theta': 1.0,
            'acc_lim_x': 2.5,
            'acc_lim_theta': 3.2,
            'sim_time': 1.7
        })

        # Navigation state
        self.current_goal = None
        self.navigation_active = False

    def execute_callback(self, goal_handle):
        """
        Execute navigation action
        """
        self.get_logger().info('Executing navigation goal')

        goal_pose = goal_handle.request.pose
        self.current_goal = goal_pose
        self.navigation_active = True

        # Plan global path
        robot_pose = self.get_current_pose()
        global_path = self.global_planner.plan_path(robot_pose, goal_pose.pose)

        if not global_path:
            self.get_logger().error('Failed to plan global path')
            goal_handle.abort()
            return NavigateToPose.Result()

        # Execute navigation
        result = self.follow_path(global_path, goal_handle)

        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            return NavigateToPose.Result()
        elif result.success:
            goal_handle.succeed()
            return result
        else:
            goal_handle.abort()
            return NavigateToPose.Result()

    def follow_path(self, global_path, goal_handle):
        """
        Follow the global path using local planner
        """
        while self.navigation_active and rclpy.ok():
            # Get current robot state
            current_pose = self.get_current_pose()
            current_vel = self.get_current_velocity()

            # Update local planner with current state
            self.local_planner.set_robot_state(current_pose, current_vel)

            # Compute velocity commands
            vel_x, vel_theta = self.local_planner.compute_velocity_commands(global_path)

            # Publish velocity commands
            cmd_vel = Twist()
            cmd_vel.linear.x = vel_x
            cmd_vel.angular.z = vel_theta
            self.cmd_vel_pub.publish(cmd_vel)

            # Check if goal reached
            goal_dist = self.distance_to_pose(current_pose, self.current_goal.pose)
            if goal_dist < 0.2:  # 20cm tolerance
                result = NavigateToPose.Result()
                result.result = 1  # SUCCESS
                return result

            # Check for obstacles
            if self.detect_immediate_obstacle():
                self.get_logger().warn('Obstacle detected, replanning...')
                # Try to replan or wait
                continue

            # Check for action cancellation
            if goal_handle.is_cancel_requested:
                self.navigation_active = False
                cmd_vel = Twist()
                cmd_vel.linear.x = 0.0
                cmd_vel.angular.z = 0.0
                self.cmd_vel_pub.publish(cmd_vel)
                return NavigateToPose.Result()

            # Sleep to control loop rate
            self.get_clock().sleep_for(Duration(seconds=0.1))

        result = NavigateToPose.Result()
        result.result = 0  # FAILURE
        return result

    def laser_callback(self, msg):
        """
        Process laser scan data for obstacle detection
        """
        # Update costmap with laser data
        self.update_costmap_from_scan(msg)

    def get_current_pose(self):
        """
        Get current robot pose (in practice, this would come from TF or localization)
        """
        # Placeholder implementation
        from geometry_msgs.msg import Pose
        pose = Pose()
        pose.position.x = 0.0
        pose.position.y = 0.0
        pose.orientation.w = 1.0
        return pose

    def get_current_velocity(self):
        """
        Get current robot velocity
        """
        from geometry_msgs.msg import Twist
        twist = Twist()
        return twist

    def distance_to_pose(self, pose1, pose2):
        """
        Calculate distance between two poses
        """
        dx = pose1.position.x - pose2.position.x
        dy = pose1.position.y - pose2.position.y
        return math.sqrt(dx*dx + dy*dy)

    def detect_immediate_obstacle(self):
        """
        Detect if there's an immediate obstacle ahead
        """
        # This would use laser scan data to detect close obstacles
        return False

    def update_costmap_from_scan(self, scan_msg):
        """
        Update costmap based on laser scan
        """
        # Process laser scan to detect obstacles and update costmap
        pass

    def cancel_callback(self, goal_handle):
        """
        Handle action cancellation
        """
        self.get_logger().info('Canceling navigation goal')
        self.navigation_active = False
        return CancelResponse.ACCEPT
```

## Navigation Safety and Recovery

### Recovery Behaviors:
```python
class NavigationRecovery:
    def __init__(self, robot_interface):
        self.robot = robot_interface
        self.recovery_behaviors = {
            'clear_costmap': self.clear_costmap_recovery,
            'rotate_in_place': self.rotate_in_place_recovery,
            'move_backward': self.move_backward_recovery,
            'wait_and_retry': self.wait_and_retry_recovery
        }

    def execute_recovery_behavior(self, behavior_type, params=None):
        """
        Execute specified recovery behavior
        """
        if behavior_type in self.recovery_behaviors:
            return self.recovery_behaviors[behavior_type](params)
        else:
            self.robot.get_logger().error(f'Unknown recovery behavior: {behavior_type}')
            return False

    def clear_costmap_recovery(self, params):
        """
        Clear the costmap to remove potential false obstacles
        """
        self.robot.get_logger().info('Executing clear costmap recovery')
        # In ROS 2 navigation, this would call the clear_costmap service
        # For simulation, reset the costmap
        return True

    def rotate_in_place_recovery(self, params):
        """
        Rotate in place to clear potential sensor issues
        """
        self.robot.get_logger().info('Executing rotate in place recovery')

        # Rotate 360 degrees slowly
        angular_vel = 0.5  # rad/s
        duration = 2 * math.pi / angular_vel  # Time to rotate 360 degrees

        start_time = self.robot.get_clock().now()
        while (self.robot.get_clock().now() - start_time).nanoseconds < duration * 1e9:
            cmd_vel = Twist()
            cmd_vel.angular.z = angular_vel
            self.robot.cmd_vel_pub.publish(cmd_vel)

            # Check for cancellation
            if self.robot.navigation_active and not self.robot.navigation_active:
                cmd_vel = Twist()
                cmd_vel.angular.z = 0.0
                self.robot.cmd_vel_pub.publish(cmd_vel)
                return False

        # Stop rotation
        cmd_vel = Twist()
        cmd_vel.angular.z = 0.0
        self.robot.cmd_vel_pub.publish(cmd_vel)

        return True

    def move_backward_recovery(self, params):
        """
        Move backward to get out of tight spaces
        """
        self.robot.get_logger().info('Executing move backward recovery')

        # Move backward slowly
        linear_vel = -0.2  # m/s (negative for backward)
        duration = 1.0  # seconds

        start_time = self.robot.get_clock().now()
        while (self.robot.get_clock().now() - start_time).nanoseconds < duration * 1e9:
            cmd_vel = Twist()
            cmd_vel.linear.x = linear_vel
            self.robot.cmd_vel_pub.publish(cmd_vel)

            # Check for cancellation
            if self.robot.navigation_active and not self.robot.navigation_active:
                cmd_vel = Twist()
                cmd_vel.linear.x = 0.0
                self.robot.cmd_vel_pub.publish(cmd_vel)
                return False

        # Stop
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        self.robot.cmd_vel_pub.publish(cmd_vel)

        return True

    def wait_and_retry_recovery(self, params):
        """
        Wait for a period then retry navigation
        """
        self.robot.get_logger().info('Executing wait and retry recovery')

        wait_time = params.get('wait_time', 5.0)  # seconds

        # Stop robot
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.robot.cmd_vel_pub.publish(cmd_vel)

        # Wait
        self.robot.get_clock().sleep_for(Duration(seconds=wait_time))

        return True

    def select_recovery_behavior(self, failure_reason):
        """
        Select appropriate recovery behavior based on failure reason
        """
        if failure_reason == 'local_planner_failed':
            return 'rotate_in_place'
        elif failure_reason == 'global_planner_failed':
            return 'clear_costmap'
        elif failure_reason == 'oscillation':
            return 'move_backward'
        elif failure_reason == 'blocked_path':
            return 'wait_and_retry'
        else:
            return 'clear_costmap'  # Default recovery
```

## Multi-floor Navigation

### Map Management for Multi-floor Environments:
```python
class MultiFloorNavigation:
    def __init__(self):
        self.floor_maps = {}  # Maps for each floor
        self.floor_transitions = {}  # Connections between floors (elevators, stairs)
        self.current_floor = 0
        self.elevator_manager = ElevatorManager()

    def load_floor_map(self, floor_number, map_data):
        """
        Load map data for a specific floor
        """
        self.floor_maps[floor_number] = map_data

    def set_floor_transition(self, floor1, floor2, transition_point):
        """
        Define a transition point between floors
        """
        if floor1 not in self.floor_transitions:
            self.floor_transitions[floor1] = {}
        self.floor_transitions[floor1][floor2] = transition_point

    def navigate_to_floor_location(self, target_floor, target_location):
        """
        Navigate to a location on a different floor
        """
        if target_floor == self.current_floor:
            # Same floor - use normal navigation
            return self.navigate_to_location(target_location)

        # Different floor - need to find path to transition
        transition_path = self.find_path_to_transition(self.current_floor, target_floor)
        if not transition_path:
            return False

        # Navigate to transition point
        success = self.navigate_to_location(transition_path['current_floor_target'])
        if not success:
            return False

        # Use elevator/stairs to change floors
        floor_change_success = self.change_floor(
            self.current_floor,
            target_floor,
            transition_path['transition_point']
        )

        if not floor_change_success:
            return False

        # Update current floor
        self.current_floor = target_floor

        # Navigate to final destination on target floor
        return self.navigate_to_location(target_location)

    def find_path_to_transition(self, current_floor, target_floor):
        """
        Find path from current location to floor transition point
        """
        # Find the best transition point between floors
        if current_floor in self.floor_transitions:
            if target_floor in self.floor_transitions[current_floor]:
                transition_point = self.floor_transitions[current_floor][target_floor]

                # Plan path to transition point using current floor map
                current_pose = self.get_current_pose()
                path_to_transition = self.plan_path_on_floor(
                    current_floor,
                    current_pose,
                    transition_point
                )

                if path_to_transition:
                    return {
                        'current_floor_target': transition_point,
                        'path': path_to_transition,
                        'transition_point': transition_point
                    }

        return None

    def change_floor(self, from_floor, to_floor, transition_point):
        """
        Handle the physical transition between floors
        """
        # This would interface with elevator/stairs system
        return self.elevator_manager.move_to_floor(to_floor)

    def plan_path_on_floor(self, floor_number, start_pose, goal_pose):
        """
        Plan path on a specific floor
        """
        if floor_number not in self.floor_maps:
            return None

        # Use the map for the specified floor to plan path
        costmap = self.create_costmap_from_map(self.floor_maps[floor_number])
        planner = AStarPlanner(costmap)
        return planner.plan_path(start_pose, goal_pose)

    def create_costmap_from_map(self, map_data):
        """
        Create costmap from map data
        """
        # Convert map data to costmap format
        # This is a simplified implementation
        resolution = map_data.get('resolution', 0.05)
        width = map_data['width']
        height = map_data['height']

        costmap = CostmapManager(
            resolution=resolution,
            size_x=width,
            size_y=height
        )

        # Populate costmap with map data
        # map_data['data'] contains occupancy grid data
        for i, value in enumerate(map_data['data']):
            x = i % width
            y = i // width
            if value > 50:  # Consider as obstacle (adjust threshold as needed)
                costmap.costmap[y, x] = min(254, value)  # Lethal or high cost

        return costmap

class ElevatorManager:
    def __init__(self):
        self.current_floor = 0
        self.target_floor = 0
        self.elevator_busy = False

    def move_to_floor(self, floor_number):
        """
        Request elevator to move to specified floor
        """
        if self.elevator_busy:
            return False

        self.elevator_busy = True
        self.target_floor = floor_number

        # Simulate elevator movement
        # In practice, this would interface with elevator control system
        success = self.simulate_elevator_movement()

        self.elevator_busy = False
        return success

    def simulate_elevator_movement(self):
        """
        Simulate elevator movement (in practice, interface with real elevator)
        """
        import time
        time.sleep(3)  # Simulate elevator travel time
        self.current_floor = self.target_floor
        return True
```

## Navigation Performance and Optimization

### Adaptive Navigation Parameters:
```python
class AdaptiveNavigation:
    def __init__(self, base_params):
        self.base_params = base_params
        self.current_params = base_params.copy()
        self.performance_history = []
        self.adaptation_enabled = True

    def adapt_parameters(self, current_performance):
        """
        Adapt navigation parameters based on performance
        """
        if not self.adaptation_enabled:
            return

        self.performance_history.append(current_performance)

        # Keep only recent performance data
        if len(self.performance_history) > 50:
            self.performance_history = self.performance_history[-50:]

        # Calculate performance metrics
        avg_success_rate = self.calculate_avg_success_rate()
        avg_time = self.calculate_avg_time()
        avg_energy = self.calculate_avg_energy()

        # Adapt parameters based on performance
        if avg_success_rate < 0.8:  # Success rate too low
            # Make navigation more conservative
            self.current_params['max_vel_x'] *= 0.8
            self.current_params['max_vel_theta'] *= 0.8
            self.current_params['inflation_radius'] *= 1.2

        elif avg_time > 1.5 * self.base_params['expected_time']:  # Too slow
            # Make navigation more aggressive
            self.current_params['max_vel_x'] = min(
                self.base_params['max_vel_x'],
                self.current_params['max_vel_x'] * 1.1
            )
            self.current_params['max_vel_theta'] = min(
                self.base_params['max_vel_theta'],
                self.current_params['max_vel_theta'] * 1.1
            )

    def calculate_avg_success_rate(self):
        """
        Calculate average navigation success rate
        """
        if not self.performance_history:
            return 1.0

        successful_navigations = sum(1 for perf in self.performance_history if perf.get('success', False))
        return successful_navigations / len(self.performance_history)

    def calculate_avg_time(self):
        """
        Calculate average navigation time
        """
        successful_times = [perf['time'] for perf in self.performance_history if perf.get('success', False)]
        if not successful_times:
            return float('inf')
        return sum(successful_times) / len(successful_times)

    def calculate_avg_energy(self):
        """
        Calculate average navigation energy consumption
        """
        successful_energies = [perf['energy'] for perf in self.performance_history if perf.get('success', False)]
        if not successful_energies:
            return float('inf')
        return sum(successful_energies) / len(successful_energies)

    def get_current_parameters(self):
        """
        Get current navigation parameters
        """
        return self.current_params.copy()

    def reset_adaptation(self):
        """
        Reset adaptation to base parameters
        """
        self.current_params = self.base_params.copy()
        self.performance_history = []
```

## Integration with Overall System

### Navigation-Perception Integration:
```python
class NavigationPerceptionIntegrator:
    def __init__(self, navigation_system, perception_system):
        self.navigation = navigation_system
        self.perception = perception_system
        self.dynamic_obstacle_tracker = DynamicObstacleTracker()

    def update_navigation_with_perception(self):
        """
        Update navigation system with perception data
        """
        # Get static obstacles from perception
        static_obstacles = self.perception.get_static_obstacles()
        self.navigation.update_static_costmap(static_obstacles)

        # Get dynamic obstacles from perception
        dynamic_obstacles = self.perception.get_dynamic_obstacles()
        self.navigation.update_dynamic_costmap(
            dynamic_obstacles,
            self.dynamic_obstacle_tracker
        )

        # Get door/window states
        door_states = self.perception.get_door_states()
        self.navigation.update_traversable_areas(door_states)

    def handle_dynamic_obstacle(self, obstacle_info):
        """
        Handle dynamic obstacle that may affect navigation
        """
        # Predict obstacle trajectory
        predicted_trajectory = self.predict_obstacle_trajectory(obstacle_info)

        # Check if obstacle will block planned path
        path_affected = self.check_path_affected(
            self.navigation.get_current_plan(),
            predicted_trajectory
        )

        if path_affected:
            # Replan if necessary
            self.navigation.replan_with_dynamic_obstacles(
                predicted_trajectory
            )

    def predict_obstacle_trajectory(self, obstacle_info):
        """
        Predict trajectory of dynamic obstacle
        """
        # Use constant velocity or acceleration model
        current_pos = obstacle_info['position']
        current_vel = obstacle_info.get('velocity', [0, 0])

        # Predict position for next few seconds
        prediction_horizon = 5.0  # seconds
        dt = 0.5  # prediction step
        trajectory = []

        for t in np.arange(0, prediction_horizon, dt):
            predicted_pos = [
                current_pos[0] + current_vel[0] * t,
                current_pos[1] + current_vel[1] * t
            ]
            trajectory.append(predicted_pos)

        return trajectory

    def check_path_affected(self, path, obstacle_trajectory):
        """
        Check if obstacle trajectory affects planned path
        """
        safety_margin = 0.5  # meters

        for path_point in path:
            for obs_pos in obstacle_trajectory:
                distance = math.sqrt(
                    (path_point[0] - obs_pos[0])**2 +
                    (path_point[1] - obs_pos[1])**2
                )
                if distance < safety_margin:
                    return True

        return False

class DynamicObstacleTracker:
    def __init__(self):
        self.tracked_obstacles = {}
        self.next_id = 0

    def update_obstacles(self, detected_obstacles):
        """
        Update tracked obstacles with new detections
        """
        # Data association and tracking
        for obs in detected_obstacles:
            best_match = self.find_best_match(obs)

            if best_match is not None:
                # Update existing track
                self.update_track(best_match, obs)
            else:
                # Create new track
                self.create_new_track(obs)

        # Remove old tracks that haven't been seen recently
        self.remove_old_tracks()

    def find_best_match(self, new_observation):
        """
        Find best matching tracked obstacle for new observation
        """
        best_match = None
        best_distance = float('inf')

        for track_id, track in self.tracked_obstacles.items():
            distance = self.calculate_observation_distance(new_observation, track)
            if distance < best_distance and distance < 1.0:  # 1m threshold
                best_distance = distance
                best_match = track_id

        return best_match

    def calculate_observation_distance(self, obs, track):
        """
        Calculate distance between observation and track
        """
        pos1 = obs['position']
        pos2 = track['position']
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def update_track(self, track_id, observation):
        """
        Update existing track with new observation
        """
        track = self.tracked_obstacles[track_id]

        # Update position with weighted average
        alpha = 0.7  # tracking parameter
        track['position'][0] = alpha * observation['position'][0] + (1 - alpha) * track['position'][0]
        track['position'][1] = alpha * observation['position'][1] + (1 - alpha) * track['position'][1]

        # Update velocity estimate
        dt = 0.1  # assume constant time step for simplicity
        if 'last_position' in track:
            dx = track['position'][0] - track['last_position'][0]
            dy = track['position'][1] - track['last_position'][1]
            track['velocity'] = [dx/dt, dy/dt]

        track['last_position'] = track['position'].copy()
        track['last_seen'] = time.time()

    def create_new_track(self, observation):
        """
        Create new track for observation
        """
        track_id = self.next_id
        self.next_id += 1

        self.tracked_obstacles[track_id] = {
            'id': track_id,
            'position': observation['position'].copy(),
            'velocity': [0.0, 0.0],
            'last_seen': time.time(),
            'observations': [observation]
        }

    def remove_old_tracks(self):
        """
        Remove tracks that haven't been seen recently
        """
        current_time = time.time()
        timeout = 3.0  # seconds

        old_tracks = [
            track_id for track_id, track in self.tracked_obstacles.items()
            if current_time - track['last_seen'] > timeout
        ]

        for track_id in old_tracks:
            del self.tracked_obstacles[track_id]
```

The navigation integration component enables the Physical AI system to move autonomously through complex environments, avoiding obstacles and reaching desired destinations. It works closely with perception systems to maintain awareness of the environment and with planning systems to execute high-level goals.