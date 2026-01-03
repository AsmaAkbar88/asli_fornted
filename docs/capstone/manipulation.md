---
sidebar_label: 'Manipulation Integration'
---

# Manipulation Integration

This section covers the integration of manipulation capabilities into the Physical AI system, enabling the robot to physically interact with objects in its environment. Manipulation represents the final step in the voice → planning → navigation → object detection → manipulation pipeline, where high-level commands are translated into precise physical actions.
 
## Manipulation Architecture Overview

The manipulation system in our integrated Physical AI system follows a hierarchical control approach:

```
High-Level Task Planner → Grasp Planner → Trajectory Generator → Low-Level Controller → Robot Hardware
```

Each layer operates at different time scales and control granularities, working together to achieve precise and safe manipulation.

## Robot Arm Control and Kinematics

### Forward and Inverse Kinematics:
```python
import numpy as np
from math import sin, cos, atan2, sqrt

class RobotKinematics:
    def __init__(self, dh_parameters):
        """
        Initialize robot kinematics with Denavit-Hartenberg parameters
        dh_parameters: list of [a, alpha, d, theta_offset] for each joint
        """
        self.dh_params = dh_parameters
        self.num_joints = len(dh_params)

    def forward_kinematics(self, joint_angles):
        """
        Calculate end-effector pose from joint angles using DH parameters
        """
        if len(joint_angles) != self.num_joints:
            raise ValueError("Number of joint angles must match number of joints")

        # Initialize transformation matrix as identity
        T = np.eye(4)

        for i in range(self.num_joints):
            a, alpha, d, theta_offset = self.dh_params[i]
            theta = joint_angles[i] + theta_offset

            # DH transformation matrix for this joint
            T_i = np.array([
                [cos(theta), -sin(theta)*cos(alpha),  sin(theta)*sin(alpha), a*cos(theta)],
                [sin(theta),  cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
                [0,          sin(alpha),             cos(alpha),             d],
                [0,          0,                      0,                      1]
            ])

            # Combine with previous transformations
            T = T @ T_i

        return T

    def inverse_kinematics(self, target_pose, current_joints=None, max_iterations=100, tolerance=1e-4):
        """
        Calculate joint angles to reach target pose using Jacobian transpose method
        """
        if current_joints is None:
            current_joints = np.zeros(self.num_joints)

        joints = np.array(current_joints, dtype=float)

        for iteration in range(max_iterations):
            # Calculate current end-effector pose
            current_pose = self.forward_kinematics(joints)

            # Calculate error
            pos_error = target_pose[:3, 3] - current_pose[:3, 3]
            rot_error = self.rotation_matrix_to_axis_angle(
                target_pose[:3, :3] @ current_pose[:3, :3].T
            )

            error = np.concatenate([pos_error, rot_error])

            if np.linalg.norm(error) < tolerance:
                break

            # Calculate Jacobian
            J = self.calculate_jacobian(joints)

            # Update joint angles using Jacobian transpose
            joints += 0.1 * J.T @ error

        return joints

    def calculate_jacobian(self, joint_angles):
        """
        Calculate geometric Jacobian matrix
        """
        J = np.zeros((6, self.num_joints))

        # End-effector position from forward kinematics
        T_total = self.forward_kinematics(joint_angles)
        ee_pos = T_total[:3, 3]

        T_so_far = np.eye(4)

        for i in range(self.num_joints):
            a, alpha, d, theta_offset = self.dh_params[i]
            theta = joint_angles[i] + theta_offset

            # Transformation matrix for this joint
            T_i = np.array([
                [cos(theta), -sin(theta)*cos(alpha),  sin(theta)*sin(alpha), a*cos(theta)],
                [sin(theta),  cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
                [0,          sin(alpha),             cos(alpha),             d],
                [0,          0,                      0,                      1]
            ])

            # Calculate z-axis of current joint frame
            z_i = T_so_far[:3, 2]

            # Calculate position of current joint to end-effector
            joint_pos = T_so_far[:3, 3]
            r = ee_pos - joint_pos

            # For revolute joint
            J[:3, i] = np.cross(z_i, r)  # Linear velocity part
            J[3:, i] = z_i               # Angular velocity part

            # Update transformation matrix
            T_so_far = T_so_far @ T_i

        return J

    def rotation_matrix_to_axis_angle(self, R):
        """
        Convert rotation matrix to axis-angle representation
        """
        angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))

        if angle < 1e-6:
            return np.zeros(3)

        scale = angle / (2 * np.sin(angle))
        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ]) * scale

        return axis

class ArmController:
    def __init__(self, robot_kinematics, joint_limits=None):
        self.kinematics = robot_kinematics
        self.joint_limits = joint_limits or [(-np.pi, np.pi)] * kinematics.num_joints
        self.current_joints = np.zeros(kinematics.num_joints)

    def move_to_pose(self, target_pose, max_speed=0.1):
        """
        Move end-effector to target pose
        """
        # Calculate inverse kinematics
        target_joints = self.kinematics.inverse_kinematics(target_pose)

        # Check joint limits
        target_joints = self.apply_joint_limits(target_joints)

        # Generate trajectory
        current_pose = self.kinematics.forward_kinematics(self.current_joints)
        current_joints = self.current_joints.copy()

        # Simple linear interpolation in joint space
        num_steps = 50
        joint_trajectory = []

        for i in range(num_steps + 1):
            ratio = i / num_steps
            joints = current_joints + ratio * (target_joints - current_joints)
            joint_trajectory.append(joints.copy())

        # Execute trajectory
        for joints in joint_trajectory:
            self.set_joint_positions(joints)
            self.current_joints = joints
            time.sleep(0.02)  # 50Hz control rate

    def apply_joint_limits(self, joints):
        """
        Apply joint limits to joint angles
        """
        limited_joints = joints.copy()
        for i, (min_limit, max_limit) in enumerate(self.joint_limits):
            limited_joints[i] = np.clip(limited_joints[i], min_limit, max_limit)
        return limited_joints

    def set_joint_positions(self, joint_positions):
        """
        Set joint positions (interface to actual robot hardware)
        """
        # In practice, this would send commands to the robot
        # For simulation, just update internal state
        print(f"Setting joint positions: {joint_positions}")
```

## Grasp Planning and Execution

### Grasp Pose Generation:
```python
class GraspPlanner:
    def __init__(self, robot_arm, gripper_specs):
        self.arm = robot_arm
        self.gripper_specs = gripper_specs
        self.approach_distance = 0.1  # meters
        self.grasp_depth = 0.05       # meters

    def plan_grasp_poses(self, object_info):
        """
        Plan approach and grasp poses for an object
        """
        object_pose = self.object_to_pose(object_info)

        # Generate multiple grasp candidates
        grasp_candidates = self.generate_grasp_candidates(object_pose, object_info)

        # Score and select best grasp
        best_grasp = self.select_best_grasp(grasp_candidates, object_info)

        return best_grasp

    def generate_grasp_candidates(self, object_pose, object_info):
        """
        Generate multiple grasp pose candidates
        """
        candidates = []

        # Generate different grasp angles around the object
        for angle in np.linspace(0, 2*np.pi, 8):  # 8 different angles
            # Approach from different directions
            approach_vec = np.array([cos(angle), sin(angle), 0])

            # Calculate grasp pose
            grasp_position = object_pose[:3, 3] + approach_vec * self.approach_distance
            grasp_orientation = self.calculate_grasp_orientation(approach_vec, object_info)

            grasp_pose = np.eye(4)
            grasp_pose[:3, 3] = grasp_position
            grasp_pose[:3, :3] = grasp_orientation

            # Calculate approach pose (before grasp)
            approach_position = object_pose[:3, 3] + approach_vec * (self.approach_distance + 0.05)
            approach_pose = np.eye(4)
            approach_pose[:3, 3] = approach_position
            approach_pose[:3, :3] = grasp_orientation

            candidate = {
                'approach_pose': approach_pose,
                'grasp_pose': grasp_pose,
                'grasp_type': self.determine_grasp_type(object_info),
                'score': self.score_grasp_candidate(grasp_pose, object_info)
            }

            candidates.append(candidate)

        return candidates

    def calculate_grasp_orientation(self, approach_vector, object_info):
        """
        Calculate appropriate grasp orientation based on approach vector
        """
        # Ensure approach vector is normalized
        approach = approach_vector / np.linalg.norm(approach_vector)

        # Define gripper orientation (typically aligned with approach vector)
        z_axis = -approach  # Gripper approaches along -z axis
        x_axis = np.array([0, 0, 1])  # Default up direction

        # Calculate y axis (orthogonal to x and z)
        y_axis = np.cross(z_axis, x_axis)
        if np.linalg.norm(y_axis) < 1e-6:  # If parallel
            x_axis = np.array([1, 0, 0])
            y_axis = np.cross(z_axis, x_axis)

        y_axis = y_axis / np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)

        # Create rotation matrix
        rotation = np.column_stack([x_axis, y_axis, z_axis])
        return rotation

    def determine_grasp_type(self, object_info):
        """
        Determine appropriate grasp type based on object properties
        """
        class_name = object_info['class_name']
        bbox = object_info['bbox']

        if class_name in ['cup', 'mug']:
            return 'cylindrical'
        elif class_name in ['book', 'box']:
            if bbox[2] > bbox[3]:  # Width > height
                return 'top_pinch'
            else:
                return 'side_grasp'
        elif class_name in ['bottle']:
            return 'parallel' if bbox[3] > bbox[2] else 'top_grasp'
        else:
            return 'pinch'

    def score_grasp_candidate(self, grasp_pose, object_info):
        """
        Score grasp candidate based on multiple criteria
        """
        score = 0.0

        # Consider object size and shape
        bbox = object_info['bbox']
        aspect_ratio = max(bbox[2], bbox[3]) / max(min(bbox[2], bbox[3]), 1e-6)
        score += 1.0 / aspect_ratio  # Better for more cube-like objects

        # Consider approach angle
        z_axis = grasp_pose[:3, 2]
        if abs(z_axis[2]) > 0.7:  # Prefer grasps approaching from above/below
            score += 1.0

        # Consider accessibility
        score += object_info.get('confidence', 0.5)  # Higher confidence objects get higher score

        return score

    def select_best_grasp(self, candidates, object_info):
        """
        Select the best grasp candidate from the list
        """
        if not candidates:
            return None

        # Sort by score (descending)
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[0]

    def execute_grasp(self, grasp_info):
        """
        Execute the grasp sequence
        """
        approach_pose = grasp_info['approach_pose']
        grasp_pose = grasp_info['grasp_pose']

        # Move to approach position
        print("Moving to approach position...")
        self.arm.move_to_pose(approach_pose)

        # Move to grasp position
        print("Moving to grasp position...")
        self.arm.move_to_pose(grasp_pose)

        # Close gripper
        print("Closing gripper...")
        self.close_gripper()

        # Lift object slightly
        print("Lifting object...")
        current_pose = self.arm.kinematics.forward_kinematics(self.arm.current_joints)
        lift_pose = current_pose.copy()
        lift_pose[2, 3] += 0.05  # Lift 5cm
        self.arm.move_to_pose(lift_pose)

        print("Grasp completed successfully")

    def close_gripper(self):
        """
        Close the robot gripper (interface to actual hardware)
        """
        print("Gripper closed")

    def open_gripper(self):
        """
        Open the robot gripper (interface to actual hardware)
        """
        print("Gripper opened")

    def object_to_pose(self, object_info):
        """
        Convert object information to a pose matrix
        """
        # This would use 3D object position and orientation
        # For now, creating a simple pose from 2D information
        pose = np.eye(4)
        pose[0, 3] = object_info['center'][0] / 100  # Convert pixels to meters (approximate)
        pose[1, 3] = object_info['center'][1] / 100
        pose[2, 3] = 0.5  # Assume object is at 0.5m height
        return pose
```

## Trajectory Generation and Execution

### Smooth Trajectory Planning:
```python
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

class TrajectoryGenerator:
    def __init__(self, max_velocity=1.0, max_acceleration=2.0):
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration

    def generate_cartesian_trajectory(self, waypoints, duration=None):
        """
        Generate smooth Cartesian trajectory through waypoints
        """
        if len(waypoints) < 2:
            return []

        # Use cubic splines for smooth interpolation
        t_waypoints = np.linspace(0, 1, len(waypoints))

        # Extract x, y, z coordinates
        x_vals = [wp[0] for wp in waypoints]
        y_vals = [wp[1] for wp in waypoints]
        z_vals = [wp[2] if len(wp) > 2 else 0 for wp in waypoints]

        # Create cubic splines
        x_spline = CubicSpline(t_waypoints, x_vals)
        y_spline = CubicSpline(t_waypoints, y_vals)
        z_spline = CubicSpline(t_waypoints, z_vals)

        # Generate more intermediate points
        if duration:
            num_points = int(duration * 50)  # 50Hz sampling
        else:
            num_points = len(waypoints) * 10

        t_fine = np.linspace(0, 1, num_points)

        trajectory = []
        for t in t_fine:
            pos = [x_spline(t), y_spline(t), z_spline(t)]
            vel = [x_spline.derivative()(t), y_spline.derivative()(t), z_spline.derivative()(t)]

            # Normalize velocity to respect limits
            vel_mag = np.linalg.norm(vel)
            if vel_mag > self.max_velocity:
                vel = [v * self.max_velocity / vel_mag for v in vel]

            trajectory.append({
                'position': pos,
                'velocity': vel,
                'time': t
            })

        return trajectory

    def generate_joint_trajectory(self, cartesian_trajectory, kinematics):
        """
        Convert Cartesian trajectory to joint space
        """
        joint_trajectory = []

        for point in cartesian_trajectory:
            # Convert Cartesian position to joint angles
            # This would involve inverse kinematics for each point
            cartesian_pos = point['position']

            # Create a pose matrix with position and a default orientation
            pose = np.eye(4)
            pose[:3, 3] = cartesian_pos

            try:
                joint_angles = kinematics.inverse_kinematics(pose, current_joints=joint_trajectory[-1]['positions'] if joint_trajectory else None)

                joint_trajectory.append({
                    'positions': joint_angles,
                    'velocities': point.get('velocity', [0, 0, 0]),  # This is approximate
                    'time': point['time']
                })
            except:
                # If IK fails, skip this point or use the previous one
                if joint_trajectory:
                    joint_trajectory.append(joint_trajectory[-1])  # Repeat last valid

        return joint_trajectory

    def optimize_trajectory(self, trajectory):
        """
        Optimize trajectory for smoothness and constraint satisfaction
        """
        # This is a simplified optimization
        # In practice, would use more sophisticated methods like CHOMP or STOMP
        optimized_trajectory = []

        for i, point in enumerate(trajectory):
            new_point = point.copy()

            # Apply simple smoothing by averaging with neighbors
            if 0 < i < len(trajectory) - 1:
                prev_pos = trajectory[i-1]['position']
                next_pos = trajectory[i+1]['position']

                # Weighted average: center has highest weight
                smoothed_pos = [
                    0.6 * point['position'][0] + 0.2 * prev_pos[0] + 0.2 * next_pos[0],
                    0.6 * point['position'][1] + 0.2 * prev_pos[1] + 0.2 * next_pos[1],
                    0.6 * point['position'][2] + 0.2 * prev_pos[2] + 0.2 * next_pos[2]
                ]

                new_point['position'] = smoothed_pos

            optimized_trajectory.append(new_point)

        return optimized_trajectory

class TrajectoryExecutor:
    def __init__(self, robot_controller, control_frequency=50.0):
        self.controller = robot_controller
        self.frequency = control_frequency
        self.dt = 1.0 / control_frequency

    def execute_trajectory(self, trajectory, blocking=True):
        """
        Execute a joint trajectory on the robot
        """
        if not trajectory:
            print("Empty trajectory, nothing to execute")
            return True

        print(f"Executing trajectory with {len(trajectory)} points")

        for point in trajectory:
            # Set joint positions
            self.controller.set_joint_positions(point['positions'])

            # Wait for control rate if blocking
            if blocking:
                time.sleep(self.dt)

        return True

    def execute_cartesian_trajectory(self, trajectory, kinematics):
        """
        Execute a Cartesian trajectory by converting to joint space in real-time
        """
        if not trajectory:
            return True

        for point in trajectory:
            # Convert Cartesian to joint space
            pose = np.eye(4)
            pose[:3, 3] = point['position']

            try:
                joint_angles = kinematics.inverse_kinematics(pose)
                self.controller.set_joint_positions(joint_angles)
                time.sleep(self.dt)
            except:
                print("Failed to find IK solution, skipping point")
                continue

        return True
```

## Manipulation Planning and Control

### High-Level Manipulation Planner:
```python
class ManipulationPlanner:
    def __init__(self, robot_arm, grasp_planner, trajectory_generator):
        self.arm = robot_arm
        self.grasp_planner = grasp_planner
        self.trajectory_gen = trajectory_generator
        self.executor = TrajectoryExecutor(robot_arm)

    def plan_manipulation_task(self, task_description, object_info, target_location=None):
        """
        Plan a complete manipulation task
        """
        task_plan = []

        # Step 1: Navigate to object (if needed)
        task_plan.append({
            'type': 'navigate_to_object',
            'object_info': object_info,
            'required': True
        })

        # Step 2: Plan grasp approach
        grasp_plan = self.grasp_planner.plan_grasp_poses(object_info)
        if not grasp_plan:
            raise Exception("Could not find valid grasp for object")

        task_plan.append({
            'type': 'plan_grasp',
            'grasp_info': grasp_plan,
            'required': True
        })

        # Step 3: Execute approach and grasp
        task_plan.append({
            'type': 'execute_grasp',
            'grasp_info': grasp_plan,
            'required': True
        })

        # Step 4: If target location specified, plan transport
        if target_location:
            task_plan.append({
                'type': 'transport_object',
                'target_location': target_location,
                'required': True
            })

            # Step 5: Place object
            task_plan.append({
                'type': 'place_object',
                'target_location': target_location,
                'required': True
            })

        return task_plan

    def execute_manipulation_task(self, task_plan):
        """
        Execute a manipulation task plan
        """
        for step in task_plan:
            print(f"Executing task step: {step['type']}")

            if step['type'] == 'navigate_to_object':
                self.execute_navigate_to_object(step['object_info'])
            elif step['type'] == 'plan_grasp':
                # Grasp plan already computed
                pass
            elif step['type'] == 'execute_grasp':
                self.grasp_planner.execute_grasp(step['grasp_info'])
            elif step['type'] == 'transport_object':
                self.execute_transport(step['target_location'])
            elif step['type'] == 'place_object':
                self.execute_placement(step['target_location'])

    def execute_navigate_to_object(self, object_info):
        """
        Execute navigation to object (this would interface with navigation system)
        """
        print(f"Navigating to object: {object_info['class_name']}")

    def execute_transport(self, target_location):
        """
        Transport the currently grasped object to target location
        """
        print(f"Transporting object to: {target_location}")

        # Move end-effector to target location while maintaining object grasp
        # This would involve complex trajectory planning to avoid obstacles
        # and maintain stable grasp

        # For simplicity, we'll just move to the target location
        target_pose = np.eye(4)
        target_pose[0, 3] = target_location[0] if len(target_location) > 0 else 0.5
        target_pose[1, 3] = target_location[1] if len(target_location) > 1 else 0.0
        target_pose[2, 3] = target_location[2] if len(target_location) > 2 else 0.2

        self.arm.move_to_pose(target_pose)

    def execute_placement(self, target_location):
        """
        Place the currently grasped object at target location
        """
        print(f"Placing object at: {target_location}")

        # Move to placement position
        place_pose = np.eye(4)
        place_pose[0, 3] = target_location[0] if len(target_location) > 0 else 0.5
        place_pose[1, 3] = target_location[1] if len(target_location) > 1 else 0.0
        place_pose[2, 3] = target_location[2] if len(target_location) > 2 else 0.2

        self.arm.move_to_pose(place_pose)

        # Open gripper to release object
        self.grasp_planner.open_gripper()

        # Lift arm slightly
        current_pose = self.arm.kinematics.forward_kinematics(self.arm.current_joints)
        lift_pose = current_pose.copy()
        lift_pose[2, 3] += 0.05  # Lift 5cm
        self.arm.move_to_pose(lift_pose)

        print("Object placement completed")
```

## Force Control and Compliance

### Force/Torque Control:
```python
class ForceController:
    def __init__(self, robot_interface, stiffness=1000, damping=20):
        self.robot = robot_interface
        self.stiffness = stiffness  # N/m or Nm/rad
        self.damping = damping      # Ns/m or Nms/rad
        self.current_forces = np.zeros(6)  # Fx, Fy, Fz, Tx, Ty, Tz
        self.desired_forces = np.zeros(6)
        self.compliance_frame = np.eye(4)  # Where compliance is applied

    def set_desired_force(self, force_vector):
        """
        Set desired end-effector forces
        force_vector: [Fx, Fy, Fz, Tx, Ty, Tz]
        """
        self.desired_forces = np.array(force_vector)

    def update_compliance(self, dt):
        """
        Update compliance control based on force feedback
        """
        # Get current forces from robot (in practice, read from force/torque sensor)
        current_forces = self.get_end_effector_forces()

        # Calculate force error
        force_error = self.desired_forces - current_forces

        # Calculate compliance adjustment
        compliance_adjustment = (force_error[:3] / self.stiffness) * dt
        compliance_rotation = (force_error[3:] / self.stiffness) * dt

        # Apply compliance adjustment to end-effector pose
        pose_adjustment = self.calculate_pose_adjustment(compliance_adjustment, compliance_rotation)

        # Apply to robot
        current_pose = self.robot.get_current_pose()
        new_pose = current_pose @ pose_adjustment

        self.robot.move_to_pose(new_pose)

    def get_end_effector_forces(self):
        """
        Get current end-effector forces (interface to force/torque sensor)
        """
        # In simulation, return zero forces or simulate force values
        # In real robot, read from actual force/torque sensor
        return np.random.normal(0, 0.1, 6)  # Simulated small forces

    def calculate_pose_adjustment(self, linear_adjustment, angular_adjustment):
        """
        Calculate pose adjustment matrix from linear and angular adjustments
        """
        # Create transformation matrix from adjustments
        adj = np.eye(4)
        adj[:3, 3] = linear_adjustment
        adj[:3, :3] = self.rotation_vector_to_matrix(angular_adjustment)

        return adj

    def rotation_vector_to_matrix(self, rotation_vector):
        """
        Convert rotation vector to rotation matrix
        """
        angle = np.linalg.norm(rotation_vector)
        if angle < 1e-6:
            return np.eye(3)

        axis = rotation_vector / angle
        x, y, z = axis
        c = np.cos(angle)
        s = np.sin(angle)
        C = 1 - c

        return np.array([
            [x*x*C + c,   x*y*C - z*s, x*z*C + y*s],
            [y*x*C + z*s, y*y*C + c,   y*z*C - x*s],
            [z*x*C - y*s, z*y*C + x*s, z*z*C + c]
        ])

class ImpedanceController:
    def __init__(self, robot_interface, mass=1.0, stiffness=1000, damping=20):
        self.robot = robot_interface
        self.mass = mass  # Effective mass of end-effector
        self.stiffness = stiffness
        self.damping = damping
        self.gravity = 9.81

        # State variables
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)

    def update_impedance_control(self, desired_pose, external_force, dt):
        """
        Update impedance control to achieve desired pose with force compliance
        """
        # Get current pose
        current_pose = self.robot.get_current_pose()
        current_pos = current_pose[:3, 3]

        # Calculate pose error
        position_error = desired_pose[:3, 3] - current_pos
        orientation_error = self.rotation_matrix_to_axis_angle(
            desired_pose[:3, :3] @ current_pose[:3, :3].T
        )

        # Impedance control law: M*a + B*v + K*x = F
        # a = (F - B*v - K*x) / M
        acceleration = (external_force[:3] -
                       self.damping * self.velocity -
                       self.stiffness * position_error) / self.mass

        # Update state (numerical integration)
        self.velocity += acceleration * dt
        self.position += self.velocity * dt

        # Calculate new pose
        new_pose = current_pose.copy()
        new_pose[:3, 3] += self.position

        # Apply to robot
        self.robot.move_to_pose(new_pose)

    def rotation_matrix_to_axis_angle(self, R):
        """
        Convert rotation matrix to axis-angle representation
        """
        angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))

        if angle < 1e-6:
            return np.zeros(3)

        scale = angle / (2 * np.sin(angle))
        axis = np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ]) * scale

        return axis
```

## Manipulation Safety and Error Handling

### Safety Monitor:
```python
class ManipulationSafetyMonitor:
    def __init__(self, robot_interface, max_force_threshold=50.0, max_speed_threshold=0.5):
        self.robot = robot_interface
        self.max_force_threshold = max_force_threshold
        self.max_speed_threshold = max_speed_threshold
        self.emergency_stop = False
        self.safety_violations = []

    def check_safety_conditions(self):
        """
        Check if current manipulation is safe
        """
        violations = []

        # Check joint limits
        current_joints = self.robot.get_current_joint_positions()
        joint_limits = self.robot.joint_limits

        for i, (joint_pos, (min_limit, max_limit)) in enumerate(zip(current_joints, joint_limits)):
            if joint_pos < min_limit or joint_pos > max_limit:
                violations.append(f"Joint {i} limit violation: {joint_pos}")

        # Check external forces
        current_forces = self.get_external_forces()
        if np.any(np.abs(current_forces) > self.max_force_threshold):
            violations.append(f"Force limit exceeded: {current_forces}")

        # Check velocity limits
        current_velocities = self.robot.get_current_joint_velocities()
        if np.any(np.abs(current_velocities) > self.max_speed_threshold):
            violations.append(f"Velocity limit exceeded: {current_velocities}")

        # Check for collisions (if collision checking is available)
        if self.check_for_collisions():
            violations.append("Collision detected")

        # Update violations list
        self.safety_violations = violations

        # Set emergency stop if critical violations
        if violations:
            for violation in violations:
                if 'collision' in violation.lower() or 'force' in violation.lower():
                    self.emergency_stop = True
                    print(f"EMERGENCY STOP: {violation}")
                    break

        return len(violations) == 0

    def get_external_forces(self):
        """
        Get external forces on the robot
        """
        # In practice, read from force/torque sensors
        # For simulation, return zero or simulated values
        return np.random.normal(0, 1, 6)

    def check_for_collisions(self):
        """
        Check for collisions (interface to collision checking system)
        """
        # In practice, this would use planning scene collision checking
        return False

    def handle_safety_violation(self, violation_type):
        """
        Handle specific safety violations
        """
        if 'collision' in violation_type.lower():
            self.execute_collision_recovery()
        elif 'force' in violation_type.lower():
            self.execute_force_recovery()
        elif 'joint' in violation_type.lower():
            self.execute_joint_limit_recovery()

    def execute_collision_recovery(self):
        """
        Execute collision recovery behavior
        """
        print("Executing collision recovery...")
        # Move away from collision direction
        # Stop all motion
        self.robot.stop_motion()

    def execute_force_recovery(self):
        """
        Execute force-based recovery
        """
        print("Executing force recovery...")
        # Reduce applied forces
        # Check if object is stuck or environment constraint
        self.robot.reduce_force_output()

    def execute_joint_limit_recovery(self):
        """
        Execute joint limit recovery
        """
        print("Executing joint limit recovery...")
        # Move joints away from limits
        current_joints = self.robot.get_current_joint_positions()
        new_joints = current_joints.copy()

        for i, (joint_pos, (min_limit, max_limit)) in enumerate(zip(current_joints, self.robot.joint_limits)):
            if joint_pos < min_limit:
                new_joints[i] = min_limit + 0.01
            elif joint_pos > max_limit:
                new_joints[i] = max_limit - 0.01

        self.robot.move_to_joint_positions(new_joints)

    def reset_safety_monitor(self):
        """
        Reset safety monitor after recovery
        """
        self.emergency_stop = False
        self.safety_violations = []
        print("Safety monitor reset")
```

## Integration with Overall System

### Manipulation-Planning Integration:
```python
class ManipulationPlanningInterface:
    def __init__(self, manipulation_system, planning_system):
        self.manipulation = manipulation_system
        self.planning = planning_system

    def handle_manipulation_request(self, task_description, object_location, target_location=None):
        """
        Handle high-level manipulation request from planning system
        """
        print(f"Handling manipulation request: {task_description}")

        # Get object information from perception system
        object_info = self.get_object_info(object_location)

        if not object_info:
            print("Object not found at specified location")
            return False

        # Plan manipulation task
        try:
            task_plan = self.manipulation.plan_manipulation_task(
                task_description,
                object_info,
                target_location
            )

            # Execute manipulation task
            self.manipulation.execute_manipulation_task(task_plan)

            print("Manipulation task completed successfully")
            return True

        except Exception as e:
            print(f"Manipulation task failed: {e}")
            return False

    def get_object_info(self, location):
        """
        Get detailed object information from perception system
        """
        # This would interface with the object detection system
        # For simulation, return a mock object
        return {
            'class_name': 'unknown_object',
            'center': [location[0], location[1]],
            'bbox': [location[0]-10, location[1]-10, 20, 20],
            'confidence': 0.9
        }

class FullSystemIntegrator:
    def __init__(self):
        # Initialize all system components
        self.dh_params = [
            [0.1, 0, 0.1, 0],      # Joint 1
            [0, np.pi/2, 0, 0],    # Joint 2
            [0.2, 0, 0, 0],        # Joint 3
            [0, np.pi/2, 0.1, 0],  # Joint 4
            [0, -np.pi/2, 0, 0],   # Joint 5
            [0, 0, 0.1, 0]         # Joint 6
        ]

        self.kinematics = RobotKinematics(self.dh_params)
        self.arm_controller = ArmController(self.kinematics)
        self.grasp_planner = GraspPlanner(self.arm_controller, {'max_width': 0.1})
        self.trajectory_generator = TrajectoryGenerator()
        self.manipulation_planner = ManipulationPlanner(
            self.arm_controller,
            self.grasp_planner,
            self.trajectory_generator
        )
        self.force_controller = ForceController(self.arm_controller)
        self.safety_monitor = ManipulationSafetyMonitor(self.arm_controller)

    def execute_voice_command(self, command, object_location, target_location=None):
        """
        Execute a voice command through the full system pipeline:
        Voice → Planning → Navigation → Object Detection → Manipulation
        """
        print(f"Executing command: '{command}'")

        # 1. Voice processing would happen at higher level
        # 2. Planning would create task plan
        # 3. Navigation would move robot to object
        # 4. Object detection would identify specific object

        # 5. Manipulation execution
        success = self.manipulation_planner.execute_manipulation_request(
            command,
            object_location,
            target_location
        )

        return success

    def demo_manipulation_sequence(self):
        """
        Demonstrate a complete manipulation sequence
        """
        print("Starting manipulation demonstration...")

        # Example object to pick up
        object_location = [0.5, 0.0, 0.1]  # x, y, z in meters
        target_location = [0.3, 0.4, 0.1]  # Place location

        # Execute pick and place task
        success = self.execute_voice_command(
            "Pick up the object and place it at target location",
            object_location,
            target_location
        )

        if success:
            print("Manipulation demonstration completed successfully!")
        else:
            print("Manipulation demonstration failed.")

    def run_safety_check(self):
        """
        Run comprehensive safety check
        """
        print("Running safety check...")

        is_safe = self.safety_monitor.check_safety_conditions()

        if is_safe:
            print("All safety checks passed.")
            return True
        else:
            print("Safety violations detected:")
            for violation in self.safety_monitor.safety_violations:
                print(f"  - {violation}")
            return False

    def shutdown(self):
        """
        Shutdown all manipulation components safely
        """
        print("Shutting down manipulation system...")

        # Stop all robot motion
        self.arm_controller.set_joint_positions([0, 0, 0, 0, 0, 0])

        # Open gripper if it was closed
        self.grasp_planner.open_gripper()

        print("Manipulation system shutdown complete.")
```

## Performance and Optimization

### Real-time Manipulation:
```python
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

class RealTimeManipulationSystem:
    def __init__(self):
        self.integrator = FullSystemIntegrator()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.command_queue = asyncio.Queue()
        self.is_running = True

    async def process_manipulation_commands(self):
        """
        Process manipulation commands in real-time
        """
        while self.is_running:
            try:
                # Get command with timeout
                command = await asyncio.wait_for(self.command_queue.get(), timeout=0.1)

                # Process command in thread pool
                future = self.executor.submit(
                    self.integrator.execute_voice_command,
                    command['task'],
                    command['object_loc'],
                    command.get('target_loc')
                )

                # Check safety in parallel
                safety_ok = self.integrator.run_safety_check()

                if not safety_ok:
                    print("Safety check failed, stopping manipulation")
                    break

                # Wait for command to complete (with timeout)
                success = future.result(timeout=30.0)  # 30 second timeout

                print(f"Command completed with success: {success}")

            except asyncio.TimeoutError:
                continue  # No commands in queue, continue loop
            except Exception as e:
                print(f"Error processing manipulation command: {e}")
                continue

    def add_command(self, task_description, object_location, target_location=None):
        """
        Add a manipulation command to the queue
        """
        command = {
            'task': task_description,
            'object_loc': object_location,
            'target_loc': target_location
        }
        asyncio.run(self.command_queue.put(command))

    def start_system(self):
        """
        Start the real-time manipulation system
        """
        # Run command processing in background
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.process_manipulation_commands())

    def stop_system(self):
        """
        Stop the real-time manipulation system
        """
        self.is_running = False
        self.executor.shutdown(wait=True)
        self.integrator.shutdown()
```

The manipulation integration component enables the Physical AI system to perform physical interactions with objects in its environment. It translates high-level goals into precise motor actions, managing everything from grasp planning to force control and safety monitoring. This component completes the full pipeline from voice command to physical action.