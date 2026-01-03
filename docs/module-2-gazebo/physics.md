---
sidebar_label: 'Physics Simulation'
---

# Physics Simulation

This section covers the physics simulation capabilities in Gazebo and Unity, including how to configure and optimize physics properties for realistic robotic simulation.

## Physics Engines Overview

Physics engines are critical components of simulation environments that compute the motion and interaction of objects according to physical laws. Both Gazebo and Unity provide sophisticated physics simulation capabilities.

### Gazebo Physics Engines:
- **ODE (Open Dynamics Engine)**: Default engine, good for general-purpose simulation
- **Bullet**: Robust engine with good performance for complex interactions
- **DART (Dynamic Animation and Robotics Toolkit)**: Advanced engine for articulated bodies

### Unity Physics Engines:
- **NVIDIA PhysX**: Default engine, widely used in game development
- **Unity Physics Package**: Part of the DOTS (Data-Oriented Technology Stack)

## Physics Configuration in Gazebo

### World Physics Settings:
Physics parameters are configured in the world file:

```xml
<physics name="default" type="ode">
  <!-- Time step controls simulation granularity -->
  <max_step_size>0.001</max_step_size>

  <!-- Real-time factor (simulation time / real time) -->
  <real_time_factor>1</real_time_factor>

  <!-- Update rate in Hz -->
  <real_time_update_rate>1000</real_time_update_rate>

  <!-- ODE-specific parameters -->
  <ode>
    <solver>
      <type>quick</type>
      <iters>10</iters>
      <sor>1.3</sor>
    </solver>
    <constraints>
      <cfm>0.0</cfm>
      <erp>0.2</erp>
      <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

### Key Physics Parameters:

#### Time Step (`max_step_size`)
- Smaller values provide more accurate simulation but require more computation
- Typical values: 0.001s to 0.01s
- Must be small enough to capture fastest dynamics in the system

#### Real-time Factor
- Ratio of simulation time to real time
- Value of 1.0 means simulation runs at real-time speed
- Values > 1.0 mean simulation runs faster than real time
- Values < 1.0 mean simulation runs slower than real time

#### Solver Iterations
- More iterations provide more accurate solutions but slower performance
- Start with 10-50 iterations and adjust based on stability

## Collision Detection and Response

### Collision Properties:
Each link in a robot model can have collision properties that define how it interacts with other objects:

```xml
<link name="link_name">
  <collision name="collision">
    <geometry>
      <box><size>0.1 0.1 0.1</size></box>
    </geometry>
    <surface>
      <friction>
        <ode>
          <mu>1.0</mu>
          <mu2>1.0</mu2>
          <fdir1>0 0 0</fdir1>
          <slip1>0.0</slip1>
          <slip2>0.0</slip2>
        </ode>
      </friction>
      <bounce>
        <restitution_coefficient>0.1</restitution_coefficient>
        <threshold>100000</threshold>
      </bounce>
      <contact>
        <ode>
          <soft_cfm>0.0</soft_cfm>
          <soft_erp>0.2</soft_erp>
          <kp>1e+13</kp>
          <kd>1.0</kd>
          <max_vel>100.0</max_vel>
          <min_depth>0.0</min_depth>
        </ode>
      </contact>
    </surface>
  </collision>
</link>
```

### Friction Parameters:
- **Static friction (mu)**: Resistance to initial motion
- **Dynamic friction (mu2)**: Resistance to continued motion
- Higher values provide more realistic grip but may cause simulation instability

### Bounce Parameters:
- **Restitution coefficient**: How bouncy the surface is (0 = no bounce, 1 = perfectly elastic)
- **Threshold**: Velocity above which bounce is applied

## Physics Optimization Strategies

### Performance Optimization:
1. **Simplify collision geometry**: Use simpler shapes (boxes, spheres, capsules) instead of complex meshes
2. **Adjust time step**: Use the largest possible time step that maintains stability
3. **Limit solver iterations**: Balance accuracy with performance
4. **Reduce contact points**: Use fewer contact points for simpler collision detection

### Stability Optimization:
1. **Increase solver iterations**: For more complex interactions
2. **Adjust ERP and CFM**: Error reduction parameter (ERP) and constraint force mixing (CFM)
3. **Use appropriate mass ratios**: Avoid extreme differences in mass between objects
4. **Set proper damping**: Use joint and link damping to reduce oscillations

### Example Optimization Configuration:
```xml
<physics name="optimized" type="ode">
  <max_step_size>0.002</max_step_size>
  <real_time_factor>1.0</real_time_factor>
  <real_time_update_rate>500</real_time_update_rate>
  <ode>
    <solver>
      <type>quick</type>
      <iters>20</iters>
      <sor>1.3</sor>
    </solver>
    <constraints>
      <cfm>1e-5</cfm>
      <erp>0.2</erp>
    </constraints>
  </ode>
</physics>
```

## Unity Physics Configuration

### Unity Physics Settings:
In Unity, physics parameters are configured through the Physics Manager:

```csharp
// Accessing physics parameters programmatically
Physics.gravity = new Vector3(0, -9.81f, 0);
Physics.defaultSolverIterations = 6;
Physics.defaultSolverVelocityIterations = 1;
```

### Key Unity Physics Parameters:
- **Gravity**: Default gravitational acceleration (typically -9.81 m/s² in y direction)
- **Solver iterations**: Position solver iterations for contacts, joints, or composite constraints
- **Velocity iterations**: Velocity solver iterations for contacts, joints, or composite constraints
- **Bounce threshold**: Minimum impact speed to consider for bouncing
- **Sleep threshold**: Minimum energy below which objects are put to sleep

## Sensor Physics Simulation

### Camera Sensors:
Camera sensors in simulation must account for:
- Field of view and resolution
- Distortion parameters
- Frame rate and latency
- Noise models

### IMU Sensors:
IMU simulation includes:
- Accelerometer noise and bias
- Gyroscope drift and noise
- Sampling rate considerations
- Mounting position and orientation

### LIDAR Sensors:
LIDAR simulation considerations:
- Range and resolution
- Angular accuracy
- Multiple reflections
- Environmental conditions (dust, rain, etc.)

## Validation and Calibration

### Simulation-to-Reality Gap:
Physics simulation inevitably differs from real-world physics. To minimize this gap:

1. **Parameter identification**: Use system identification techniques to determine real robot parameters
2. **Validation experiments**: Compare simulation and real robot behavior under controlled conditions
3. **Iterative refinement**: Adjust physics parameters based on validation results

### Common Validation Tests:
- **Free fall**: Compare fall times and trajectories
- **Inclined plane**: Test friction coefficients
- **Pendulum motion**: Validate joint dynamics
- **Collision response**: Test impact behavior

## Best Practices

### For Robotic Simulation:
- Start with default physics parameters and adjust incrementally
- Use real robot parameters when available (mass, dimensions, etc.)
- Validate simulation behavior against real robot when possible
- Consider computational constraints when setting parameters
- Document all physics parameters used for reproducibility
- Test edge cases to ensure simulation stability

### Performance Guidelines:
- Use fixed time steps for deterministic behavior
- Balance accuracy and performance based on simulation goals
- Use appropriate collision geometries for your use case
- Consider using multiple physics worlds for different simulation requirements

Physics simulation is fundamental to creating realistic robotic simulations that can be used for development, testing, and training. Proper configuration and validation are essential for ensuring simulation accuracy.