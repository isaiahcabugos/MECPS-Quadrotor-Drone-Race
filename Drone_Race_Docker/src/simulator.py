# src/simulator.py
"""
Clover drone racing simulator with PX4-style cascaded control.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from typing import Tuple, Dict


@dataclass
class CloverSpecs:
    mass: float = 0.52
    arm_length: float = 0.092
    motor_thrust_max: float = 3.5
    motor_time_constant: float = 0.05
    Ixx: float = 0.0032
    Iyy: float = 0.0032
    Izz: float = 0.0055
    linear_drag: float = 0.3
    angular_drag: float = 0.02
    max_velocity_xy: float = 5.0
    max_velocity_z: float = 2.0
    max_yaw_rate: float = 1.0

    @property
    def total_max_thrust(self):
        return 4 * self.motor_thrust_max

    @property
    def hover_thrust(self):
        return self.mass * 9.81


@dataclass
class ControllerGains:
    vel_p_xy: float = 1.8
    vel_i_xy: float = 0.4
    vel_p_z: float = 4.0
    vel_i_z: float = 2.0
    att_p: float = 6.5
    max_tilt_angle: float = np.radians(45)
    max_rate_rp: float = np.radians(220)
    max_rate_yaw: float = np.radians(180)


class ImprovedCopterSimulator(gym.Env):
    """
    Clover racing simulator with PX4-style cascaded control.

    This matches the behavior of the Coex Clover's set_velocity service,
    which internally uses PX4's multicopter position controller.
    """
    def __init__(self, specs: CloverSpecs = None, gains: ControllerGains = None,
                 dt: float = 0.004,
                 domain_randomization: bool = False, add_noise: bool = False,
                 use_gimbal: bool = True):
        super().__init__()
        self.specs = specs or CloverSpecs()
        self.dt = dt
        self.control_dt = 0.04
        self.physics_steps_per_control = int(self.control_dt / self.dt)
        self.domain_randomization = domain_randomization
        self.add_noise = add_noise

        # Controller gains
        self.gains = ControllerGains()

        self.action_space = spaces.Box(
            low=-np.ones(3, dtype=np.float32),
            high=np.ones(3, dtype=np.float32)
        )

        # Swift observation: position(3) + velocity(3) + rotation_matrix(9) + gate_corners(12) + prev_action(3) = 30D
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(30,), dtype=np.float32
        )

        # Camera configuration for r_perc
        self.camera_fov = np.radians(60)  # Field of view
        self.camera_direction = np.array([1.0, 0.0, 0.0])  # Forward-facing in body frame

        self._init_state()

        # Racing track configuration (7 gates like Swift)
        self.num_gates = 7
        self.gate_size = 1.0  # 1.0m x 1.0m square gates (Swift standard)
        self.gates = self._create_racing_track()
        self.current_gate_idx = 0
        self.gates_passed = 0

        # Add tracking for reward components
        self._last_r_progress = 0.0
        self._last_r_perc = 0.0
        self._last_r_smooth = 0.0
        self._last_r_crash = 0.0

        # Episode statistics
        self.episode_gates_passed = 0
        self.episode_crashes = 0
        self.episode_max_altitude = 0.0

        self.use_gimbal = use_gimbal

        # Camera configuration
        self.camera_fov = np.radians(60)
        if use_gimbal:
            # Gimbal keeps camera level regardless of drone pitch/roll
            self.camera_direction = np.array([1.0, 0.0, 0.0])  # Still forward
        else:
            # Body-fixed camera (original behavior)
            self.camera_direction = np.array([1.0, 0.0, 0.0])

    def _create_racing_track(self):
        """Create a 7-gate racing circuit similar to Swift's track (75m lap)"""
        gates = [
            np.array([5.0, 0.0, 2.0]),    # Gate 1: Start/Finish
            np.array([10.0, -5.0, 2.5]),  # Gate 2: Right turn
            np.array([8.0, -8.0, 3.0]),   # Gate 3: Descending turn
            np.array([0.0, -10.0, 2.0]),  # Gate 4: Split-S entry
            np.array([-5.0, -5.0, 1.5]),  # Gate 5: Split-S exit
            np.array([-8.0, 2.0, 2.5]),   # Gate 6: Left turn
            np.array([0.0, 5.0, 2.0]),    # Gate 7: Final approach
        ]
        return gates

    def _init_state(self):
        """Initialize drone state"""
        self.position = np.zeros(3, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.orientation = np.eye(3, dtype=np.float32)
        self.angular_velocity = np.zeros(3, dtype=np.float32)

        # Velocity controller state
        self.velocity_integral_xy = np.zeros(2, dtype=np.float32)
        self.velocity_integral_z = 0.0

        # Motor state
        self.motor_thrusts = np.ones(4) * self.specs.hover_thrust / 4

        # Previous action (for observation)
        self.last_action = np.zeros(3, dtype=np.float32)
        self.previous_action = np.zeros(3, dtype=np.float32)
        self.steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_state()

        if self.domain_randomization:
            self.specs.mass *= np.random.uniform(0.9, 1.1)
            self.specs.linear_drag *= np.random.uniform(0.8, 1.2)

        # CRITICAL FIX: Spawn at gate X, target gate X+1
        spawn_gate_idx = np.random.randint(0, self.num_gates)
        self.current_gate_idx = (spawn_gate_idx + 1) % self.num_gates  # TARGET NEXT GATE!
        self.gates_passed = 0

        # Position near the SPAWN gate (not target gate)
        spawn_pos = self.gates[spawn_gate_idx]
        perturbation = np.random.uniform(-0.5, 0.5, size=3)
        perturbation[2] = np.random.uniform(-0.2, 0.2)
        self.position = spawn_pos + perturbation
        self.position[2] = max(0.1, self.position[2])

        # Small initial velocity TOWARD next gate
        to_next_gate = self.gates[self.current_gate_idx] - self.position
        to_next_gate_normalized = to_next_gate / (np.linalg.norm(to_next_gate) + 1e-6)
        self.velocity = to_next_gate_normalized * 0.5 + np.random.normal(0, 0.1, size=3)
        self.velocity = self.velocity.astype(np.float32)

        # Reset episode stats
        self.episode_gates_passed = 0
        self.episode_crashes = 0
        self.episode_max_altitude = self.position[2]

        return self._get_observation(), {}

    def _get_gate_corners(self, gate_idx: int) -> np.ndarray:
        """
        Get 4 corners of a gate in world frame.
        Swift uses square gates; corners are arranged: top-left, top-right, bottom-right, bottom-left.

        Returns: (4, 3) array of corner positions
        """
        gate_center = self.gates[gate_idx]
        half_size = self.gate_size / 2.0

        # Calculate gate normal (direction to next gate)
        next_idx = (gate_idx + 1) % self.num_gates
        to_next = self.gates[next_idx] - gate_center

        # Gate normal: perpendicular to gate plane, pointing along racing line
        gate_normal = to_next / (np.linalg.norm(to_next) + 1e-6)

        # Construct local frame for gate
        # Assume gates are upright (vertical edges aligned with Z axis)
        world_up = np.array([0.0, 0.0, 1.0])

        # Right vector: perpendicular to both normal and up
        gate_right = np.cross(gate_normal, world_up)
        gate_right_norm = np.linalg.norm(gate_right)

        # Handle edge case: gate normal parallel to world up
        if gate_right_norm < 1e-6:
            # Use arbitrary perpendicular vector
            gate_right = np.array([1.0, 0.0, 0.0])
        else:
            gate_right = gate_right / gate_right_norm

        # Up vector: perpendicular to normal and right (ensures orthogonal frame)
        gate_up = np.cross(gate_right, gate_normal)
        gate_up = gate_up / (np.linalg.norm(gate_up) + 1e-6)

        # 4 corners in world frame (counter-clockwise from top-right when viewed from behind)
        corners = np.array([
            gate_center + half_size * gate_up + half_size * gate_right,   # Top-right
            gate_center + half_size * gate_up - half_size * gate_right,   # Top-left
            gate_center - half_size * gate_up - half_size * gate_right,   # Bottom-left
            gate_center - half_size * gate_up + half_size * gate_right,   # Bottom-right
        ])

        return corners
    def _passed_through_gate(self) -> bool:
        """
        Check if drone passed through the current gate plane.
        Uses signed distance to detect plane crossing.
        """
        gate_center = self.gates[self.current_gate_idx]

        # Calculate gate normal (pointing toward next gate)
        next_idx = (self.current_gate_idx + 1) % self.num_gates
        to_next = self.gates[next_idx] - gate_center
        gate_normal = to_next / (np.linalg.norm(to_next) + 1e-6)

        # Current position relative to gate
        to_drone = self.position - gate_center

        # Previous position
        prev_position = self.position - self.velocity * self.control_dt
        to_drone_prev = prev_position - gate_center

        # Signed distances from gate plane (positive = in front of gate)
        dist_current = np.dot(to_drone, gate_normal)
        dist_prev = np.dot(to_drone_prev, gate_normal)

        # Check if crossed plane: prev behind (negative), current ahead (positive)
        crossed_plane = (dist_prev <= 0) and (dist_current > 0)

        if not crossed_plane:
            return False

        # Verify crossing happened within gate bounds
        # Project position onto gate plane
        # Point on plane = position - (distance along normal) * normal
        crossing_point = self.position - dist_current * gate_normal

        # Vector from gate center to crossing point (in gate plane)
        in_plane_offset = crossing_point - gate_center

        # Get gate basis vectors
        world_up = np.array([0.0, 0.0, 1.0])
        gate_right = np.cross(gate_normal, world_up)
        gate_right_norm = np.linalg.norm(gate_right)

        if gate_right_norm < 1e-6:
            gate_right = np.array([1.0, 0.0, 0.0])
        else:
            gate_right = gate_right / gate_right_norm

        gate_up = np.cross(gate_right, gate_normal)
        gate_up = gate_up / (np.linalg.norm(gate_up) + 1e-6)

        # Project offset onto gate's local axes
        offset_right = np.dot(in_plane_offset, gate_right)
        offset_up = np.dot(in_plane_offset, gate_up)

        # Check if within square bounds (gate is 1m × 1m)
        half_size = self.gate_size / 2.0
        within_bounds = (abs(offset_right) <= half_size and
                        abs(offset_up) <= half_size)

        return within_bounds

    def _get_camera_pointing_angle(self) -> float:
        """
        Calculate angle between camera optical axis and center of next gate.
        If gimbal is enabled, camera is pitch/roll stabilized.
        """
        if self.use_gimbal:
            # Gimbal: Camera only rotates with yaw (stays horizontal)
            yaw = self._rotation_to_euler(self.orientation)[2]
            camera_direction_world = np.array([
                np.cos(yaw),
                np.sin(yaw),
                0.0  # Always horizontal
            ])
        else:
            # Body-fixed: Camera rotates with drone
            camera_direction_world = self.orientation @ self.camera_direction

        # Vector from drone to gate center
        gate_center = self.gates[self.current_gate_idx]
        to_gate = gate_center - self.position

        to_gate_norm = np.linalg.norm(to_gate)
        if to_gate_norm < 1e-6:
            return 0.0

        to_gate_normalized = to_gate / to_gate_norm

        # Angle between camera and gate
        dot_product = np.dot(camera_direction_world, to_gate_normalized)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle = np.arccos(dot_product)

        return float(angle)

    def step(self, action: np.ndarray):
        """
        Execute one control step.

        Args:
            action: Velocity setpoint in body frame, range [-1, 1]
                   Maps to: vx ∈ [-5, 5] m/s, vy ∈ [-5, 5] m/s, vz ∈ [-2, 2] m/s
        """
        action = np.clip(action, -1, 1).astype(np.float32)

        if self.add_noise:
            action = action + np.random.normal(0, 0.05, size=3).astype(np.float32)
            action = np.clip(action, -1, 1)

        self.previous_action = action.copy()

        # FIX: Convert normalized action DIRECTLY to world-frame velocity setpoint
        # NO rotation needed - action already specifies world-frame velocity
        velocity_setpoint_world = np.array([
            action[0] * self.specs.max_velocity_xy,  # World X (forward)
            action[1] * self.specs.max_velocity_xy,  # World Y (right)
            action[2] * self.specs.max_velocity_z    # World Z (up)
        ])

        # Run physics simulation (10 physics steps per control step)
        for _ in range(self.physics_steps_per_control):
            self._physics_step(velocity_setpoint_world)

        # Check gate passage
        if self._passed_through_gate():
            self.gates_passed += 1
            self.current_gate_idx = (self.current_gate_idx + 1) % self.num_gates

        # Compute reward and check termination
        reward, terminated, truncated, info = self._compute_reward_and_termination()
        self.steps += 1
        obs = self._get_observation()

        return obs, float(reward), bool(terminated), bool(truncated), info

    def _physics_step(self, velocity_setpoint_world: np.ndarray):
        """
        CORRECTED: PX4-style cascaded control

        Key insight: When tilting for horizontal acceleration,
        must INCREASE thrust to maintain vertical component.
        """

        # =====================================================================
        # LAYER 1: VELOCITY CONTROLLER (unchanged)
        # =====================================================================
        velocity_error = velocity_setpoint_world - self.velocity

        velocity_error_xy = velocity_error[:2]
        velocity_error_z = velocity_error[2]

        self.velocity_integral_xy += velocity_error_xy * self.dt
        self.velocity_integral_z += velocity_error_z * self.dt

        integral_limit_xy = 2.0
        integral_limit_z = 1.0
        self.velocity_integral_xy = np.clip(self.velocity_integral_xy,
                                            -integral_limit_xy, integral_limit_xy)
        self.velocity_integral_z = np.clip(self.velocity_integral_z,
                                          -integral_limit_z, integral_limit_z)

        desired_accel_xy = (self.gains.vel_p_xy * velocity_error_xy +
                          self.gains.vel_i_xy * self.velocity_integral_xy)
        desired_accel_z = (self.gains.vel_p_z * velocity_error_z +
                          self.gains.vel_i_z * self.velocity_integral_z)

        desired_accel_world = np.array([
            desired_accel_xy[0],
            desired_accel_xy[1],
            desired_accel_z
        ])

        max_accel_xy = 5.0
        max_accel_z_up = 3.0
        max_accel_z_down = 2.0

        desired_accel_world[0] = np.clip(desired_accel_world[0], -max_accel_xy, max_accel_xy)
        desired_accel_world[1] = np.clip(desired_accel_world[1], -max_accel_xy, max_accel_xy)
        desired_accel_world[2] = np.clip(desired_accel_world[2], -max_accel_z_down, max_accel_z_up)

        # =====================================================================
        # LAYER 2: THRUST AND ATTITUDE CONTROL (FIXED!)
        # =====================================================================

        # CRITICAL FIX: Separate horizontal and vertical control
        # Horizontal acceleration → determines tilt angle
        # Vertical acceleration → determines thrust magnitude

        # Desired horizontal acceleration (in world frame)
        desired_accel_horizontal = desired_accel_world[:2]
        accel_horizontal_magnitude = np.linalg.norm(desired_accel_horizontal)

        # Desired tilt angle to achieve horizontal acceleration
        # From F_horizontal = m * a_horizontal = T * sin(tilt)
        # Therefore: sin(tilt) = m * a_horizontal / T
        # For small angles: tilt ≈ a_horizontal / g
        if accel_horizontal_magnitude > 0.1:
            # Use simplified tilt calculation (valid for small angles)
            desired_tilt_angle = np.arctan2(accel_horizontal_magnitude, 9.81)

            # Limit tilt angle
            desired_tilt_angle = np.clip(desired_tilt_angle, 0, self.gains.max_tilt_angle)

            # Direction of tilt (in XY plane)
            tilt_direction = desired_accel_horizontal / accel_horizontal_magnitude

            # Compute desired pitch and roll from tilt
            # Pitch = tilt in X direction (forward/back)
            # Roll = tilt in Y direction (left/right)
            desired_pitch = desired_tilt_angle * tilt_direction[0]  # Negative because pitch forward is negative
            desired_roll = desired_tilt_angle * tilt_direction[1]
        else:
            desired_pitch = 0.0
            desired_roll = 0.0
            desired_tilt_angle = 0.0

        # CRITICAL FIX: Thrust magnitude must compensate for tilt
        # T_vertical = T_total * cos(tilt)
        # We want: T_vertical = m * (g + a_z_desired)
        # Therefore: T_total = m * (g + a_z_desired) / cos(tilt)

        desired_vertical_accel = desired_accel_world[2]

        # Thrust needed for vertical acceleration (before tilt compensation)
        thrust_vertical = self.specs.mass * (9.81 + desired_vertical_accel)

        # Compensate for tilt (thrust must be larger to maintain vertical component)
        cos_tilt = np.cos(desired_tilt_angle)
        if cos_tilt > 0.01:  # Avoid division by zero
            thrust_magnitude = thrust_vertical / cos_tilt
        else:
            thrust_magnitude = self.specs.total_max_thrust  # Max thrust if near-horizontal

        # Limit thrust
        min_thrust = 0.3 * self.specs.total_max_thrust
        max_thrust = 0.9 * self.specs.total_max_thrust
        thrust_magnitude = np.clip(thrust_magnitude, min_thrust, max_thrust)

        # =====================================================================
        # LAYER 3: ATTITUDE CONTROLLER (unchanged)
        # =====================================================================

        # Smooth transition to desired attitude
        desired_yaw = 0.0  # Keep yaw at zero for now

        tau_attitude = 0.05 # Changed from 0.05
        alpha = self.dt / (tau_attitude + self.dt)

        current_rpy = self._rotation_to_euler(self.orientation)

        new_roll = (1 - alpha) * current_rpy[0] + alpha * desired_roll
        new_pitch = (1 - alpha) * current_rpy[1] + alpha * desired_pitch
        new_yaw = (1 - alpha) * current_rpy[2] + alpha * desired_yaw

        self.orientation = self._euler_to_rotation(new_roll, new_pitch, new_yaw)

        # Re-orthogonalize
        self.orientation[:, 2] = self.orientation[:, 2] / np.linalg.norm(self.orientation[:, 2])
        self.orientation[:, 0] = (self.orientation[:, 0] -
                                  np.dot(self.orientation[:, 0], self.orientation[:, 2]) *
                                  self.orientation[:, 2])
        self.orientation[:, 0] = self.orientation[:, 0] / np.linalg.norm(self.orientation[:, 0])
        self.orientation[:, 1] = np.cross(self.orientation[:, 2], self.orientation[:, 0])

        # =====================================================================
        # LAYER 4: PHYSICS SIMULATION (unchanged)
        # =====================================================================

        thrust_body = np.array([0, 0, thrust_magnitude])
        thrust_world_actual = self.orientation @ thrust_body

        drag_force = -self.specs.linear_drag * self.velocity * np.abs(self.velocity)

        total_accel = (thrust_world_actual + drag_force) / self.specs.mass
        total_accel[2] -= 9.81

        total_accel = np.clip(total_accel, -50.0, 50.0)

        self.velocity += total_accel * self.dt
        self.velocity = np.clip(self.velocity, -20.0, 20.0)

        self.position += self.velocity * self.dt

        # # Diagnostic output (optional - remove for production)
        # if self.steps % 10 == 0 and self.steps < 30:
        #     self._print_diagnostics(velocity_setpoint_world, desired_accel_world,
        #                            thrust_magnitude, total_accel)

    def _print_diagnostics(self, vel_setpoint, accel_cmd, thrust, accel_actual):
        """Print control diagnostics for debugging"""
        print(f"\n{'='*70}")
        print(f"CONTROL DEBUG - Step {self.steps}")
        print(f"{'='*70}")
        print(f"Velocity:")
        print(f"  Setpoint: {vel_setpoint}")
        print(f"  Current:  {self.velocity}")
        print(f"  Error:    {vel_setpoint - self.velocity}")
        print(f"\nAcceleration:")
        print(f"  Commanded: {accel_cmd}")
        print(f"  Actual:    {accel_actual}")
        print(f"\nOrientation:")
        euler = self._rotation_to_euler(self.orientation)
        print(f"  Roll:  {np.degrees(euler[0]):6.1f}°")
        print(f"  Pitch: {np.degrees(euler[1]):6.1f}°")
        print(f"  Yaw:   {np.degrees(euler[2]):6.1f}°")
        print(f"\nThrust: {thrust:.2f} N (hover: {self.specs.hover_thrust:.2f} N)")
        print(f"Altitude: {self.position[2]:.3f} m")
        print(f"Vertical accel: {accel_actual[2]:.3f} m/s² (should be ~0 for level flight)")
        print(f"{'='*70}")

    def _compute_reward_and_termination(self):
      """Fixed reward function with component tracking"""

      current_gate = self.gates[self.current_gate_idx]
      current_distance = float(np.linalg.norm(current_gate - self.position))

      prev_position = self.position - self.velocity * self.control_dt
      prev_distance = float(np.linalg.norm(current_gate - prev_position))

      # Track max altitude
      self.episode_max_altitude = max(self.episode_max_altitude, self.position[2])

      # r_progress: Reward getting closer (CLAMP to prevent extreme values)
      lambda1 = 1.0
      r_progress = lambda1 * (prev_distance - current_distance)

      # r_perc: Camera pointing at gate
      delta_cam = self._get_camera_pointing_angle()
      lambda2 = 0.02
      lambda3 = 10.0
      r_perc = lambda2 * np.exp(-lambda3 * (delta_cam ** 4))

      # r_smooth: Action smoothness
      action_change = self.previous_action - self.last_action
      lambda4 = 0.0001
      lambda5 = 0.0002
      r_smooth = -(lambda4 * np.linalg.norm(self.previous_action)**2 +
                  lambda5 * np.linalg.norm(action_change))

      # Crash penalty
      r_crash = 0.0
      terminated = False
      truncated = False

      info = {
          'distance_to_gate': current_distance,
          'current_gate': self.current_gate_idx,
          'gates_passed': self.gates_passed,
          'progress': prev_distance - current_distance,
          'camera_angle_deg': np.degrees(delta_cam),
          'r_progress': r_progress,
          'r_perc': r_perc,
          'r_smooth': r_smooth,
          'max_altitude': self.episode_max_altitude
      }

      # Termination conditions with tracking
      if self.steps >= 1500:  # Swift episode length
          truncated = True
          info['termination_reason'] = 'timeout'
      elif np.any(np.abs(self.position[:2]) > 15): # Changed from 15
          terminated = True
          info['termination_reason'] = 'out_of_bounds'
          r_crash = 5.0
          self.episode_crashes += 1
      elif self.position[2] < 0.05:
          terminated = True
          info['termination_reason'] = 'ground_crash'
          r_crash = 5.0
          self.episode_crashes += 1
      elif self.position[2] > 6:
          terminated = True
          info['termination_reason'] = 'ceiling'
          r_crash = 5.0
          self.episode_crashes += 1
      elif self.orientation[2, 2] < 0.5:
          terminated = True
          info['termination_reason'] = 'flipped'
          r_crash = 5.0
          self.episode_crashes += 1
      elif not np.all(np.isfinite(self.position)) or not np.all(np.isfinite(self.velocity)):
          terminated = True
          info['termination_reason'] = 'numerical_instability'
          r_crash = 5.0
          self.episode_crashes += 1

      # Store for logging
      self._last_r_progress = r_progress
      self._last_r_perc = r_perc
      self._last_r_smooth = r_smooth
      self._last_r_crash = r_crash


      total_reward = r_progress + r_perc + r_smooth - r_crash
      total_reward = np.clip(total_reward, -10.0, 10.0)  # Prevent extreme rewards

      return float(total_reward), terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """
        Swift-style observation (Page 7, Methods section):
        - Robot state: position (3) + velocity (3) + rotation matrix (9) = 15D
        - Next gate: 4 corners relative to drone (12D)
        - Previous action (3D)
        Total: 30D
        """

        # Get 4 corners of next gate in world frame
        gate_corners_world = self._get_gate_corners(self.current_gate_idx)

        # Transform corners to drone's body frame
        gate_corners_relative = []
        for corner in gate_corners_world:
            # Relative position in world frame
            rel_world = corner - self.position
            # Rotate to body frame
            rel_body = self.orientation.T @ rel_world
            gate_corners_relative.extend(rel_body)

        obs = np.concatenate([
            self.position,                  # 3D
            self.velocity,                  # 3D
            self.orientation.flatten(),     # 9D (rotation matrix)
            np.array(gate_corners_relative, dtype=np.float32),  # 12D (4 corners × 3D)
            self.previous_action            # 3D
        ]).astype(np.float32)

        if self.add_noise:
            # Add sensor noise (Swift paper mentions noisy observations)
            obs[:3] += np.random.normal(0, 0.01, size=3)    # Position noise
            obs[3:6] += np.random.normal(0, 0.02, size=3)   # Velocity noise
            obs[12:24] += np.random.normal(0, 0.05, size=12)  # Gate detection noise

        return obs

    @staticmethod
    def _euler_to_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """
        Convert Euler angles (roll, pitch, yaw) to rotation matrix.

        CRITICAL FIX: Pitch convention for quadrotors:
        - Positive pitch = nose UP (tilt backward)
        - Negative pitch = nose DOWN (tilt forward)

        When nose tilts forward (negative pitch), thrust points forward,
        accelerating drone in +X direction.

        Aircraft convention (NASA standard):
        - Roll: rotation about X-axis (right-hand rule)
        - Pitch: rotation about Y-axis (right-hand rule)
        - Yaw: rotation about Z-axis (right-hand rule)

        Rotation order: Yaw → Pitch → Roll (ZYX Euler angles)
        """
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        # ZYX Euler angle rotation matrix (NASA standard)
        # This is the CORRECT formula for aircraft/quadrotor orientation
        R = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp,     cp * sr,                cp * cr               ]
        ], dtype=np.float32)

        return R

    @staticmethod
    def _rotation_to_euler(R: np.ndarray) -> np.ndarray:
        pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
        if np.cos(pitch) > 1e-6:
            roll = np.arctan2(R[2, 1], R[2, 2])
            yaw = np.arctan2(R[1, 0], R[0, 0])
        else:
            roll = 0
            yaw = np.arctan2(-R[0, 1], R[1, 1])
        return np.array([roll, pitch, yaw])