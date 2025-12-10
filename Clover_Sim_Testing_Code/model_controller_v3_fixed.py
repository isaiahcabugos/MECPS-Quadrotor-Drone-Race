#!/usr/bin/env python3
"""
Model-based velocity controller for Clover drone.
Runs RL model in a closed-loop to navigate to a goal position.

CRITICAL FIX APPLIED (v3_fixed):
- Y-axis coordinate frame mismatch corrected
- Training simulator used Y=RIGHT convention
- ROS/PX4 uses Y=LEFT convention
- Solution: Flip Y-axis sign in action-to-velocity conversion (line ~418)
- This ensures policy commands match actual drone behavior
"""
import os
import rospy
import numpy as np
import math
from geometry_msgs.msg import PoseStamped
from clover import srv
from std_srvs.srv import Trigger
import sys

# TensorFlow and TF-Agents imports
try:
    import tensorflow as tf
    import tensorflow_probability as tfp
    # Suppress TF warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
    HAS_TF = True
    rospy.loginfo("TensorFlow available: v%s", tf.__version__)
except ImportError:
    HAS_TF = False
    rospy.logwarn("TensorFlow not available - model loading will fail")


class TFAgentsPolicyWrapper:
    """Wrapper for TF-Agents saved policy that mimics SB3 interface"""
    
    def __init__(self, policy_path):
        """
        Load TF-Agents saved policy.
        
        Args:
            policy_path: Path to saved policy directory 
                        (e.g., 'checkpoints/policy/policy_step_5000000')
        """
        if not HAS_TF:
            raise RuntimeError("TensorFlow not available")
        
        rospy.loginfo(f"Loading TF-Agents policy from: {policy_path}")
        
        # Load the saved policy using TensorFlow SavedModel
        self.policy = tf.saved_model.load(policy_path)
        
        rospy.loginfo("TF-Agents policy loaded successfully!")
    
    def predict(self, observation, deterministic=True):
        """
        Predict action from observation (SB3-compatible interface).
        
        Args:
            observation: numpy array shape (30,)
            deterministic: if True, use deterministic action
            
        Returns:
            action: numpy array shape (3,)
            state: None (for compatibility with SB3 interface)
        """
        # Convert observation to TensorFlow format
        # TF-Agents expects batch dimension: (batch_size, obs_dim)
        obs_tensor = tf.constant([observation], dtype=tf.float32)
        
        # Create time_step (TF-Agents format)
        # time_step has: step_type, reward, discount, observation
        time_step = self._create_time_step(obs_tensor)
        
        # Get action from policy
        policy_step = self.policy.action(time_step)
        
        # Extract action - policy_step.action is a tensor
        action = policy_step.action.numpy()[0]  # Remove batch dim
        
        return action, None
    
    def _create_time_step(self, observation):
        """Create a TF-Agents TimeStep from observation."""
        import collections
        
        TimeStep = collections.namedtuple(
            'TimeStep', 
            ['step_type', 'reward', 'discount', 'observation']
        )
        
        # Fixed: Don't use .numpy() on symbolic tensor
        return TimeStep(
            step_type=tf.constant([1], dtype=tf.int32),      # MID=1, batch_size=1
            reward=tf.constant([0.0], dtype=tf.float32),
            discount=tf.constant([1.0], dtype=tf.float32),
            observation=observation
        )

class ModelController:
    def __init__(self, model_path=None, control_rate=10.0, \
                 num_gates=1, goal_tolerance=0.3):
        """
        Initialize model-based controller.
        
        Args:
            goal_position: [x, y, z] target position in map frame
            model_path: Path to saved TF-Agents policy directory
            control_rate: Hz, how often to run model and update velocity command
            goal_tolerance: meters, distance to goal considered "reached"
        """
        rospy.init_node('model_controller')

        self.num_gates = num_gates
        self.goal = np.array([0,0,0], dtype=np.float32)
        self.goal_tolerance = goal_tolerance
        self.gate_size = 1.0
        self.current_gate_idx = 0
        self.gates = self._create_racing_track()
        self.goal = self.gates[self.current_gate_idx]
        self.gates_passed = 0
    
        self.control_rate = control_rate
        self.control_period = 1.0 / control_rate
        
        # Service proxies
        self.get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
        self.set_velocity = rospy.ServiceProxy('set_velocity', srv.SetVelocity)
        self.navigate = rospy.ServiceProxy('navigate', srv.Navigate)
        self.land = rospy.ServiceProxy('land', Trigger)
        
        # State tracking
        self.current_pos = np.zeros(3)
        self.current_vel = np.zeros(3)
        self.current_yaw = 0.0
        self.current_pitch = 0.0
        self.current_roll = 0.0
        self.prev_action = np.zeros(3)
        
        # Load your trained model
        # Load TF-Agents policy
        self.model = None
        if model_path:
            try:
                self.model = TFAgentsPolicyWrapper(model_path)
                rospy.loginfo("Model loaded successfully!")
            except Exception as e:
                rospy.logerr(f"Failed to load model: {e}")
                rospy.logwarn("Falling back to placeholder controller")
        else:
            rospy.logwarn("No model_path provided, using placeholder controller")
        
        rospy.loginfo(f"Model Controller initialized")
        rospy.loginfo(f"Current Goal: {self.goal}, Tolerance: {self.goal_tolerance}m")
        rospy.loginfo(f"Control rate: {control_rate} Hz")

    def _create_racing_track(self, random=False):
        """Create a 7-gate racing circuit similar to Swift's track (75m lap)"""
        if not random:
            gates = [
                np.array([5.0,  0.0,    2.0],    dtype=np.float32),    # Gate 1: Start/Finish
                np.array([10.0, -5.0,   2.5],  dtype=np.float32),  # Gate 2: Right turn
                np.array([8.0,  -8.0,   3.0],   dtype=np.float32),   # Gate 3: Descending turn
                np.array([0.0,  -9.0,  2.0],  dtype=np.float32),  # Gate 4: Split-S entry (modified to fit in aruco map)
                np.array([-5.0, -5.0,   1.5],  dtype=np.float32),  # Gate 5: Split-S exit
                np.array([-8.0, 2.0,    2.5],   dtype=np.float32),   # Gate 6: Left turn
                np.array([0.0,  5.0,    2.0],    dtype=np.float32),    # Gate 7: Final approach
            ]

        else:
            gates = []
        return gates
    
    def get_observation_with_debug(self, frame_id='aruco_map'):
        try:
            telem = self.get_telemetry(frame_id=frame_id)
            self.current_pos = np.array([telem.x, telem.y, telem.z], dtype=np.float32)
            self.current_vel = np.array([telem.vx, telem.vy, telem.vz], dtype=np.float32)
            
            R = self._euler_to_rotation_matrix(
                telem.roll, 
                telem.pitch, 
                telem.yaw
            )
            
        except rospy.ServiceException as e:
            rospy.logerr(f"Telemetry service call failed: {e}")
            return None
        
        # CRITICAL FIX: Use _get_gate_corners instead of _create_virtual_gate
        gate_corners_world = self._get_gate_corners(self.current_gate_idx)
        
        # Transform to body frame (like training)
        gate_corners_body = []
        for corner in gate_corners_world:
            rel_world = corner - self.current_pos
            rel_body = R.T @ rel_world
            gate_corners_body.extend(rel_body)
        
        obs = np.concatenate([
            self.current_pos,
            self.current_vel,
            R.flatten(),
            np.array(gate_corners_body, dtype=np.float32),  # Body frame corners
            self.prev_action
        ], dtype=np.float32)
        
        # # === DETAILED DEBUG LOGGING ===
        # rospy.loginfo("=" * 60)
        # rospy.loginfo("OBSERVATION DEBUG")
        # rospy.loginfo("=" * 60)
        
        # # 1. Raw telemetry
        # rospy.loginfo(f"Position (world): [{self.current_pos[0]:.3f}, {self.current_pos[1]:.3f}, {self.current_pos[2]:.3f}]")
        # rospy.loginfo(f"Velocity (world): [{self.current_vel[0]:.3f}, {self.current_vel[1]:.3f}, {self.current_vel[2]:.3f}]")
        # rospy.loginfo(f"Euler (r,p,y): [{telem.roll:.3f}, {telem.pitch:.3f}, {telem.yaw:.3f}]")
        
        # # 2. Rotation matrix diagnostics
        # rospy.loginfo(f"R[2,2] (upright check): {R[2,2]:.3f} (should be ~1.0 when level)")
        # rospy.loginfo(f"R determinant: {np.linalg.det(R):.3f} (should be 1.0)")
        
        # # 3. Gate info
        gate_center = self.gates[self.current_gate_idx]
        rospy.loginfo(f"Gate {self.current_gate_idx} center: [{gate_center[0]:.2f}, {gate_center[1]:.2f}, {gate_center[2]:.2f}]")
        rospy.loginfo(f"Distance to gate: {np.linalg.norm(gate_center - self.current_pos):.2f}m")
        
        # # 4. Gate corners in world frame
        # rospy.loginfo("Gate corners (world):")
        # for i, corner in enumerate(gate_corners_world):
        #     rospy.loginfo(f"  Corner {i}: [{corner[0]:.2f}, {corner[1]:.2f}, {corner[2]:.2f}]")
        
        # # 5. Gate corners in body frame (CRITICAL - what policy sees)
        # rospy.loginfo("Gate corners (body frame - POLICY INPUT):")
        # for i in range(4):
        #     idx = i * 3
        #     rospy.loginfo(f"  Corner {i}: [{gate_corners_body[idx]:.2f}, {gate_corners_body[idx+1]:.2f}, {gate_corners_body[idx+2]:.2f}]")
        
        # # 6. Observation vector breakdown
        # rospy.loginfo(f"Observation shape: {obs.shape}")
        # rospy.loginfo(f"Obs[0:3] position: {obs[0:3]}")
        # rospy.loginfo(f"Obs[3:6] velocity: {obs[3:6]}")
        # rospy.loginfo(f"Obs[6:15] R flat: {obs[6:15]}")
        # rospy.loginfo(f"Obs[15:27] corners: {obs[15:27]}")
        # rospy.loginfo(f"Obs[27:30] prev_act: {obs[27:30]}")
        
        # # 7. Sanity checks
        # corner_distances = [np.linalg.norm(gate_corners_body[i*3:(i+1)*3]) for i in range(4)]
        # rospy.loginfo(f"Body-frame corner distances: {[f'{d:.2f}' for d in corner_distances]}")
        
        # if max(corner_distances) > 50:
        #     rospy.logwarn("WARNING: Gate corners very far! Check frame transformation.")
        
        # if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
        #     rospy.logerr("ERROR: NaN or Inf in observation!")
        
        # rospy.loginfo("=" * 60)

        assert obs.shape == (30,), f"Observation shape mismatch: {obs.shape}"
        return obs

    def _get_gate_corners(self, gate_idx):
        """
        Get 4 corners of a gate in world frame - matches training simulator.
        """
        gate_center = self.gates[gate_idx]
        half_size = self.gate_size / 2.0
        
        # Direction to next gate
        next_idx = (gate_idx + 1) % self.num_gates
        to_next = self.gates[next_idx] - gate_center
        gate_normal = to_next / (np.linalg.norm(to_next) + 1e-6)
        
        # Build orthogonal frame
        world_up = np.array([0.0, 0.0, 1.0])
        gate_right = np.cross(gate_normal, world_up)
        gate_right_norm = np.linalg.norm(gate_right)
        
        if gate_right_norm < 1e-6:
            gate_right = np.array([1.0, 0.0, 0.0])
        else:
            gate_right = gate_right / gate_right_norm
        
        gate_up = np.cross(gate_right, gate_normal)
        gate_up = gate_up / (np.linalg.norm(gate_up) + 1e-6)
        
        # 4 corners (same order as simulator)
        corners = np.array([
            gate_center + half_size * gate_up + half_size * gate_right,
            gate_center + half_size * gate_up - half_size * gate_right,
            gate_center - half_size * gate_up - half_size * gate_right,
            gate_center - half_size * gate_up + half_size * gate_right,
        ])
        
        return corners
    
    def _euler_to_rotation_matrix(self, roll, pitch, yaw):
        """Convert Euler angles to 3×3 rotation matrix."""
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr           ]
        ], dtype=np.float32)
        return R
    
    def _verify_gate_corners(self):
        """Verify gate corners form a proper square"""
        corners = self._get_gate_corners(self.current_gate_idx)
        
        # Check all corners are unique
        for i in range(4):
            for j in range(i+1, 4):
                dist = np.linalg.norm(corners[i] - corners[j])
                if dist < 0.01:
                    rospy.logerr(f"DEGENERATE: Corners {i} and {j} are identical!")
                    return False
        
        # Check corner distances (should be ~gate_size apart)
        edge_01 = np.linalg.norm(corners[0] - corners[1])
        edge_12 = np.linalg.norm(corners[1] - corners[2])
        edge_23 = np.linalg.norm(corners[2] - corners[3])
        edge_30 = np.linalg.norm(corners[3] - corners[0])
        
        rospy.loginfo(f"Gate edge lengths: {edge_01:.3f}, {edge_12:.3f}, {edge_23:.3f}, {edge_30:.3f}")
        rospy.loginfo(f"Expected: {self.gate_size:.3f}")
        
        # All edges should be approximately gate_size
        if not all(abs(e - self.gate_size) < 0.1 for e in [edge_01, edge_12, edge_23, edge_30]):
            rospy.logerr("Gate corners don't form a proper square!")
            return False
        
        rospy.loginfo("✓ Gate corners verified")
        return True

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
        to_drone = self.current_pos - gate_center

        # Previous position
        prev_position = self.current_pos - self.current_vel * self.control_period
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
        crossing_point = self.current_pos - dist_current * gate_normal

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
    
    def distance_to_goal(self):
        """Calculate Euclidean distance to goal."""
        return np.linalg.norm(self.current_pos - self.goal)
    
    def run_model(self, obs):
        if self.model is None:
            error = self.goal - self.current_pos
            gain = 0.5
            action = np.clip(error * gain, -1, 1)
        else:
            action, _ = self.model.predict(obs, deterministic=True)
        
        # === ACTION DEBUG LOGGING ===
        rospy.loginfo("ACTION DEBUG")
        # rospy.loginfo(f"Raw action ([-1,1]): [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}]")
        # rospy.loginfo(f"Action magnitude: {np.linalg.norm(action):.3f}")
        
        # Check for saturated actions (indicates policy confusion)
        if np.any(np.abs(action) > 0.95):
            rospy.logwarn(f"WARNING: Saturated action detected!")
        
        # CRITICAL FIX: Flip Y-axis sign to match ROS/PX4 convention
        # Training simulator used Y=RIGHT, but ROS uses Y=LEFT
        velocity_cmd = action * np.array([5.0, -5.0, 2.0])  # Note: -5.0 for Y!
        rospy.loginfo(f"Velocity cmd (m/s): [{velocity_cmd[0]:.2f}, {velocity_cmd[1]:.2f}, {velocity_cmd[2]:.2f}]")
        
        return velocity_cmd
    
    def send_velocity_command(self, velocity):
        """Send velocity command to drone."""
        try:
            vx, vy, vz = velocity
            response = self.set_velocity(
                vx=float(vx),
                vy=float(vy),
                vz=float(vz),
                frame_id='aruco_map',  # World frame (matches training)
                yaw=float('nan')  # Don't control yaw
            )
            
            if not response.success:
                rospy.logwarn(f"set_velocity failed: {response.message}")
                return False
            return True
            
        except rospy.ServiceException as e:
            rospy.logerr(f"set_velocity service call failed: {e}")
            return False
        
    def navigate_wait(self, x=0, y=0, z=0, yaw=float('nan'), speed=0.5, frame_id='map', tolerance=0.2, auto_arm=False):
        res = self.navigate(x=x, y=y, z=z, yaw=yaw, speed=speed, frame_id=frame_id, auto_arm=auto_arm)

        if not res.success:
            return res

        while not rospy.is_shutdown():
            telem = self.get_telemetry(frame_id='navigate_target')
            if math.sqrt(telem.x ** 2 + telem.y ** 2 + telem.z ** 2) < tolerance:
                return res
            rospy.sleep(0.2)
    
    def execute(self, timeout=30.0):
        """
        Main control loop: run model until goal is reached or timeout.
        
        Args:
            timeout: maximum time in seconds
            
        Returns:
            success: True if goal reached, False if timeout/error
        """
        rospy.loginfo("Starting model control loop...")
        rate = rospy.Rate(self.control_rate)
        start_time = rospy.Time.now()
        
        # IMPORTANT: Initial position hold to ensure OFFBOARD mode is active
        rospy.sleep(0.5)  # Let telemetry stabilize
        self.send_velocity_command([0, 0, 0])  # Start with hover
        rospy.sleep(0.5)
        
        consecutive_at_goal = 0
        required_consecutive = int(0.5 * self.control_rate)  # Stay at goal for 0.5s
        prev_cmd = [0, 0, 0]
        while not rospy.is_shutdown():
            loop_start = rospy.Time.now()
            
            # Check timeout
            elapsed = (rospy.Time.now() - start_time).to_sec()
            if elapsed > timeout:
                rospy.logwarn(f"Timeout reached ({timeout}s)")
                # Stop motion before returning
                self.send_velocity_command([0,0,0])
                rospy.sleep(0.5)
                return False
            
            # Get observation
            obs = self.get_observation_with_debug(frame_id='aruco_map')
            if obs is None:
                rospy.logwarn("Failed to get observation, retrying...")
                # CRITICAL: Still send zero velocity to maintain OFFBOARD
                self.send_velocity_command([0, 0, 0])
                rate.sleep()
                continue
            
            # Check if goal reached
            dist = self.distance_to_goal()
            rospy.loginfo_throttle(1.0, 
                f"Distance to goal: {dist:.2f}m | Velocity: "
                f"[{self.current_vel[0]:.1f}, {self.current_vel[1]:.1f}, "
                f"{self.current_vel[2]:.1f}] m/s"
                f"\n Current Gate {self.current_gate_idx}: Location: {self.goal}")
            
            if self._passed_through_gate():
                self.gates_passed += 1
                self.current_gate_idx = (self.current_gate_idx + 1) % self.num_gates
                self.goal = self.gates[self.current_gate_idx]  # Update goal to next gate
                rospy.loginfo(f"Gate {self.gates_passed} passed! Moving to gate {self.current_gate_idx} at {self.goal}")
            
            # Always compute velocity command (whether gate was passed or not)
            velocity_cmd = self.run_model(obs)

            # if dist < self.goal_tolerance:
            #     consecutive_at_goal += 1
            #     if consecutive_at_goal >= required_consecutive:
            #         rospy.loginfo(f"Goal reached and held! (distance: {dist:.3f}m)")
            #         # Stop motion
            #         self.send_velocity_command([0, 0, 0])
            #         rospy.sleep(0.5)
            #         return True
            #     # Continue sending small corrections even near goal
            #     velocity_cmd = self.run_model(obs) * 0.3  # Reduce velocity near goal
            # else:
            #     consecutive_at_goal = 0
            #     # Run model to get velocity command
            #     velocity_cmd = self.run_model(obs)
            
            self.prev_action = velocity_cmd / np.array([5.0, 5.0, 2.0])  # Store normalized
            rospy.loginfo(f" Velocity_cmd: {velocity_cmd}")
            # ALWAYS send velocity command to maintain OFFBOARD mode
            success = self.send_velocity_command(velocity_cmd)
            prev_cmd = velocity_cmd
            if not success:
                rospy.logwarn("Failed to send velocity command")
                # Try to hold position if command fails
                try:
                    self.set_velocity(vx=0, vy=0, vz=0, frame_id='map', yaw=float('nan'))
                except:
                    pass
            
            # Log velocity command
            rospy.logdebug(f"Velocity command: [{velocity_cmd[0]:.2f}, "
                          f"{velocity_cmd[1]:.2f}, {velocity_cmd[2]:.2f}] m/s")
            
            # Measure actual loop time
            loop_time = (rospy.Time.now() - loop_start).to_sec()
            if loop_time > self.control_period * 1.5:
                rospy.logwarn(f"Control loop running slow: {loop_time*1000:.1f}ms "
                             f"(target: {self.control_period*1000:.1f}ms)")
            
            rate.sleep()
        
        return False
    
    def land_and_shutdown(self):
        """Land the drone and shut down."""
        rospy.loginfo("Landing...")
        try:
            response = self.land()
            if response.success:
                rospy.loginfo("Landed successfully")
            else:
                rospy.logwarn(f"Landing failed: {response.message}")
        except rospy.ServiceException as e:
            rospy.logerr(f"Land service call failed: {e}")


def main():
    # Parse command line arguments
    if len(sys.argv) < 4:
        print("Usage: python3 model_controller.py <x> <y> <z> [model_path]")
        print("Example: python3 model_controller.py 3.0 2.0 1.5")
        print("   or:   python3 clover model_controller.py 3.0 2.0 1.5 /path/to/policy")
    
    goal_x = float(sys.argv[1])
    goal_y = float(sys.argv[2])
    goal_z = float(sys.argv[3])
    
    # Model path can come from command line or ROS parameter
    if len(sys.argv) > 4:
        model_path = sys.argv[4]
    else:
        model_path = None
    

    # Create controller
    controller = ModelController(
        model_path=model_path,
        control_rate=50.0,      # 50 Hz - good balance        
        goal_tolerance=0.3 ,     # 30cm tolerance
        num_gates=7,
    )
    
    try:
        # Run the control loop
        controller.navigate_wait(z=2, frame_id='body', auto_arm=True)
        gate_6_pos = controller.gates[6]  # [0, 5, 2]
        controller.navigate_wait(x=gate_6_pos[0], y=gate_6_pos[1], z=gate_6_pos[2], 
                                frame_id='aruco_map')
        controller._verify_gate_corners()
        success = controller.execute(timeout=30.0)
        
        if success:
            rospy.loginfo("Mission completed successfully!")
        else:
            rospy.logwarn("Mission failed or timed out")
    
    finally:
        # Always land at the end
        controller.land_and_shutdown()


if __name__ == '__main__':
    main()