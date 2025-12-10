#!/usr/bin/env python3
"""
Model-based velocity controller with dynamics residual collection.

RESIDUAL COLLECTION MODE:
- Logs ground truth state and commanded velocities
- Saves data for KNN residual model fitting
- Use --collect-residuals flag to enable

COORDINATE FRAME NOTE:
- Training used Y=RIGHT convention
- ROS/PX4 uses Y=LEFT convention  
- Y-axis flip applied at line ~450 (action to velocity conversion)
- Residuals collected in ROS frame (post-flip)
"""
import os
import rospy
import numpy as np
import math
import pickle
from datetime import datetime
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelStates
from clover import srv
from std_srvs.srv import Trigger
import sys

# TensorFlow and TF-Agents imports
try:
    import tensorflow as tf
    import tensorflow_probability as tfp
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
        if not HAS_TF:
            raise RuntimeError("TensorFlow not available")
        
        rospy.loginfo(f"Loading TF-Agents policy from: {policy_path}")
        self.policy = tf.saved_model.load(policy_path)
        rospy.loginfo("TF-Agents policy loaded successfully!")
    
    def predict(self, observation, deterministic=True):
        obs_tensor = tf.constant([observation], dtype=tf.float32)
        time_step = self._create_time_step(obs_tensor)
        policy_step = self.policy.action(time_step)
        action = policy_step.action.numpy()[0]
        return action, None
    
    def _create_time_step(self, observation):
        import collections
        TimeStep = collections.namedtuple(
            'TimeStep', 
            ['step_type', 'reward', 'discount', 'observation']
        )
        return TimeStep(
            step_type=tf.constant([1], dtype=tf.int32),
            reward=tf.constant([0.0], dtype=tf.float32),
            discount=tf.constant([1.0], dtype=tf.float32),
            observation=observation
        )


class ResidualDataCollector:
    """Collects ground truth and command data for dynamics residual identification"""
    
    def __init__(self):
        self.data = {
            'timestamp': [],
            'position': [],
            'velocity': [],
            'attitude_quat': [],
            'commanded_velocity': [],  # Post-flip velocities (ROS frame)
        }
        self.collecting = False
        rospy.loginfo("Residual data collector initialized")
    
    def log_sample(self, position, velocity, attitude_quat, commanded_velocity):
        """
        Log a single data sample.
        
        Args:
            position: [x, y, z] in world frame
            velocity: [vx, vy, vz] in world frame
            attitude_quat: [qx, qy, qz, qw]
            commanded_velocity: [vx, vy, vz] POST-FLIP (ROS frame)
        """
        if self.collecting:
            self.data['timestamp'].append(rospy.get_time())
            self.data['position'].append(position.copy())
            self.data['velocity'].append(velocity.copy())
            self.data['attitude_quat'].append(attitude_quat.copy())
            self.data['commanded_velocity'].append(commanded_velocity.copy())
    
    def start_collection(self):
        """Start collecting data"""
        self.collecting = True
        rospy.loginfo("✓ Residual data collection STARTED")
    
    def stop_collection(self):
        """Stop collecting data"""
        self.collecting = False
        rospy.loginfo("✓ Residual data collection STOPPED")
    
    def save_data(self, filename=None):
        """Save collected data to pickle file"""
        if not self.data['timestamp']:
            rospy.logwarn("No data collected!")
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"residual_data_{timestamp}.pkl"
        
        # Convert to numpy arrays
        processed_data = {
            'timestamp': np.array(self.data['timestamp']),
            'position': np.array(self.data['position']),
            'velocity': np.array(self.data['velocity']),
            'attitude_quat': np.array(self.data['attitude_quat']),
            'commanded_velocity': np.array(self.data['commanded_velocity']),
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(processed_data, f)
        
        n_samples = len(processed_data['timestamp'])
        duration = processed_data['timestamp'][-1] - processed_data['timestamp'][0]
        
        rospy.loginfo(f"✓ Residual data saved: {filename}")
        rospy.loginfo(f"  Samples: {n_samples}")
        rospy.loginfo(f"  Duration: {duration:.1f} seconds")
        rospy.loginfo(f"  Rate: {n_samples/duration:.1f} Hz")
        
        return filename


class ModelController:
    def __init__(self, model_path=None, control_rate=10.0, 
                 num_gates=1, goal_tolerance=0.3, collect_residuals=False):
        """
        Initialize model-based controller with optional residual collection.
        
        Args:
            model_path: Path to saved TF-Agents policy directory
            control_rate: Hz, control loop frequency
            num_gates: Number of gates in track
            goal_tolerance: Distance to goal considered "reached"
            collect_residuals: If True, enable residual data collection
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
        
        # Ground truth from Gazebo (for residual collection)
        self.gt_pos = None
        self.gt_vel = None
        self.gt_quat = None
        self.gt_ready = False
        self.drone_name = 'clover'  # Adjust if your drone has different name
        
        rospy.Subscriber('/gazebo/model_states', ModelStates, 
                        self._ground_truth_callback)
        
        # Residual collection
        self.collect_residuals = collect_residuals
        self.residual_collector = None
        if collect_residuals:
            self.residual_collector = ResidualDataCollector()
            rospy.loginfo("=" * 60)
            rospy.loginfo("RESIDUAL COLLECTION MODE ENABLED")
            rospy.loginfo("=" * 60)
        
        # Load model
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
        """Create a 7-gate racing circuit"""
        if not random:
            # gates = [
            #     np.array([5.0,  0.0,  2.0],  dtype=np.float32),
            #     np.array([10.0, -5.0, 2.5],  dtype=np.float32),
            #     np.array([8.0,  -8.0, 3.0],  dtype=np.float32),
            #     np.array([0.0,  -9.0, 2.0],  dtype=np.float32),
            #     np.array([-5.0, -5.0, 1.5],  dtype=np.float32),
            #     np.array([-8.0, 2.0,  2.5],  dtype=np.float32),
            #     np.array([0.0,  5.0,  2.0],  dtype=np.float32),
            # ]
            gates = [
        np.array([6.50, 5.50, 2.00]),   # Gate 1: Start/Finish
        np.array([9.00, 3.00, 2.50]),   # Gate 2: Right turn
        np.array([8.00, 1.50, 3.00]),   # Gate 3: Descending turn
        np.array([4.00, 1.00, 2.00]),   # Gate 4: Split-S entry
        np.array([1.50, 3.00, 1.50]),   # Gate 5: Split-S exit
        np.array([0.00, 6.50, 2.50]),   # Gate 6: Left turn
        np.array([4.00, 8.00, 2.00]),   # Gate 7: Final approach
        ]
        else:
            gates = []
        return gates
    
    def _euler_to_rotation_matrix(self, roll, pitch, yaw):
        """Convert Euler angles to rotation matrix"""
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr           ]
        ])
        return R
    
    def _get_gate_corners(self, gate_idx):
        """Get the 4 corners of a gate in world frame"""
        gate_center = self.gates[gate_idx]
        half_size = self.gate_size / 2.0
        
        corners = np.array([
            [gate_center[0] - half_size, gate_center[1] - half_size, gate_center[2]],
            [gate_center[0] + half_size, gate_center[1] - half_size, gate_center[2]],
            [gate_center[0] + half_size, gate_center[1] + half_size, gate_center[2]],
            [gate_center[0] - half_size, gate_center[1] + half_size, gate_center[2]],
        ], dtype=np.float32)
        
        return corners
    
    def _ground_truth_callback(self, msg):
        """Extract ground truth from Gazebo model_states"""
        try:
            idx = msg.name.index(self.drone_name)
            
            pose = msg.pose[idx]
            twist = msg.twist[idx]
            
            self.gt_pos = np.array([
                pose.position.x,
                pose.position.y,
                pose.position.z
            ])
            
            self.gt_vel = np.array([
                twist.linear.x,
                twist.linear.y,
                twist.linear.z
            ])
            
            self.gt_quat = np.array([
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w
            ])
            
            if not self.gt_ready:
                rospy.loginfo("✓ Ground truth initialized from Gazebo")
            self.gt_ready = True
            
        except (ValueError, IndexError):
            if not self.gt_ready:
                rospy.logwarn_throttle(5.0, 
                    f"Drone '{self.drone_name}' not found in /gazebo/model_states. "
                    f"Available: {msg.name}")

    
    def get_observation_with_debug(self, frame_id='aruco_map'):
        """
        Build observation vector for policy.
        
        MUST MATCH TRAINING FORMAT (30D):
        - Position (3D)
        - Velocity (3D)
        - Rotation matrix flattened (9D)
        - Gate corners in body frame (12D)
        - Previous action (3D)
        """
        try:
            telem = self.get_telemetry(frame_id=frame_id)
            self.current_pos = np.array([telem.x, telem.y, telem.z], dtype=np.float32)
            self.current_vel = np.array([telem.vx, telem.vy, telem.vz], dtype=np.float32)
            
            R = self._euler_to_rotation_matrix(telem.roll, telem.pitch, telem.yaw)
            
        except rospy.ServiceException as e:
            rospy.logerr(f"Telemetry service call failed: {e}")
            return None
        
        # Get gate corners in body frame
        gate_corners_world = self._get_gate_corners(self.current_gate_idx)
        gate_corners_body = []
        for corner in gate_corners_world:
            rel_world = corner - self.current_pos
            rel_body = R.T @ rel_world
            gate_corners_body.extend(rel_body)
        
        # CRITICAL: Must include rotation matrix (9D) to match training!
        obs = np.concatenate([
            self.current_pos,           # 3D
            self.current_vel,           # 3D
            R.flatten(),                # 9D - rotation matrix flattened
            np.array(gate_corners_body), # 12D - gate corners in body frame
            self.prev_action            # 3D
        ]).astype(np.float32)
        
        # Verify observation dimension
        assert obs.shape == (30,), f"Observation shape mismatch: {obs.shape} != (30,)"
        
        return obs
    
    def distance_to_goal(self):
        """Calculate Euclidean distance to current goal"""
        return np.linalg.norm(self.current_pos - self.goal)
    
    def _passed_through_gate(self):
        """Check if drone passed through current gate"""
        dist = self.distance_to_goal()
        return dist < self.goal_tolerance
    
    def run_model(self, observation):
        """
        Run the trained model to get velocity command.
        Returns velocity in m/s in world frame.
        """
        if self.model is None:
            # Placeholder: move toward goal
            direction = self.goal - self.current_pos
            dist = np.linalg.norm(direction)
            if dist > 0.01:
                direction = direction / dist
            velocity = direction * 2.0
            velocity[2] = max(min(velocity[2], 1.0), -1.0)
            return velocity
        
        # Get action from model
        action, _ = self.model.predict(observation, deterministic=True)
        
        # Check for saturated actions
        if np.any(np.abs(action) > 0.95):
            rospy.logwarn(f"WARNING: Saturated action detected!")
        
        # CRITICAL: Y-axis flip + scaling
        # Training used Y=RIGHT, ROS uses Y=LEFT
        velocity_cmd = action * np.array([5.0, -5.0, 2.0])
        
        rospy.loginfo(f"Velocity cmd (m/s): [{velocity_cmd[0]:.2f}, "
                     f"{velocity_cmd[1]:.2f}, {velocity_cmd[2]:.2f}]")
        
        return velocity_cmd
    
    def send_velocity_command(self, velocity):
        """Send velocity command to drone"""
        try:
            vx, vy, vz = velocity
            response = self.set_velocity(
                vx=float(vx),
                vy=float(vy),
                vz=float(vz),
                frame_id='aruco_map',
                yaw=float('nan')
            )
            
            if not response.success:
                rospy.logwarn(f"set_velocity failed: {response.message}")
                return False
            return True
            
        except rospy.ServiceException as e:
            rospy.logerr(f"set_velocity service call failed: {e}")
            return False
    
    def navigate_wait(self, x=0, y=0, z=0, yaw=float('nan'), speed=0.5, 
                     frame_id='map', tolerance=0.2, auto_arm=False):
        """Navigate to position and wait until reached"""
        res = self.navigate(x=x, y=y, z=z, yaw=yaw, speed=speed, 
                          frame_id=frame_id, auto_arm=auto_arm)

        if not res.success:
            return res

        while not rospy.is_shutdown():
            telem = self.get_telemetry(frame_id='navigate_target')
            if math.sqrt(telem.x ** 2 + telem.y ** 2 + telem.z ** 2) < tolerance:
                return res
            rospy.sleep(0.2)
    
    def execute(self, timeout=30.0):
        """
        Main control loop with optional residual collection.
        
        Args:
            timeout: maximum time in seconds
            
        Returns:
            success: True if goal reached, False if timeout/error
        """
        rospy.loginfo("Starting model control loop...")
        rate = rospy.Rate(self.control_rate)
        start_time = rospy.Time.now()
        
        # Initial position hold
        rospy.sleep(0.5)
        self.send_velocity_command([0, 0, 0])
        rospy.sleep(0.5)
        
        # Start residual collection if enabled
        if self.residual_collector:
            # Wait for ground truth to be ready
            if not self.gt_ready:
                rospy.logwarn("Waiting for ground truth from Gazebo...")
                wait_start = rospy.Time.now()
                while not self.gt_ready and not rospy.is_shutdown():
                    rospy.sleep(0.1)
                    if (rospy.Time.now() - wait_start).to_sec() > 5.0:
                        rospy.logerr("Ground truth not available after 5 seconds!")
                        rospy.logerr("Make sure /gazebo/model_states is publishing")
                        rospy.logerr(f"and drone name is '{self.drone_name}'")
                        return None
                rospy.loginfo("✓ Ground truth ready")
            
            self.residual_collector.start_collection()
        
        # Loop timing diagnostics
        loop_times = []
        max_loop_time = 0.0
        slow_loops = 0
        
        while not rospy.is_shutdown():
            loop_start = rospy.Time.now()
            
            # Check timeout
            elapsed = (rospy.Time.now() - start_time).to_sec()
            if elapsed > timeout:
                rospy.logwarn(f"Timeout reached ({timeout}s)")
                self.send_velocity_command([0,0,0])
                rospy.sleep(0.5)
                break
            
            # Get observation
            obs = self.get_observation_with_debug(frame_id='aruco_map')
            if obs is None:
                rospy.logwarn("Failed to get observation, retrying...")
                self.send_velocity_command([0, 0, 0])
                rate.sleep()
                continue
            
            # Check gate passage
            dist = self.distance_to_goal()
            
            # Throttled status logging
            if len(loop_times) % 50 == 0:  # Every 1 second at 50Hz
                avg_loop = np.mean(loop_times[-50:]) if loop_times else 0
                rospy.loginfo(
                    f"Distance: {dist:.2f}m | Vel: [{self.current_vel[0]:.1f}, "
                    f"{self.current_vel[1]:.1f}, {self.current_vel[2]:.1f}] m/s | "
                    f"Gate {self.current_gate_idx} | Loop: {avg_loop*1000:.1f}ms"
                )
            
            if self._passed_through_gate():
                self.gates_passed += 1
                self.current_gate_idx = (self.current_gate_idx + 1) % self.num_gates
                self.goal = self.gates[self.current_gate_idx]
                rospy.loginfo(f"✓ Gate {self.gates_passed} passed! "
                            f"Moving to gate {self.current_gate_idx}")
            
            # Get velocity command
            velocity_cmd = self.run_model(obs)
            
            # LOG DATA FOR RESIDUAL COLLECTION using Gazebo ground truth
            if self.residual_collector and self.gt_ready:
                self.residual_collector.log_sample(
                    position=self.gt_pos.copy(),
                    velocity=self.gt_vel.copy(),
                    attitude_quat=self.gt_quat.copy(),
                    commanded_velocity=velocity_cmd  # POST-flip velocity
                )
            
            # Send command
            self.prev_action = velocity_cmd / np.array([5.0, 5.0, 2.0])
            success = self.send_velocity_command(velocity_cmd)
            
            if not success:
                rospy.logwarn("Failed to send velocity command")
            
            # Measure loop timing
            loop_time = (rospy.Time.now() - loop_start).to_sec()
            loop_times.append(loop_time)
            max_loop_time = max(max_loop_time, loop_time)
            
            if loop_time > self.control_period * 1.5:
                slow_loops += 1
                if slow_loops < 5:  # Only warn first few times
                    rospy.logwarn(f"Control loop slow: {loop_time*1000:.1f}ms "
                                f"(target: {self.control_period*1000:.1f}ms)")
            
            rate.sleep()
        
        # Print timing summary
        if loop_times:
            avg_time = np.mean(loop_times)
            actual_rate = 1.0 / avg_time if avg_time > 0 else 0
            rospy.loginfo("="*60)
            rospy.loginfo("CONTROL LOOP TIMING SUMMARY")
            rospy.loginfo("="*60)
            rospy.loginfo(f"Target rate: {self.control_rate:.1f} Hz")
            rospy.loginfo(f"Actual rate: {actual_rate:.1f} Hz")
            rospy.loginfo(f"Average loop time: {avg_time*1000:.1f} ms")
            rospy.loginfo(f"Max loop time: {max_loop_time*1000:.1f} ms")
            rospy.loginfo(f"Slow loops (>1.5x target): {slow_loops}/{len(loop_times)}")
            rospy.loginfo("="*60)
        
        # Stop collection and save
        if self.residual_collector:
            self.residual_collector.stop_collection()
            filename = self.residual_collector.save_data()
            return filename
        
        return None
    
    def land_and_shutdown(self):
        """Land the drone"""
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
    # Parse arguments
    if len(sys.argv) < 4:
        print("Usage: python3 model_controller_with_residuals.py <x> <y> <z> "
              "[model_path] [--collect-residuals]")
        print("\nExamples:")
        print("  Normal flight:")
        print("    python3 model_controller_with_residuals.py 3.0 2.0 1.5 /path/to/policy")
        print("\n  Residual collection:")
        print("    python3 model_controller_with_residuals.py 3.0 2.0 1.5 "
              "/path/to/policy --collect-residuals")
        return
    
    goal_x = float(sys.argv[1])
    goal_y = float(sys.argv[2])
    goal_z = float(sys.argv[3])
    
    # Check for model path and flags
    model_path = None
    collect_residuals = False
    
    for arg in sys.argv[4:]:
        if arg == '--collect-residuals':
            collect_residuals = True
        elif not arg.startswith('--'):
            model_path = arg
    
    # Create controller
    controller = ModelController(
        model_path=model_path,
        control_rate=10.0,
        goal_tolerance=0.3,
        num_gates=7,
        collect_residuals=collect_residuals
    )
    
    try:
        # Navigate to starting position
        controller.navigate_wait(z=2, frame_id='body', auto_arm=True)
        gate_6_pos = controller.gates[6]
        controller.navigate_wait(x=gate_6_pos[0], y=gate_6_pos[1], 
                               z=gate_6_pos[2], frame_id='aruco_map', speed=2.0)
        
        # Run control loop (with residual collection if enabled)
        result = controller.execute(timeout=120.0)
        
        if collect_residuals and result:
            rospy.loginfo("=" * 60)
            rospy.loginfo(f"Residual data saved: {result}")
            rospy.loginfo("Next step: Process data with fit_dynamics_residual.py")
            rospy.loginfo("=" * 60)
        elif not collect_residuals:
            if result:
                rospy.loginfo("Mission completed successfully!")
            else:
                rospy.logwarn("Mission failed or timed out")
    
    finally:
        controller.land_and_shutdown()


if __name__ == '__main__':
    main()
