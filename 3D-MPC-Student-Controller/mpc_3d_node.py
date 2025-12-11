import rospy
import numpy as np
import cvxpy as cp
from clover.srv import GetTelemetry, SetVelocity# <--- CHANGED: Import the service
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
import os
import pandas as pd
import time

class MPC3DRaceNodeAuto:
    def __init__(self):
        rospy.init_node("mpc_3d_race_node_auto")

        # ===== MPC setup =====
        self.dt = rospy.get_param("~dt", 0.05)
        self.N = rospy.get_param("~N", 40)
        self.nx = 6
        self.nu = 3
        self.data_log = []

        A_block = np.array([[1.0, self.dt],
                            [0.0, 1.0]])
        B_block = np.array([[0.0],
                            [1.0]])

        self.A = np.zeros((self.nx, self.nx))
        self.B = np.zeros((self.nx, self.nu))
        for i in range(3):
            r = 2*i
            self.A[r:r+2, r:r+2] = A_block
            self.B[r:r+2, i:i+1] = B_block
        self.c = np.zeros(self.nx)

        # safer defaults for debug; increase later for racing
        self.v_max = rospy.get_param("~v_max", 3.0)       # internal model max velocity (m/s)
        self.a_max = rospy.get_param("~a_max", 0.4)       # velocity increment per step (m/s per solve)
        self.v_cmd_limit = rospy.get_param("~v_cmd_limit", 1.0)  # clamp published velocity (m/s)
        self.z_min = rospy.get_param("~z_min", 0.5)

        pos_lim = 1000.0
        self.x_min = np.array([-pos_lim, -self.v_max, -pos_lim, -self.v_max, self.z_min, -self.v_max])
        self.x_max = np.array([pos_lim, self.v_max, pos_lim, self.v_max, 10.0, self.v_max])
        self.u_min = np.array([-self.a_max]*3)
        self.u_max = np.array([self.a_max]*3)

        # Weights: tune if Y oscillates more, increase corresponding R or delta-u penalty
        self.Q = np.diag([150.0, 30.0, 150.0, 60.0, 100.0, 20.0]) # Pos weights higher, but velocity weights *significantly* increased
        self.R = np.diag([2.0, 4.0, 2.0]) # Increase R to dampen control
        
        self.rho_s = 1e4

        # delta-u penalty matrix (helps smooth abrupt control changes)
        self.S = np.diag([20.0, 40.0, 20.0])

        # Optimization vars
        self.x = cp.Variable((self.nx, self.N+1))
        self.u = cp.Variable((self.nu, self.N))
        self.s = cp.Variable((self.nx, self.N), nonneg=True)

        # Params
        self.x0 = cp.Parameter(self.nx)
        self.ref = cp.Parameter(self.nx)

        self.prob = self._build_mpc()

        # ===== ROS state =====
        self.state = np.zeros(self.nx)  # [x, vx, y, vy, z, vz]
        self.odom_received = False # <--- Flag is now used to check for *initial* telemetry

        self.current_ref = np.zeros(3)
        self.start_ref = np.zeros(3)
        self.target_ref = np.zeros(3)
        self.ramp_duration = rospy.get_param("~ramp_duration", 4.0)
        self.ramp_start_time = rospy.Time.now().to_sec()
        self.ready_for_new_target = True
        
        # --- CHANGED: Add service proxy ---
        rospy.loginfo("Waiting for get_telemetry service...")
        rospy.wait_for_service('get_telemetry')
        self.get_telemetry_proxy = rospy.ServiceProxy('get_telemetry', GetTelemetry)
        rospy.loginfo("Service 'get_telemetry' found.")
        # --- End of Change ---

        # # publish ENU Twist (this topic uses geometry_msgs/Twist in your setup)

        # --- MODIFIED: Replace Publisher with Service Proxy ---
        rospy.loginfo("Waiting for set_velocity service...")
        rospy.wait_for_service('set_velocity')
        self.set_velocity_proxy = rospy.ServiceProxy('set_velocity', SetVelocity)
        rospy.loginfo("Service 'set_velocity' found.")
        # --- End of MODIFIED ---

        self.rate = rospy.Rate(1.0 / self.dt)

        # ===== Predefined waypoint sequence =====
        base_waypoints = [
            np.array([0.0, 2.0, 1.3]),  # gate1
            np.array([0.0, 4.0, 1.3]),  # gate2
            np.array([0.0, 6.0, 1.0])   # landing (hover before landing; actual landing procedure not included)
        ]
        # Add random offsets to each waypoint
        self.waypoints = [] # Start with an empty list

        for wp in base_waypoints: # Iterate all waypoints since takeoff is removed
            # Add a random offset between -15cm and +15cm on x, y, z
            offset = (np.random.rand(3) * 0.3) - 0.15 
            # Make sure z offset doesn't go too low
            offset[2] = max(offset[2], -0.1) # e.g., don't go lower than -10cm
            
            self.waypoints.append(wp + offset)
            rospy.loginfo(f"Randomized waypoint: {self.waypoints[-1]}")


        self.wp_index = 0
        rospy.loginfo("3D racing MPC node (ENU-Twist) with predefined waypoints ready.")

    def _build_mpc(self):
        cost = 0
        cons = [self.x[:,0] == self.x0]
        for k in range(self.N):
            cost += cp.quad_form(self.x[:,k] - self.ref, self.Q)
            cost += cp.quad_form(self.u[:,k], self.R)
            cost += self.rho_s * cp.norm1(self.s[:,k])
            # delta-u penalty (smoothness)
            if k > 0:
                cost += cp.quad_form(self.u[:,k] - self.u[:,k-1], self.S)
            cons += [self.x[:,k+1] == self.A @ self.x[:,k] + self.B @ self.u[:,k] + self.c]
            cons += [self.x[:,k] >= self.x_min - self.s[:,k]]
            cons += [self.x[:,k] <= self.x_max + self.s[:,k]]
            cons += [self.u[:,k] >= self.u_min, self.u[:,k] <= self.u_max]
        cost += cp.quad_form(self.x[:,self.N] - self.ref, self.Q)
        return cp.Problem(cp.Minimize(cost), cons)

    def _next_waypoint(self):
        if self.wp_index < len(self.waypoints):
            return self.waypoints[self.wp_index]
        return None

    def _solve_mpc(self, xyz_ref):
        # --- Solve MPC ---
        ref_full = np.array([xyz_ref[0], 0.0, xyz_ref[1], 0.0, xyz_ref[2], 0.0])
        self.x0.value = self.state
        self.ref.value = ref_full

        try:
            self.prob.solve(solver=cp.CLARABEL, warm_start=True, verbose=False)
        except Exception:
            self.prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)

        if self.prob.status in ("optimal","optimal_inaccurate"):
            du = self.u.value[:,0]
            v_now = np.array([self.state[1], self.state[3], self.state[5]])
            v_cmd = v_now + du
            return v_cmd
        else:
            rospy.logwarn_throttle(2.0, "MPC failed: %s", self.prob.status)
            return np.zeros(3)

    def _publish_target_twist(self, v_cmd):
        """Publish ENU velocity command using the set_velocity service."""
        # clamp to safe published limits
        v_safe = np.clip(v_cmd, -self.v_cmd_limit, self.v_cmd_limit)

        try:
            # Call the "smart" service, specifying the frame_id.
            # This will correctly transform your aruco_map (ENU) velocity
            # into the drone's base_link (FRD) frame for you.
            self.set_velocity_proxy(
                vx=float(v_safe[0]),
                vy=float(v_safe[1]),
                vz=float(v_safe[2]),
                frame_id='aruco_map'  # <--- This is the key
            )
        except rospy.ServiceException as e:
            rospy.logwarn_throttle(1.0, f"Failed to call set_velocity service: {e}")

        # The loginfo is still correct, as it's logging the ENU command
        rospy.loginfo_throttle(0.5,
            "pos=(%.2f, %.2f, %.2f) target=(%.2f, %.2f, %.2f) v_cmd_enu=(%.2f, %.2f, %.2f)",
            self.state[0], self.state[2], self.state[4],
            self.current_ref[0], self.current_ref[1], self.current_ref[2],
            v_safe[0], v_safe[1], v_safe[2]
        )
    def _update_ramp(self):
        if self.wp_index >= len(self.waypoints):
            return self.current_ref  # finished all waypoints

        xyz_target = self.waypoints[self.wp_index]
        t_now = rospy.Time.now().to_sec()
        progress = min((t_now - self.ramp_start_time)/self.ramp_duration, 1.0)
        self.current_ref = self.start_ref + progress*(xyz_target - self.start_ref)

        pos_err = np.linalg.norm([self.state[0]-xyz_target[0],
                                  self.state[2]-xyz_target[1],
                                  self.state[4]-xyz_target[2]])
        speed = np.linalg.norm([self.state[1], self.state[3], self.state[5]])

        # Condition for switching to next waypoint
        is_final_waypoint = (self.wp_index == len(self.waypoints) - 1)
        
        # --- RACING LOGIC ---
        # Gate fulfillment: Pass the gate position with any speed.
        # Final point fulfillment: Must be near target AND slow down.
        
        # Condition 1: Position check (0.2m proximity)
        position_met = (pos_err < 0.20) 
        
        # Condition 2: Speed check (strict only for final point)
        speed_met = (speed < 0.12) or not is_final_waypoint 
        
        # Condition 3: Must be past the virtual ramp time (progress=1.0)
        time_met = (progress >= 1.0) 

        # We switch if we're past the ramp time AND close to the target.
        # For intermediate points, this encourages high speed and forward motion.
        if time_met and position_met and speed_met:
            rospy.loginfo("Reached waypoint %d: (%.2f, %.2f, %.2f)", self.wp_index, *xyz_target)
            self.wp_index += 1
            if self.wp_index < len(self.waypoints):
                # set new start_ref to current position (helps ramp continuity)
                self.start_ref = np.array([self.state[0], self.state[2], self.state[4]])
                self.ramp_start_time = rospy.Time.now().to_sec()
        
        # Ensure that if we are past the ramp duration, the target is FIXED at the waypoint center
        if progress >= 1.0:
            self.current_ref = xyz_target

        return self.current_ref

    def run(self):
        # --- CHANGED: Replace "wait for odom" with a single service call ---
        rospy.loginfo("Waiting for first telemetry from 'aruco_map'...")
        try:
            # Call the service once to get the initial state
            telem = self.get_telemetry_proxy(frame_id='aruco_map')
            
            # --- NEW: Initial state validation ---
            initial_state = np.array([
                telem.x, telem.vx,
                telem.y, telem.vy,
                telem.z, telem.vz
            ])
            if np.isnan(initial_state).any():
                rospy.logerr("Initial telemetry is NaN. Drone cannot see ArUco map at startup. Exiting.")
                return # Exit the script
            # --- END OF NEW CHECK ---

            self.state[:] = initial_state
            self.odom_received = True # Use the same flag
            rospy.loginfo("Initial telemetry received.")
        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to get initial telemetry: {e}. Exiting.")
            return # Exit the script
        # --- End of Change ---

        # Also randomize the starting position slightly for each run
        start_offset = (np.random.rand(3) * 0.2) - 0.1 # +/- 10cm
        start_offset[2] = 0 # Don't offset Z at start
        rospy.loginfo(f"Applying start offset: {start_offset}")
        # --- END OF DATA AUGMENTATION ---

        # initialize start_ref to current position for first waypoint
        self.start_ref = np.array([self.state[0], self.state[2], self.state[4]]) + start_offset
        self.ramp_start_time = rospy.Time.now().to_sec()

        while not rospy.is_shutdown() and self.wp_index < len(self.waypoints):
            
            # --- CHANGED: Get Current State from Service on each loop ---
            try:
                telem = self.get_telemetry_proxy(frame_id='aruco_map')
                new_state = np.array([
                    telem.x, telem.vx,
                    telem.y, telem.vy,
                    telem.z, telem.vz
                ])

                # --- NEW: CRITICAL NAN CHECK ---
                # Check if any value in the new state is NaN
                if np.isnan(new_state).any():
                    rospy.logwarn_throttle(1.0, "Telemetry contains NaN! ArUco map lost. Sending zero velocity.")
                    # Publish a zero velocity command for safety
                    self._publish_target_twist(np.zeros(3))
                    self.rate.sleep()
                    continue # Skip the rest of this loop and try again
                # --- END OF NEW CHECK ---

                # If the state is valid (no NaNs), update it
                self.state[:] = new_state

            except rospy.ServiceException as e:
                rospy.logwarn(f"Failed to call get_telemetry service: {e}")
                self.rate.sleep() # Wait a moment
                continue # Skip this loop iteration
            # --- End of Change ---

            xyz_cmd = self._update_ramp()
            v_cmd = self._solve_mpc(xyz_cmd)

            # safety: if the z-target is below z_min, clip (avoid crash)
            if xyz_cmd[2] < self.z_min:
                xyz_cmd[2] = self.z_min

            self.current_ref = xyz_cmd
            # Collect input-output pair for ML training
            self.data_log.append([
                self.state[0],  # x
                self.state[2],  # y
                self.state[4],  # z
                self.state[1],  # vx
                self.state[3],  # vy
                self.state[5],  # vz
                self.current_ref[0],  # xt
                self.current_ref[1],  # yt
                self.current_ref[2],  # zt
                v_cmd[0],  # vmx (MPC output)
                v_cmd[1],  # vmy
                v_cmd[2]   # vmz
            ])
            self._publish_target_twist(v_cmd)
            self.rate.sleep()

        rospy.loginfo("Waypoint sequence finished or node shutdown.")
        df = pd.DataFrame(self.data_log, columns=[
        "x","y","z",
        "vx","vy","vz",
        "xt","yt","zt",
        "vmx","vmy","vmz"
         ])
        save_dir = os.path.expanduser("~/mpc_data")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"run_{int(time.time() * 1000)}.csv")
        df.to_csv(save_path, index=False)
        rospy.loginfo(f"Saved data log to {save_path}")

if __name__=="__main__":
    try:
        MPC3DRaceNodeAuto().run()
    except rospy.ROSInterruptException:
        pass
