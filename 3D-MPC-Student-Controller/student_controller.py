#!/usr/bin/env python3


#!/usr/bin/env python3

import rospy
import numpy as np
from tflite_runtime.interpreter import Interpreter
import pickle
import os

# --- 1. ADDED IMPORTS ---
from clover.srv import GetTelemetry, SetVelocity, Navigate
from std_srvs.srv import Trigger

class StudentControllerNode:
    def __init__(self):
        rospy.init_node("student_controller_node")

        # --- Load the trained model and scalers ---
        model_dir = os.path.expanduser("~/drone_student_model")
        scaler_dir = os.path.expanduser("~/mpc_data/processed_drone_data")
        model_path = os.path.join(model_dir, 'student_controller.tflite')
        scaler_x_path = os.path.join(scaler_dir, 'scaler_X.pkl')
        scaler_y_path = os.path.join(scaler_dir, 'scaler_y.pkl')

        rospy.loginfo(f"Loading TFLite model from: {model_path}")
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        rospy.loginfo(f"Loading X scaler from: {scaler_x_path}")
        with open(scaler_x_path, 'rb') as f:
            self.scaler_X = pickle.load(f)
            
        rospy.loginfo(f"Loading y scaler from: {scaler_y_path}")
        with open(scaler_y_path, 'rb') as f:
            self.scaler_y = pickle.load(f)
            
        rospy.loginfo("Student model and scalers loaded successfully.")

        # --- ROS State and I/O ---
        self.state = np.zeros(6)  # [x, vx, y, vy, z, vz]
        self.v_cmd_limit = rospy.get_param("~v_cmd_limit", 1.0) 

        # --- RAMP LOGIC ---
        self.current_ref = np.zeros(3)
        self.start_ref = np.zeros(3)
        self.target_ref = np.zeros(3)
        self.ramp_duration = rospy.get_param("~ramp_duration", 4.0)
        self.ramp_start_time = rospy.Time.now().to_sec()
        self.odom_received = False 

        # These are the BASE waypoints the student will follow
        self.waypoints = [
            np.array([0.0, 2.0, 0.5]), # gate1
            np.array([0.0, 4.0, 0.5]),  # gate2
            np.array([0.0, 6.0, 0.5])
        ]
        self.wp_index = 0

        # --- Telemetry Service ---
        rospy.loginfo("Waiting for get_telemetry service...")
        rospy.wait_for_service('get_telemetry')
        self.get_telemetry_proxy = rospy.ServiceProxy('get_telemetry', GetTelemetry)
        rospy.loginfo("Service 'get_telemetry' found.")
        
        # --- Velocity Service ---
        rospy.loginfo("Waiting for set_velocity service...")
        rospy.wait_for_service('set_velocity')
        self.set_velocity_proxy = rospy.ServiceProxy('set_velocity', SetVelocity)
        rospy.loginfo("Service 'set_velocity' found.")

        # --- 2. ADDED NAVIGATION SERVICES AND SHUTDOWN HOOK ---
        rospy.loginfo("Waiting for navigation services (navigate, land)...")
        rospy.wait_for_service('navigate')
        rospy.wait_for_service('land')
        self.navigate_proxy = rospy.ServiceProxy('navigate', Navigate)
        self.land_proxy = rospy.ServiceProxy('land', Trigger)
        rospy.loginfo("Navigation services found.")
        
        # Add safety landing on shutdown
        rospy.on_shutdown(self._handle_shutdown)
        # --- END OF ADDED SERVICES ---

        self.rate = rospy.Rate(1.0 / 0.05) # 20 Hz

        # --- 3. ADDED TAKEOFF CALL ---
        # This will execute takeoff and block until it's done
        self._perform_takeoff()
        # --- END OF ADDED TAKEOFF ---

    # --- 4. NEW METHOD: Handle Takeoff ---
    def _perform_takeoff(self):
        try:
            # Take off to 0.5 meters. This matches your first waypoint height.
            rospy.loginfo('Taking off to z=0.5m...')
            self.navigate_proxy(x=0, y=0, z=0.5, frame_id='body', auto_arm=True)
            
            # Wait 5 seconds for the takeoff to stabilize
            rospy.sleep(5.0) 
            rospy.loginfo('Takeoff complete. Proceeding to MPC mode.')

        except rospy.ServiceException as e:
            rospy.logerr(f"Takeoff service call failed: {e}")
            # If takeoff fails, we should not continue.
            rospy.signal_shutdown("Failed to take off")
            
        except rospy.ROSInterruptException:
            rospy.loginfo("Takeoff interrupted. Landing...")
            self._handle_shutdown() # Attempt to land if interrupted

    # --- 5. NEW METHOD: Handle Shutdown ---
    def _handle_shutdown(self):
        rospy.loginfo('Shutdown requested. Landing drone...')
        try:
            # Call the land service
            self.land_proxy()
        except rospy.ServiceException as e:
            rospy.logerr(f"Land service call failed on shutdown: {e}")
        rospy.loginfo('Landed.')

    # --- RAMP LOGIC (Unchanged) ---
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

        is_final_waypoint = (self.wp_index == len(self.waypoints) - 1)
        position_met = (pos_err < 0.20) 
        speed_met = (speed < 0.12) or not is_final_waypoint 
        time_met = (progress >= 1.0) 

        if time_met and position_met and speed_met:
            rospy.loginfo("Reached waypoint %d: (%.2f, %.2f, %.2f)", self.wp_index, *xyz_target)
            self.wp_index += 1
            if self.wp_index < len(self.waypoints):
                self.start_ref = np.array([self.state[0], self.state[2], self.state[4]])
                self.ramp_start_time = rospy.Time.now().to_sec()
        
        if progress >= 1.0:
            self.current_ref = xyz_target

        return self.current_ref

    # --- PUBLISH LOGIC (Unchanged) ---
    def _publish_target_twist(self, v_cmd):
        """Publish ENU velocity command using the set_velocity service."""
        v_safe = np.clip(v_cmd, -self.v_cmd_limit, self.v_cmd_limit)

        try:
            self.set_velocity_proxy(
                vx=float(v_safe[0]),
                vy=float(v_safe[1]),
                vz=float(v_safe[2]),
                frame_id='aruco_map'
            )
        except rospy.ServiceException as e:
            rospy.logwarn_throttle(1.0, f"Failed to call set_velocity service: {e}")
        
        rospy.loginfo_throttle(0.5,
            "pos=(%.2f, %.2f, %.2f) target=(%.2f, %.2f, %.2f) v_cmd_enu=(%.2f, %.2f, %.2f)",
            self.state[0], self.state[2], self.state[4],
            self.current_ref[0], self.current_ref[1], self.current_ref[2],
            v_safe[0], v_safe[1], v_safe[2]
        )

    # --- MAIN RUN METHOD (Unchanged, but execution context is new) ---
    # This method will now be called *after* takeoff is complete.
    def run(self):
        rospy.loginfo("Waiting for first telemetry from 'aruco_map'...")
        timeout = rospy.Time.now().to_sec() + 10.0 # 10 second timeout
        
        while not rospy.is_shutdown() and not self.odom_received and rospy.Time.now().to_sec() < timeout:
            try:
                telem = self.get_telemetry_proxy(frame_id='aruco_map')
                initial_state = np.array([
                    telem.x, telem.vx, telem.y, telem.vy, telem.z, telem.vz
                ])
                
                if not np.isnan(initial_state).any():
                    self.state[:] = initial_state
                    self.odom_received = True
                    rospy.loginfo("Initial telemetry received. Starting Student Controller.")
                else:
                    rospy.logwarn_throttle(1.0, "Telemetry is NaN. Waiting for ArUco lock...")
                    self.rate.sleep()
                    
            except rospy.ServiceException as e:
                rospy.logwarn(f"Failed to get initial telemetry: {e}. Retrying...")
                self.rate.sleep()
        
        if not self.odom_received:
            rospy.logerr("No valid telemetry from 'aruco_map' after 10 seconds. Landing.")
            self._handle_shutdown() # Try to land if we can't get telemetry
            return 

        # Initialize the ramp logic *after* getting the first state
        # This will correctly use the drone's post-takeoff position as the start.
        self.start_ref = np.array([self.state[0], self.state[2], self.state[4]])
        self.ramp_start_time = rospy.Time.now().to_sec()

        rospy.loginfo("Starting student controller main loop.")
        
        while not rospy.is_shutdown() and self.wp_index < len(self.waypoints):
            
            # --- Get Current State ---
            try:
                telem = self.get_telemetry_proxy(frame_id='aruco_map')
                new_state = np.array([
                    telem.x, telem.vx, telem.y, telem.vy, telem.z, telem.vz
                ])

                if np.isnan(new_state).any():
                    rospy.logwarn_throttle(1.0, "Telemetry contains NaN! ArUco map lost. Sending zero velocity.")
                    self._publish_target_twist(np.zeros(3)) # Send safety command
                    self.rate.sleep()
                    continue 
                
                self.state[:] = new_state

            except rospy.ServiceException as e:
                rospy.logwarn(f"Failed to call get_telemetry service: {e}")
                self.rate.sleep() 
                continue 
            
            # --- MODEL INFERENCE ---
            
            # 1. Get the ramping target
            target_pos = self._update_ramp()
            
            # 2. Construct the input vector
            current_pos = np.array([self.state[0], self.state[2], self.state[4]]) # [x, y, z]
            current_vel = np.array([self.state[1], self.state[3], self.state[5]]) # [vx, vy, vz]
            input_vector_numpy = np.hstack([current_pos, current_vel, target_pos]).reshape(1, -1)

            # 3. Scale the input
            input_vector_scaled = self.scaler_X.transform(input_vector_numpy)

            # 4. Predict with TFLite
            input_data = input_vector_scaled.astype(np.float32)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            predicted_scaled = self.interpreter.get_tensor(self.output_details[0]['index'])

            # 5. Inverse transform the output
            v_cmd = self.scaler_y.inverse_transform(predicted_scaled.reshape(1, -1)).flatten()
            
            # --- Publishing and Waypoint Logic ---
            self._publish_target_twist(v_cmd)
            
            self.rate.sleep()
            
        rospy.loginfo("Student controller finished waypoint sequence. Landing.")
        
        # Call the safe landing function
        self._handle_shutdown()
        
        # Keep the node alive briefly to ensure land command is processed
        rospy.sleep(2.0)


if __name__ == "__main__":
    try:
        # __init__ will now perform the takeoff
        node = StudentControllerNode()
        # run() will start after takeoff is complete
        node.run()
    except rospy.ROSInterruptException:
        pass
    except rospy.service.ServiceException as e:
        rospy.logerr(f"Service-related error during initialization: {e}")
    except Exception as e:
        rospy.logerr(f"An unhandled error occurred: {e}")