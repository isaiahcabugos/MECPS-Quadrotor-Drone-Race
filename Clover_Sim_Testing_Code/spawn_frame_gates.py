#!/usr/bin/env python3
"""
Spawn 1m x 1m FRAME gates (hollow rectangles) for racing track
More realistic appearance with actual gate frames
"""
import rospy
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose
import numpy as np


def create_frame_gate_sdf(name, size=1.0, bar_width=0.05, color_rgba=(0, 1, 0, 0.8)):
    """
    Generate SDF for a hollow frame gate (4 bars forming a square)
    
    Args:
        name: Model name
        size: Width and height of the gate opening (default 1.0m)
        bar_width: Thickness of frame bars (default 5cm)
        color_rgba: (r, g, b, a) color tuple
    """
    r, g, b, a = color_rgba
    half_size = size / 2.0
    
    sdf = f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <static>true</static>
    
    <!-- Top bar -->
    <link name="top_bar">
      <pose>0 0 {half_size} 0 0 0</pose>
      <visual name="visual">
        <geometry>
          <box>
            <size>{size + 2*bar_width} {bar_width} {bar_width}</size>
          </box>
        </geometry>
        <material>
          <ambient>{r} {g} {b} {a}</ambient>
          <diffuse>{r} {g} {b} {a}</diffuse>
          <emissive>{r*0.5} {g*0.5} {b*0.5} 1</emissive>
        </material>
      </visual>
    </link>
    
    <!-- Bottom bar -->
    <link name="bottom_bar">
      <pose>0 0 -{half_size} 0 0 0</pose>
      <visual name="visual">
        <geometry>
          <box>
            <size>{size + 2*bar_width} {bar_width} {bar_width}</size>
          </box>
        </geometry>
        <material>
          <ambient>{r} {g} {b} {a}</ambient>
          <diffuse>{r} {g} {b} {a}</diffuse>
          <emissive>{r*0.5} {g*0.5} {b*0.5} 1</emissive>
        </material>
      </visual>
    </link>
    
    <!-- Left bar -->
    <link name="left_bar">
      <pose>-{half_size} 0 0 0 0 0</pose>
      <visual name="visual">
        <geometry>
          <box>
            <size>{bar_width} {bar_width} {size}</size>
          </box>
        </geometry>
        <material>
          <ambient>{r} {g} {b} {a}</ambient>
          <diffuse>{r} {g} {b} {a}</diffuse>
          <emissive>{r*0.5} {g*0.5} {b*0.5} 1</emissive>
        </material>
      </visual>
    </link>
    
    <!-- Right bar -->
    <link name="right_bar">
      <pose>{half_size} 0 0 0 0 0</pose>
      <visual name="visual">
        <geometry>
          <box>
            <size>{bar_width} {bar_width} {size}</size>
          </box>
        </geometry>
        <material>
          <ambient>{r} {g} {b} {a}</ambient>
          <diffuse>{r} {g} {b} {a}</diffuse>
          <emissive>{r*0.5} {g*0.5} {b*0.5} 1</emissive>
        </material>
      </visual>
    </link>
    
  </model>
</sdf>"""
    return sdf


def calculate_gate_orientation(prev_pos, curr_pos, next_pos):
    """Calculate yaw angle to orient gate perpendicular to race path"""
    incoming = np.array(curr_pos) - np.array(prev_pos)
    outgoing = np.array(next_pos) - np.array(curr_pos)
    path_direction = incoming + outgoing
    yaw = np.arctan2(path_direction[1], path_direction[0]) + np.pi/2
    return yaw


def spawn_frame_gate(position, marker_id, yaw=0.0, gate_size=1.0, color=None):
    """Spawn a single frame gate marker in Gazebo"""
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    
    try:
        spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        
        if color is None:
            progress = marker_id / 6.0
            r = progress
            g = 1.0 - progress
            b = 0.0
            a = 0.9
            color = (r, g, b, a)
        
        model_name = f"gate_frame_{marker_id}"
        sdf_string = create_frame_gate_sdf(model_name, size=gate_size, 
                                          bar_width=0.05, color_rgba=color)
        
        pose = Pose()
        pose.position.x = position[0]
        pose.position.y = position[1]
        pose.position.z = position[2]
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = np.sin(yaw / 2.0)
        pose.orientation.w = np.cos(yaw / 2.0)
        
        resp = spawn_model(model_name, sdf_string, "", pose, "world")
        
        if resp.success:
            yaw_deg = np.degrees(yaw)
            rospy.loginfo(f"✓ Spawned {model_name} at ({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f}), yaw={yaw_deg:.1f}°")
        else:
            rospy.logwarn(f"✗ Failed to spawn {model_name}: {resp.status_message}")
            
        return resp.success
        
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
        return False


def main():
    rospy.init_node('spawn_race_gate_frames', anonymous=True)
    
    gates = [
        (5.0,   0.0,  2.0),   # Gate 1: Start/Finish
        (10.0, -5.0,  2.5),   # Gate 2: Right turn
        (8.0,  -8.0,  3.0),   # Gate 3: Descending turn
        (0.0,  -9.0,  2.0),   # Gate 4: Split-S entry
        (-5.0, -5.0,  1.5),   # Gate 5: Split-S exit
        (-8.0,  2.0,  2.5),   # Gate 6: Left turn
        (0.0,   5.0,  2.0),   # Gate 7: Final approach
    ]
    
    gate_labels = [
        "Start/Finish",
        "Right turn",
        "Descending turn",
        "Split-S entry",
        "Split-S exit",
        "Left turn",
        "Final approach"
    ]
    
    rospy.loginfo("=" * 60)
    rospy.loginfo("Spawning 7 FRAME gates (1m x 1m hollow) for racing track")
    rospy.loginfo("=" * 60)
    
    # Calculate orientations
    num_gates = len(gates)
    orientations = []
    
    for i in range(num_gates):
        prev_idx = (i - 1) % num_gates
        next_idx = (i + 1) % num_gates
        yaw = calculate_gate_orientation(gates[prev_idx], gates[i], gates[next_idx])
        orientations.append(yaw)
    
    # Spawn all gates
    success_count = 0
    for i, (pos, label, yaw) in enumerate(zip(gates, gate_labels, orientations)):
        rospy.loginfo(f"\nGate {i+1}: {label}")
        if spawn_frame_gate(pos, i, yaw=yaw, gate_size=1.0):
            success_count += 1
        rospy.sleep(0.2)
    
    rospy.loginfo("\n" + "=" * 60)
    rospy.loginfo(f"✓ Successfully spawned {success_count}/{len(gates)} frame gates")
    rospy.loginfo("=" * 60)
    rospy.loginfo("\nGates are 1m x 1m hollow frames")
    rospy.loginfo("Frame bars are 5cm thick")
    rospy.loginfo("Oriented perpendicular to race path")
    rospy.loginfo("Colors: Green (start) → Red (finish)")


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
