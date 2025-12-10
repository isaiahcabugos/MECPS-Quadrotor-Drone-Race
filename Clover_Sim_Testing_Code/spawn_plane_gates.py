#!/usr/bin/env python3
"""
Spawn 1m x 1m plane markers for the 7-gate racing track in Gazebo
Gates are vertical planes oriented perpendicular to the race path
"""
import rospy
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose
import numpy as np


def create_plane_gate_sdf(name, size=1.0, thickness=0.02, color_rgba=(0, 1, 0, 0.8)):
    """
    Generate SDF string for a flat plane gate marker
    
    Args:
        name: Model name
        size: Width and height of the gate (default 1.0m)
        thickness: Thickness of the plane (default 2cm)
        color_rgba: (r, g, b, a) color tuple
    """
    r, g, b, a = color_rgba
    
    sdf = f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry>
          <box>
            <size>{size} {thickness} {size}</size>
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
    """
    Calculate yaw angle to orient gate perpendicular to race path
    
    Args:
        prev_pos: Position of previous gate (or last gate if this is first)
        curr_pos: Position of current gate
        next_pos: Position of next gate (or first gate if this is last)
    
    Returns:
        yaw: Rotation angle in radians
    """
    # Calculate incoming and outgoing direction vectors
    incoming = np.array(curr_pos) - np.array(prev_pos)
    outgoing = np.array(next_pos) - np.array(curr_pos)
    
    # Average direction (path direction at this gate)
    path_direction = incoming + outgoing
    
    # Gate should be perpendicular to path direction (rotate 90 degrees)
    # Calculate yaw from x-y components only
    yaw = np.arctan2(path_direction[1], path_direction[0]) + np.pi/2
    
    return yaw


def spawn_plane_gate(position, marker_id, yaw=0.0, gate_size=1.0, color=None):
    """
    Spawn a single plane gate marker in Gazebo
    
    Args:
        position: (x, y, z) tuple
        marker_id: Unique integer ID for this marker
        yaw: Rotation angle in radians (gate orientation)
        gate_size: Gate width and height (default 1.0m)
        color: (r, g, b, a) tuple, or None for gradient
    """
    rospy.wait_for_service('/gazebo/spawn_sdf_model')
    
    try:
        spawn_model = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        
        # Use color gradient from green to red if not specified
        if color is None:
            progress = marker_id / 6.0  # 7 gates, 0-6 index
            r = progress
            g = 1.0 - progress
            b = 0.0
            a = 0.7  # Slightly transparent to see through
            color = (r, g, b, a)
        
        model_name = f"gate_plane_{marker_id}"
        sdf_string = create_plane_gate_sdf(model_name, size=gate_size, 
                                          thickness=0.02, color_rgba=color)
        
        pose = Pose()
        pose.position.x = position[0]
        pose.position.y = position[1]
        pose.position.z = position[2]
        
        # Set orientation (gate perpendicular to path)
        # Convert yaw to quaternion
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
    rospy.init_node('spawn_race_gate_planes', anonymous=True)
    
    # # Your exact gate positions
    # gates = [
    #     (5.0,   0.0,  2.0),   # Gate 1: Start/Finish
    #     (10.0, -5.0,  2.5),   # Gate 2: Right turn
    #     (8.0,  -8.0,  3.0),   # Gate 3: Descending turn
    #     (0.0,  -9.0,  2.0),   # Gate 4: Split-S entry
    #     (-5.0, -5.0,  1.5),   # Gate 5: Split-S exit
    #     (-8.0,  2.0,  2.5),   # Gate 6: Left turn
    #     (0.0,   5.0,  2.0),   # Gate 7: Final approach
    # ]

    gates = [
    (6.50, 5.50, 2.00),   # Gate 1: Start/Finish
    (9.00, 3.00, 2.50),   # Gate 2: Right turn
    (8.00, 1.50, 3.00),   # Gate 3: Descending turn
    (4.00, 1.00, 2.00),   # Gate 4: Split-S entry
    (1.50, 3.00, 1.5),    # Gate 5: Split-S exit
    (0.00, 6.50, 2.50),   # Gate 6: Left turn
    (4.00, 8.00, 2.00),   # Gate 7: Final approach
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
    rospy.loginfo("Spawning 7 plane gates (1m x 1m) for racing track")
    rospy.loginfo("=" * 60)
    
    # Calculate orientations for each gate
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
        if spawn_plane_gate(pos, i, yaw=yaw, gate_size=1.0):
            success_count += 1
        rospy.sleep(0.2)
    
    rospy.loginfo("\n" + "=" * 60)
    rospy.loginfo(f"✓ Successfully spawned {success_count}/{len(gates)} plane gates")
    rospy.loginfo("=" * 60)
    rospy.loginfo("\nGates are 1m x 1m vertical planes")
    rospy.loginfo("Oriented perpendicular to race path")
    rospy.loginfo("Colors: Green (start) → Red (finish)")


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
