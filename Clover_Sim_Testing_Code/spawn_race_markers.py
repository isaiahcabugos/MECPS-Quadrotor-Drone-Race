#!/usr/bin/env python3
"""
Spawn visual markers for the 7-gate racing track in Gazebo
Ready to run - just execute this script with Gazebo running
"""
import rospy
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose
import numpy as np


def create_sphere_sdf(name, radius=0.15, color_rgba=(0, 1, 0, 0.8)):
    """Generate SDF string for a colored sphere marker"""
    r, g, b, a = color_rgba
    
    sdf = f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry>
          <sphere>
            <radius>{radius}</radius>
          </sphere>
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


def spawn_goal_marker(position, marker_id, radius=0.15, color=None):
    """
    Spawn a single goal marker in Gazebo
    
    Args:
        position: (x, y, z) tuple
        marker_id: Unique integer ID for this marker
        radius: Sphere radius in meters
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
            a = 0.8
            color = (r, g, b, a)
        
        model_name = f"gate_marker_{marker_id}"
        sdf_string = create_sphere_sdf(model_name, radius=radius, color_rgba=color)
        
        pose = Pose()
        pose.position.x = position[0]
        pose.position.y = position[1]
        pose.position.z = position[2]
        pose.orientation.w = 1.0
        
        resp = spawn_model(model_name, sdf_string, "", pose, "world")
        
        if resp.success:
            rospy.loginfo(f"✓ Spawned {model_name} at ({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})")
        else:
            rospy.logwarn(f"✗ Failed to spawn {model_name}: {resp.status_message}")
            
        return resp.success
        
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
        return False


def main():
    rospy.init_node('spawn_race_gate_markers', anonymous=True)
    
    # Your exact gate positions
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
    rospy.loginfo("Spawning 7 gate markers for racing track")
    rospy.loginfo("=" * 60)
    
    success_count = 0
    for i, (pos, label) in enumerate(zip(gates, gate_labels)):
        rospy.loginfo(f"\nGate {i+1}: {label}")
        if spawn_goal_marker(pos, i, radius=0.15):
            success_count += 1
        rospy.sleep(0.2)  # Small delay between spawns
    
    rospy.loginfo("\n" + "=" * 60)
    rospy.loginfo(f"✓ Successfully spawned {success_count}/{len(gates)} markers")
    rospy.loginfo("=" * 60)
    rospy.loginfo("\nMarkers are colored from green (start) to red (finish)")
    rospy.loginfo("Markers are static and will persist in Gazebo")


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
