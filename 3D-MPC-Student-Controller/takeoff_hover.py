#!/usr/bin/env python3

import rospy
from clover.srv import Navigate
from std_srvs.srv import Trigger

rospy.init_node('takeoff_hover_node')

# --- Service Proxies ---
rospy.loginfo('Waiting for services...')
rospy.wait_for_service('navigate')
rospy.wait_for_service('land')

navigate = rospy.ServiceProxy('navigate', Navigate)
land = rospy.ServiceProxy('land', Trigger)

# --- Safety Land on Shutdown ---
def handle_shutdown():
    rospy.loginfo('Shutdown requested. Landing drone...')
    land()
    rospy.loginfo('Landed.')

rospy.on_shutdown(handle_shutdown)

# --- Main Action ---
try:
    rospy.loginfo('Taking off to 1.5m...')
    # We use frame_id='body' for a simple takeoff.
    # It will climb 1.5m straight up from its current position.
    navigate(x=0, y=0, z=1.5, frame_id='body', auto_arm=True)
    
    rospy.sleep(5.0) # Give it 5 seconds to reach the altitude
    
    rospy.loginfo('Now hovering at 1.5m. Ready for MPC node.')
    
    # Keep the node alive to maintain the hover
    rospy.spin()

except rospy.ServiceException as e:
    rospy.logerr(f"Service call failed: {e}")
except rospy.ROSInterruptException:
    rospy.loginfo("Interrupted. Landing...")
    land()