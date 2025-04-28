import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Quaternion
from rosgraph_msgs.msg import Log
from std_srvs.srv import Empty 

import math

def clear_costmaps():
    """Calls the /move_base/clear_costmaps service to reset the costmap."""
    rospy.wait_for_service('/move_base/clear_costmaps')
    try:
        clear_costmaps_srv = rospy.ServiceProxy('/move_base/clear_costmaps', Empty)
        clear_costmaps_srv()
        rospy.loginfo("Cleared costmaps successfully!")
    except rospy.ServiceException as e:
        rospy.logerr("Failed to clear costmaps: %s" % e)

def set_initial_pose(x, y, yaw):
    pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=10)
    rospy.sleep(1)

    msg = PoseWithCovarianceStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "map"

    msg.pose.pose.position.x = x
    msg.pose.pose.position.y = y
    msg.pose.pose.position.z = 0.0


    qx, qy, qz, qw = quaternion_from_euler(0, 0, yaw)
    quaternion = Quaternion(qx, qy, qz, qw)
    msg.pose.pose.orientation = quaternion

    pub.publish(msg)
    rospy.loginfo("initial pose set in Rviz")
    rospy.sleep(1)
    clear_costmaps()

def quaternion_from_euler(roll, pitch, yaw):
    qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
    qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
    qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
    qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
    return [qx, qy, qz, qw]

def euler_from_quaternion(quaternion):
    x = quaternion.x
    y = quaternion.y
    z = quaternion.z
    w = quaternion.w

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    return roll, pitch, yaw