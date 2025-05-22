from rrt import rrt
import math

import cv2
import numpy as np 
import yaml
import matplotlib.pyplot as plt

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Quaternion
from rosgraph_msgs.msg import Log
from std_srvs.srv import Empty 

goalReached = False 
alreadyThere = False
setInitPose = False

def log_callback(msg):
    global goalReached
    if "Goal reached" in msg.msg:  
        rospy.loginfo("Obiectivul a fost atins")
        goalReached = True
    

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


def odom_callback(msg):
    # extrage coordonatele x, y si unghiul theta din mesajul Odometry
    global goalReached, setInitPose 
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    theta = msg.pose.pose.orientation.z  

    quaternion = msg.pose.pose.orientation
    _, _, yaw = euler_from_quaternion(quaternion)


    if setInitPose == False:
        set_initial_pose(x, y, yaw)
        setInitPose = True

    rospy.loginfo("Robot pose: x=%.2f, y=%.2f, theta=%.2f", x, y, theta)

    fig, ax = plt.subplots()
    x_goal = 1.73
    y_goal = -1.3
    ax.plot(x_goal, y_goal, 'ro', markersize=5, markerfacecolor='b')

    q_min = rrt(x, y, ax, x_goal, y_goal)
    #coordPath = [{'coord': [1, 2], 'cost': 0, 'parent': 0}, {'coord': [-1, 2], 'cost': 0, 'parent': 0} ]
    velPub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)

    x = q_min['coord'][0]
    y = q_min['coord'][1]

    print(q_min)

    while (x > abs(x_goal)+0.1 or x < abs(x_goal)-0.1) or (y > abs(y_goal)+0.1 or y < abs(y_goal)-0.1):
        ax.plot(x , y, 'go', markersize=5, markerfacecolor='g')

        cmd = PoseStamped()
        cmd.header.frame_id = "odom"

        cmd.pose.position.x = x
        cmd.pose.position.y = y

        cmd.pose.orientation.x = 0
        cmd.pose.orientation.y = 0
        cmd.pose.orientation.z = 0
        cmd.pose.orientation.w = 1

        rospy.sleep(1)
        goalReached = False
        velPub.publish(cmd)
        print(goalReached)

        while not goalReached:
            print("Asteptam sa ajungem la obiectiv...") 
            rospy.sleep(0.5)  
            velPub.publish(cmd)

        #goalReached = False 
        rospy.sleep(1)
        
        q_min = rrt(x, y, ax, x_goal, y_goal)
        x = q_min['coord'][0]
        y = q_min['coord'][1]
    
    ax.plot(x , y, 'go', markersize=5, markerfacecolor='g')
    
    rospy.signal_shutdown("reached goal")
        

def listener():
    rospy.init_node('pose_listener', anonymous=True)

    rospy.Subscriber("/odom", Odometry, odom_callback)
    rospy.Subscriber("/rosout", Log, log_callback)

    rospy.spin()

if __name__ == '__main__':
    listener()
    
