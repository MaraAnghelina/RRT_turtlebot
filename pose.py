#from rrt import rrt
from rrt_intreg import rrt
from set_init_pose import set_initial_pose, euler_from_quaternion

import cv2
import numpy as np 
import yaml
import matplotlib.pyplot as plt

import rospy
from nav_msgs.msg import Odometry

coordPath = []
setInitPose = False

# Callback function pentru procesarea mesajelor de localizare
def odom_callback(msg):
    # Extrage coordonatele x, y si unghiul theta din mesajul Odometry
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
    ax.plot(1.73, -1.3, 'ro', markersize=5, markerfacecolor='b')

    x_goal = 1.73
    y_goal = -1.3
    q_min = rrt(x, y, ax, x_goal, y_goal)
    for point in path:
        ax.plot(point[0], point[1], 'ro')
    rospy.signal_shutdown("got pose")
    #coordPath = [{'coord': [1, 2], 'cost': 0, 'parent': 0}, {'coord': [-1, 2], 'cost': 0, 'parent': 0} ]

    x = q_min['coord'][0]
    y = q_min['coord'][1]

    print(q_min)

    while (x > abs(x_goal)+0.1 or x < abs(x_goal)-0.1) or (y > abs(y_goal)+0.1 or y < abs(y_goal)-0.1):
        q_min = rrt(x, y, ax, x_goal, y_goal)
        x = q_min['coord'][0]
        y = q_min['coord'][1]
    
    ax.plot(x , y, 'go', markersize=5, markerfacecolor='g')
    
    rospy.signal_shutdown("got pose")
        

def listener():
    rospy.init_node('pose_listener', anonymous=True)

    rospy.Subscriber("/odom", Odometry, odom_callback)

    rospy.spin()

if __name__ == '__main__':
    listener()
    
