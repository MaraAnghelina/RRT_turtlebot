from rrt import rrt

import cv2
import numpy as np 
import yaml

import rospy
from nav_msgs.msg import Odometry


# Callback function pentru procesarea mesajelor de localizare
def odom_callback(msg):
    # Extrage coordonatele x, y și unghiul theta din mesajul Odometry
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    theta = msg.pose.pose.orientation.z  # depinde de cum e orientată, poate fi quaternion

    rospy.loginfo("Robot pose: x=%.2f, y=%.2f, theta=%.2f", x, y, theta)

    rrt(x, y)


    rospy.signal_shutdown("got pose")
    

def listener():
    rospy.init_node('pose_listener', anonymous=True)

    rospy.Subscriber("/odom", Odometry, odom_callback)

    rospy.spin()

if __name__ == '__main__':
    listener()
    
