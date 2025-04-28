#!/usr/bin/env python

import tf2_ros
import tf_conversions
import rospy
import numpy as np
import cv2
import matplotlib.pyplot as plt
import subprocess

from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Quaternion, PointStamped
from std_srvs.srv import Empty 
from visualization_msgs.msg import Marker
from rosgraph_msgs.msg import Log
from copy import copy

robot_position = (0, 0)
goalReached = False 
setInitPose = False
plannerTrigger = False
    

# Subscribers' callbacks------------------------------
mapData = OccupancyGrid()

def mapCallBack(data):
    global mapData
    mapData = data


def log_callback(msg):
    global goalReached, plannerTrigger
    if "Goal reached" in msg.msg:  
        rospy.loginfo("Obiectivul a fost atins")
        goalReached = True
    if "DWA planner failed to produce path." or "Rotation cmd in collision" in msg.msg:
        #rospy.loginfo("miscare imposibila")
        rospy.sleep(3)
        plannerTrigger = True

#-------------------------------------------------------------


#Clear costmap -------------------------------------------------
def clear_costmaps():
    """Calls the /move_base/clear_costmaps service to reset the costmap."""
    rospy.wait_for_service('/move_base/clear_costmaps')
    try:
        clear_costmaps_srv = rospy.ServiceProxy('/move_base/clear_costmaps', Empty)
        clear_costmaps_srv()
        rospy.loginfo("Cleared costmaps successfully!")
    except rospy.ServiceException as e:
        rospy.logerr("Failed to clear costmaps: %s" % e)

#------------------------------------------------------------

def test(frontier):
    global robot_position
    print(robot_position)
    if (abs(robot_position[0]) >= abs(frontier[0]) - 0.30 and abs(robot_position[0]) <= abs(frontier[0]) + 0.30) and (abs(robot_position[1]) >= abs(frontier[1]) - 0.30 and abs(robot_position[1]) <= abs(frontier[1]) + 0.30):
        return True
    return False

#Go to a point----------------------------------------------------------------
def go_to_point(frontier):
    global goalReached, mapData, plannerTrigger, robot_position

    velPub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
    cmd = PoseStamped()
    cmd.header.frame_id = mapData.header.frame_id

    cmd.pose.position.x = frontier[0]
    cmd.pose.position.y = frontier[1]

    cmd.pose.orientation.x = 0
    cmd.pose.orientation.y = 0
    cmd.pose.orientation.z = 0
    cmd.pose.orientation.w = 1

    goalReached = False
    plannerTrigger = False
    velPub.publish(cmd)

    while not test(frontier):
        print("Asteptam sa ajungem la obiectiv...") 
        rospy.sleep(0.5)  
        velPub.publish(cmd)    

    rospy.sleep(1)
    if plannerTrigger == True:
        return
#-----------------------------------------------------------------------------


#Save the explored map -------------------------------------------------------
def save_map(filename):
    try:
        retcode = subprocess.call(['rosrun', 'map_server', 'map_saver', '-f', filename])
        if retcode == 0:
            rospy.loginfo("Map saved successfully!")
        else:
            rospy.logerr("Map saving failed with return code: {}".format(retcode))
    except OSError as e:
        rospy.logerr("Execution failed" % e)

#-----------------------------------------------------------------------------

#Show the map that the robot sees at th epoint function is called -----------
def draw_map(mapData):
    data = mapData.data
    w = mapData.info.width
    h = mapData.info.height
    resolution = mapData.info.resolution
    Xstartx = mapData.info.origin.position.x
    Xstarty = mapData.info.origin.position.y
	 
    img = np.zeros((h, w, 1), np.uint8)
	
    for i in range(0,h):
    	for j in range(0,w):
    		if data[i*w+j] == 100:
    			img[i,j] = 0
    		elif data[i*w+j] == 0:
    			img[i,j] = 255
    		elif data[i*w+j] == -1:
    			img[i,j] = 205
    
	
   	o = cv2.inRange(img,0,1)
    edges = cv2.Canny(img,0,255)
    im2, contours, hierarchy = cv2.findContours(o,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(o, contours, -1, (255,255,255), 5)
    o = cv2.bitwise_not(o) 
    res = cv2.bitwise_and(o, edges)

    #plt.imshow(im2)
    #plt.show()

    return res

#-----------------------------------------------------------------------------

#EXploration function -------------------------------------------------------
def explore():
    global mapData, robot_position

    exploration_goal = PointStamped()
    map_topic= rospy.get_param('~map_topic','/map')
    targetspub = rospy.Publisher('/detected_points', PointStamped, queue_size=10)
    pub = rospy.Publisher('/shapes', Marker, queue_size=10)

    # wait until map is received, when a map is received, mapData.header.seq will not be < 1
    while mapData.header.seq < 1 or len(mapData.data) <1 :
        rospy.loginfo("Waiting for map data...")
        rospy.sleep(1)
    	   	
    rate = rospy.Rate(50)	
    points = Marker()

	#Set the frame ID and timestamp.  See the TF tutorials for information on these.
    points.header.frame_id=mapData.header.frame_id
    points.header.stamp=rospy.Time.now()

    points.ns= "markers"
    points.id = 0

    points.type = Marker.POINTS
	#Set the marker action.  Options are ADD, DELETE, and new in ROS Indigo: 3 (DELETEALL)
    points.action = Marker.ADD

    points.pose.orientation.w = 1.0
    points.scale.x = points.scale.y = 0.3
    points.color.r = 255.0/255.0
    points.color.g = 0.0/255.0
    points.color.b = 0.0/255.0
    points.color.a = 1
    points.lifetime = rospy.Duration(0)  #0 = forever


    while not rospy.is_shutdown():

        frontiers = frontier_detection()

        rx = robot_position[0]
        ry = robot_position[1]
        print(robot_position)

        #Compute distances to all frontiers
        if frontiers is None:
            save_map('/home/internship/explored_map')
            rospy.sleep(2)
            rospy.loginfo("Explored all map")
            rospy.signal_shutdown("Explored all map")
            exit()
        dists = [np.linalg.norm(np.array([rx, ry]) - np.array([pt[0], pt[1]])) for pt in frontiers]

        #Choose the closest
        min_idx = np.argmin(dists)
        closestFrontier = frontiers[min_idx]
        print("Closest frontier: ", closestFrontier)

        exploration_goal.header.frame_id = mapData.header.frame_id
        exploration_goal.header.stamp = rospy.Time.now()
        exploration_goal.point.x = closestFrontier[0] - 0.2
        exploration_goal.point.y = closestFrontier[1]
        exploration_goal.point.z = 0

        targetspub.publish(exploration_goal)
        points.points = [exploration_goal.point]
        pub.publish(points)
        rospy.sleep(1)

        go_to_point(closestFrontier)

#---------------------------------------------------------------------------------

def world_to_map_index(x, y, mapData):
    res = mapData.info.resolution
    origin_x = mapData.info.origin.position.x
    origin_y = mapData.info.origin.position.y
    width = mapData.info.width

    mx = int((x - origin_x) / res)
    my = int((y - origin_y) / res)

    #Check bounds
    if mx < 0 or my < 0 or mx >= width or my >= mapData.info.height:
        return None  #out of bounds

    index = my * width + mx
    return index


def is_point_occupied(x, y, mapData):
    index = world_to_map_index(x, y, mapData)
    if index is None:
        return True  #out of bounds = unsafe

    value = mapData.data[index]
    return value == 100  #100 = occupied


#Function for getting the frontier -----------------------------------------------
def get_frontier(res):
    global mapData
    resolution = mapData.info.resolution
    Xstartx = mapData.info.origin.position.x
    Xstarty = mapData.info.origin.position.y

    frontier = copy(res)
    im2, contours, hierarchy = cv2.findContours(frontier,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frontier, contours, -1, (255,255,255), 2)

    im2, contours, hierarchy = cv2.findContours(frontier,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    all_pts = []
    if len(contours) > 0:
    	upto = len(contours) - 1
    	i = 0
    	maxx = 0
    	maxind = 0
		
    	for i in range(0,len(contours)):
                cnt = contours[i]
                M = cv2.moments(cnt)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                xr = cx * resolution + Xstartx
                yr = cy * resolution + Xstarty
                pt = [np.array([xr,yr])]
                if is_point_occupied(xr, yr, mapData) == True:
                    print("Frontiera ", xr, yr, "este obstacol")

                if len(all_pts) > 0:
                    all_pts = np.vstack([all_pts,pt])
                else:
                    all_pts = pt
	
	return all_pts
#--------------------------------------------------------------------------------


#Function for frontier detection ----------------------------
def frontier_detection():
    global mapData

    res = draw_map(mapData)
    all_pts = get_frontier(res)
    #print(all_pts)

    return all_pts

#-------------------------------------------------------------------------------


#Init pose in Rvizz so that you don t use anymore 2D PoseEstimate --------------
def set_initial_pose(x, y, yaw):
    pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=10)
    rospy.sleep(1)

    msg = PoseWithCovarianceStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "map"

    msg.pose.pose.position.x = x
    msg.pose.pose.position.y = y
    msg.pose.pose.position.z = 0.0


    quaternion = tf_conversions.transformations.quaternion_from_euler(0, 0, yaw)
    msg.pose.pose.orientation = Quaternion(*quaternion)

    pub.publish(msg)
    rospy.loginfo("initial pose set in Rviz")
    clear_costmaps()

#-----------------------------------------------------------


def odom_callback(msg):
    global robot_position, setInitPose

    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    w = msg.pose.pose.orientation

    quaternion = msg.pose.pose.orientation
    yaw = tf_conversions.transformations.euler_from_quaternion([
        quaternion.x, quaternion.y, quaternion.z, quaternion.w
    ])[2]

    if setInitPose == False:
        set_initial_pose(x, y, yaw)
        setInitPose = True

    robot_position = (x, y)

    #rospy.signal_shutdown("frontier found")


def main():
    rospy.init_node('frontier_exploration', anonymous=True)

    rospy.Subscriber("/odom", Odometry, odom_callback)
    rospy.Subscriber("/map", OccupancyGrid, mapCallBack)
    rospy.Subscriber("/rosout", Log, log_callback)

    explore()

    rospy.spin()

if __name__ == '__main__':
    main()
    
