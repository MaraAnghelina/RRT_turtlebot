import tf2_ros
import tf_conversions
import rospy
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import math  

from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Quaternion, PointStamped
from std_srvs.srv import Empty 
from visualization_msgs.msg import Marker
from rosgraph_msgs.msg import Log
from geometry_msgs.msg import Twist
from copy import copy

from set_init_pose import set_initial_pose, euler_from_quaternion
from get_frontier import get_closest_frontier, get_furthest_frontier, is_point_occupied, get_closest_frontier_failed, get_furthest_frontier_failed

TIME_LIMIT = 15

robot_position = (0, 0)
goalReached = False 
setInitPose = False
plannerTrigger = False
lastFrontier = None
lastFrontierTime = time.time()
failed_frontiers = []
    

# Subscribers' callbacks------------------------------
mapData = OccupancyGrid()

def mapCallBack(data):
    global mapData
    mapData = data


def log_callback(msg):
    global goalReached, plannerTrigger
    

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
    global goalReached, mapData, robot_position, lastFrontierTime, failed_frontiers

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
    lastFrontierTime = time.time()

    if test(frontier):
        frontier = get_closest_frontier_failed(robot_position, mapData, frontier)
        cmd.pose.position.x = frontier[0] 
        cmd.pose.position.y = frontier[1]

    velPub.publish(cmd)

    while not test(frontier):
        print("Asteptam sa ajungem la obiectiv...") 
        print(test(frontier))
        rospy.sleep(0.5)  

        if is_point_occupied(frontier[0], frontier[1], mapData):
            print("Frontier is in object")
            rospy.sleep(0.5)

            failed_frontiers = frontier

            frontier = get_furthest_frontier_failed(robot_position, mapData, frontier)
            cmd.pose.position.x = frontier[0] 
            cmd.pose.position.y = frontier[1]

            lastFrontierTime = time.time()

        #if time.time() - lastFrontierTime > TIME_LIMIT:
         #   print("More than  15sec")
          #  rospy.sleep(0.5)

#            frontier = get_furthest_frontier(robot_position, mapData)
#            cmd.pose.position.x = frontier[0]
#            cmd.pose.position.y = frontier[1]

#            lastFrontierTime = time.time()

        velPub.publish(cmd)    

    rospy.sleep(1)
    #print("Start rotating")
    #rotate_360()
    #rospy.sleep(1)
    
#-----------------------------------------------------------------------------



#EXploration function -------------------------------------------------------
def explore():
    global mapData, robot_position, lastFrontier, lastFrontierTime

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

    rotate_360()

    while not rospy.is_shutdown():

        frontier = get_furthest_frontier(robot_position, mapData)

        if lastFrontier is None:
            lastFrontier = frontier
            lastFrontierTime = time.time()

        exploration_goal.header.frame_id = mapData.header.frame_id
        exploration_goal.header.stamp = rospy.Time.now()
        exploration_goal.point.x = frontier[0]
        exploration_goal.point.y = frontier[1]
        exploration_goal.point.z = 0

        targetspub.publish(exploration_goal)
        points.points = [exploration_goal.point]
        pub.publish(points)
        rospy.sleep(1)

        #adjusted_goal = shift_frontier_towards_robot(frontier, robot_position, offset=0.3)
        
        go_to_point(frontier)

#---------------------------------------------------------------------------------

def shift_frontier_towards_robot(frontier_point, robot_pos, offset=0.3):
    global mapData

    vector = np.array(frontier_point) - np.array(robot_pos)
    dist = np.linalg.norm(vector)
    if dist == 0:
        return frontier_point  # avoid division by zero

    # Normalize vector
    direction = vector / dist

    # New target is offset meters closer to robot along that vector
    new_target = np.array(frontier_point) - direction * offset

    if is_point_occupied(new_target[0], new_target[1], mapData):
        print("OBIECT")

    return new_target.tolist()


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

def rotate_360():
    
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    twist = Twist()

    #Rotate in place (angular z)
    twist.angular.z = 0.5  #radians per second (positive = left)
    duration = 2 * 3.14159 / twist.angular.z  #12.57 seconds for full 360

    rate = rospy.Rate(10)
    start_time = rospy.Time.now().to_sec()

    while rospy.Time.now().to_sec() - start_time < duration:
        pub.publish(twist)
        rospy.sleep(0.5)

    # Stop rotation
    twist.angular.z = 0.0
    pub.publish(twist)
    print("Rotation complete.")
    rospy.sleep(1)


def main():
    rospy.init_node('frontier_exploration', anonymous=True)

    rospy.Subscriber("/odom", Odometry, odom_callback)
    rospy.Subscriber("/map", OccupancyGrid, mapCallBack)
    rospy.Subscriber("/rosout", Log, log_callback)

    explore()

    rospy.spin()

if __name__ == '__main__':
    main()
    