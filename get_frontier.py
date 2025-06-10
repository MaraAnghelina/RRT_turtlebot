import tf2_ros
import tf_conversions
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math  
import subprocess
import rospy

from copy import copy
from random import randrange

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


#Function for getting the frontier -----------------------------------------------
def get_frontiers(res, mapData):
    resolution = mapData.info.resolution
    Xstartx = mapData.info.origin.position.x
    Xstarty = mapData.info.origin.position.y

    frontier = copy(res)
    im2, contours, hierarchy = cv2.findContours(frontier,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frontier, contours, -1, (255,255,255), 5)

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
                if is_point_occupied(xr, yr, mapData):
                    print("Frontier", xr, yr, "is in obstacle, skipping.")
                    continue  #skip this point

                #Additional safety check: test a small surrounding area
                safe = True
                for dx in np.linspace(-0.2, 0.2, 5):
                    for dy in np.linspace(-0.2, 0.2, 5):
                        if is_point_occupied(xr + dx, yr + dy, mapData):
                            safe = False
                            break
                    if not safe:
                        break

                if not safe:
                    print("Frontier", xr, yr, "is too close to obstacle, skipping.")
                    continue

                #Safe point add it
                if len(all_pts) > 0:
                    all_pts = np.vstack([all_pts, pt])
                else:
                    all_pts = pt
	
	return all_pts
#--------------------------------------------------------------------------------


#Function for frontier detection ----------------------------
def frontier_detection(mapData):

    res = draw_map(mapData)
    all_pts = get_frontiers(res, mapData)
    #print(all_pts)

    return all_pts

#-------------------------------------------------------------------------------

def finish_exploring():
    save_map('/home/internship/explored_map')
    rospy.sleep(2)
    rospy.loginfo("Explored all map")
    rospy.signal_shutdown("Explored all map")

#Fuction for deciding the closest frontier ---------------------------------------------------
def get_closest_frontier(robot_position, mapData):

    frontiers = frontier_detection(mapData)

    rx = robot_position[0]
    ry = robot_position[1]
    print(robot_position)

    #Compute distances to all frontiers
    if frontiers is None:
        finish_exploring()
        exit()
    #dists = [np.linalg.norm(np.array([rx, ry]) - np.array([pt[0], pt[1]])) for pt in frontiers]
    dists = [math.sqrt((rx - pt[0])**2 + (ry - pt[1])**2) for pt in frontiers]
    print("DISTS: ", dists)

    if not dists:
        finish_exploring()
        exit()

    #Choose the closest
    min_idx = np.argmin(dists)
    closestFrontier = frontiers[min_idx]
    print("Closest frontier: ", closestFrontier)

    return closestFrontier


def get_closest_frontier_failed(robot_position, mapData, failedFrontier):
    frontiers = frontier_detection(mapData)

    rx = robot_position[0]
    ry = robot_position[1]

    x = np.array(failedFrontier)
    filtered = np.array([pt for pt in frontiers if not np.array_equal(pt, x)])

    #Compute distances to all frontiers
    if filtered is None:
        finish_exploring()
        exit()
    #dists = [np.linalg.norm(np.array([rx, ry]) - np.array([pt[0], pt[1]])) for pt in frontiers] 

    dists = [math.sqrt((rx - pt[0])**2 + (ry - pt[1])**2) for pt in filtered]
    print("DISTS: ", dists)

    if not dists:
        finish_exploring()
        exit()

    #Choose the closest
    min_idx = np.argmin(dists)
    closestFrontier = frontiers[min_idx]
    print("Closest frontier: ", closestFrontier)

    return closestFrontier

#------------------------------------------------------------------------------------------------------------

#Fuction for deciding the closest frontier ---------------------------------------------------
def get_furthest_frontier(robot_position, mapData):

    frontiers = frontier_detection(mapData)

    rx = robot_position[0]
    ry = robot_position[1]
    print(robot_position)

    #Compute distances to all frontiers
    if frontiers is None:
        finish_exploring()
        exit()
    #dists = [np.linalg.norm(np.array([rx, ry]) - np.array([pt[0], pt[1]])) for pt in frontiers]
    dists = [math.sqrt((rx - pt[0])**2 + (ry - pt[1])**2) for pt in frontiers]
    print("DISTS: ", dists)

    if not dists:
        finish_exploring()
        exit()

    #Choose the closest
    max_idx = np.argmax(dists)
    furthestFrontier = frontiers[max_idx]
    print("Furthest frontier: ", furthestFrontier)

    return furthestFrontier

def get_furthest_frontier_failed(robot_position, mapData, failedFrontier):
    frontiers = frontier_detection(mapData)

    rx = robot_position[0]
    ry = robot_position[1]

    x = np.array(failedFrontier)
    filtered = np.array([pt for pt in frontiers if not np.array_equal(pt, x)])

    #Compute distances to all frontiers
    if filtered is None:
        finish_exploring()
        exit()
    #dists = [np.linalg.norm(np.array([rx, ry]) - np.array([pt[0], pt[1]])) for pt in frontiers] 

    dists = [math.sqrt((rx - pt[0])**2 + (ry - pt[1])**2) for pt in filtered]
    print("DISTS: ", dists)

    if not dists:
        finish_exploring()
        exit()

    #Choose the closest
    max_idx = np.argmax(dists)
    furthestFrontier = frontiers[max_idx]
    print("Furthest frontier: ", furthestFrontier)

    return furthestFrontier

#------------------------------------------------------------------------------------------------------------

#Function for deciding the random frontier ------------------------------------------------------------------
def get_random_frontier(robot_position, mapData):

    frontiers = frontier_detection(mapData)

    rx = robot_position[0]
    ry = robot_position[1]
    print(robot_position)

    if frontiers is None:
        finish_exploring()
        exit()
    

    #Choose the random frontier
    idx = randrange(0, len(frontiers))
    randomFrontier = frontiers[idx]
    print("Random frontier: ", randomFrontier)

    return randomFrontier


#------------------------------------------------------------------------------------------------------------

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


#See if the frontier is a object or not ---------------------------------------------------------------------
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
    #print(value)
    return value == 100  #100 = occupied

#--------------------------------------------------------------------------------------------------

