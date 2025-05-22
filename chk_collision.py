import cv2
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy.spatial import ConvexHull

#Helper function to compute the orientation of three points
def orientation(p, q, r):
    #Cross product of vectors pq and pr
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

#Helper function to check if point q lies on the segment pr
def on_segment(p, q, r):
    if min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1]):
        return True
    return False

#Function to check if two line segments (p1q1 and p2q2) intersect
def do_intersect(p1, q1, p2, q2):
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    #General case: if the orientations are different, the segments intersect
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True

    # Special case: when the points are collinear, we check if they lie on the segment
    if o1 == 0 and on_segment(p1, p2, q1): return True
    if o2 == 0 and on_segment(p1, q2, q1): return True
    if o3 == 0 and on_segment(p2, p1, q2): return True
    if o4 == 0 and on_segment(p2, q1, q2): return True

    return False

#Function to check if a segment intersects a convex polygon
def segment_intersects_convex_polygon(segment, polygon):
    #Segment is defined by two points
    p1, q1 = segment

    #Loop over each edge of the convex polygon
    n = len(polygon)
    for i in range(n):
        p2 = polygon[i]
        p3 = polygon[(i + 1) % n]  # Next point (wrapping around)

        #Check if the segment intersects with the current edge of the polygon
        if do_intersect(p1, q1, p2, p3):
            return True

    return False

def chk_collision(line, poly):
  line = np.array(line)

  for obstacle in poly:
    for i in range(len(obstacle)):
      C = obstacle[i]
      D = obstacle[(i+1) % len(obstacle)]
      if ccw(line[0], C, D) != ccw(line[1], C, D) and ccw(line[0], line[1], C) != ccw(line[0], line[1], D):
        return 1
  return 0


def chk_collision_rrt(segment, hull_list):
    for hull in hull_list:
        if segment_intersects_convex_polygon(segment, hull):
            return 1  #collision
    return 0  #no collison


def map_hull_points(img_path):
  #Load the image
  image = cv2.imread(img_path, -1)

  #apply threshold to decide which bits are obstacles and which not
  _, image    = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
  image = cv2.bitwise_not(image)

  #%% in general, a good idea to play with erosion and dilation (depends on specific image; careful how you choose the kernel)
  kernel = np.ones((2, 2), np.uint8)  # a square kernel
  #Perform dilation (enlargement)
  image = cv2.dilate(image, kernel, iterations=5)
  #Perform erosion (shrinking)
  image = cv2.erode(image, kernel, iterations=2)
  image = cv2.bitwise_not(image)

  # Display the output image
  #plt.figure(figsize=(8, 8))
  #plt.imshow(image, cmap='gray')
  #plt.title('Filtered Image')
  #plt.axis('off')
  #plt.show()

  #%% opencv magic to detect blobs; many parameters that I don't really understand...

  #Setup the blob detector parameters 
  params = cv2.SimpleBlobDetector_Params()    
  params.filterByArea = True
  params.minArea = 5
  params.filterByCircularity = False
  params.filterByConvexity = False
  params.filterByInertia = False

  #Create the blob detector
  detector = cv2.SimpleBlobDetector_create(params)

  #Detect blobs <== contours are not selected; not cleat if walls are identified
  keypoints = detector.detect(image)

  #Draw detected blobs on the image
  im_with_keypoints = cv2.drawKeypoints(image, keypoints, None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

  #Display the output image
  #plt.figure(figsize=(8, 8))
  #plt.imshow(im_with_keypoints, cmap='gray')
  #plt.title('Blobs with Keypoints')
  #plt.axis('off')
  #plt.show()


  #%% use scipy.ConvexHull to get the points that define the 
  #convex region containing a blob; not sure if this is 
  #really necessaary or f we can use directly the boundary pixels of the blob

        
  #fig = plt.figure(figsize=(8, 8))

  #a polygon is defined as either A*x <=b or as convex hull of vertices stored in V; I save both
  A = []
  b = []
  V = []
  #Iterate over each detected keypoint (blob) and get pixel positions
  for keypoint in keypoints:
    #Get the coordinates of the blob center
    center = keypoint.pt  # (x, y)
    radius = keypoint.size * 0.75  # Approximate radius of the blob <== from chatGPT, why is it necessaary? I suspect that I am not using correctly the mask/color
    
    #Create a circular mask around the blob center to extract the blob pixels
    mask = np.zeros_like(image, dtype=np.uint8)
    cv2.circle(mask, (int(center[0]), int(center[1])), int(radius), 255, -1)

    #Extract the blob pixels using the mask
    blob_pixels = cv2.bitwise_and(image, image, mask=mask)
    
    #Get the actual pixel positions of the blob
    blob_positions = np.column_stack(np.where(blob_pixels == 255))  # Get all non-zero (white) pixels
    
    #Compute the convex hull using SciPy
    if len(blob_positions) >= 3:  #Convex hull requires at least 3 points
        hull = ConvexHull(blob_positions)
        hull_points = blob_positions[hull.vertices]  #Get the points forming the convex hull
        hull_points = hull_points[:, [1, 0]] # <== necessary because opencv works in a 90 degrees rotated frame with respect to matplotlib... 

        polygon = patches.Polygon(hull_points, closed=True, facecolor='red', edgecolor='black', alpha=0.5)
        plt.gca().add_patch(polygon)  #Add the polygon patch to the current axes
    
        A.append(hull.equations[:, :-1])  #All columns except the last one (normal vectors)
        b.append(-hull.equations[:, -1])  #Last column (constants), negate it to match inequality form
        V.append(hull_points)

  #Show the image with convex hulls
  #plt.imshow(image, cmap='gray')
  #plt.title('Convex Hulls of Detected Blobs')
  #plt.axis('off')  # Hide axes for better visualization

  #plt.show()

  return V