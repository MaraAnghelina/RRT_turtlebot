import numpy as np
from ccw import ccw

# def chk_collision(line, poly) -> bool:
#   result = False

#   shape_count = len(poly)

#   for i in range(shape_count):
#     crnt_poly = poly[i]
#     poly_size = crnt_poly.shape

#     # polygon should consist of at least 3 points 
#     #
#     # if poly_size(1) <= 2:
#     #  result = -2
#     #  return         
#     #

#     # Check if line segment returns to the initial point.
#     # If not, add the initial point as the ending point to encircle the polygon.
#     #
#     # if not np.array_equal(crnt_poly[-1, :], crnt_poly[0, :]):
#     #   crnt_poly = np.vstack((crnt_poly, crnt_poly[0, :]))
#     #   poly_size = crnt_poly.shape

#     seg_count = poly_size[0] - 1

#     dA = line[1, :] - line[0, :] 
#     dB = crnt_poly[1:poly_size[0], :] - crnt_poly[0:poly_size[0]-1, :]
#     dA1B1 = np.tile(line[0, :], (seg_count, 1)) - crnt_poly[0:seg_count, :]
#     denominator = dB[:, 1] * dA[0] - dB[:, 0] * dA[1]

#     if np.all(denominator == 0):
#       result = False
#       return result

#     ua = (dB[:, 0] *dA1B1[:, 1] - dB[:, 1] * dA1B1[:, 0]) / denominator
#     ub = (dA1B1[:, 1] * dA[0] - dA1B1[:, 0] *dA[1]) / denominator

#     if np.all((ua < 0) | (ua > 1) | (ub < 0) | (ub > 1)):
#       result = False
#     else: 
#       result = True
#       return result

#   return result

def chk_collision(line, poly):
  line = np.array(line)

  for obstacle in poly:
    for i in range(len(obstacle)):
      C = obstacle[i]
      D = obstacle[(i+1) % len(obstacle)]
      if ccw(line[0], C, D) != ccw(line[1], C, D) and ccw(line[0], line[1], C) != ccw(line[0], line[1], D):
        return 1
  return 0