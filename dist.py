import numpy as np

def dist(q1, q2) -> float:
  return np.sqrt((q1[0] - q2[0])**2 +(q1[1] - q2[1])**2)
