from ccw import ccw

def no_collision(n2, n1, o):
  A = [n1[0], n1[1]]
  B = [n2[0], n2[1]]
  obs = [o[0], o[1], o[0] + o[2], o[1] + o[3]]
  
  C1 = [obs[0], obs[1]]
  D1 = [obs[0], obs[3]]
  C2 = [obs[0], obs[1]]
  D2 = [obs[2], obs[1]]
  C3 = [obs[2], obs[3]]
  D3 = [obs[2], obs[1]]
  C4 = [obs[2], obs[3]]
  D4 = [obs[0], obs[3]]

  # Check if the path from n1 to n2 intersects any of the four edges of the obstacle
  ints1 = ccw(A, C1, D1) != ccw(B, C1, D1) and ccw(A, B, C1) != ccw(A, B, D1) 
  ints2 = ccw(A, C2, D2) != ccw(B, C2, D2) and ccw(A, B, C2) != ccw(A, B, D2)
  ints3 = ccw(A, C3, D3) != ccw(B, C3, D3) and ccw(A, B, C3) != ccw(A, B, D3)
  ints4 = ccw(A, C4, D4) != ccw(B, C4, D4) and ccw(A, B, C4) != ccw(A, B, D4)

  if not ints1 and not ints2 and not ints3 and not ints4:
    return True 
  else:
    return False
