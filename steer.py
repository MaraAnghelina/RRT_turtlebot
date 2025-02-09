from typing import Tuple
import dist

def steer(qr, qn, val, eps) -> Tuple[int, int]:
  qnew = [0, 0]

  # Steer towards qn with maximum step size of eps
  if val >= eps:
    qnew[0] = qn[0] + ((qr[0] - qn[0]) * eps) / dist.dist(qr, qn)
    qnew[1] = qn[1] + ((qr[1] - qn[1]) * eps) / dist.dist(qr, qn)
  else:
    qnew[0] = qr[0]
    qnew[1] = qr[1]
    
  return qnew