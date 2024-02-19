import utils as u
import numpy as np


A = [
    [2,0,0,0,0],
    [3,2,0,0,0],
    [4,3,2,0,0],
    [5,4,0,2,0],
    [6,5,4,3,2]
]

P, Q, L, U = u.lu(A, "partial pivoting")

print('L:', np.array(L),'\n\n', 'U:', np.array(U), '\n\n', 'P_r:', np.array(P), '\n\n', 'P_c:', np.array(Q))

















