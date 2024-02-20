import utils as u
import numpy as np

A = [
    [1,0,0,1],
    [-1,1,0,1],
    [-1,-1,1,1],
    [-1,-1,-1,1]
]

P, Q, L, U = u.lu(A, "complete pivoting")


print('L:', np.array(L),'\n\n', 'U:', np.array(U), '\n\n', 'P_r:', np.array(P), 
      '\n\n', 'P_c:', np.array(Q), '\n\n')

print(np.array(L)@np.array(U))

print(u.growth_factor(L, U, A))

