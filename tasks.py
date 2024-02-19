# import utils as u
import numpy as np


A = np.array([
    [0,0,0,0,1],
    [0,0,0,2,0],
    [0,0,3,0,0],
    [0,4,0,0,0],
    [5,0,0,0,0]
])

# P, Q, L, U = u.lu(A, "no pivoting")

# print('L:', np.array(L),'\n\n', 'U:', np.array(U), '\n\n', 'P_r:', np.array(P), '\n\n', 'P_c:', np.array(Q))

def LUdecomp(a):
    n = len(a)
    for k in range(0,n-1):
        for i in range(k+1,n):
            if a[i,k] != 0.0:
                lam = a [i,k]/a[k,k]
                a[i,k+1:n] = a[i,k+1:n] - lam*a[k,k+1:n]
                a[i,k] = lam
    return a

print(LUdecomp(A))
















