import utils as u


A = [
    [0,0,0,0,1],
    [0,0,0,2,0],
    [0,0,3,0,0],
    [0,4,0,0,0],
    [5,0,0,0,0]
]

L, U, P = u.lu(A, "complete pivoting")

print(u.matrix_multiplication(L, U, 'LU'))
print(u.apply_permutation_matrix(P, A))
print(L)
'hi'

















