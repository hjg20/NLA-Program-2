import numpy as np
import random


def is_full_rank(matrix):
    if len(matrix) > len(matrix[0]):
        return all(any(row) for row in zip(*matrix))
    else:
        return all(any(row) for row in matrix)

def generate_matrix(n):
    matrix = []
    for i in range(n):
        while True:
            new_row = [random.randint(1, 10) for _ in range(n)]
            matrix.append(new_row)
            if is_full_rank(matrix):
                break 
            else:
                matrix.pop()

    return matrix


def lu(A, selection):
    if selection == "no pivoting":
        L, U = lu_basic(A)
        P = [i for i in range(len(A))]
        return L, U, P
    elif selection == "partial pivoting":
        L, U = lu_pp(A)
        P = [i for i in range(len(A))]
        return L, U, P
    elif selection == "complete pivoting":
        L, U, P = lu_cpp(A)
        return L, U, P
    else:
        raise ValueError("Invalid input.")


def lu_basic(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    for i in range(n):
        L[i][i] = 1.0
        for j in range(i, n):
            sum_upper = sum(L[i][k] * U[k][j] for k in range(i))
            U[i][j] = A[i][j] - sum_upper
        for j in range(i+1, n):
            sum_lower = sum(L[j][k] * U[k][i] for k in range(i))
            L[j][i] = (A[j][i] - sum_lower) / U[i][i]
    return L, U


def lu_pp(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    P = list(range(n))
    for i in range(n):
        pivot_value = 0
        pivot_row = i
        for row in range(i, n):
            if abs(A[P[row]][i]) > pivot_value:
                pivot_value = abs(A[P[row]][i])
                pivot_row = row
        P[i], P[pivot_row] = P[pivot_row], P[i]
        for j in range(i, n):
            sum_upper = sum(L[i][k] * U[k][j] for k in range(i))
            U[i][j] = A[P[i]][j] - sum_upper
        for j in range(i + 1, n):
            sum_lower = sum(L[j][k] * U[k][i] for k in range(i))
            L[j][i] = (A[P[j]][i] - sum_lower) / U[i][i]
    for i in range(n):
        L[i][i] = 1.0
    return L, U


def lu_cpp(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    P = list(range(n))
    for i in range(n):
        max_index = max(((k, j) for k in range(i, n) for j in range(i, n)), key=lambda x: abs(A[x[0]][x[1]]))
        if A[max_index[0]][max_index[1]] == 0:
            print("Complete pivoting failed. Zero pivot encountered.")
            return None, None, None
        A[i], A[max_index[0]] = A[max_index[0]], A[i]
        for k in range(n):
            A[k][i], A[k][max_index[1]] = A[k][max_index[1]], A[k][i]
        P[i], P[max_index[0]] = P[max_index[0]], P[i]
        L[i][i] = 1.0
        for j in range(i, n):
            U[i][j] = A[i][j] - sum(L[i][k] * U[k][j] for k in range(i))
        for j in range(i+1, n):
            L[j][i] = (A[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i][i]
    return L, U, P


def apply_permutation_matrix(P, A):
    n = len(A)
    result = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            result[i][j] = A[P[i]][j]
    return result


def subtract_matrices(A, B):
    n = len(A)
    result = [[A[i][j] - B[i][j] for j in range(n)] for i in range(n)]
    return result


def matrix_multiplication(L, U, product_type):
    n = len(L)
    result = [[0.0] * n for _ in range(n)]
    if product_type == 'LU':
        for i in range(n):
            for j in range(n):
                result[i][j] = sum(L[i][k] * U[k][j] for k in range(n))
    elif product_type == '|LU|':
        for i in range(n):
            for j in range(n):
                result[i][j] = L[i][j] * U[j][i]
    return result


def matrix_norm(W, norm_type):
    n = len(W)
    if norm_type == '1':
        return max(sum(abs(W[j][i]) for j in range(n)) for i in range(n))
    elif norm_type == 'inf':
        return max(sum(abs(W[i][j]) for j in range(n)) for i in range(n))
    elif norm_type == 'F':
        return sum(sum(W[i][j] ** 2 for j in range(n)) for i in range(n)) ** 0.5
    else:
        raise ValueError("Invalid norm type. Choose from '1', 'inf', or 'F'")


def factorization_accuracy(A, L, U, P):
    PA = apply_permutation_matrix(P, A)
    LU = matrix_multiplication(L, U, 'LU')
    error = matrix_norm(subtract_matrices(PA, LU), 'F') / max(1, matrix_norm(A, 'F'))
    return error


def growth_factor(L, U, A):
    LU_norm = matrix_norm(matrix_multiplication(L, U, 'LU'), 'F')
    A_norm = matrix_norm(A, 'F')
    return LU_norm / max(1, A_norm)

