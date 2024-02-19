import numpy as np
import random


def generate_matrix(n):
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            if i == j:
                L[i][j] = 1.0
            else:
                L[i][j] = random.uniform(0.1, 1.0)
    for i in range(n):
        for j in range(i, n):
            if i == j:
                U[i][j] = random.uniform(1.0, 10.0)
            else:
                U[i][j] = random.uniform(0.1, 1.0)
    A = [[sum(L[i][k] * U[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
    return A


def lu(A, selection):
    if selection == "no pivoting":
        L, U = lu_basic(A)
        P = [i for i in range(len(A))]
        Q = [i for i in range(len(A))]
        return P, Q, L, U
    elif selection == "partial pivoting":
        Q = [i for i in range(len(A))]
        P, L, U = lu_pp(A)
        return P, Q, L, U
    elif selection == "complete pivoting":
        P, Q, L, U = lu_cp(A)
        return P, Q, L, U
    else:
        raise ValueError("Invalid input.")


def lu_basic(A):
    n = len(A)
    L = [[0.0 for _ in range(n)] for _ in range(n)]
    U = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for k in range(i, n):
            sum_upper = sum(L[i][j] * U[j][k] for j in range(i))
            U[i][k] = A[i][k] - sum_upper
        for k in range(i, n):
            if i == k:
                L[i][i] = 1
            else:
                sum_lower = sum(L[k][j] * U[j][i] for j in range(i))
                if U[i][i] == 0:
                    L[k][i] = 0
                else:
                    L[k][i] = (A[k][i] - sum_lower) / U[i][i]            
    return L, U


def lu_pp(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    P = [i for i in range(n)]
    for i in range(n):
        pivot_index = max(range(i, n), key=lambda x: abs(A[x][i]))
        if A[pivot_index][i] == 0:
            print("Partial pivoting failed. Zero pivot encountered.")
            return None, None, None
        A[i], A[pivot_index] = A[pivot_index], A[i]
        P[i], P[pivot_index] = P[pivot_index], P[i]
        L[i][i] = 1.0
        for j in range(i, n):
            U[i][j] = A[i][j] - sum(L[i][k] * U[k][j] for k in range(i))
        for j in range(i+1, n):
            L[j][i] = (A[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i][i]
    return P, L, U
    


def swap_rows(matrix, i, j):
    matrix[i], matrix[j] = matrix[j], matrix[i]

def swap_cols(matrix, i, j):
    for row in matrix:
        row[i], row[j] = row[j], row[i]

def find_max_pivot(matrix, start):
    max_val = 0
    pivot = (start, start)
    for i in range(start, len(matrix)):
        for j in range(start, len(matrix[0])):
            if abs(matrix[i][j]) > max_val:
                max_val, pivot = abs(matrix[i][j]), (i, j)
    return pivot

def lu_cp(A):
    n = len(A)
    L = [[0 if i != j else 1 for j in range(n)] for i in range(n)]
    U = [[0 for _ in range(n)] for _ in range(n)]
    P = [i for i in range(n)]
    Q = [i for i in range(n)]

    for k in range(n):
        pivot_row, pivot_col = find_max_pivot(A, k)
        swap_rows(A, k, pivot_row)
        swap_cols(A, k, pivot_col)
        P[k], P[pivot_row] = P[pivot_row], P[k]
        Q[k], Q[pivot_col] = Q[pivot_col], Q[k]

        for i in range(k, n):
            sum_upper = sum(L[k][s] * U[s][i] for s in range(k))
            U[k][i] = A[k][i] - sum_upper

        for i in range(k + 1, n):
            sum_lower = sum(L[i][s] * U[s][k] for s in range(k))
            L[i][k] = (A[i][k] - sum_lower) / U[k][k]

    return P, Q, L, U


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

