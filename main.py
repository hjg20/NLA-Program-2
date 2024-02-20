import utils as u
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

sizes = [10, 25, 50, 100]
iterations = 10                                                                                                                        
total_acc = []
total_gf = []
total_times = []
selection = input("Select factorization method: no pivoting, partial pivoting, or complete pivoting. ")
for i in sizes:
    accuracies = []
    gf = []
    times = []
    for _ in range(iterations):
        A = u.generate_matrix(i)
        start_time = time.time()
        P, Q, L, U = u.lu(A, selection)
        end_time = time.time()
        error = u.factorization_accuracy(A, L, U, P)
        gamma = u.growth_factor(L, U, A)
        accuracies.append(error)
        gf.append(gamma)
        times.append(end_time - start_time)
    total_acc.append(sum(accuracies) / len(accuracies))
    total_gf.append(sum(gf) / len(gf))
    total_times.append(sum(times) / len(times))

df = pd.DataFrame()
df['Matrix Sizes'] = sizes
df['Accuracies'] = total_acc
df['Growth Factors'] = total_gf
df['Times'] = total_times

print(df)
