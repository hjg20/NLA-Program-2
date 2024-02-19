import utils as u
import numpy as np
import matplotlib.pyplot as plt
import time

problem_sizes = [10, 25, 50]
num_samples = 10
results = []

selection = input("Select factorization method: no pivoting, partial pivoting, or complete pivoting. ")

for i in problem_sizes:
    accuracies = []
    growth_factors = []
    execution_times = []

    for _ in range(num_samples):
        A = u.generate_matrix(i)
        start_time = time.time()
        L, U, P = u.lu(A, selection)
        end_time = time.time()
        error = u.factorization_accuracy(A, L, U, P)
        gamma = u.growth_factor(L, U, A)

        accuracies.append(error)
        growth_factors.append(gamma)
        execution_times.append(end_time - start_time)

    avg_accuracy = sum(accuracies) / len(accuracies)
    avg_growth_factor = sum(growth_factors) / len(growth_factors)
    avg_execution_time = sum(execution_times) / len(execution_times)

    results.append({
        'n': i,
        'avg_accuracy': avg_accuracy,
        'avg_growth_factor': avg_growth_factor,
        'avg_execution_time': avg_execution_time

    })
    print(f"n = {i} done")

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.plot([r['n'] for r in results], [r['avg_accuracy'] for r in results], marker='o')
plt.xlabel('Problem Size (n)')
plt.ylabel('Average Factorization Accuracy')
plt.title('Factorization Accuracy vs. Problem Size')

plt.subplot(1, 3, 2)
plt.plot([r['n'] for r in results], [r['avg_growth_factor'] for r in results], marker='o')
plt.xlabel('Problem Size (n)')
plt.ylabel('Average Growth Factor')
plt.title('Growth Factor vs. Problem Size')

plt.subplot(1, 3, 3)
plt.plot([r['n'] for r in results], [r['avg_execution_time'] for r in results], marker='o')
plt.xlabel('Problem Size (n)')
plt.ylabel('Average Execution Time (s)')
plt.title('Execution Time vs. Problem Size')

plt.tight_layout()
plt.show()
