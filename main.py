import utils as u
import numpy as np
import matplotlib.pyplot as plt
import time

sizes = [10, 25, 50]
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


plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.plot(sizes, total_acc)
plt.xlabel('Matrix sizes')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Matrix sizes')

plt.subplot(1, 3, 2)
plt.plot(sizes, total_gf)
plt.xlabel('Matrix sizes')
plt.ylabel('Growth Factor')
plt.title('Growth Factor vs. Matrix sizes')

plt.subplot(1, 3, 3)
plt.plot(sizes, total_times)
plt.xlabel('Matrix sizes')
plt.ylabel('Time')
plt.title('Time vs. Matrix sizes')

plt.tight_layout()
plt.show()
