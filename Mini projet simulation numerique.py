import numpy as np
import scipy.linalg as la
import time
import pandas as pd
import matplotlib.pyplot as plt
import platform
import psutil

# Gaussian Elimination Function
def gaussian_elimination(A, b):
    n = len(b)
    for i in range(n):
        for j in range(i+1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]
    
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
    return x

# Measure Time Function
def measure_time(func, A, b, trials=5):
    times = []
    for _ in range(trials):
        A_copy = A.copy()
        b_copy = b.copy()
        start = time.perf_counter()
        func(A_copy, b_copy)
        end = time.perf_counter()
        times.append(end - start)
    return np.mean(times)

# Main experiment function
def run_experiment(sizes):
    results = []
    
    for n in sizes:
        print(f"Running for size {n}...")
        A = np.random.rand(n, n)
        b = np.random.rand(n)

        # Time the Gaussian Elimination
        gauss_time = measure_time(gaussian_elimination, A, b)
        
        # Time the LU factorization method
        lu_time = measure_time(lambda A, b: la.lu_solve(la.lu_factor(A), b), A, b)
        
        results.append((n, gauss_time, lu_time))

    return results

# Run the experiment for specific matrix sizes
sizes = [100, 400, 500, 700, 1000, 1500, 2000]
experiment_results = run_experiment(sizes)

# Convert results to a DataFrame
df = pd.DataFrame(experiment_results, columns=["Size", "Gauss Time", "LU Time"])

# Display the results
print("Experiment Results:")
print(df)

# Save results to a CSV file
df.to_csv("performance_results.csv", index=False)
print("CSV file saved as 'performance_results.csv'.")

# Plotting the results
plt.plot(df["Size"], df["Gauss Time"], label="Gauss", marker='o')
plt.plot(df["Size"], df["LU Time"], label="LU", marker='s')
plt.xlabel("Matrix Size")
plt.ylabel("Execution Time (s)")
plt.legend()
plt.title("Performance of Gauss vs LU")
plt.grid()

# Save the plot as an image
plt.savefig('output_plot.png')
print("Plot saved as 'output_plot.png'.")

# Show the plot
plt.show()

# Print system specifications
system_info = {
    "OS": platform.system() + " " + platform.release(),
    "Processor": platform.processor(),
    "RAM": str(round(psutil.virtual_memory().total / (1024 ** 3), 2)) + " GB",
    "Python Version": platform.python_version()
}

print("System Specifications:")
for key, value in system_info.items():
    print(f"{key}: {value}")
