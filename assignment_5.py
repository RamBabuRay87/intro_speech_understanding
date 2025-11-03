import numpy as np
import matplotlib.pyplot as plt

# User input
nums = input("Enter numbers separated by spaces: ").split()
x = np.array([float(n) for n in nums])

# Center of gravity
c = np.dot(np.arange(len(x)), x) / np.sum(x)
print("Center of gravity:", c)

# Identity matrix
I = np.eye(len(x))
print("Identity matrix:\n", I)

# Sine and cosine plot
t_start = float(input("Start (e.g. 0): "))
t_end = float(input("End (e.g. 6.28): "))
t_steps = int(input("Steps (e.g. 100): "))

t = np.linspace(t_start, t_end, t_steps)
plt.plot(t, np.cos(t), label='cos(t)')
plt.plot(t, np.sin(t), label='sin(t)')
plt.legend()
plt.show()



