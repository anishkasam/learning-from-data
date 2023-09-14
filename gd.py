import numpy as np

# error surface
def E(x):
    u = x[0]
    v = x[1]
    return (u * np.exp(v) - 2 * v * np.exp(-u)) ** 2

# partial derivative with respect to u, partial derivative with respect to v
def grad(x):
    u = x[0]
    v = x[1]
    return np.array([
        2 * (u * np.exp(v) - 2 * v * np.exp(-u)) * (np.exp(v) + 2 * v * np.exp(-u)),
        2 * (u * np.exp(v) - 2 * v * np.exp(-u)) * (u * np.exp(v) - 2 * np.exp(-u))
    ])

# gradient functions for coordinate descent
def gradu(x):
    u = x[0]
    v = x[1]
    return np.array([2 * (u * np.exp(v) - 2 * v * np.exp(-u)) * (np.exp(v) + 2 * v * np.exp(-u)), 0])

def gradv(x):
    u = x[0]
    v = x[1]
    return np.array([0, 2 * (u * np.exp(v) - 2 * v * np.exp(-u)) * (u * np.exp(v) - 2 * np.exp(-u))])

# gradient descent
def gd(limit, lr):
    x = np.array([1, 1])
    error = E(x)
    iterations = 0

    while error > limit:
        iterations += 1
        gradient = grad(x)
        x = x - gradient * lr
        error = E(x)

        if iterations > 100:
            print("failed")
            break

    return iterations, x

# coordinate descent
def cd(iter_limit, lr):
    x = np.array([1, 1])
    iterations = 0

    while iterations < iter_limit:
        iterations += 1
        gradient = gradu(x)
        x = x - gradient * lr
        gradient = gradv(x)
        x = x - gradient * lr

    return E(x)

print(gd(1e-14, 0.1))
# (10, array([0.04473629, 0.02395871]))
print(cd(15, 0.1))
# 0.13981379199615315