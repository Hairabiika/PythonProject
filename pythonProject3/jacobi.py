import numpy as np


def jacobi(A, b, tolerance=1e-10, max_iterations=1):
    x = np.zeros_like(b, dtype=np.double)

    T = A - np.diag(np.diagonal(A))

    for k in range(max_iterations):

        x_old = x.copy()

        x[:] = (b - np.dot(T, x)) / np.diagonal(A)

        if np.linalg.norm(x - x_old, ord=np.inf) / np.linalg.norm(x, ord=np.inf) < tolerance:
            break

    return x
