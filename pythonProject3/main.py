from scipy.optimize import minimize, rosen
import numpy as np

import gauss
import jacobi

A = np.array([[15, -1], [28, 44]])
b = np.array([5, 13])
g = gauss.gauss_seidel(A, b, tolerance=1e-10, max_iterations=4)
f = jacobi.jacobi(A, b, tolerance=1e-10, max_iterations=2)
x0 = np.array([13.8,21.7])
sol = minimize(rosen,x0,method='Nelder-Mead')
print(sol.x)
t = sol.x
for x in t:
    print(14*x*x+4*x+56)
print(f)
for x in f:
    print(x*x+2*x+7)
print(g)
for x in g:
    print(x*x-63*x+413)
