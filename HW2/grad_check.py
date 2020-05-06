import numpy as np
from scipy.optimize import approx_fprime

"""
This short script demonstrates the usage of scipy.optimize.approx_fprime for 
a scalar function.
It should be used to numerically check gradients.

In this case, it is implement for a simple input function 
f(x)=\mathbf{1}^T @ sin(x).
"""

def f(x):
    return np.ones_like(x).T @ np.sin(x)

def grad_f(x):
    return np.cos(x)

x_ = np.random.rand(3)
print(grad_f(x_))
print(approx_fprime(x_,f,1e-6))