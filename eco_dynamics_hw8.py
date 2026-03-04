# eco_dynamics_hw8.py
# In this script I simulate the environmental insecticide system
# using the matrix exponential solution x(t) = exp(At)x0.
# I also compute eigenvalues and eigenvectors to understand stability.

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

# I define the rate constants given in the problem
alpha_WP = 0.388
alpha_PW = 0.0515
alpha_WF = 0.136
alpha_FS = 0.067
alpha_SF = 0.0254
beta_FW = 0.0788
beta_WP = 0.0068
beta_PW = 0.001

# I build the dynamics matrix A directly from the equations
A = np.array([
    [-(alpha_WF + alpha_WP),  alpha_PW,                   0,                0,             0,          0],
    [ alpha_WP,              -alpha_PW,                   0,                0,             0,          0],
    [ alpha_WF,               0,       -(alpha_FS+beta_FW), alpha_SF,       0,          0],
    [ 0,                      0,         alpha_FS,      -alpha_SF,           0,          0],
    [ 0,                      0,         beta_FW,           0,       -beta_WP,    beta_PW],
    [ 0,                      0,             0,              0,        beta_WP,   -beta_PW]
])

# I define the initial condition (100 g in water only)
x0 = np.array([100, 0, 0, 0, 0, 0])

# I define time from 0 to 1000 hours
t = np.arange(0, 1001)
n = len(t)

# I initialize the state response matrix
X = np.full((6, n), np.nan)

# I compute the solution using the matrix exponential
for k in range(n):
    X[:, k] = expm(A * t[k]).dot(x0)

# I plot full 1000 hour response
plt.figure()
plt.plot(t, X.T, linewidth=3)
plt.xlabel("time t hours")
plt.ylabel("insecticide X and metabolite M (g)")
plt.legend(["X_W","X_P","X_F","X_S","M_W","M_P"])
plt.title("Full 1000 hour response")
plt.show()

# I zoom into the first 150 hours
plt.figure()
plt.plot(t[:150], X[:, :150].T, linewidth=3)
plt.xlabel("time t hours")
plt.ylabel("insecticide X and metabolite M (g)")
plt.legend(["X_W","X_P","X_F","X_S","M_W","M_P"])
plt.title("Zoom: first 150 hours")
plt.show()

# I compute eigenvalues and eigenvectors
eVal, eVec = np.linalg.eig(A)

print("Eigenvalues:")
print(eVal)

print("\nEigenvectors (columns):")
print(eVec)