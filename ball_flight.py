# ball_flight.py
# I fill in the dynamics for a spinning basketball with drag, lift (Magnus), and gravity,
# then I simulate the trajectory using ode4u and plot px vs py.

import numpy as np
import matplotlib.pyplot as plt
from multivarious.ode import ode4u


def ball_flight_dyn(t, x, u, c):
    """
    ODE function for ball flight with backspin.

    Parameters:
        t : float
            Time (not used explicitly here but required for ODE solvers)
        x : array_like, shape (4,)
            State vector [position_x, position_y, velocity_x, velocity_y]
        u : array_like
            Input vector (not used in this function but included for interface)
        c : list or tuple
            Constants: [g, r, m, rho, omega, CD, CL]

    Returns:
        dxdt : ndarray, shape (4,)
            Time derivative of the state vector
        y : ndarray, shape (3,)
            Magnitudes of forces [gravity, drag, lift]
    """
    vx = x[2]  # I store velocity in x-direction, m/s
    vy = x[3]  # I store velocity in y-direction, m/s

    g, r, m, rho, omega, CD, CL = c  # I unpack constants

    # I build unit vectors and 3D vectors so I can use norms and cross products
    j = np.array([0.0, 1.0, 0.0])          # I define the unit vector in y direction
    v = np.array([vx, vy, 0.0])            # I embed velocity as a 3-vector
    w = np.array([0.0, 0.0, omega])        # I define the backspin vector (about +k)

    # I compute gravity, drag, and lift (Magnus) force vectors
    fG = -m * g * j

    # I compute drag: -CD * (pi r^2) * (1/2 rho ||v|| v)
    fD = -CD * (np.pi * r**2) * (0.5 * rho * np.linalg.norm(v) * v)

    # I compute lift: CL * (16/3) * (pi^2/2) * r^3 * rho * (w x v)
    fL = CL * (16.0 / 3.0) * (np.pi**2 / 2.0) * (r**3) * rho * np.cross(w, v)

    # I sum forces to get the net force
    f = fG + fD + fL

    # I return the state derivatives: p_dot = v, v_dot = f/m
    dxdt = np.array([
        vx,        # derivative of position_x
        vy,        # derivative of position_y
        f[0] / m,  # derivative of velocity_x
        f[1] / m   # derivative of velocity_y
    ])

    # I also return magnitudes of each force for debugging/inspection
    y = np.array([
        np.linalg.norm(fG),
        np.linalg.norm(fD),
        np.linalg.norm(fL)
    ])

    return dxdt, y


# -------------------------
# I simulate ball flight below
# -------------------------

# Constants
g = 9.806          # gravitational acceleration, m/s^2
r = 0.1192         # I use size-7 basketball radius ~ 0.119 m
m = 0.62           # I use a typical size-7 mass ~ 0.62 kg
rho = 1.225        # I use standard sea-level air density, kg/m^3
omega = 6 * np.pi  # I pick a reasonable backspin (rad/s)
CD = 0.47          # I use a typical sphere drag coefficient (order ~0.4-0.5)
CL = 0.20          # I choose a moderate lift coefficient for spin

c = (g, r, m, rho, omega, CD, CL)

# Time parameters
T = 8.0
dt = 0.01
nT = int(np.floor(T / dt))
t_eval = np.linspace(0, T, nT)

# Initial state vector: [px, py, vx, vy]
# I choose an initial shot with forward + upward velocity
x0 = np.array([0.0, 0.0, 8.0, 7.0])

# External forcing (not used)
u = np.zeros(nT)

# Solve ODE
time, x, _, _ = ode4u(ball_flight_dyn, t_eval, x0, u, c)

# Extract solution
px = x[0, :]
py = x[1, :]

# Plot the trajectory
plt.figure(1)
plt.clf()
plt.plot(px, py, linewidth=3, label="Trajectory")
plt.xlabel("x-position (m)")
plt.ylabel("y-position (m)")
plt.title("Ball Flight Trajectory")
plt.legend()
plt.grid(True)
plt.show()