import torch
import numpy as np
import numba as nb

@nb.njit
def derivatives(state, l1, l2, m1, m2, g)
    theta1, omega1, theta2, omega2 = state

    delta = theta2 - theta1
    den1 = (m1 + m2) * l1 - m2 * l1 * np.cos(delta) ** 2
    den2 = (l2 / l1) * den1

    domega1 = (m2 * l1 * omega1**2 * np.sin(delta) * np.cos(delta) + 
              m2 * g * np.sin(theta2) * np.sin(delta) + 
              m2 * l2 * omega2**2 * np.sin(delta) - 
              (m1 + m2) * g * np.sin(theta1)) / den1

    domega2 = (-m2 * l2 * omega2**2 * np.sin(delta) * np.cos(delta) + 
              (m1 + m2) * g * np.sin(theta1) * np.sin(delta) + 
              (m1 + m2) * l1 * omega1**2 * np.sin(delta) - 
              (m1 + m2) * g * np.sin(theta2)) / den2


    return np.array([omega1, domega1, omega2, domega2])

@njit
def gen_traj(state0, dt, timesteps):
    states = np.xeros(timesteps, 4)

    state0 = states[0]

    for i in range(timesteps - 1):
        s = states[i]
        k = derivatives(s)




