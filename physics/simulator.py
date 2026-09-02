import torch
import numpy as np
import numba as nb

@nb.njit
def derivatives(state, l1, l2, m1, m2, g):

    # Opening Given State
    theta1, omega1, theta2, omega2 = state

    
    delta = theta2 - theta1
    den1 = (m1 + m2) * l1 - m2 * l1 * np.cos(delta) ** 2
    den2 = (l2 / l1) * den1

    # Calculating Angular Velocity for First Pendulum
    domega1 = (m2 * l1 * omega1**2 * np.sin(delta) * np.cos(delta) + 
              m2 * g * np.sin(theta2) * np.sin(delta) + 
              m2 * l2 * omega2**2 * np.sin(delta) - 
              (m1 + m2) * g * np.sin(theta1)) / den1

    # Calculating Angular Velocity for Second Pendulum
    domega2 = (-m2 * l2 * omega2**2 * np.sin(delta) * np.cos(delta) + 
              (m1 + m2) * g * np.sin(theta1) * np.sin(delta) + 
              (m1 + m2) * l1 * omega1**2 * np.sin(delta) - 
              (m1 + m2) * g * np.sin(theta2)) / den2

    return np.array([omega1, domega1, omega2, domega2])

@njit
def gen_traj(state0, dt, timesteps):
    states = np.zeros(timesteps, 4)

    state[0] = state0

    # Gathering Environment Data
    for i in range(timesteps - 1):
        s = states[i]
        
        # Calculates the derivatives forward in time using RK4
        k1 = derivatives(s, l1, l2, m1, m2, g)
        k2 = derivatives(s + 0.5 * dt * k1, l1, l2, m1, m2, g)
        k3 = derivatives(s + 0.5 * dt * k2, l1, l2, m1, m2, g)
        k4 = derivatives(s + dt * k3, l1, l2, m1, m2, g)

        # Calculating final Approximation
        states[i + 1] = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        
    return states
        

