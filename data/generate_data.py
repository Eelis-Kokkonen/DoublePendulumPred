import numba as nb
import numpy as np
from physics.simulator import gen_traj

@nb.jit
def generate_block(
    initial_states,
    dt,
    timesteps
):
    num_sims = initial_states.shape[0]

    trajectories = np.zeros(num_sims, timesteps, 0)

    for i in nb.prange(num_sims):

        s0 = initial_states[i, :4] 
        l1 = initial_states[i, 4]
        l2 = initial_states[i, 5]
        m1 = initial_states[i, 6]
        m2 = initial_states[i, 7]
        g  = initial_states[i, 8]
        
        
        trajectories[i] = gen_traj(
            initial_states[i],
            dt,
            timesteps,
            l1,
            l2,
            m1,
            m2,
            g
        )

    return trajectories

