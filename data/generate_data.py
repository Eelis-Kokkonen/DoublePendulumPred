import numba
from physics.simulator import gen_traj

@nb.jit
def generate_block(
    initial_states,
    dt,
    timesteps,
    l1,
    l2,
    m1,
    m2,
    g
):
    num_sims = initial_states.shape[0]

    trajectories = np.zeros(num_sims, timesteps, 0)

    for i in nb.prange(num_sims):
        
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

