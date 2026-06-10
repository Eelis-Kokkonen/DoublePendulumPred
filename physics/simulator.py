import torch
import numpy as np
import numba as nb

@nb.njit
def derivatives(state, l1, l2, m1, m2, g)
    theta1, omega1, theta2, omega2 = state

    dtheta1 = omega1
    dtheta2 = omega2
    domega1 = 0.0
    domega2 = 0.0

    return np.array([dtheta1, domega1, dtheta2, domega2])


@nb.njit
def step(state, l1, l2, m1, m2, g):


@nb.njit
def rollout(state, l1, l2, m1, m2, g, seq_len):
    traj = np.zeros((seq_len, 4))
    state = state

    for i in range(seq_len):
        traj[i] = state
        state = step(state, l1, l2, m1, m2, g)

    return traj


class PendulumSimulation():
    def __init__(self, l1, l2, m1, m2, theta1, theta2, vel1, vel2, g):
        super().__init__()

        self.l1, self.l2 = l1, l2
        self.m1, self.m2 = m1, m2
      
        self.theta1, self.theta2 = theta1, theta2
        self.vel1, self.vel2 = vel1, vel2
      
        self.g = g

    def simulate(self, state, seq_len)
        return rollout(
            state,
            l1,
            l2,
            m1,
            m2,
            g,
            seq_len
        )


class PenddulumDataset(torch.utils.data.Dataset):
    def __init__(self, simulator, seq_len=1000, size=10000):
      super().__init__()

      self.size = size
      self.seq_len = seq_len
      self.sim = simulator

    def __createstate__(self):
        return np.array([
            np.random.uniform(-np.pi, np.pi),
            np.random.uniform(-1, 1),
            np.random.uniform(-np.pi, np.pi),      
            np.random.uniform(-1, 1),
        ], dtype=np.float64)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        state = self.__createstate__()
        traj = self.sim.simulate(state, self.seq_len)

        return torch.tensor(traj, dtype=torch.float32)
        
