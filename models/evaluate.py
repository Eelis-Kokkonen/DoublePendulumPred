from model import Model
from physics.simulator import sim

import torch





class Evaluation:
    def __init__(self, simulation=sim, model=Model):


        self.simulation = simulation
        self.model = Model


    def evaluate(self, steps=10_000):


        for step in steps:
            model.train()

