from models.model import Model
from models.training import Training
from data.generate_data import generate_block


if __name__ == "__main__":

    trainer = Training(data_gen=generate_block, model=Model)

    trainer.train(steps=1_000)
