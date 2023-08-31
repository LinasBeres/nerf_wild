import os
import torch

from trainer import Trainer
from evaluator import Evaluator

if __name__ == '__main__':
    #  image_path = './lake-small-small'
    #  image_size = (512, 512)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #  trainer = Trainer("lake-small-small", image_path, image_size, device)
    #  trainer.run()

    #  image_path = './lake-small'
    #  trainer = Trainer("lake-small", image_path, image_size, device)
    #  trainer.run()

    image_size = (256, 256)
    #  image_path = './lake-small-small'
    #  trainer = Trainer("lake-small-small-256", image_path, image_size, device)
    #  trainer.run()

    image_path = './lake-small'
    trainer = Trainer("lake-small-256", image_path, image_size, device)

    image_path = '/home/lberesna/Downloads/lake'
    trainer = Trainer("lake-256", image_path, image_size, device)

    #  image_size = (512, 512)
    #  evaluator = Evaluator("lake-small-small", image_path, image_size, device)
    #  evaluator.plot()
