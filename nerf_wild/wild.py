import os
import torch
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import cv2

from pathlib import Path

from PIL import Image
import numpy as np
from tqdm import tqdm

from models.mlp import FCNet, NERF_W
from models.glm import CombinedModel

import pytorch_lightning as pl

# pytorch lightning module

class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, dataloader, **kwargs):
        super().__init__()
        self.learner = CombinedModel(dataloader, 2).to('cpu')

    def forward(self, images):
        return self.learner(images)

    def training_step(self, images, _):
        loss = self.forward(images)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)

    def on_before_zero_grad(self, _):
        if self.learner.use_momentum:
            self.learner.update_moving_average()

def expandGreyscale(t):
    return t.expand(3, -1, -1)

class ImagesDataset(Dataset):
    def __init__(self, folder, image_size, device = 'cuda'):
        super().__init__()
        self.folder = folder
        self.paths = []

        for path in Path(f'{folder}').glob('**/*'):
            _, ext = os.path.splitext(path)
            if ext.lower() in ['.jpg', '.png', '.jpeg']:
                self.paths.append(path)

        print(f'{len(self.paths)} images found')

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(expandGreyscale)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image = Image.open(path)
        image = image.convert('RGB')
        return (index, self.transform(image))

if __name__ == '__main__':
    image_path = './images'
    image_size = (256, 256)
    device = 'cpu'

    dataset = ImagesDataset(image_path, image_size, device)
    dataloader = DataLoader(dataset, batch_size=4096, shuffle=True)

    model = SelfSupervisedLearner(
        dataloader,
        image_size = image_size,
        hidden_layer = 'avgpool',
        projection_size = 256,
        projection_hidden_size = 4096,
        moving_average_decay = 0.99
    )

    trainer = pl.Trainer(
        max_epochs = 20,
        accumulate_grad_batches = 1,
        sync_batchnorm = True
    )

    trainer.fit(model, train_loader)

