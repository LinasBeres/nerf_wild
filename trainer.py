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

def get_coords(res, normalize = False):
    x = y = torch.arange(res)
    xx, yy = torch.meshgrid(x, y)
    coords = torch.stack([xx, yy], dim=-1)
    if normalize:
        coords = coords / (res - 1)
    return coords

class ImagesDataset(Dataset):
    def __init__(self, images_path, image_size, device = 'cuda'):
        self.totalPixelsPerImage = image_size[0] * image_size[1]
        self.images = []
        self.rgb_vals = torch.zeros(0).to(device)
        self.coords = get_coords(image_size[0], normalize=True).reshape(-1, 2).to(device)

        for path in Path(f'{images_path}').glob('**/*'):
            _, ext = os.path.splitext(path)
            if ext.lower() in ['.jpeg', '.jpg', '.png']:
                image = Image.open(path).resize(image_size)
                self.images.append(image)

                rgb_vals = torch.from_numpy(np.array(image)).reshape(-1, 3).to(device)
                rgb_vals = rgb_vals.float() / 255

                self.rgb_vals = torch.cat((self.rgb_vals, rgb_vals))

    def __len__(self):
        return len(self.rgb_vals)
    def __getitem__(self, idx):
        imageIndex = int( idx / self.totalPixelsPerImage )
        coordIndex = int( idx % (self.totalPixelsPerImage) )
        return self.coords[coordIndex], self.rgb_vals[idx], imageIndex

class Trainer:
    def __init__(self, name, imagePath, imageSize, device = 'cuda', visualizeResults = True):
        self.name = name
        self.imageSize = imageSize
        self.visualizeResults = visualizeResults

        self.dataset = ImagesDataset(imagePath, imageSize, device)
        self.dataloader = DataLoader(self.dataset, batch_size=4096, shuffle=True)

        self.useZ = True

        if (self.useZ):
            self.embeddingSize = 1
        else:
            self.embeddingSize = 0

        self.model = NERF_W(self.dataset, imageSize, self.embeddingSize, self.useZ).to(device)

        lr = 1e-3
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()

        self.nepochs = 200

    def toImage(self, prediction):
        image = prediction.cpu().numpy().reshape(*self.imageSize, 3)
        image = (image * 255).astype(np.uint8)
        return image

    def run(self):
        pbar = tqdm(range(self.nepochs))
        for epoch in pbar:
            self.model.train()
            for coord, rgb_vals, imageIndex in self.dataloader:
                self.optimizer.zero_grad()
                pred = self.model(coord, imageIndex)
                loss = self.criterion(pred, rgb_vals)
                loss.backward()
                self.optimizer.step()

            torch.save(self.model, "model" + self.name)

            if self.visualizeResults:
                self.model.eval()
                with torch.no_grad():
                    coords = self.dataset.coords
                    pred0 = self.model(coords, torch.IntTensor([0 for _ in range(len(coords))]), True, 0.05)
                    pred1 = self.model(coords, torch.IntTensor([0 for _ in range(len(coords))]), True, 0.25)
                    pred2 = self.model(coords, torch.IntTensor([0 for _ in range(len(coords))]), True, 0.5)
                    pred3 = self.model(coords, torch.IntTensor([0 for _ in range(len(coords))]), True, 0.75)
                    pred4 = self.model(coords, torch.IntTensor([0 for _ in range(len(coords))]), True, 0.95)

                pbar.set_description(f'Epoch: {epoch}')
                pred0 = self.toImage(pred0)
                pred1 = self.toImage(pred1)
                pred2 = self.toImage(pred2)
                pred3 = self.toImage(pred3)
                pred4 = self.toImage(pred4)

                save_image = np.hstack([pred0, pred1, pred2, pred3, pred4])
                save_image = Image.fromarray(save_image)
                #  save_image.save(f'output_{epoch}.png')
                self.visualize(np.array(save_image), text = '# Epochs: {}, Embedding Vector Size: {}'.format(epoch, self.embeddingSize))

    def get_num_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def visualize(self, image, text):
        save_image = np.ones((44 + self.imageSize[0], self.imageSize[0] * 5, 3), dtype=np.uint8) * 255
        img_start = (44)
        save_image[img_start:img_start + self.imageSize[0], :, :] = image
        save_image = cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR)
        position = (100, 20)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        color = (255, 0, 0)
        thickness = 2

        cv2.putText(save_image, text, position, font, scale, color, thickness)
        cv2.imshow('image', save_image)

        cv2.waitKey(1)
