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

import pytorch_lightning as pl

def get_coords(res, normalize = False):
    x = y = torch.arange(res)
    xx, yy = torch.meshgrid(x, y)
    coords = torch.stack([xx, yy], dim=-1)
    if normalize:
        coords = coords / (res - 1)
    return coords

def get_psnr(pred, gt):
    mse = torch.mean((pred - gt) ** 2)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr

class ImageDataset(Dataset):
    def __init__(self, image_path, image_size, device = 'cuda'):
        self.image = Image.open(image_path).resize(image_size)
        self.rgb_vals = torch.from_numpy(np.array(self.image)).reshape(-1, 3).to(device)
        self.rgb_vals = self.rgb_vals.float() / 255
        self.coords = get_coords(image_size[0], normalize=True).reshape(-1, 2).to(device)

    def __len__(self):
        return len(self.rgb_vals)
    def __getitem__(self, idx):
        return self.coords[idx], self.rgb_vals[idx]

class ImagesDataset(Dataset):
    def __init__(self, images_path, image_size, device = 'cuda'):
        self.totalPixelsPerImage = image_size[0] * image_size[1]
        self.images = []
        self.rgb_vals = torch.zeros(0)
        self.coords = torch.zeros(0)

        for path in Path(f'{images_path}').glob('**/*'):
            _, ext = os.path.splitext(path)
            if ext.lower() in ['.jpeg', '.jpg', '.png']:
                image = Image.open(path).resize(image_size)
                self.images.append(image)
                rgb_vals = torch.from_numpy(np.array(image)).reshape(-1, 3).to(device)
                rgb_vals = rgb_vals.float() / 255

                self.rgb_vals = torch.cat((self.rgb_vals, rgb_vals))

                coords = get_coords(image_size[0], normalize=True).reshape(-1, 2).to(device)
                self.coords = torch.cat((self.coords, coords))
                self.coords = coords

                print("LENGTH OF RGB VALS:", len(self.rgb_vals))

        print("TOTAL PIXELS:", self.totalPixelsPerImage)

    def __len__(self):
        return len(self.rgb_vals)
    def __getitem__(self, idx):
        imageIndex = int( idx / self.totalPixelsPerImage )
        #  print(idx, self.totalPixelsPerImage)
        coordIndex = int( idx % (self.totalPixelsPerImage) )
        return self.coords[coordIndex], self.rgb_vals[idx], imageIndex

class Trainer:
    def __init__(self, image_path, image_size, use_pe = True, device = 'cpu'):
        self.dataset = ImagesDataset(image_path, image_size, device)
        self.dataloader = DataLoader(self.dataset, batch_size=4096, shuffle=True)

        self.model = NERF_W(self.dataset).to(device)

        lr = 1e-3
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()

        self.nepochs = 20


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

            self.model.eval()
            with torch.no_grad():
                coords = self.dataset.coords
                pred = self.model(coords, 0)
                gt = self.dataset.rgb_vals
                #  psnr = get_psnr(pred, gt)

            pbar.set_description(f'Epoch: {epoch}')
            pred = pred.cpu().numpy().reshape(*self.dataset.images[0].size[::-1], 3)
            pred = (pred * 255).astype(np.uint8)

            gt = self.dataset.rgb_vals[0:65536].cpu().numpy().reshape(*self.dataset.images[0].size[::-1], 3)
            gt = (gt * 255).astype(np.uint8)

            save_image = np.hstack([gt, pred])
            save_image = Image.fromarray(save_image)
            #  save_image.save(f'output_{epoch}.png')
            self.visualize(np.array(save_image), text = '# params: {}'.format(self.get_num_params()))

    def get_num_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def visualize(self, image, text):
        save_image = np.ones((300, 512, 3), dtype=np.uint8) * 255
        img_start = (300 - 256)
        save_image[img_start:img_start + 256, :, :] = image
        save_image = cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR)
        position = (100, 20)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        color = (255, 0, 0)
        thickness = 2

        cv2.putText(save_image, text, position, font, scale, color, thickness)
        cv2.imshow('image', save_image)
        cv2.waitKey(1)

if __name__ == '__main__':
    image_path = './images'
    image_size = (256, 256)
    device = 'cpu'

    trainer = Trainer(image_path, image_size, device)
    print('# params: {}'.format(trainer.get_num_params()))
    trainer.run()

