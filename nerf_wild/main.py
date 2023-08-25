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
                self.coords = coords

    def __len__(self):
        return len(self.rgb_vals)
    def __getitem__(self, idx):
        imageIndex = int( idx / self.totalPixelsPerImage )
        coordIndex = int( idx % (self.totalPixelsPerImage) )
        return self.coords[coordIndex], self.rgb_vals[idx], imageIndex

class Trainer:
    def __init__(self, imagePath, imageSize, use_pe = True, device = 'cpu'):
        self.imageSize = imageSize

        self.dataset = ImagesDataset(imagePath, imageSize, device)
        self.dataloader = DataLoader(self.dataset, batch_size=4096, shuffle=True)

        self.useZ = True

        if (self.useZ):
            self.embeddingSize = 100
        else:
            self.embeddingSize = 0

        self.model = NERF_W(self.dataset, imageSize, self.embeddingSize, self.useZ).to(device)

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
                pred0 = self.model(coords, torch.IntTensor([0 for _ in range(len(coords))]), True, 0.25)
                pred1 = self.model(coords, torch.IntTensor([0 for _ in range(len(coords))]), True, 0.5)
                interp = self.model(coords, torch.IntTensor([0 for _ in range(len(coords))]), True, 0.72)
                gt = self.dataset.rgb_vals

            pbar.set_description(f'Epoch: {epoch}')
            pred0 = pred0.cpu().numpy().reshape(*self.dataset.images[0].size[::-1], 3)
            pred0 = (pred0 * 255).astype(np.uint8)

            pred1 = pred1.cpu().numpy().reshape(*self.dataset.images[0].size[::-1], 3)
            pred1 = (pred1 * 255).astype(np.uint8)

            interp = interp.cpu().numpy().reshape(*self.dataset.images[0].size[::-1], 3)
            interp = (interp * 255).astype(np.uint8)

            totalImageSize = self.imageSize[0] * self.imageSize[1]
            gt1 = self.dataset.rgb_vals[0:totalImageSize].cpu().numpy().reshape(*self.dataset.images[0].size[::-1], 3)
            gt1 = (gt1 * 255).astype(np.uint8)
            gt2 = self.dataset.rgb_vals[totalImageSize:].cpu().numpy().reshape(*self.dataset.images[0].size[::-1], 3)
            gt2 = (gt2 * 255).astype(np.uint8)

            save_image = np.hstack([gt1, gt2, pred0, pred1, interp])
            save_image = Image.fromarray(save_image)
            #  save_image.save(f'output_{epoch}.png')
            self.visualize(np.array(save_image), text = '# Epochs: {}, Embedding Vector Size: {}'.format(epoch, self.embeddingSize))

        self.visualize(np.array(save_image), text = '# Epochs: {}, Embedding Vector Size: {}'.format(epoch, self.embeddingSize))
        while(1):
            if cv2.waitKey(33) == ord('q'):
                return


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

if __name__ == '__main__':
    image_path = './bran_gate'
    image_size = (256, 256)
    device = 'cpu'

    trainer = Trainer(image_path, image_size, device)
    print('# params: {}'.format(trainer.get_num_params()))
    trainer.run()

