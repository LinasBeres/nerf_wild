import os
import time
import torch
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import cv2

from pathlib import Path

from PIL import Image
import numpy as np
from tqdm import tqdm

from models.mlp import FCNet, NERF_W
from trainer import get_coords, ImagesDataset

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

class Evaluator:
    def __init__(self, name, imageSize, device = 'cuda'):
        self.name = name
        self.imageSize = imageSize

        self.model = torch.load("model" + self.name, map_location=torch.device(device)).to(device)
        self.model.eval()

        self.coords = get_coords(imageSize[0], normalize=True).reshape(-1, 2).to(device)

    def toImage(self, prediction):
        image = prediction.cpu().numpy().reshape(*self.imageSize, 3)
        image = (image * 255).astype(np.uint8)
        return image

    def getImage(self, interpolation = 0.5):
        with torch.no_grad():
            coords = self.coords
            prediction = self.model(coords, torch.IntTensor([0 for _ in range(len(coords))]), True, interpolation)

        prediction = self.toImage(prediction)

        return prediction

    def visualiseImage(self, image):
        image = np.array(image)

        save_image = np.ones((44 + self.imageSize[0], self.imageSize[0], 3), dtype=np.uint8) * 255

        img_start = (44)

        save_image[img_start:img_start + self.imageSize[0], :, :] = image
        save_image = cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR)

        position = (100, 20)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        color = (255, 0, 0)
        thickness = 2

        cv2.putText(save_image, "Interpolation", position, font, scale, color, thickness)
        cv2.imshow('image', save_image)

        while True:
            if cv2.waitKey(33) == ord('q'):
                return

    def plot(self):
        image = self.getImage()

        fig, axs = plt.subplots(1, 1)

        im = axs.imshow(image)

        axinter = fig.add_axes([0.20, 0.05, 0.60, 0.03])
        inter_slider = Slider(
            ax=axinter,
            label='Interpolation',
            valmin=0.1,
            valmax=1,
            valinit=0.5,
        )

        def update(val):
            image = self.getImage(inter_slider.val)
            axs.imshow(image)
            fig.canvas.draw_idle()

        inter_slider.on_changed(update)

        axs.set_yticklabels([])
        axs.set_xticklabels([])

        plt.show()

    def visualiseAll(self):
        plt.ion()

        fig, axs = plt.subplots(1, 1)

        image = self.getImage(1 / 100)

        im = axs.imshow(image)

        axs.set_yticklabels([])
        axs.set_xticklabels([])

        plt.axis('off')

        fig.canvas.draw()
        fig.canvas.flush_events()

        time.sleep(10)

        for i in range(0, 100, 1):
            print(i)

            image = self.getImage(i / 100)
            axs.imshow(image)
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.5)
