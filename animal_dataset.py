# define AnimalDataset

import torch
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision

class AnimalDataset(Dataset):
    def __init__(self, csv_file, S=5, B=2, C=4):
        #number of grids S*S
        self.S = S
        #B number of predicted bounding boxes each grid
        self.B = B
        #C number of classes
        self.C = C
        #csv file containg the path of images and label_txt
        self.csv_file = pd.read_csv(csv_file)

        # Add some transforms for data augmentation.
        self.tensor_transform = torchvision.transforms.ToTensor()
        self.normalize_transform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])
        self.resize = torchvision.transforms.Resize(size=(448, 448))
        self.transform = torchvision.transforms.Compose([self.resize,
                                                         self.tensor_transform,
                                                         self.normalize_transform])

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        #read in image
        img = Image.open(self.csv_file.iloc[idx, 0])
        path = self.csv_file.iloc[idx, 0]
        img_tensor = self.transform(img)

        # txt file --> label boxes
        boxes = []
        with open(self.csv_file.iloc[idx, 1]) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [float(x) if float(x) != int(float(x)) else int(x) for x in
                                                    label.replace("\n", "").split()]
                boxes.append([class_label, x, y, width, height])

        #label for each grid containing B boxes
        # (0, 1, 2, 3) --> class label
        # (4) --> object probability
        # (5, 6, 7 ,8) --> (x_location_wrt_cell [0,1], y_location_wrt_cell [0,1], width_wrt_cell, hight_wrt_cell)
        # different from original paper, width/height label is wrt whole image
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))

        for box in boxes:
            class_label, x, y, width, height = box
            class_label = int(class_label)

            # S cells in x and y direction
            i = int(self.S * y)
            j = int(self.S * x)
            # center of box relative to the boundary of cell (i,j)
            x_cell = self.S * x - j
            y_cell = self.S * y - i
            width_wrt_cell = self.S * width
            height_wrt_cell = self.S * height

            #set class label to 1
            label_matrix[i, j, class_label] = 1
            # exist object = 1
            label_matrix[i, j, 4] = 1
            # x, y, width, height
            label_matrix[i, j, 5] = x_cell
            label_matrix[i, j, 6] = y_cell
            label_matrix[i, j, 7] = width_wrt_cell
            label_matrix[i, j, 8] = height_wrt_cell

        return img_tensor, label_matrix, path

# def test():
#     batch_size = 8
#     train_dataset = AnimalDataset("F:\Jupyter-Notebook\I2DL_WS20\ObjectDetection\\animal_dataset\\train.csv")
#     train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     img, label = next(iter(train_dataloader))
#     print(img.shape)
#     print(label.shape)
#
# test()