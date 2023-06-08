import json
import os
import natsort
import torch.utils.data
import cv2
import numpy as np
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, label_file, transform=None):
        self.image_folder = image_folder
        self.label_file = label_file
        self.transform = transform
        self.data = self._load_data()

    def _load_data(self):
        labels_df = pd.read_csv(self.label_file)
        data = []
        for _, row in labels_df.iterrows():
            image_filename = row['StudyInstanceUID']
            label = []
            label.append(row['ETT'])
            label.append(row['NGT'])
            label.append(row['CVC'])
            image_path = os.path.join(self.image_folder, image_filename)
            data.append((image_path, label))
        return data

    def __getitem__(self, index):
        image_path, label = self.data[index]
        image_path = image_path + '.jpg'
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return (image, label)

    def __len__(self):
        return len(self.data)