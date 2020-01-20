import os
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import pandas as pd

class breast_classify_inbreast(Dataset):
    def __init__(self, root, csv_file, transform=None):
        self.root = root
        img_label = pd.read_csv(csv_file)
        self.imgs = img_label['img_file_name'].values.tolist()
        self.labels = img_label['category_name'].values.tolist()
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root, self.imgs[idx]))
        label = self.labels[idx]
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)