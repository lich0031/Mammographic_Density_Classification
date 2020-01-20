import os
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import pandas as pd

class breast_classify_hospital(Dataset):
    def __init__(self, root, csv_file, transform=None):
        self.root = root
        img_label = pd.read_csv(csv_file)
        self.imgfolders = img_label['img_folder'].values.tolist()
        self.imgsview1 = img_label['img_file_name_view1'].values.tolist()
        self.imgsview2 = img_label['img_file_name_view2'].values.tolist()
        self.labels = img_label['category'].values.tolist()
        self.transform = transform

    def __getitem__(self, idx):
        imgview1 = Image.open(os.path.join(self.root, self.imgfolders[idx], self.imgsview1[idx]))
        imgview2 = Image.open(os.path.join(self.root, self.imgfolders[idx], self.imgsview2[idx]))
        label = self.labels[idx]
        if imgview1.mode != 'RGB':
            imgview1 = imgview1.convert('RGB')
        if imgview2.mode != 'RGB':
            imgview2 = imgview2.convert('RGB')
        if self.transform:
            imgview1 = self.transform(imgview1)
            imgview2 = self.transform(imgview2)
        return imgview1, imgview2, label

    def __len__(self):
        return len(self.imgsview1)