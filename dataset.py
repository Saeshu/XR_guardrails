import os
import cv2
import torch
from torch.utils.data import Dataset
from augmentations.quality_transforms import apply_augmentation

class XrayDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (224, 224))

        image, metadata = apply_augmentation(image)

        image = image / 255.0
        image = torch.tensor(image).unsqueeze(0).float()

        label = torch.tensor(metadata["label"]).float()

        return image, label, metadata
