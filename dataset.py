import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np


class TrafficSignDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_names = sorted(os.listdir(img_dir))
        self.label_names = sorted(os.listdir(label_dir))
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        label_path = os.path.join(self.label_dir, self.label_names[idx])

        img = Image.open(img_path).convert('RGB')
        labels = np.loadtxt(label_path)

        # Ensure labels is a 2D array, even if there's only one object in the image
        if labels.ndim == 1:
            labels = labels.reshape(1, -1)

        boxes = labels[:, 1:]  # (center_x, center_y, width, height)
        labels = labels[:, 0].astype(int)  # (class_id)

        if self.transform:
            img = self.transform(img)

        return img, boxes, labels


def collate_fn(batch):
    """
    Custom collate function to handle variable-sized bounding boxes and labels.
    It pads the boxes and labels to the size of the largest item in the batch.
    """
    imgs, boxes, labels = zip(*batch)
    imgs = torch.stack(imgs, 0)
    max_boxes = max(box.shape[0] for box in boxes)

    padded_boxes = []
    padded_labels = []
    for box, label in zip(boxes, labels):
        # Calculate padding size
        padding_size = max_boxes - box.shape[0]
        padded_box = np.pad(box, ((0, padding_size), (0, 0)), mode='constant', constant_values=0)
        padded_boxes.append(torch.tensor(padded_box, dtype=torch.float32))
        padded_label = np.pad(label, (0, padding_size), mode='constant', constant_values=-1)
        padded_labels.append(torch.tensor(padded_label, dtype=torch.int64))

    padded_boxes = torch.stack(padded_boxes, 0)
    padded_labels = torch.stack(padded_labels, 0)

    return imgs, padded_boxes, padded_labels
