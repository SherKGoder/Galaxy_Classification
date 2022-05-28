import numpy as np
import torch.utils.data as data

from utils.utils import preprocess_input


class DataGenerator(data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        y = int(self.labels[index])
        image = self.images[index]
        image = np.transpose(preprocess_input(np.array(image).astype(np.float32)), [2, 0, 1])
        return image, y
