
import torch
from torch.utils.data import Dataset
import collections.abc
import numpy as np
from PIL import Image
from torchvision import transforms as T

class ImageDataset(Dataset):
    def __init__(self, data, annotations=None, transformations=None, data_reader=None):
        """
        Image dataset class
        :param data: The samples. Can be a matrix, list of paths to the images, or a data type compatible with the data_reader
        :param annotations: If not None it should be a list or vector with the corresponding labels for each sample
        :param data_reader: Optional, an object to read the data. Use for custom data structures (e.g. h5 files)
        """
        self.transformations = transformations or T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.data_reader = data_reader or ImageReader(data, annotations)

    def __len__(self):
        return self.data_reader.__len__()

    def __getitem__(self, index):
        image, annotation = self.data_reader.__getitem__(index)
        if self.transformations is not None:
            image = self.transformations(image)
        return index, image, annotation

    def _get_data_type(self, data):
        data_type = None
        if isinstance(data, torch.Tensor) or isinstance(data, np.ndarray):
            data_type = "tensor"
        elif isinstance(data, collections.abc.Sequence):
            if isinstance(data[0], str):
                data_type = "files"
        assert data_type is not None

        return data_type

class ImageReader:
    def __init__(self, data, annotations=None):
        self.data = data
        self.annotations=annotations
        self.tensor2PIL = T.ToPILImage()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        if isinstance(image, str):
            image = load_image(image)
        elif isinstance(image, torch.Tensor):
            image = self.tensor2PIL(image)
        else:
            image = Image.fromarray(image)

        if self.annotations is not None:
            annotation = int(self.annotations[index])
        else:
            annotation = None

        return image, annotation

def load_image(path):
    """Loads an image from given path"""
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


