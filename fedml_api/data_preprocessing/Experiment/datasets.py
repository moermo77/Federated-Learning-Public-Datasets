import logging

import numpy as np
import torch.utils.data as data
from PIL import Image
import torch
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def Mydata(root, transform, train):
    if train:
        data_file = '\\train\\train_data.txt'
        label_file = '\\train\\train_label.txt'
    else:
        data_file = '\\test\\test_data.txt'
        label_file = '\\test\\test_label.txt'
    data = np.loadtxt(root + data_file).astype(np.float32)
    label = np.loadtxt(root + label_file).astype(np.float32)
    data_trans = torch.tensor(data).unsqueeze(1)
    label_trans = torch.tensor(label).unsqueeze(1)
    return data_trans, label_trans

class Mydata_truncated(data.Dataset):

    def __init__(self, root_data, transform=None,train=True,  target_transform=None, dataidxs=None):

        self.root_data = root_data
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.dataidxs= dataidxs
        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        data, label = Mydata(self.root_data, self.transform, self.train)
        logging.info("global_mean:" + str(torch.mean(data, 0)) + "\n global_var:"+str(torch.var(data, 0)))
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            label = label[self.dataidxs]
            logging.info("local_mean:" + str(torch.mean(data, 0)) + "\n local_var:" + str(torch.var(data, 0)))
        return data, label

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, target = self.data[index], self.target[index]
        '''
        if self.transform is not None:
            data = data

        if self.target_transform is not None:
            target = self.target_transform(target)
        '''
        return data, target

    def __len__(self):
        return len(self.data)
