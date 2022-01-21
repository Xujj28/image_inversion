import glob
import os

import numpy as np
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms as T

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None

class Caltech256(Dataset):
    """Dataset Caltech 256
    Class number: 257
    Train data number: 24582
    Test data number: 6027
    """
    def __init__(self, dataroot, transform=None, train=True, inversion=False):
        # Initial parameters
        self.dataroot_param = dataroot
        self.dataroot = os.path.join(dataroot, "Caltech-256")
        self.train = train
        self.inversion = inversion
        if transform: # Set default transforms if no transformation provided.
            self.transform = transform
        else:
            self.transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomRotation((0, 30)),
                T.Resize((256, 256)),
                T.RandomResizedCrop((224, 224)),
                T.ToTensor(),
                T.Normalize((.485, .456, .406), (.229, .224, .225))
            ])
        
        # Metadata of dataset
        classes = [i.split('/')[-1] for i in glob.glob(os.path.join(self.dataroot, 'train' if train else 'test', '*'))]
        self.class_num = len(classes)
        self.classes = [i.split('.')[1] for i in classes]
        self.class_to_idx = {i.split('.')[1]: int(i.split('.')[0])-1 for i in classes}
        self.idx_to_class = {int(i.split('.')[0])-1: i.split('.')[1] for i in classes}

        self.img_paths = glob.glob(os.path.join(self.dataroot, 'train' if train else 'test', '*', '*'))
        self.targets = [self.class_to_idx[p.split('/')[-2].split('.')[1]] for p in self.img_paths]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        target = self.targets[idx]

        if self.inversion:
            return (img_tensor, target, img_path)
        else:
            return (img_tensor, target)

    def __repr__(self):
        repr = """Caltech-256 Dataset:
\tRoot location: {}
\tSplit: {}
\tClass num: {}
\tData num: {}""".format(self.dataroot, 'Train' if self.train else 'Test', self.class_num, self.__len__())
        return repr

class miniImageNet(Dataset):
    """ ImageNet dataset with subset and MAX #data-per-class settings.
    If use default parameters, it will just return a dataset with all ImageNet data.
    Otherwise, it will return a subset of ImageNet dataset.
    """
    def __init__(self, dataroot, transform=None, train=True, inversion=False):
        # self.dataroot = os.path.join(dataroot, "miniImageNet")
        self.dataroot = os.path.join(dataroot)
        self.train = train
        self.inversion = inversion
        if transform: # Set default transforms if no transformation provided.
            self.transform = transform
        else:
            self.transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomRotation((0, 30)),
                T.Resize((256, 256)),
                T.RandomResizedCrop((224, 224)),
                T.ToTensor(),
                T.Normalize((.485, .456, .406), (.229, .224, .225))
            ])
        
        # Metadata of dataset
        self.classes = [i.split('/')[-1] for i in glob.glob(os.path.join(self.dataroot, 'train' if train else 'test', '*'))]
        self.class_num = len(self.classes)
        self.class_to_idx = {self.classes[i]: i for i in range(self.class_num)}
        self.idx_to_class = {i: self.classes[i] for i in range(self.class_num)}

        self.img_paths = glob.glob(os.path.join(self.dataroot, 'train' if train else 'test', '*', '*'))
        self.targets = [self.class_to_idx[p.split('/')[-2]] for p in self.img_paths]


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        target = self.targets[idx]

        if self.inversion:
            return (img_tensor, target, img_path)
        else:
            return (img_tensor, target)

    def __repr__(self):
        repr = """ImageNet Dataset:
\tRoot location: {}
\tSplit: {}
\tClass num: {}
\tData num: {}""".format(self.dataroot, 'Train' if self.train else 'Test', self.class_num, self.__len__())
        return repr

class cifar100(Dataset):
    """ ImageNet dataset with subset and MAX #data-per-class settings.
    If use default parameters, it will just return a dataset with all ImageNet data.
    Otherwise, it will return a subset of ImageNet dataset.
    """
    def __init__(self, dataroot, transform=None, train=True, inversion=False):
        # self.dataroot = os.path.join(dataroot, "miniImageNet")
        self.dataroot = os.path.join(dataroot, "train")
        
        self.train = train
        self.inversion = inversion
        self.transform = transform
        
        # Metadata of dataset
        self.classes = [i for i in glob.glob(os.path.join(self.dataroot, 'train' if train else 'test', '*'))]
        self.class_num = len(self.classes)
        self.class_to_idx = {self.classes[i]: i for i in range(self.class_num)}
        self.idx_to_class = {i: self.classes[i] for i in range(self.class_num)}

        self.img_paths = glob.glob(os.path.join(self.dataroot, 'train' if train else 'test', '*', '*'))
        self.targets = [self.class_to_idx[p.split('/')[-2]] for p in self.img_paths]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        target = self.targets[idx]

        if self.inversion:
            return (img_tensor, target, img_path)
        else:
            return (img_tensor, target)

    def __repr__(self):
        repr = """ImageNet Dataset:
\tRoot location: {}
\tSplit: {}
\tClass num: {}
\tData num: {}""".format(self.dataroot, 'Train' if self.train else 'Test', self.class_num, self.__len__())
        return repr

    
