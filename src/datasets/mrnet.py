import numpy as np
import os
import torch
import torch.utils.data as data
import pandas as pd
from base.torchvision_dataset import TorchvisionDataset


MEAN = 58.81274059207973
STDDEV = 48.56406668573295

class MRDataset(TorchvisionDataset):
    def __init__(self, root_dir, plane='sagittal'):
        super().__init__(root_dir)



        self.test_set = MyMRDataset(root_dir, plane, 'test')
        self.train_set = MyMRDataset(root_dir, plane, 'train')




class MyMRDataset(data.Dataset):
    def __init__(self, root_dir, plane, task):
        super().__init__()

        self.plane = plane
        self.root_dir = root_dir

        self.folder_path = self.root_dir + '/train/{0}/'.format(plane)
        self.records = pd.read_csv('./datasets/metadata.csv')

        if task == 'train':
          self.records = self.records.loc[self.records['ref_set']==1]
          self.labels = list(self.records.loc[self.records['ref_set'] ==1, 'label'])
        else:
          self.records = self.records.loc[self.records['test']==1]
          self.labels = list(self.records.loc[self.records['test'] ==1, 'label'])

        self.records['id'] = self.records['id'].map(
            lambda i: '0' * (4 - len(str(i))) + str(i))
        self.paths1 = [self.folder_path + filename +
                      '.npy' for filename in self.records.loc[self.records['mrnet_split'] == 0, 'id'].tolist()]
        self.folder_path2 = self.root_dir + '/valid/{0}/'.format(plane)
        self.paths2 = [self.folder_path2 + filename +
                      '.npy' for filename in self.records.loc[self.records['mrnet_split'] == 1, 'id'].tolist()]
        self.paths = self.paths1 + self.paths2

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        array = np.load(self.paths[index])
        array = np.stack((array,)*3, axis=1)
        array = torch.FloatTensor(array)
        array = (array - MEAN) / STDDEV
        label = torch.FloatTensor([self.labels[index]])
        return array, label, index
