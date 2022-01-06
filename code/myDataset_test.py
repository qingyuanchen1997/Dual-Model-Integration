import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from skimage import io
import numpy as np

class myDataset(Dataset):

    def __init__(self, path, totensor, transform=None):
        self.data = self.get_data(path)
        self.transform = transform
        self.totensor = totensor

    def get_data(self, path):
        data = []

        for file in sorted(os.listdir(path)):
            data.append(self.get_pairs(path, file))
        return data

    def get_pairs(self, cur_path, path):
        file_name_list = cur_path.split('/')
        file_name1 = os.path.splitext(os.path.basename(path))[0]
        file_name= file_name1[:-9]
        file_name2 = file_name + 'magnetogram' + '.jpg'
        file_path=''
        for part in file_name_list[:-2]:
            file_path+=part+'/'
        #file_path = file_name_list[0] + '/' + file_name_list[1] + '/' + file_name_list[2] + '/' + file_name_list[3]
        file_path1 = file_path + 'continuum/' + path
        file_path2 = file_path + 'magnetogram/'  + file_name2
        file_path3 = file_path + 'magnetogram/' +  file_name2
        return file_path1, file_path2, file_path3


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        img_path1, img_path2, img_path3 = img_path[0], img_path[1], img_path[2]
        img1, img2, img3 = io.imread(img_path1), io.imread(img_path2), io.imread(img_path3)
        img1, img2, img3 = Image.fromarray(img1.astype('uint8')), Image.fromarray(img2.astype('uint8')), Image.fromarray(img3.astype('uint8'))
        '''if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        '''
        img1 = self.totensor(img1)
        img2 = self.totensor(img2)
        img3 = self.totensor(img3)
        img = torch.cat([img1, img2, img3], dim=0)
        if self.transform:
            img = self.transform(img)
        return img
