import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
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
        label_name = []
        for name in sorted(os.listdir(path)):
            label_name.append(name)

        for i in range(len(label_name)):
            cur_path = path + str(label_name[i])
            for file in os.listdir(cur_path):
                data.append([self.get_pairs(cur_path, file, label_name[i]), i])
        return data

    def get_pairs(self, cur_path, path, label_name):
        file_name_list = cur_path.split('/')
        file_name1 = os.path.splitext(os.path.basename(path))[0]
        file_name= file_name1[:-9]
        file_name2 = file_name + 'magnetogram' + '.jpg'
        file_path=''
        for part in file_name_list[:-2]:
            file_path+=part+'/'
        #file_path = file_name_list[0] + '/' + file_name_list[1] + '/' + file_name_list[2] + '/' + file_name_list[3]
        file_path1 = file_path + 'continuum/' + label_name + '/' + path
        file_path2 = file_path + 'magnetogram/' + label_name + '/' + file_name2
        return file_path1, file_path2, file_path2


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img_path1, img_path2, img_path3 = img_path[0], img_path[1], img_path[2]

        img_c=Image.open(img_path1).resize((224,224))
        img_m=Image.open(img_path2).resize((224,224))
        img = Image.merge('RGB', (img_c, img_m, img_m))

        if self.transform:
            img = self.transform(img)
        return img, label

if __name__ == "__main__":
    transform = transforms.Compose([transforms.Grayscale(),
                                    transforms.Resize([224, 224]),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5])])
    train_dataset = myDataset('G:/Tianchi/project/trainset/continuum/', transform) #使用绝对路径
    # test_dataset = myDataset('E:/python/myOCTv5/val', 'E:/python/myOCTv5/val_list.txt', transform)

    print("数据个数：", len(train_dataset))    # 707 pairs
    # print("数据个数：", len(test_dataset))     # 104 pairs
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=32,
                                               shuffle=True)
    # test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
    #                                           batch_size=8,
    #                                           shuffle=True)
    #
    for image,label in train_loader:
        print(image.shape)
        print(label)