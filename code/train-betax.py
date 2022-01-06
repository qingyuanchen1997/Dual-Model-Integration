from PIL import Image
from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data as data
import os

# set up gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 图片文件夹目录 ps 最后不要加 "/" 例 /train/img  而不是 /train/img/
basedir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
IMG_DIR_PATH = ''
# 保存模型路径
checkpoint_path = basedir+'/user_data/model_data/test_betax.pt'
# 训练 txt file
train_txt = basedir+'/user_data/tmp_data/train_input/continuum_train.txt'
# 验证 txt file
val_txt = basedir+'/user_data/tmp_data/train_input/continuum_val.txt'


# data loader
class ImageDataset(data.Dataset):
    def __init__(self, path, transform=None):
        with open(path, 'r') as f:
            self.annotations = f.read().split("\n")
        self.transform = transform

    def __getitem__(self, index):
        img_path, label = self.annotations[index].split(' ')
        image = Image.open(basedir + '/user_data/tmp_data/train_input/continuum' + os.path.splitext(img_path)[0] + '.jpg').convert('RGB').resize((500, 375))

        if self.transform is not None:
            image = self.transform(image)
        return image, int(label)

    def __len__(self):
        return len(self.annotations)


class Net(nn.Module):
    def __init__(self, alexnet):
        super(Net, self).__init__()
        self.alexnet = alexnet
        self.backbone = alexnet.features
        self.avgpool = alexnet.avgpool
        # in_features = alexnet.classifier[1].in_features
        self.logit = nn.Linear(2304, 3)
        # self.classifer[6].out_features = 3

    def forward(self,x):
        batch_size, C, H, W = x.shape
        x = self.backbone(x)
        # print(x.shape)
        x = self.avgpool(x)
        # print(x.shape)
        # x = F.dropout(x, 0.25, self.training)
        x = F.adaptive_avg_pool2d(x, 3).reshape(batch_size, -1)
        # print(x.shape)
        x = self.logit(x)
        return x


train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
trainset = ImageDataset(train_txt, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=4)
validset = ImageDataset(val_txt, transform=valid_transform)
validloader = torch.utils.data.DataLoader(validset, batch_size=8, shuffle=False, num_workers=4)

# model build
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
alexnet = models.alexnet(pretrained=True)
alexnet = Net(alexnet)

# train
alexnet.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(alexnet.parameters(), lr=0.0008, momentum=0.9)
valid_accuracy = 0
for epoch in range(10):
    running_loss = 0
    train_correct = 0
    train_total = 0
    for i, data in enumerate(trainloader):
        images, labels = data[0].to(device), data[1].to(device, dtype=torch.int64)
        optimizer.zero_grad()
        outputs = alexnet(images)

        m = nn.LogSoftmax()
        logit = m(outputs)

        t_logit = logit/torch.Tensor([1.0, 1.0, 1.0]).to(device)
        softmax = t_logit / 20.00
        nllloss = nn.NLLLoss()
        loss = nllloss(softmax, labels)

        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 199:
            print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/200))
            running_loss = 0.0
    train_accuracy = 100 * train_correct / train_total
    print('train dataset accuracy %.4f' % train_accuracy)

    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for data in validloader:
            images, labels = data[0].to(device), data[1].to(device, dtype=torch.int64)
            outputs = alexnet(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    valided_accuracy = 100*test_correct/test_total
    print('valid dataset accuracy %.4f' % valided_accuracy)
    if valided_accuracy > valid_accuracy:
        torch.save(alexnet.state_dict(), checkpoint_path)
        valid_accuracy = valided_accuracy
