import torch
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
from torch import nn
import numpy as np
import scipy
from myDataset2 import myDataset
import os

num_epoch_no_improvement = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_features1 = models.resnet18(pretrained=False)#torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')

model_features1.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

model_features1.layer2[0].downsample=nn.Sequential(
    nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), padding=0),
    nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False),
    nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
)
model_features1.layer3[0].downsample=nn.Sequential(
    nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), padding=0),
    nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
    nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
)
model_features1.layer4[0].downsample=nn.Sequential(
    nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), padding=0),
    nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
    nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
)

model_features1.fc = nn.Sequential(
    nn.Linear(512, 128),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),
    nn.Linear(128, 3)
)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features1 = model_features1

    def forward(self, x):
        y = self.features1(x)
        return y
#model = Model()
model = model_features1
#for param in model.features.parameters():
#    param.requires_grad = False
model = model.to(device)

m = nn.Softmax(dim=1)

# data loading
batch_size = 64

transform = transforms.Compose([#transforms.Grayscale(),
                                #transforms.RandomGrayscale(p=0.1),
                                #transforms.ToPILImage(mode=None),
                                #transforms.Resize([224, 224]),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomVerticalFlip(p=0.5),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

transform_val = transforms.Compose([#transforms.Grayscale(),
                                #transforms.RandomGrayscale(p=0.1),
                                #transforms.ToPILImage(mode=None),
                                #transforms.Resize([224, 224]),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

totensor = transforms.Compose([
    transforms.ToTensor(),
])
basedir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
train_dataset = myDataset(basedir+'/user_data/tmp_data/train_input/continuum/',totensor, transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# loss function and optimizer
loss_f = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.56,1.0,3.05])).float().to(device))
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = 2e-6)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=5, verbose=True, threshold=0.05)

# training and validating

train_acc = []
train_loss = []

val_loss = []
val_acc = []

initial_epoch = 0
num_epoch = 100

best_acc = 0.0
patience = 5
for epoch in range(initial_epoch, num_epoch):

    training_loss = 0.0
    training_cors = 0
    validating_loss = 0.0
    validating_cors = 0
    model.train()
    print("model is in training phase...")
    stop=0
    std_sum = 0
    for data in train_loader:
        X, y = data
        X1=X[:,0,:,:]
        X1 = X1.unsqueeze(1)
        X1, y = Variable(X1.to(device)), Variable(y.to(device))
        pred = model(X1)

        output = m(pred.data)
        std_sum = std_sum + output.std()
        _, res = torch.max(output, 1)

        optimizer.zero_grad()
        loss = loss_f(pred, y)# + 4*output.std()

        loss.backward()
        optimizer.step()

        training_loss += loss.data * len(y)
        training_cors += torch.sum(res == y.data)

    epoch_loss = training_loss / len(train_dataset)
    epoch_acc =  training_cors.float() / len(train_dataset)
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    print('std_sum=',std_sum)
    print("Train Loss:{:.4f} Acc:{:.4f}".format(epoch_loss, epoch_acc))
    if epoch_acc>0.90:
        torch.save({
                'epoch': epoch+1,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()

        }, basedir+'/user_data/model_data/resnet18x2_change_c.pt')
        print("Saving continuum model!")
        break