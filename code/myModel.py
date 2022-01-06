import torch
import torchvision
from torchvision import models, datasets, transforms
from torch import nn

# model loading
model_features1 = models.resnet18(pretrained=False)#torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
model_features2 = models.resnet18(pretrained=False)

model_features1.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model_features2.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
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

model_features2.layer2[0].downsample=nn.Sequential(
    nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), padding=0),
    nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False),
    nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
)
model_features2.layer3[0].downsample=nn.Sequential(
    nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), padding=0),
    nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
    nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
)
model_features2.layer4[0].downsample=nn.Sequential(
    nn.AvgPool2d(kernel_size=(2,2), stride=(2,2), padding=0),
    nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False),
    nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
)
model_features1.fc = nn.Sequential(

)
model_features2.fc = nn.Sequential(

)
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features1 = model_features1
        self.features2 = model_features2
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 3)
        )

    def forward(self, x, y):
        x = self.features1(x)
        y = self.features2(y)
        xy=torch.cat((x,y),dim=1)
        #x = x.view(-1, 2000)
        xy = self.classifier(xy)
        return xy