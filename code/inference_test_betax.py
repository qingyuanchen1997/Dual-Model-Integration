from PIL import Image
from torchvision import models
from astropy.io import fits
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import warnings
warnings.filterwarnings('ignore')

# relevant path      
basedir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
data_path = basedir + '/data/test_input/continuum/'
save_path = basedir + '/user_data/tmp_data/images/'
model_path = basedir + '/user_data/model_data/test_betax.pt'
result_path = basedir + '/user_data/tmp_data/results/result_2.txt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


paths = os.listdir(data_path)
paths = sorted(paths)
res_name = [p[29: 35] for p in paths]
paths = [data_path+p for p in paths]
if not os.path.exists(save_path):
    os.makedirs(save_path)
print("fits to jpg... start")
for i, p in enumerate(paths):
    with fits.open(p) as f:
        f.verify("fix")
        img = f[1].data
        name = plt.imsave(save_path + res_name[i] + '.jpg', img, cmap='gray')
        if i % 20 == 19:
            print("doing...{}".format(i))
print('fits to jpg... end')
print('start test...')


# data loading
class ImageDataset(data.Dataset):
    def __init__(self, paths, transform=None):
        self.annotations = paths
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.annotations[index]
        image = Image.open(img_path).convert('RGB').resize((500, 375))
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.annotations)


test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_paths = [save_path + p + ".jpg" for p in res_name]
testset = ImageDataset(test_paths, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)


# model building
class Net(nn.Module):
    def __init__(self, alexnet):
        super(Net, self).__init__()
        self.alexnet = alexnet
        self.backbone = alexnet.features
        self.avgpool = alexnet.avgpool
        self.logit = nn.Linear(2304, 3)

    def forward(self,x):
        batch_size, C, H, W = x.shape
        x = self.backbone(x)
        x = self.avgpool(x)
        x = F.adaptive_avg_pool2d(x, 3).reshape(batch_size, -1)
        x = self.logit(x)
        return x
    

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
alexnet = models.alexnet(pretrained=False)
alexnet_changed = Net(alexnet)
PATH = model_path
alexnet_changed.load_state_dict(torch.load(PATH))
alexnet_changed.to(device)

res = ""
alpha = 0.5
beta = 0.5
with torch.no_grad():
    for i, data in enumerate(testloader):
        images = data.to(device)
        outputs = alexnet_changed(images)

        if abs(outputs[:, 1][0] - outputs[:, 2][0]) < beta:
            outputs[:, 1] = outputs[:, 1] + alpha

        _, predicted = torch.max(outputs.data, 1)
        res += res_name[i] + " " + str(int(predicted[0])+1) + "\n"

res = res.strip()
with open(result_path, 'w') as f:
    f.write(res)
print('finished...')