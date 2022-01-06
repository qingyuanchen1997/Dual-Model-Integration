import torch
import torchvision
from torchvision import models, datasets, transforms
from torch import nn
from myDataset_test import myDataset
from torch.autograd import Variable
import save_result
import numpy as np
from myModel import Model
import os

# relevant path      
basedir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
data_path = basedir + '/user_data/tmp_data/test_input/continuum/'
model_path = basedir + '/user_data/model_data/test_alpha.pt'
result_path = basedir + '/user_data/tmp_data/results/result_1.txt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model establishing and weights loading
model = Model()
weight_dir = model_path
checkpoint = torch.load(weight_dir)
state_dict = checkpoint['state_dict']

unParalled_state_dict = {}
for key in state_dict.keys():
    unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
model.load_state_dict(state_dict)
model = model.to(device)

# data loading
batch_size = 64

transform = transforms.Compose([#transforms.Grayscale(),
                                transforms.ToPILImage(mode=None),
                                transforms.Resize([224, 224]),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])
totensor = transforms.Compose([
    transforms.ToTensor(),
])
test_dataset = myDataset(data_path,totensor , transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# inference
ind = 0
preds = torch.zeros(len(test_dataset), 1)
m = nn.Softmax(dim=1)
model.eval()
for param in model.parameters():
    param.requires_grad = False
for data in test_loader:
    print('testing')
    X = data
    X1=X[:,0,:,:]
    X2=X[:,1,:,:]
    X1 = X1.unsqueeze(1)
    X2 = X2.unsqueeze(1)
    X1, X2= Variable(X1.to(device)), Variable(X2.to(device))
    pred = model(X1,X2)
    output = m(pred.data)
    _, res = torch.max(output, 1)
    len = X.shape[0]
    preds[ind:ind + len, 0] = res[0:len]
    ind = ind + len

# saving result
label_name = ['alpha', 'beta', 'betax']
save_result.save_pred(y_pred=preds, res_path=result_path)