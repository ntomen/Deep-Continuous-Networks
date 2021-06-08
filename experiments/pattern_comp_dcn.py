import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
#------------------------------------------------------------------------------
# Set working directory
print('\nSetting the directory of '+__file__+' as the working directory...')
script_dir=os.path.dirname(os.path.realpath(__file__))

if os.getcwd() != script_dir:
    os.chdir(script_dir)
import sys
sys.path.append(os.path.abspath(script_dir+"/../"))
from srf.structured_conv_layer import Srf_layer_shared
#------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, choices=['resnet', 'odenet'], default='odenet')
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', type=eval, default=True, choices=[True, False])
parser.add_argument('--downsampling-method', type=str, default='conv', choices=['conv', 'res'])
parser.add_argument('--nepochs', type=int, default=1)
parser.add_argument('--data_aug', type=eval, default=True, choices=[True, False])
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=1000)

parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()
#------------------------------------------------------------------------------
if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint
#------------------------------------------------------------------------------

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,\
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,\
                     bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x

        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + shortcut


class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, init_k, init_order, init_scale,\
                   transpose=False):                                              
          super(ConcatConv2d, self).__init__()
          module = nn.ConvTranspose2d if transpose else Srf_layer_shared
          self._layer = module(inC=dim_in + 1,\
                                 outC=dim_out,\
                                 init_k=init_k,\
                                 init_order=init_order,\
                                 init_scale=init_scale,\
                                 learn_sigma=True,\
                                 use_cuda=torch.cuda.is_available())

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.CELU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 2.0, 2.0, 0.0)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 2.0, 2.0, 0.0)
        self.norm3 = norm(dim)
        
        self.trajectories = []
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        self.trajectories.append((t.to('cpu').detach().numpy(),\
                                  x[0].to('cpu').detach().numpy()))

        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out


class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 2]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=args.tol, atol=args.tol)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


def get_mnist_loaders(data_aug=False, batch_size=128, test_batch_size=1000, perc=1.0):
    if data_aug:
        transform_train = transforms.Compose([\
            transforms.RandomHorizontalFlip(),\
            transforms.RandomCrop(32, padding=4),\
            transforms.ToTensor(),\
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(
        datasets.CIFAR10(root='./', train=True, download=True, transform=transform_train),
        batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
    )

    train_eval_loader = DataLoader(
        datasets.CIFAR10(root='./', train=True, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    test_loader = DataLoader(
        datasets.CIFAR10(root='./', train=False, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    return train_loader, test_loader, train_eval_loader


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


if __name__ == '__main__':

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    # Define model
    is_odenet = args.network == 'odenet'

    if args.downsampling_method == 'conv':
        downsampling_layer1 = [
            norm(32),
            nn.CELU(inplace=True),
            nn.Conv2d(32,64,4,2,1),
        ]
        downsampling_layer2 = [
            norm(64),
            nn.CELU(inplace=True),
            nn.Conv2d(64,128,4,2,1),
        ]

    elif args.downsampling_method == 'res':
        downsampling_layer1 = [
            ResBlock(32, 64, stride=2, downsample=conv1x1(32, 32, 2)),
        ]
        downsampling_layer2 = [
            ResBlock(64, 128, stride=2, downsample=conv1x1(64, 64, 2)),
        ]

    block1=ODEfunc(32)
    block2=ODEfunc(64)
    block3=ODEfunc(128)
    feature_layer1 = \
    [ODEBlock(block1)] if is_odenet else [ResBlock(32, 32) for _ in range(6)]
    feature_layer2 = \
    [ODEBlock(block2)] if is_odenet else [ResBlock(64, 64) for _ in range(6)]
    feature_layer3 = \
    [ODEBlock(block3)] if is_odenet else [ResBlock(128, 128) for _ in range(6)]

    fc_layers = [norm(128), nn.CELU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)),\
                 Flatten(), nn.Linear(128,10)]

    model = nn.Sequential(nn.Conv2d(3,32,3,1),\
                          *feature_layer1,\
                          *downsampling_layer1,\
                          *feature_layer2,\
                          *downsampling_layer2,\
                          *feature_layer3,\
                          *fc_layers).to(device)
    # Load trained model
    A=torch.load('experiment_cifar10_dcn_ode/model.pth')
    model=model.to(device)
    model.load_state_dict(A['state_dict'])
    model.eval()

    # Define dataset loaders
    _, test_loader, _ = get_mnist_loaders(
        args.data_aug, args.batch_size, args.test_batch_size
    )

    data_gen = inf_generator(test_loader)
    batches_per_epoch = len(test_loader)

    accuracy=np.zeros((1,2))

    for itr in range(args.nepochs * batches_per_epoch):
        # Forward pass
        x, y = data_gen.__next__()

        x[:,:,9:23,9:23]=0
        x = x.to(device)
        y = y.to(device)
        
        with torch.no_grad():
            logits = model(x)
            for j in range(int(y.shape[0])):
                i_class=y[j].cpu().numpy().item()

                accuracy[0,1]+=1
                accuracy[0,0]+=(torch.argmax(logits[j,:].cpu()).numpy()==i_class)

        if is_odenet:
            nfe_forward = feature_layer1[0].nfe
            feature_layer1[0].nfe = 0

        # Concatanate trajectories        
        traj1=np.stack([block1.trajectories[i][1]\
                        for i in range(len(block1.trajectories))])
        traj2=np.stack([block2.trajectories[i][1]\
                        for i in range(len(block2.trajectories))])
        traj3=np.stack([block3.trajectories[i][1]\
                        for i in range(len(block3.trajectories))])
        
        t1=np.stack([block1.trajectories[i][0]\
                        for i in range(len(block1.trajectories))])
        t2=np.stack([block2.trajectories[i][0]\
                        for i in range(len(block2.trajectories))])
        t3=np.stack([block3.trajectories[i][0]\
                        for i in range(len(block3.trajectories))])
        
        # Reset trajectories
        block1.trajectories=[]
        block2.trajectories=[]
        block3.trajectories=[]

        # Print val accuracy
        print(c, accuracy[0,0]/accuracy[0,1]*100)



