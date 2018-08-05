'''ZipNet in PyTorch.

Custom trial
'''

'''ZipNet in PyTorch.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArgMax(nn.Module):
    def __init__(self, channels, groups):
        super(ArgMax, self).__init__()
        self.groups = groups
        self.gC = int(channels / groups)

        self.index = torch.nn.Parameter(torch.FloatTensor(1, self.gC, 1, 1, 1), requires_grad=False)
        self.index.data = torch.arange(1, self.gC + 1).view(1, self.gC, 1, 1, 1).float() / self.gC

    def forward(self, x):
        '''Computes Maximum Values/ Index and Concatenate them'''
        N,C,H,W = x.size()
        g = self.groups
        y = x.view(N,C/g,g,H,W)
        if self.training:
            scores = F.softmax(y,1)
            coord = torch.sum(self.index*scores, 1)
            val,_ = torch.max(y, 1)
        else:
            val, coord = torch.max(y, 1)
            coord = (coord + 1).float() / self.gC

        result = torch.cat([val, coord], 1)
        return result

class Bottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, stride, groups):
        super(Bottleneck, self).__init__()
        self.stride = stride

        mid_planes = out_planes/4
        g = mid_planes/4
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, groups=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.amax = ArgMax(channels=mid_planes, groups=g)
        self.conv2 = nn.Conv2d(g*2, mid_planes, kernel_size=7, stride=stride, padding=3, groups=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, groups=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 2:
            self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.amax(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        res = self.shortcut(x)
        out = F.relu(torch.cat([out,res], 1)) if self.stride==2 else F.relu(out+res)
        return out

class ZipNet(nn.Module):
    def __init__(self, cfg):
        super(ZipNet, self).__init__()
        out_planes = cfg['out_planes']
        num_blocks = cfg['num_blocks']
        groups = cfg['groups']

        self.conv1 = nn.Conv2d(3, 32, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.in_planes = 32
        self.layer1 = self._make_layer(out_planes[0], num_blocks[0], groups)
        self.layer2 = self._make_layer(out_planes[1], num_blocks[1], groups)
        self.layer3 = self._make_layer(out_planes[2], num_blocks[2], groups)
        self.linear = nn.Linear(out_planes[2], 10)

    def _make_layer(self, out_planes, num_blocks, groups):
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            cat_planes = self.in_planes if i == 0 else 0
            layers.append(Bottleneck(self.in_planes, out_planes-cat_planes, stride=stride, groups=groups))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ZipNetG2():
    cfg = {
        'out_planes': [256,512,1024],
        'num_blocks': [4,8,4,4],
        'groups': 16
    }
    return ZipNet(cfg)

def ZipNet3():
    cfg = {
        'out_planes': [240,480,960],
        'num_blocks': [4,8,4],
        'groups': 3
    }
    return ZipNet(cfg)


if __name__ == '__main__':
    from torch.autograd import Variable

    # C = 40
    # G = 4
    # amax = ArgMax(C, G)
    # amax.cuda()
    # amax.eval()
    #
    # x = torch.rand(1, C, 3, 3) * 100
    # x = Variable(x).cuda()
    # y = amax(x)
    # print(y.size())
    # print(y[:,:2])
    # print(y[:,2:])

    net = ZipNetG2()
    x = torch.randn(1, 3, 32, 32)
    y = net(x)
    print(y)



