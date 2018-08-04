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

        self.index = torch.nn.Parameter(torch.FloatTensor(1,self.gC,1,1,1),requires_grad=False)
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



if __name__ == '__main__':
    from torch.autograd import Variable

    C = 40
    G = 4
    amax = ArgMax(C, G)
    amax.cuda()
    amax.eval()

    x = torch.rand(1, C, 3, 3) * 100
    x = Variable(x).cuda()
    y = amax(x)
    print(y.size())
    print(y[:,:2])
    print(y[:,2:])



