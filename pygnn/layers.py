import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):

    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, batch_size, in_features, out_features, mu0, sigma0, scale):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.batch_size = batch_size
        self.mu0 = mu0
        self.sigma0 = sigma0
        self.scale = scale
        if self.scale:
            self.weight = Parameter(torch.ones(batch_size, in_features, out_features))
        else:
            self.weight = Parameter(torch.FloatTensor(batch_size, in_features, out_features))

        self.reset_parameters()

    def reset_parameters(self):

        if self.scale:
            lr1,lr2 = torch.chunk(self.weight,chunks=2,dim=2)
            with torch.no_grad():
                lr1.mul_(self.mu0**0.5)
                lr2.mul_(self.sigma0**0.5)
        else:
            torch.nn.init.xavier_uniform_(self.weight, gain=1.0)

    def forward(self, input, adj):
        support = torch.bmm(input, self.weight)
        output = torch.bmm(adj, support)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class InnerProduct(Module):

    def __init__(self, in_dim):
        super(InnerProduct, self).__init__()
        self.in_dim = in_dim

    def forward(self, input):

        x,y = torch.chunk(input,chunks=2,dim=2)
        y = y.permute(0,2,1)
        xy = torch.bmm(x,y)
        xy = torch.flatten(xy)
        return xy

    def __repr__(self):
        return self.__class__.__name__