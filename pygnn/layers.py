import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):

    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, batch_size, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.batch_size = batch_size
        self.weight = Parameter(torch.FloatTensor(batch_size, in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
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