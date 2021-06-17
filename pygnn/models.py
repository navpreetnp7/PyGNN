import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, InnerProduct
from utils import norm_embed
import torch


class GNN(nn.Module):

    def __init__(self, batch_size, nfeat, nhid, ndim):
        super(GNN, self).__init__()

        self.gc1 = GraphConvolution(batch_size, nfeat, nhid)
        self.embeddings = GraphConvolution(batch_size, nhid, 4 * ndim)
        self.reconstructions = InnerProduct(2 * ndim)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = self.embeddings(x, adj)
        x = norm_embed(x)
        lr1, lr2 = torch.chunk(x, chunks=2, dim=2)
        mu = F.relu(self.reconstructions(lr1))
        sigma = F.relu(self.reconstructions(lr2))
        return mu, sigma
