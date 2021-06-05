import numpy as np
import torch
import pickle as pkl
import networkx as nx
import pycombo
from sklearn.metrics import normalized_mutual_info_score as nmi

def load_data(daily=False):

    if daily:
        graph = pkl.load(open("data/Taxi2017_daily_net.pkl", "rb"))
        g = [x[0] for x in graph[0]]
        adj = np.array([np.array(nx.adjacency_matrix(x).todense(), dtype=float) for x in g])
        adj = adj[:,1:, 1:]
        return np.array(adj)
    else:
        graph = pkl.load(open("data/Taxi2017_total_net.pkl", "rb"))
        adj = np.array([nx.adjacency_matrix(graph[0]).todense()], dtype=float)
        adj = adj[:,1:, 1:]
        return np.array(adj)


def normalize(adj):

    adj = torch.FloatTensor(adj)
    adj_id = torch.FloatTensor(torch.eye(adj.shape[1]))
    adj_id = adj_id.reshape((1, adj.shape[1], adj.shape[1]))
    adj_id = adj_id.repeat(adj.shape[0], 1, 1)
    adj = adj + adj_id
    rowsum = torch.FloatTensor(adj.sum(2))
    degree_mat_inv_sqrt = torch.diag_embed(torch.float_power(rowsum,-0.5), dim1=-2, dim2=-1).float()
    adj_norm = torch.bmm(torch.transpose(torch.bmm(adj,degree_mat_inv_sqrt),1,2),degree_mat_inv_sqrt)

    return adj_norm

def norm_embed(embed):
    embedx,embedy = torch.chunk(embed,chunks=2,dim=2)
    ES = (embedx ** 2).sum(axis=1) / (embedy ** 2).sum(axis=1)
    ES = ES.unsqueeze(dim=1)
    embedx = embedx / (ES ** 0.25)
    embedy = embedy / (ES ** 0.25)
    embed_norm = torch.cat((embedx,embedy),dim=2)
    return embed_norm

def toy_data():

    graph = nx.DiGraph()
    graph.add_nodes_from([1, 2, 3, 4, 5])
    graph.add_edge(1, 2, weight=10)
    graph.add_edge(1, 5, weight=57)
    graph.add_edge(2, 1, weight=8)
    graph.add_edge(2, 4, weight=34)
    graph.add_edge(2, 5, weight=75)
    graph.add_edge(4, 1, weight=24)
    graph.add_edge(5, 4, weight=14)
    graph.add_edge(5, 1, weight=73)
    graph.add_edge(5, 2, weight=48)

    adj = np.array([nx.adjacency_matrix(graph).todense()], dtype=float)

    return adj

def nmi_score(adj1,adj2):

    G1 = nx.from_numpy_matrix(np.array(adj1))
    G2 = nx.from_numpy_matrix(np.array(adj2))
    partition1 = pycombo.execute(G1)
    partition2 = pycombo.execute(G2)
    return nmi(list(partition1[0]),list(partition2[0]))