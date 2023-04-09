import networkx as nx
import numpy as np
import torch
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_networkx, dropout_adj


def get_multihop(data, kmax, prune=True, adj_dprate=0):
    assert kmax >= 2
    multihop_list = [data['edge_index']]

    G = to_networkx(data, to_undirected=True)
    adj = nx.adjacency_matrix(G)
    adjk = adj
    for k in range(2, kmax+1):
        adjk *= adj
        min_value = k if prune else 1
        # adj_dprate = 0.5 if drop else 0
        edge_index_khop = torch.tensor(np.vstack((adjk>=min_value).nonzero()), dtype=torch.long)
        edge_index_khop, _ = dropout_adj(edge_index_khop, p=adj_dprate, force_undirected=True)

        multihop_list.append(edge_index_khop)
        # data[f'edge_index_{k}hop'] = edge_index_khop
    return multihop_list


# def get_knn(data):
#     edge_index_knn = knn_graph(data['x'], k=data['avg_degree'])
#     return edge_index_knn

def get_knn(data, attr):
    assert hasattr(data, attr)
    edge_index_knn = knn_graph(data[attr], k=data['avg_degree'])
    return edge_index_knn