import torch
from torch import Tensor, norm, nn, einsum
from torch_sparse import SparseTensor, fill_diag, mul
from torch_sparse import sum as sparsesum
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add
from torch_geometric.utils import add_self_loops, dropout_adj
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import warnings
warnings.filterwarnings('ignore')
from searchspace import *


def rel_pe_norm(edge_index, pe, edge_mlp):
    if isinstance(edge_index, Tensor):
        rel_coors = pe[edge_index[0]] - pe[edge_index[1]]

    elif isinstance(edge_index, SparseTensor):
        rel_coors = pe[edge_index.to_torch_sparse_coo_tensor()._indices()[0]] - \
                    pe[edge_index.to_torch_sparse_coo_tensor()._indices()[1]]

    rel_dist = (rel_coors ** 2).sum(dim=-1, keepdim=True)
    norm = edge_mlp(rel_dist)
    return norm.squeeze()


def fagcn_like_norm(edge_index, x, g):
    h_i_j = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
    norm = einsum('Ed,Id->E', h_i_j, g)
    norm = torch.tanh(norm)
    return norm


def sage_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv = 1. / deg
        deg_inv[deg_inv == float('inf')] = 0
        edge_weight = edge_weight * deg_inv[row]
        return edge_index, edge_weight



def skip_trans(x, xs, skip_ws, layer_idx):
    def skip_map(x, xs, skip_mode):
        if skip_mode == None:
            return x
        if skip_mode == 'add':
            return x + xs[layer_idx-1]

    l = [skip_map(x, xs, skip_mode) for skip_mode in SKIP_CANDIDATES]
    out = einsum('i,ijk->jk', skip_ws[layer_idx-1], torch.stack(l))
    return out


def comb_trans(m, x, comb_lns, comb_ws, layer_idx):
    def comb_map(m, x, comb_mode):
        if comb_mode == None:
            return m
        if comb_mode == 'add':
            return m + x
        if comb_mode == 'cat':
            return comb_lns[layer_idx](torch.cat([x, m], dim=-1))

    l = [comb_map(m, x, comb_mode) for comb_mode in COMB_CANDIDATES]
    out = einsum('i,ijk->jk', comb_ws[layer_idx-1], torch.stack(l))
    return out


def jk_trans(xs, jks, jk_ws):
    l = [jk(xs) for jk in jks]
    out = einsum('i,ijk->jk', jk_ws[0], torch.stack(l))
    return out


def tsfm_trans(x, lns, tsfm_ws, layer_idx):
    def tsfm_map(x, ln, tsfm_mode):
        if tsfm_mode == None:
            return x
        if tsfm_mode == 'linear':
            return ln(x)

    l = [tsfm_map(x, lns[layer_idx], tsfm_mode) for tsfm_mode in TSFM_CANDIDATES]
    out = einsum('i,ijk->jk', tsfm_ws[layer_idx], torch.stack(l))
    return out


# def neigh_prop_trans(x, data, propss, neigh_ws, aggr_ws, norm_ws, layer_idx, adj_dprate):
def neigh_prop_trans(x, data, propss, neigh_ws, aggr_ws, norm_ws, layer_idx, max_degree):
    rwpe, lepe, avg_degree = data.random_walk_pe, data.laplacian_eigenvector_pe, data.avg_degree
    # rwpe, lepe, avg_degree = None, None, data.avg_degree
    adj_dprate = 0 if avg_degree <= max_degree else (1-max_degree/avg_degree)

    for neigh_mode in NEIGH_CANDIDATES:
        data[neigh_mode], _ = dropout_adj(data[neigh_mode], p=adj_dprate, force_undirected=True)

    l = [prop_trans(x, data[neigh_mode], propss, rwpe, lepe, aggr_ws, norm_ws, layer_idx)
         for neigh_mode in NEIGH_CANDIDATES]
    out = einsum('i,ijk->jk', neigh_ws[layer_idx], torch.stack(l))
    return out


def prop_trans(x, edge_index, propss, rwpe, lepe, aggr_ws, norm_ws, layer_idx):
    l = []
    for props in propss:
        _ = [prop(x, edge_index, rwpe, lepe) for prop in props]
        out = einsum('i,ijk->jk', aggr_ws[layer_idx], torch.stack(_))
        l.append(out)
    out = einsum('i,ijk->jk', norm_ws[layer_idx], torch.stack(l))
    return out


class Prop(MessagePassing):
    # Prop is parameter free
    def __init__(self, aggr_mode, norm_mode, args=None):
        super().__init__(aggr=aggr_mode)
        self.args = args
        self.aggr_mode = aggr_mode
        self.norm_mode = norm_mode
        self.edge_mlp = nn.Sequential(nn.Linear(1, self.args.rwpe_dim),
                                      nn.Linear(self.args.rwpe_dim, 1),
                                      nn.Sigmoid())
        # self.g = nn.Parameter(torch.Tensor(1, 2*self.args.hidden_dim))
        self.g = nn.Parameter(torch.Tensor(1, 2*self.args.num_classes))
        nn.init.xavier_uniform_(self.g)


    def forward(self, x, edge_index, rwpe, lepe):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        norm = self.get_norm(self.norm_mode, x, edge_index, rwpe, lepe)
        out = self.propagate(edge_index, x=x, norm=norm)
        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]
        return norm.view(-1, 1) * x_j

    # def update(self):
    #    pass

    def get_norm(self, norm_mode, x, edge_index, rwpe, lepe):

        assert norm_mode in NORM_CANDIDATES
        if norm_mode is None:
            norm = torch.ones(edge_index.size(1), device=edge_index.device)
        if norm_mode == 'degree_sys':
            _, norm = gcn_norm(edge_index=edge_index, num_nodes=x.size(0), add_self_loops=False)
        if norm_mode == 'degree_row':
            _, norm = sage_norm(edge_index=edge_index, num_nodes=x.size(0), add_self_loops=False)
        if norm_mode == 'rel_rwpe':
            norm = rel_pe_norm(edge_index, rwpe, self.edge_mlp)
        if norm_mode == 'rel_lepe':
            norm = rel_pe_norm(edge_index, lepe, self.edge_mlp)
        if norm_mode == 'fagcn_like':
            norm = fagcn_like_norm(edge_index, x, self.g)
        # norm has shape size(E, )
        return norm

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.aggr_mode, self.norm_mode)
