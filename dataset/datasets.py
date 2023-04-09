import json
import sys
import os.path as osp
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.transforms import ToUndirected, AddRandomWalkPE
from torch_sparse import coalesce, SparseTensor
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import read_planetoid_data, read_npz
from torch_geometric.datasets import Planetoid, CitationFull, Amazon, WebKB, WikipediaNetwork, Airports, Actor, Flickr, Coauthor
from neigh import get_multihop, get_knn
from lepe import MyAddLaplacianEigenvectorPE
from gdc import MyGDC


def add_node_attr_nc(data, rw_dim=16, le_dim=16):
    toundirected = ToUndirected()
    toundirected(data)
    # data['random_walk_pe'], data['laplacian_eigenvector_pe'] = None, None
    rwpe = AddRandomWalkPE(walk_length=rw_dim)
    lepe = MyAddLaplacianEigenvectorPE(k=le_dim)
    rwpe(data)
    lepe(data)

    data['avg_degree'] = int(np.rint(data['edge_index'].size(1) / data['x'].size(0)))
    data['edge_index_2hop'] = get_multihop(data, kmax=2, prune=True)[-1]
    data['edge_index_knn'] = get_knn(data, attr='x')
    data['edge_index_knn_rwpe'] = get_knn(data, attr='random_walk_pe')
    data['edge_index_knn_lepe'] = get_knn(data, attr='laplacian_eigenvector_pe')

    # gdc_ppr = MyGDC(diffusion_kwargs=dict(method='ppr', alpha=0.15),
    #                 sparsification_kwargs=dict(method='topk', k=10, dim=0),
    #                 exact=True)
    # gdc_heat = MyGDC(diffusion_kwargs=dict(method='heat', t=4),
    #                 sparsification_kwargs=dict(method='topk', k=10, dim=0),
    #                 exact=True)
    # gdc_ppr(data)
    # gdc_heat(data)
    return data


class MyPlanetoid(Planetoid):
    def process(self):
        data = read_planetoid_data(self.raw_dir, self.name)

        if self.split == 'geom-gcn':
            train_masks, val_masks, test_masks = [], [], []
            for i in range(10):
                name = f'{self.name.lower()}_split_0.6_0.2_{i}.npz'
                splits = np.load(osp.join(self.raw_dir, name))
                train_masks.append(torch.from_numpy(splits['train_mask']))
                val_masks.append(torch.from_numpy(splits['val_mask']))
                test_masks.append(torch.from_numpy(splits['test_mask']))
            data.train_mask = torch.stack(train_masks, dim=1)
            data.val_mask = torch.stack(val_masks, dim=1)
            data.test_mask = torch.stack(test_masks, dim=1)

        data = data if self.pre_transform is None else self.pre_transform(data)
        data = add_node_attr_nc(data)
        print('Project specific processing...', file=sys.stderr)
        torch.save(self.collate([data]), self.processed_paths[0])


class MyCitationFull(CitationFull):
    def process(self):
        data = read_npz(self.raw_paths[0])
        data = data if self.pre_transform is None else self.pre_transform(data)

        data = add_node_attr_nc(data)
        print('Project specific processing...', file=sys.stderr)
        torch.save(self.collate([data]), self.processed_paths[0])


class MyCoauthor(Coauthor):
    def process(self):
        data = read_npz(self.raw_paths[0])
        data = data if self.pre_transform is None else self.pre_transform(data)
        data = add_node_attr_nc(data)
        print('Project specific processing...', file=sys.stderr)
        torch.save(self.collate([data]), self.processed_paths[0])


class MyAmazon(Amazon):
    def process(self):
        data = read_npz(self.raw_paths[0])
        data = data if self.pre_transform is None else self.pre_transform(data)
        # data = add_node_attr_nc(data)
        data = add_node_attr_nc(data)
        print('Project specific processing...', file=sys.stderr)
        torch.save(self.collate([data]), self.processed_paths[0])


class MyWebKB(WebKB):
    def process(self):
        with open(self.raw_paths[0], 'r') as f:
            data = f.read().split('\n')[1:-1]
            x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
            x = torch.tensor(x, dtype=torch.float)

            y = [int(r.split('\t')[2]) for r in data]
            y = torch.tensor(y, dtype=torch.long)

        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        train_masks, val_masks, test_masks = [], [], []
        for f in self.raw_paths[2:]:
            tmp = np.load(f)
            train_masks += [torch.from_numpy(tmp['train_mask']).to(torch.bool)]
            val_masks += [torch.from_numpy(tmp['val_mask']).to(torch.bool)]
            test_masks += [torch.from_numpy(tmp['test_mask']).to(torch.bool)]
        train_mask = torch.stack(train_masks, dim=1)
        val_mask = torch.stack(val_masks, dim=1)
        test_mask = torch.stack(test_masks, dim=1)

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)
        data = data if self.pre_transform is None else self.pre_transform(data)
        data = add_node_attr_nc(data)
        print('Project specific processing...', file=sys.stderr)
        torch.save(self.collate([data]), self.processed_paths[0])


class MyWikipediaNetwork(WikipediaNetwork):
    def process(self):
            if self.geom_gcn_preprocess:
                with open(self.raw_paths[0], 'r') as f:
                    data = f.read().split('\n')[1:-1]
                x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
                x = torch.tensor(x, dtype=torch.float)
                y = [int(r.split('\t')[2]) for r in data]
                y = torch.tensor(y, dtype=torch.long)

                with open(self.raw_paths[1], 'r') as f:
                    data = f.read().split('\n')[1:-1]
                    data = [[int(v) for v in r.split('\t')] for r in data]
                edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
                edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

                train_masks, val_masks, test_masks = [], [], []
                for filepath in self.raw_paths[2:]:
                    f = np.load(filepath)
                    train_masks += [torch.from_numpy(f['train_mask'])]
                    val_masks += [torch.from_numpy(f['val_mask'])]
                    test_masks += [torch.from_numpy(f['test_mask'])]
                train_mask = torch.stack(train_masks, dim=1).to(torch.bool)
                val_mask = torch.stack(val_masks, dim=1).to(torch.bool)
                test_mask = torch.stack(test_masks, dim=1).to(torch.bool)

                data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                            val_mask=val_mask, test_mask=test_mask)

            else:
                data = np.load(self.raw_paths[0], 'r', allow_pickle=True)
                x = torch.from_numpy(data['features']).to(torch.float)
                edge_index = torch.from_numpy(data['edges']).to(torch.long)
                edge_index = edge_index.t().contiguous()
                edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))
                y = torch.from_numpy(data['target']).to(torch.float)

                data = Data(x=x, edge_index=edge_index, y=y)

            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data = add_node_attr_nc(data)
            print('Project specific processing...', file=sys.stderr)
            torch.save(self.collate([data]), self.processed_paths[0])


class MyActor(Actor):
    def process(self):

        with open(self.raw_paths[0], 'r') as f:
            data = [x.split('\t') for x in f.read().split('\n')[1:-1]]

            rows, cols = [], []
            for n_id, col, _ in data:
                col = [int(x) for x in col.split(',')]
                rows += [int(n_id)] * len(col)
                cols += col
            x = SparseTensor(row=torch.tensor(rows), col=torch.tensor(cols))
            x = x.to_dense()

            y = torch.empty(len(data), dtype=torch.long)
            for n_id, _, label in data:
                y[int(n_id)] = int(label)

        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        train_masks, val_masks, test_masks = [], [], []
        for f in self.raw_paths[2:]:
            tmp = np.load(f)
            train_masks += [torch.from_numpy(tmp['train_mask']).to(torch.bool)]
            val_masks += [torch.from_numpy(tmp['val_mask']).to(torch.bool)]
            test_masks += [torch.from_numpy(tmp['test_mask']).to(torch.bool)]
        train_mask = torch.stack(train_masks, dim=1)
        val_mask = torch.stack(val_masks, dim=1)
        test_mask = torch.stack(test_masks, dim=1)

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)
        data = data if self.pre_transform is None else self.pre_transform(data)
        data = add_node_attr_nc(data)
        print('Project specific processing...', file=sys.stderr)
        torch.save(self.collate([data]), self.processed_paths[0])


class MyFlickr(Flickr):
    def process(self):
        f = np.load(osp.join(self.raw_dir, 'adj_full.npz'))
        adj = sp.csr_matrix((f['data'], f['indices'], f['indptr']), f['shape'])
        adj = adj.tocoo()
        row = torch.from_numpy(adj.row).to(torch.long)
        col = torch.from_numpy(adj.col).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)

        x = np.load(osp.join(self.raw_dir, 'feats.npy'))
        x = torch.from_numpy(x).to(torch.float)

        ys = [-1] * x.size(0)
        with open(osp.join(self.raw_dir, 'class_map.json')) as f:
            class_map = json.load(f)
            for key, item in class_map.items():
                ys[int(key)] = item
        y = torch.tensor(ys)

        with open(osp.join(self.raw_dir, 'role.json')) as f:
            role = json.load(f)

        train_mask = torch.zeros(x.size(0), dtype=torch.bool)
        train_mask[torch.tensor(role['tr'])] = True

        val_mask = torch.zeros(x.size(0), dtype=torch.bool)
        val_mask[torch.tensor(role['va'])] = True

        test_mask = torch.zeros(x.size(0), dtype=torch.bool)
        test_mask[torch.tensor(role['te'])] = True

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)

        data = data if self.pre_transform is None else self.pre_transform(data)
        data = add_node_attr_nc(data)
        print('Project specific processing...', file=sys.stderr)

        torch.save(self.collate([data]), self.processed_paths[0])


class MyAirports(Airports):
    def process(self):
        index_map, ys = {}, []
        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            for i, row in enumerate(data):
                idx, y = row.split()
                index_map[int(idx)] = i
                ys.append(int(y))
        y = torch.tensor(ys, dtype=torch.long)
        x = torch.eye(y.size(0))

        edge_indices = []
        with open(self.raw_paths[0], 'r') as f:
            data = f.read().split('\n')[:-1]
            for row in data:
                src, dst = row.split()
                edge_indices.append([index_map[int(src)], index_map[int(dst)]])
        edge_index = torch.tensor(edge_indices).t().contiguous()
        edge_index, _ = coalesce(edge_index, None, y.size(0), y.size(0))

        data = Data(x=x, edge_index=edge_index, y=y)
        data = data if self.pre_transform is None else self.pre_transform(data)

        data = add_node_attr_nc(data)
        print('Project specific processing...', file=sys.stderr)

        torch.save(self.collate([data]), self.processed_paths[0])