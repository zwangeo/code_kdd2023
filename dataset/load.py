import torch
from datasets import MyPlanetoid, MyCitationFull, MyAmazon, MyWebKB, MyWikipediaNetwork, MyActor, MyFlickr, MyCoauthor, MyAirports


def get_data(args):
    dataset, data, num_features, num_classes = load_raw(args.data)
    data = permute_masks(data, num_classes, args)
    # if args.data.lower() in ['computers', 'photo']:
    #     args.adj_dprate = 0.7
    return data, num_features, num_classes


def load_raw(name):
    name = name.lower()
    if name in ['cora', 'citeseer', 'pubmed']:
        dataset = MyPlanetoid(root='./data', name=name)

    elif name in ['corafull']:
        dataset = MyCitationFull(root='./data', name='cora')

    elif name in ['dblp']:
        dataset = MyCitationFull(root='./data', name=name)

    elif name in ['computers', 'photo']:
        dataset = MyAmazon(root='./data', name=name)

    elif name in ['cs', 'physics']:
        dataset = MyCoauthor(root='./data', name=name)

    elif name in ['actor']:
        dataset = MyActor(root='./data/actor')

    elif name in ['cornell', 'texas', 'wisconsin']:
        dataset = MyWebKB(root='./data', name=name)

    elif name in ['chameleon', 'squirrel']:
        dataset = MyWikipediaNetwork(root='./data', name=name)
        # preProcDs = MyWikipediaNetwork(root='./data', name=name, geom_gcn_preprocess=False)
        # dataset = MyWikipediaNetwork(root='./data', name=name, geom_gcn_preprocess=True)
        # num_features = dataset.num_features
        # num_classes = dataset.num_classes
        # data = dataset[0]
        # data.edge_index = preProcDs[0].edge_index
        # return dataset, data, num_features, num_classes

    elif name in ['flickr']:
        dataset = MyFlickr(root='./data/flickr')

    elif name in ['brazil', 'europe', 'usa']:
        dataset = MyAirports(root='./data', name=name)

    else:
        raise ValueError(f'dataset {name} not supported')

    num_features = dataset.num_features
    num_classes = dataset.num_classes
    data = dataset[0]
    return dataset, data, num_features, num_classes


# adopted from GPRGNN:
# https://github.com/jianhao2016/GPRGNN/blob/dc246833865af87ae5d4e965d189608f8832ddac/src/utils.py
# f2gnn utilizes StratifiedKFold, which is equivalent (except that mask setting is a bit different)

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def permute_masks(data, num_classes, args):
    # if args.data == 'chameleon':
    #     data.train_mask = data.train_mask[:,0]
    #     data.val_mask = data.val_mask[:,0]
    #     data.test_mask = data.test_mask[:,0]

    # else:
    num_train_per_class = int(round(args.train_rate * len(data.y) / num_classes))
    num_val = int(round(args.val_rate * len(data.y)))

    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        # index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:num_train_per_class] for i in indices], dim=0)
    rest_index = torch.cat([i[num_train_per_class:] for i in indices], dim=0)
    train_index = train_index[torch.randperm(train_index.size(0))]
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(rest_index[:num_val], size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index[num_val:], size=data.num_nodes)
    return data
