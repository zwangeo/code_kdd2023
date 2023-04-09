import sys
sys.path.append('./dataset')
from copy import deepcopy
from torch.optim import Adam, SGD
import argparse
import warnings
warnings.filterwarnings('ignore')
from model import *
from utils import *
from dataset.load import get_data
from log import *


parser = argparse.ArgumentParser('Interface')
parser.add_argument('--K', type=str, default='s')
parser.add_argument('--space', type=str, default='')
parser.add_argument('--data', type=str, default='texas', help='data name')
parser.add_argument('--rwpe_dim', type=int, default=16, help='random walk encoding dimension')
parser.add_argument('--lepe_dim', type=int, default=16, help='laplacian eigenmap dimension')
parser.add_argument('--train_rate', type=float, default=0.6)
parser.add_argument('--val_rate', type=float, default=0.2)
# parser.add_argument('--adj_dprate', type=float, default=0, help='random drop edges for dense graphs')
parser.add_argument('--max_degree', type=int, default=10, help='max ave_degree not to drop edges')
parser.add_argument('--dprate', type=float, default=0.5)

parser.add_argument('--num_layers', type=int, default=4, help='framework layers')
parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension')
parser.add_argument('--num_classes', type=int, default=0, help='num_classes')
parser.add_argument('--temp', type=float, default=0.5, help='temp in softmax')
parser.add_argument('--loc_mean', type=float, default=1, help='initial mean value to generate the location')
parser.add_argument('--loc_std', type=float, default=0.01, help='initial std to generate the location')

parser.add_argument('--num_runs', type=int, default=30, help='number of runs with different seeds')
parser.add_argument('--seed', type=int, default=2531, help='random seed')
parser.add_argument('--epochs', type=int, default=500, help='num of training epochs')
parser.add_argument('--lr', type=float, default=1e-2, help='init learning rate')
# parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--clip', type=float, default=1, help='gradient clipping')

parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--kill_cnt', type=int, default=100, help='early stopping')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')

parser.add_argument('--log_dir', type=str, default='./log', help='log directory')
parser.add_argument('--file_path', type=str, default=None, help='log path, to be set in set_up_log()')
parser.add_argument('--summary_file', type=str, default='summary.log', help='results summary')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)
global device
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
sys_argv = sys.argv


def run():
    if args.K == 's':
        k_min, k_max = 2, 6
    elif args.K == 'm':
        k_min, k_max = 6, 9
    else:
        k_min, k_max = 9, 11

    for num_layers in range(k_min, k_max):
        args.num_layers = num_layers
        for run in range(args.num_runs):
            seed = np.random.randint(0, 10000)
            args.seed = seed
            main()


def main():
    set_random_seed(args.seed)
    logger = set_up_log(args, sys_argv)
    data, num_features, num_classes = get_data(args)

    model = SuperNet(num_features, num_classes, args)
    data, model = data.to(device), model.to(device)
    count_parameters_in_MB(model, args, logger)
    show_search_space(logger)

    model, model_max = search(data, model, logger)
    standalone_train(data, model_max, logger)


def search(data, model, logger):
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model_max = deepcopy(model)  # save model_max on val, refer to https://github.com/Graph-COM/PEG/blob/c6900146d0a4444f742366f299f189971aa2da29/task1/train.py

    for epoch in range(args.epochs):
        model.train()

        t1 = time.time()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        t2 = time.time()
        model.time_se += (t2-t1)

        model.eval()
        with torch.no_grad():
            out = model(data)
            pred = out.argmax(dim=1)
            correct = (pred[data.val_mask] == data.y[data.val_mask]).sum()
            acc = int(correct) / int(data.val_mask.sum())

            if acc > model.max_acc_se:
                model.max_acc_se = np.round(acc, 4)
                model.max_epoch_se = epoch
                model_max = deepcopy(model)
            # print(f'Epoch: {epoch} || Best Val Acc: {model.max_acc_se:.4f} || Current Val Acc: {acc:.4f}')
            logger.info(f'Epoch: {epoch} || Best Val Acc: {model.max_acc_se:.4f} || Current Val Acc: {acc:.4f} || Time Cost: {model.time_se:.4f}')

        if (epoch - model.max_epoch_se) > args.kill_cnt:
            logger.info(f'[search] Early Stopping at Epoch {epoch}!')
            break
    return model, model_max


def standalone_train(data, model_max, logger):
    model_max.derive_arch()
    show_searched_ops(model_max, logger)
    optimizer = Adam(model_max.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        t1 = time.time()
        model_max.train()
        optimizer.zero_grad()
        out = model_max(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        # nn.utils.clip_grad_norm_(model_max.parameters(), args.clip)
        optimizer.step()
        t2 = time.time()
        model_max.time_re += (t2-t1)

        model_max.eval()
        with torch.no_grad():
            out = model_max(data)
            pred = out.argmax(dim=1)
            correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
            acc = int(correct) / int(data.test_mask.sum())

            if acc > model_max.max_acc_re:
                model_max.max_acc_re = np.round(acc, 4)
                model_max.max_epoch_re = epoch
            # print(f'Epoch: {epoch} || Best Test Acc: {model_max.max_acc_re:.4f} || Current Test Acc: {acc:.4f}')
            logger.info(f'Epoch: {epoch} || Best Test Acc: {model_max.max_acc_re:.4f} || Current Test Acc: {acc:.4f} || Time Cost: {model_max.time_re:.4f}')

        if (epoch - model_max.max_epoch_re) > args.kill_cnt:
            logger.info(f'[re_train] Early Stopping at Epoch {epoch}!')
            break
    save_performance_result(args, logger, model_max)



if __name__ == '__main__':
    run()
    # main()






