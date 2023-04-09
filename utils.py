import numpy as np
import random
import torch
import os
import warnings
warnings.filterwarnings('ignore')
from searchspace import *


def set_random_seed(seed):
    seed = seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def count_parameters_in_MB(net, args, logger):
    args.num_parameters_in_MB = np.sum(np.prod(p.size()) for n, p in net.named_parameters())/1e6
    logger.info(f'Total number of parameters: {args.num_parameters_in_MB} (MB)')


def show_search_space(logger):
    logger.info('Candidates in SuperNet:')
    # logger.info(f'SKIP ({(len(SKIP_CANDIDATES))}): {SKIP_CANDIDATES}')
    # logger.info(f'TSFM ({(len(TSFM_CANDIDATES))}): {TSFM_CANDIDATES}')
    logger.info(f'NEIGH ({(len(NEIGH_CANDIDATES))}): {NEIGH_CANDIDATES}')
    logger.info(f'NORM ({(len(NORM_CANDIDATES))}): {NORM_CANDIDATES}')
    logger.info(f'AGGR ({(len(AGGR_CANDIDATES))}): {AGGR_CANDIDATES}')
    logger.info(f'COMB ({(len(COMB_CANDIDATES))}): {COMB_CANDIDATES}')
    logger.info(f'JK ({(len(JK_CANDIDATES))}): {JK_CANDIDATES}')


def show_searched_ops(best_model, logger):
    logger.info('Searched operators:')
    # logger.info(f'SKIP: {best_model.searched_arch["SKIP"]}')
    # logger.info(f'TSFM: {best_model.searched_arch["TSFM"]}')
    logger.info(f'NEIGH: {best_model.searched_arch["NEIGH"]}')
    logger.info(f'NORM: {best_model.searched_arch["NORM"]}')
    logger.info(f'AGGR: {best_model.searched_arch["AGGR"]}')
    logger.info(f'COMB: {best_model.searched_arch["COMB"]}')
    logger.info(f'JK: {best_model.searched_arch["JK"]}')


