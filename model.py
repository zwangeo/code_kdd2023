import numpy as np
from torch.nn import Linear, Parameter, ParameterList
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
from jumpingknowledge import JumpingKnowledge
from model_utils import *
from collections import defaultdict as ddict
from searchspace import *


class SuperNet(nn.Module):
    def __init__(self, num_features, num_classes, args):
        super().__init__()
        self.args = args
        self.num_features = num_features
        self.num_classes = num_classes
        self.args.num_classes = num_classes
        global device
        device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

        # self.pre_mp = nn.Sequential(nn.Linear(self.num_features, self.args.hidden_dim), nn.ReLU(),
        #                             nn.Linear(self.args.hidden_dim, self.args.hidden_dim))
        # self.pre_mp = nn.Sequential(nn.Linear(self.num_features, self.args.hidden_dim), nn.ReLU(), nn.Dropout(p=0.5),
        #                             nn.Linear(self.args.hidden_dim, self.num_classes))
        self.pre_mp = nn.Sequential(nn.Linear(self.num_features, self.args.hidden_dim), nn.ReLU(), nn.Dropout(p=self.args.dprate),
                                    nn.Linear(self.args.hidden_dim, self.num_classes))

        # self.lns = nn.ModuleList(
        #     [nn.Linear(self.args.hidden_dim, self.args.hidden_dim) for _ in range(self.args.num_layers)])
        self.comb_lns = nn.ModuleList([nn.Linear(2 * self.num_classes, self.num_classes) for _ in range(self.args.num_layers)])

        self._get_propss()

        # self.post_mp = nn.Sequential(nn.Linear(self.args.hidden_dim, self.args.hidden_dim), nn.ReLU(),
        #                              nn.Linear(self.args.hidden_dim, self.num_classes))

        self._get_jks()
        self._initialize_alphas()
        self._initialize_records()
        self.reset_parameters()


    def reset_parameters(self):
        pass

    def _get_propss(self):
        self.propss = nn.ModuleList([nn.ModuleList([]) for _ in range(len(NORM_CANDIDATES))])
        for idx, norm_mode in enumerate(NORM_CANDIDATES):
            for aggr_mode in AGGR_CANDIDATES:
                self.propss[idx].append(Prop(aggr_mode=aggr_mode, norm_mode=norm_mode, args=self.args))

    def _get_jks(self):
        self.jks = nn.ModuleList([])
        # self.jk_cat_ln = nn.Linear(self.args.hidden_dim * (self.args.num_layers + 1), self.args.hidden_dim)
        self.jk_cat_ln = nn.Linear(self.num_classes * (self.args.num_layers + 1), self.num_classes)

        for jk_mode in JK_CANDIDATES:

            if jk_mode == 'last':
                self.jks.append(JumpingKnowledge(mode='last'))
            elif jk_mode == 'cat':
                self.jks.append(nn.Sequential(JumpingKnowledge(mode='cat'), self.jk_cat_ln))
            elif jk_mode == 'max':
                self.jks.append(JumpingKnowledge(mode='max'))
            elif jk_mode == 'mean':
                self.jks.append(JumpingKnowledge(mode='mean'))
            # elif jk_mode == 'ppr':
            #     self.jks.append(JumpingKnowledge(mode='ppr', K=self.args.num_layers))
            elif 'ppr' in jk_mode:
                self.jks.append(JumpingKnowledge(mode=jk_mode, K=self.args.num_layers))
            elif jk_mode == 'gpr':
                self.jks.append(JumpingKnowledge(mode='gpr', K=self.args.num_layers))
            elif jk_mode == 'lstm':
                # self.jks.append(JumpingKnowledge(mode='lstm', channels=self.args.hidden_dim, num_layers=2))
                self.jks.append(JumpingKnowledge(mode='lstm', channels=self.num_classes, num_layers=2))
            elif jk_mode == 'node_adaptive':
                # self.jks.append(JumpingKnowledge(mode='node_adaptive', channels=self.args.hidden_dim))
                self.jks.append(JumpingKnowledge(mode='node_adaptive', channels=self.num_classes))
            else:
                raise NotImplementedError

    def _initialize_alphas(self):
        # num_skip_ops = len(SKIP_CANDIDATES)
        # num_tsfm_ops = len(TSFM_CANDIDATES)
        num_neigh_ops = len(NEIGH_CANDIDATES)
        num_aggr_ops = len(AGGR_CANDIDATES)
        num_norm_ops = len(NORM_CANDIDATES)
        num_comb_ops = len(COMB_CANDIDATES)
        num_jk_ops = len(JK_CANDIDATES)

        # self.skip_alphas = Parameter(
        #     torch.ones((self.args.num_layers, num_skip_ops), device=device).normal_(self.args.loc_mean,
        #                                                                             self.args.loc_std))
        # self.tsfm_alphas = Parameter(
        #     torch.ones((self.args.num_layers, num_tsfm_ops), device=device).normal_(self.args.loc_mean,
        #                                                                             self.args.loc_std))
        self.neigh_alphas = Parameter(
            torch.ones((self.args.num_layers, num_neigh_ops), device=device).normal_(self.args.loc_mean,
                                                                                     self.args.loc_std))
        self.aggr_alphas = Parameter(
            torch.ones((self.args.num_layers, num_aggr_ops), device=device).normal_(self.args.loc_mean,
                                                                                    self.args.loc_std))
        self.norm_alphas = Parameter(
            torch.ones((self.args.num_layers, num_norm_ops), device=device).normal_(self.args.loc_mean,
                                                                                    self.args.loc_std))
        self.comb_alphas = Parameter(
            torch.ones((self.args.num_layers, num_comb_ops), device=device).normal_(self.args.loc_mean,
                                                                                    self.args.loc_std))
        self.jk_alphas = Parameter(
            torch.ones((1, num_jk_ops), device=device).normal_(self.args.loc_mean, self.args.loc_std))

    def _initialize_records(self):
        self.max_acc_se, self.max_acc_re = 0, 0
        self.max_epoch_se, self.max_epoch_re = 0, 0
        self.time_se, self.time_re = 0, 0
        self.searched_arch = ddict(list)



    # def arch_parameters(self):
    #     self._arch_parameters = [self.tsfm_alphas, self.neigh_alphas, self.aggr_alphas, self.norm_alphas]
    #     return self._arch_parameters
    #
    # def supernet_parameters(self):
    #     self._supernet_parameters = []
    #     for n, p in self.named_parameters():
    #         if n not in ['tsfm_alphas', 'neigh_alphas', 'aggr_alphas', 'norm_alphas', 'jk_alphas']:
    #             self._supernet_parameters.append(p)
    #     return self._supernet_parameters

    def _get_categ_mask(self, alphas):
        log_alphas = alphas
        u = torch.zeros_like(log_alphas).uniform_()
        softmax = torch.nn.Softmax(-1)
        ws = softmax((log_alphas + (-((-(u.log())).log()))) / self.args.temp)
        # return ws

        values, indices = ws.max(dim=1)
        ws_onehot = torch.zeros_like(ws).scatter_(1, indices.view(-1, 1), 1)
        ws_onehot = (ws_onehot - ws).detach() + ws
        return ws_onehot

    def MAP(self, key):
        # if key == 'SKIP':
        #     return SKIP_CANDIDATES
        # if key == 'TSFM':
        #     return TSFM_CANDIDATES
        if key == 'NEIGH':
            return NEIGH_CANDIDATES
        if key == 'NORM':
            return NORM_CANDIDATES
        if key == 'AGGR':
            return AGGR_CANDIDATES
        if key == 'COMB':
            return COMB_CANDIDATES
        if key == 'JK':
            return JK_CANDIDATES

    def _get_alphas_onehot(self, alphas, key): #_get_searched_ops
        values, indices = alphas.max(dim=1)
        alphas_onehot = torch.zeros_like(alphas).scatter_(1, indices.view(-1, 1), 1)
        self.searched_arch[key] = np.array(self.MAP(key))[indices.cpu()]
        return alphas_onehot

    def derive_arch(self):
        # self.skip_alphas = Parameter(self._get_alphas_onehot(self.skip_alphas.clone().detach(), 'SKIP'), requires_grad=False)
        # self.tsfm_alphas = Parameter(self._get_alphas_onehot(self.tsfm_alphas.clone().detach(), 'TSFM'), requires_grad=False)
        self.neigh_alphas = Parameter(self._get_alphas_onehot(self.neigh_alphas.clone().detach(), 'NEIGH'), requires_grad=False)
        self.norm_alphas = Parameter(self._get_alphas_onehot(self.norm_alphas.clone().detach(), 'NORM'), requires_grad=False)
        self.aggr_alphas = Parameter(self._get_alphas_onehot(self.aggr_alphas.clone().detach(), 'AGGR'), requires_grad=False)
        self.comb_alphas = Parameter(self._get_alphas_onehot(self.comb_alphas.clone().detach(), 'COMB'), requires_grad=False)
        self.jk_alphas = Parameter(self._get_alphas_onehot(self.jk_alphas.clone().detach(), 'JK'), requires_grad=False)


    def forward(self, data):
        # self.skip_ws = self._get_categ_mask(self.skip_alphas)
        # self.tsfm_ws = self._get_categ_mask(self.tsfm_alphas)
        self.neigh_ws = self._get_categ_mask(self.neigh_alphas)
        self.aggr_ws = self._get_categ_mask(self.aggr_alphas)
        self.norm_ws = self._get_categ_mask(self.norm_alphas)
        self.comb_ws = self._get_categ_mask(self.comb_alphas)
        self.jk_ws = self._get_categ_mask(self.jk_alphas)

        x = self.forward_model(data)
        return x

    def forward_model(self, data):
        x = data.x
        x = self.pre_mp(x)
        xs = [x]

        for layer_idx in range(self.args.num_layers):
            # x = tsfm_trans(x, self.lns, self.tsfm_ws, layer_idx)
            # print(layer_idx)
            # print(x.shape)
            m = neigh_prop_trans(x, data, self.propss, self.neigh_ws, self.aggr_ws, self.norm_ws, layer_idx, self.args.max_degree)

            # *********************************************************************************************
            # try to add following 2 lines on March 08 (to see whether we can improve results on chameleon)
            m = F.relu(m)
            m = F.dropout(m, training=self.training, p=self.args.dprate)
            # *********************************************************************************************

            x = comb_trans(m, xs[-1], self.comb_lns, self.comb_ws, layer_idx)
            xs.append(x)

        # print(xs.shape)
        x = jk_trans(xs, self.jks, self.jk_ws)
        # x = self.post_mp(x)
        return F.log_softmax(x, dim=1)
