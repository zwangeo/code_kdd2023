# SKIP_CANDIDATES = [None, 'add']
# SKIP_CANDIDATES = [None]
# TSFM_CANDIDATES = [None, 'linear']
# TSFM_CANDIDATES = [None]
NEIGH_CANDIDATES = ['edge_index', 'edge_index_knn',
                    'edge_index_2hop',
                    'edge_index_knn_rwpe', 'edge_index_knn_lepe']
# NEIGH_CANDIDATES = ['edge_index', 'edge_index_knn']
AGGR_CANDIDATES = ['add', 'mean', 'max', 'min']
NORM_CANDIDATES = [None, 'degree_sys', 'degree_row', 'fagcn_like',
                   'rel_rwpe', 'rel_lepe']
COMB_CANDIDATES = [None, 'add', 'cat']
# COMB_CANDIDATES = [None, 'cat']
JK_CANDIDATES = ['last', 'cat', 'mean', 'max', 'ppr_0.1', 'ppr_0.5', 'gpr', 'lstm', 'node_adaptive']



