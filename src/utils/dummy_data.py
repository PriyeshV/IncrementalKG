import pickle
import numpy as np
from os import path
import networkx as nx
from tqdm import tqdm
import scipy.sparse as sps


class IncrementalDataset:
    def __init__(self, path=None):
        self.emb_rel, self.emb_old_ent, self.emb_new_ent = None, None, None
        self.adj, self.rel_in, self.rel_out = None, None, None
        self.deg_adj = None
        self.n_ent_add = 0
        # self.load_data_from_path(path)

    def load_data(self, embs, graphs):
        self.emb_rel, self.emb_old_ent, self.emb_new_ent = embs
        self.adj, self.rel_in, self.rel_out = graphs
        self.deg_adj = self.adj.sum(axis=1)
        self.n_ent_add = self.emb_new_ent.shape[0] - self.emb_old_ent.shape[0]

    def slice_graphs(self, nodes):
        # node_ids are numpy ids in new graph
        adj = self.adj[nodes, :].tocsc()[:, nodes]
        rel_in = self.rel_in[nodes, :]
        rel_out = self.rel_out[nodes, :]
        graph_data = [adj, rel_in, rel_out]

        return graph_data

    def slice_emb(self, new_ent_ids, old_ent_ids, old_ent_neigh_ids):
        # all are numpy ids
        new_ids = np.concatenate([new_ent_ids, old_ent_ids], axis=0)  # old_ent_ids in new graph
        emb_new_ent = self.emb_new_ent[new_ids, :]

        old_ids = np.concatenate([old_ent_ids, old_ent_neigh_ids], axis=0)
        old_ids = old_ids - self.n_ent_add
        emb_old_ent = self.emb_old_ent[old_ids, :]

        return emb_old_ent, emb_new_ent

    def load_data_from_path(self, path):
        return


path_prefix = '../../Datasets'
dataset = 'Dummy'
data_creation = False
data_formatting = False

# Create Raw data
n_data = 10
n_rel = 100
emb_dim = 64
n_ent_old = 1500
n_ent_new = 2000

if data_creation:

    for i in tqdm(range(n_data), ascii=True, desc='Preparing dummy dataset'):
        emb_rel = np.random.rand(n_rel, emb_dim)
        emb_old_ent= np.random.rand(n_ent_old, emb_dim)
        emb_new_ent = np.random.rand(n_ent_new, emb_dim)
        emb_data = [emb_rel, emb_old_ent, emb_new_ent]

        # node_ids are ordered in adj such that the old nodes appear first, i.e [old_nodes, new_nodes]
        # adj = sps.random(n_ent_new, n_ent_new, density=0.05, data_rvs=np.ones).tocsr()  # directed graph
        G = nx.powerlaw_cluster_graph(n_ent_new, 5, 0.6)
        adj = nx.to_scipy_sparse_matrix(G).tocsr()
        rel_out = sps.random(n_ent_new, n_rel, density=0.02, data_rvs=np.ones).tocsr()
        rel_in = rel_out.copy()
        graph_data = [adj, rel_in, rel_out]

        data = IncrementalDataset()
        data.load_data(emb_data, graph_data)

        with open(path.join(*[path_prefix, dataset], str(i)+'_data_obj_b.pkl'), 'wb') as out:  # dump as binary data
            pickle.dump(data, out, pickle.HIGHEST_PROTOCOL)

# if data_formatting:
#
#     data = np.ones([150000, 200], dtype=np.float32)
#     print(data.nbytes)

