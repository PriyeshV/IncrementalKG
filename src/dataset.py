import pickle
import itertools
import numpy as np
from os import path


class Dataset:
    def __init__(self, config):
        self.config = config
        self.n_nodes_batch = self.config.n_nodes_batch

        self.n_relations, self.emb_dim = 0, 0
        self.n_batches, self.batch_size = {}, {}
        self.n_entities_added, self.n_entities_new, self.n_entities_old = {}, {}, {}

        self.initalize_vars()
        # Automatic compute this from data directory
        self.n_sets = {}
        self.n_sets['train'] = 4
        self.n_sets['val'] = 1
        self.n_sets['test'] = 1

    def initalize_vars(self):
        dataset = ['train', 'val', 'test']
        for file in dataset:
            with open(path.join(self.config.paths['data'], file + '_0_data_obj_b.pkl'), 'rb') as inp:
                data = pickle.load(inp)
                self.n_entities_added[file] = data.n_ent_add
                self.n_entities_new[file] = data.emb_new_ent.shape[0]
                self.n_entities_old[file] = self.n_entities_new[file] - self.n_entities_added[file]

                self.n_batches[file] = np.ceil(self.n_entities_added[file] / self.n_nodes_batch).astype(int)
                self.batch_size[file] = self.config.n_nodes_batch * self.config.n_data_augments

                if file == 'test':
                    self.n_batches[file] = 1
                    self.batch_size[file] = self.n_entities_added[file]

                if file == 'train':
                    self.n_relations, self.emb_dim = data.emb_rel.shape
                    self.config.emb_dim = self.emb_dim

    def load_data(self, dataset):
        n_set = min(self.config.n_data_augments, self.n_sets[dataset])
        data = []
        for i in range(n_set):
            with open(path.join(self.config.paths['data'], dataset+'_'+str(i)+'_data_obj_b.pkl'), 'rb') as inp:
                data.append(pickle.load(inp))
        return data

    def get_config(self):
        return self.config

    def get_connected_nodes(self, data, nodes, degrees, adjlist):
        max_depth = self.config.max_depth
        new_nodes = set(range(self.n_entities_added[data]))
        neighbors = [None]*(max_depth+1)
        neighbors[0] = np.asarray(nodes, dtype=int)
        all_nodes = set(nodes)
        for h in range(1, max_depth+1):
            new_k = set()
            for n in neighbors[h-1]:
                if degrees[n] > 0:
                    new_k.update(adjlist[n])
            new_k = new_k - all_nodes
            new_k = new_k - new_nodes
            neighbors[h] = np.asarray(list(new_k), dtype=int)
            all_nodes.update(new_k)
        return neighbors

    def batch_generator(self, file='train', shuffle=True):
        n_nodes_batch = self.n_nodes_batch
        n_batches = self.n_batches[file]
        if file == 'test':
            n_nodes_batch = self.n_entities_added['test']
            n_batches = 1

        dataset = self.load_data(file)
        node_order = np.arange(self.n_entities_added[file])
        if shuffle:
            node_order = np.arange(self.n_entities_added[file])
            np.random.shuffle(node_order)

        for batch_id in range(n_batches):
            start = batch_id * n_nodes_batch
            end = np.min([(batch_id+1) * n_nodes_batch, self.n_entities_added[file]])

            n_new_ent = end - start
            new_ent_ids = list(node_order[start:end])
            old_ent_ids = []

            mask_new = []
            mask_old = []
            mask_old_neigh = []

            emb_rel = np.zeros([0, self.emb_dim])
            op_ent_emb = np.zeros([0, self.emb_dim])   # contains embedding of both newly added entities and updated neighboring entities
            ip_ent_emb = np.zeros([0, self.emb_dim])   # emb of entity in old graph

            adj_ind, rel_in_ind, rel_out_ind = np.zeros([0, 2], dtype=int), np.zeros([0, 2], dtype=int), np.zeros([0, 2], dtype=int)
            adj_data, rel_in_data, rel_out_data = [], [], []

            n_samples = 0
            n_relations = 0
            t_old_ent = 0
            for data_id, data in enumerate(dataset):

                sg_neighbors = self.get_connected_nodes(file, new_ent_ids, dataset[data_id].deg_adj, dataset[data_id].adj.tolil().rows)
                old_ent_ids.append(sg_neighbors[1])

                all_nodes = list(itertools.chain.from_iterable(sg_neighbors))
                n_all_nodes = len(all_nodes)
                n_old_ent = len(sg_neighbors[1])
                t_old_ent += n_old_ent
                n_old_neigh_ent = len(sg_neighbors[2])

                # Mask
                t_mask_new = np.full((n_all_nodes), False)
                t_mask_old = np.full((n_all_nodes), False)  # old_ent
                t_mask_old_neigh = np.full((n_all_nodes), False)
                t_mask_new[:n_new_ent] += True
                t_mask_old[n_new_ent:n_new_ent+n_old_ent] += True
                t_mask_old_neigh[-n_old_neigh_ent:] += True
                mask_new = np.concatenate([mask_new, t_mask_new], axis=0)
                mask_old = np.concatenate([mask_old, t_mask_old], axis=0)
                mask_old_neigh = np.concatenate([mask_old_neigh, t_mask_old_neigh], axis=0)

                # Get Graph data
                graphs = dataset[data_id].slice_graphs(all_nodes)
                G = graphs[0].tocoo()
                ind = np.vstack([G.row, G.col]) + n_samples
                adj_ind = np.concatenate([adj_ind, ind.T], axis=0)
                adj_data = np.concatenate([adj_data, G.data], axis=0)

                G = graphs[1].tocoo()
                ind = np.vstack([G.row + n_samples, G.col + n_relations])
                rel_in_ind = np.concatenate([rel_in_ind, ind.T], axis=0)
                rel_in_data = np.concatenate([rel_in_data, G.data], axis=0)

                G = graphs[2].tocoo()
                ind = np.vstack([G.row + n_samples, G.col + n_relations])
                rel_out_ind = np.concatenate([rel_out_ind, ind.T], axis=0)
                rel_out_data = np.concatenate([rel_out_data, G.data], axis=0)

                # Get Emb data
                input_emb, output_emb = dataset[data_id].slice_emb(*sg_neighbors)
                emb_rel = np.concatenate([emb_rel, dataset[data_id].emb_rel], axis=0)
                ip_ent_emb = np.concatenate([ip_ent_emb, np.concatenate([np.zeros([n_new_ent, self.emb_dim]), input_emb], axis=0)], axis=0)
                op_ent_emb = np.concatenate([op_ent_emb, np.concatenate([output_emb, np.zeros([n_old_neigh_ent, self.emb_dim])], axis=0)], axis=0)

                # update total n_samples in this batch
                n_samples += n_all_nodes
                n_relations += self.n_relations

            adj_shape = (n_samples, n_samples)
            rel_shape = (n_samples, self.n_relations*len(dataset))

            new_ent_ids = np.array(new_ent_ids, dtype=int)
            old_ent_ids = np.unique(np.hstack(old_ent_ids))

            yield new_ent_ids, old_ent_ids, mask_new, mask_old, mask_old_neigh, emb_rel, ip_ent_emb, op_ent_emb,\
                  adj_ind, adj_data, adj_shape, rel_in_ind, rel_in_data, rel_out_ind, rel_out_data, rel_shape,