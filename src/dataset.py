import pickle
import itertools
import numpy as np
from os import path


class Dataset:
    def __init__(self, config):
        self.config = config

        self.data = self.load_data(config)
        self.n_relations, self.emb_dim = self.data[0].emb_rel.shape
        self.config.emb_dim = self.emb_dim
        self.n_entities_added = self.data[0].n_ent_add
        self.n_entities_new = self.data[0].emb_new_ent.shape[0]
        self.n_entities_old = self.n_entities_new - self.n_entities_added

        self.n_nodes_batch = config.n_nodes_batch
        self.n_batches = np.ceil(self.n_entities_added / self.n_nodes_batch).astype(int)
        self.batch_size = config.n_nodes_batch * config.n_data_augments

        # self.batch_generator()

    def load_data(self, config):
        data = []
        for i in range(config.n_data_augments):
            with open(path.join(config.paths['data'], str(i)+'_data_obj_b.pkl'), 'rb') as inp:
                data.append(pickle.load(inp))
        return data

    def get_config(self):
        return self.config

    def get_connected_nodes(self, nodes, degrees, adjlist):
        max_depth = self.config.max_depth
        new_nodes = set(range(self.n_entities_added))
        neighbors = [None]*(max_depth+1)
        neighbors[0] = np.asarray(nodes, dtype=int)
        all_nodes = set(nodes)
        for h in range(1, max_depth+1):
            new_k = set()
            for n in neighbors[h-1]:
                if degrees[n] > 0:
                    new_k.update(adjlist[n])
            # new_k = new_k - all_nodes
            new_k = new_k - new_nodes
            neighbors[h] = np.asarray(list(new_k), dtype=int)
            all_nodes.update(new_k)
        return neighbors

    def batch_generator(self, data='train', shuffle=True):

        if shuffle:
            node_order = np.arange(self.n_entities_added)
            np.random.shuffle(node_order)

        for batch_id in range(self.n_batches):
            start = batch_id * self.n_nodes_batch
            end = np.min([(batch_id+1) * self.n_nodes_batch, self.n_entities_added])
            # curr_b_size = end - start

            new_ent_ids = list(node_order[start:end])
            n_new_ent = end - start

            # mask_new = np.zeros([0, 1], dtype=bool)
            mask_new = []
            mask_old = []
            mask_old_neigh = []

            emb_rel = np.zeros([0, self.emb_dim])
            op_ent_emb = np.zeros([0, self.emb_dim])   # contains embedding of both newly added entities and updated neighboring entities
            ip_ent_emb = np.zeros([0, self.emb_dim])   # emb of entity in old graph

            adj_ind, rel_in_ind, rel_out_ind = np.zeros([0, 2], dtype=int), np.zeros([0, 2], dtype=int), np.zeros([0, 2], dtype=int)
            adj_data, rel_in_data, rel_out_data = [], [], []

            n_samples = 0
            t_old_ent = 0
            for data_id, data in enumerate(self.data):

                sg_neighbors = self.get_connected_nodes(new_ent_ids, self.data[data_id].deg_adj , self.data[data_id].adj.tolil().rows)

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
                graphs = self.data[data_id].slice_graphs(all_nodes)
                G = graphs[0].tocoo()
                ind = np.vstack([G.row, G.col]) + n_samples
                adj_ind = np.concatenate([adj_ind, ind.T], axis=0)
                adj_data = np.concatenate([adj_data, G.data], axis=0)

                G = graphs[1].tocoo()
                ind = np.vstack([G.row, G.col]) + n_samples
                rel_in_ind = np.concatenate([rel_in_ind, ind.T], axis=0)
                rel_in_data = np.concatenate([rel_in_data, G.data], axis=0)

                G = graphs[2].tocoo()
                ind = np.vstack([G.row, G.col]) + n_samples
                rel_out_ind = np.concatenate([rel_out_ind, ind.T], axis=0)
                rel_out_data = np.concatenate([rel_out_data, G.data], axis=0)

                # Get Emb data
                input_emb, output_emb = self.data[data_id].slice_emb(*sg_neighbors)
                ip_ent_emb = np.concatenate([ip_ent_emb, np.concatenate([np.zeros([n_new_ent, self.emb_dim]), input_emb], axis=0)], axis=0)
                op_ent_emb = np.concatenate([op_ent_emb, np.concatenate([output_emb, np.zeros([n_old_neigh_ent, self.emb_dim])], axis=0)], axis=0)

                emb_rel = np.concatenate([emb_rel, self.data[data_id].emb_rel], axis=0)

                n_samples += n_all_nodes

            adj_shape = (n_samples, n_samples)
            rel_shape = (n_samples, self.n_relations*len(self.data))

            yield mask_new, mask_old, mask_old_neigh, emb_rel, ip_ent_emb, op_ent_emb,\
                  adj_ind, adj_data, adj_shape, rel_in_ind, rel_in_data, rel_out_ind, rel_out_data, rel_shape,

            # adj_mat = tf.SparseTensor(indices=adj_ind, values=adj_data, dense_shape=adj_shape)
            # rel_in_mat = tf.SparseTensor(indices=rel_in_ind, values=rel_in_data, dense_shape=rel_shape)
            # rel_out_mat = tf.SparseTensor(indices=rel_out_ind, values=rel_out_data, dense_shape=rel_shape)
            #
            # yield mask_new, mask_old, mask_old_neigh, \
            #       emb_rel, ip_ent_emb, op_ent_emb,\
            #       adj_mat, rel_in_mat, rel_out_mat