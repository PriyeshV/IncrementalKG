import numpy as np
from os import path
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sps
import matplotlib.pyplot as plt

path_prefix = '../../Datasets'
dataset = 'FB20K'
files = ['train', 'val', 'test', 'test_zeroshot']
files = ['val']

for file in files:
    file_path = path.join(*[path_prefix, dataset], file+'.txt')
    if not path.exists(file_path):
        continue
    else:
        print('Processing ', file_path)

    with open(file_path, 'r') as fid:
        triples = np.loadtxt(fid, delimiter='\t', dtype=np.str)

        heads = np.unique(triples[:, 0])
        tails = np.unique(triples[:, 1])
        relations = np.unique(triples[:, 2])
        entities = np.unique(np.hstack([heads, tails]))
        n_triples, n_entities, n_heads, n_tails, n_relations = len(triples), len(entities), len(heads), len(tails), len(relations)

        # Print Basic statistics
        pd.set_option('display.max_columns', 10)
        headers = ['|Triples|', '|Entities|', '|Heads|', '|Tails|', '|Relations|', '|H - T|', '|T - H|']
        output = [n_triples, n_entities, n_heads, n_tails, n_relations, len(set(heads) - set(tails)), len(set(tails) - set(heads))]
        print(pd.DataFrame([output], columns=headers))

        # Compute Adj Tensor
        entities = dict(zip(entities, list(range(n_entities))))
        relations = dict(zip(relations, list(range(n_relations))))

        adj = [None]*n_relations
        for i in tqdm(range(n_relations), ascii=True, desc='Creating Sparse Tensor'):
            adj[i] = sps.lil_matrix((n_entities, n_entities), dtype=np.int8)

        rel = []
        for i in tqdm(range(n_triples), ascii=True, desc='Computing Adjacency Tensor'):
            head, tail, relation = triples[i, 0], triples[i, 1], triples[i, 2]
            adj[relations[relation]][entities[head], entities[tail]] = 1
            rel.append(relations[relation])
        del triples

        for i in tqdm(range(n_relations), ascii=True, desc='Converting to CSR matrix'):
            adj[i] = adj[i].tocsr()

        # Compute Relation statistics
        rel_n_edges = dict()
        for i, relation in tqdm(enumerate(relations.keys()), ascii=True, desc='Relation count stats'):
            rel_n_edges[relation] = adj[relations[relation]].nnz

        plt.hist(rel_n_edges.values(), log=True, bins=30)  # axis is scaled in the log space and not the values
        plt.title('Relation: Histogram of #edges\n Min: {}, Max: {}, Mean: {}'.format(min(rel_n_edges.values()), max(rel_n_edges.values()), round(np.mean(list(rel_n_edges.values()))), 1))
        plt.show()

        # Compute Entity statistics
        ent_n_relations = dict()
        sum_adj = adj[0].copy()
        ent_rel_type = np.full((n_entities, n_relations), True, dtype=np.bool)
        for i in tqdm(range(1, n_relations), ascii=True, desc='Entity count stats'):
            sum_adj += adj[i]
            ent_rel_type[:, i] = adj[i].sum(axis=1).astype(bool).reshape((n_entities,))

        ent_head_n_rels = sum_adj.sum(axis=1)  # out-edge
        ent_tail_n_rels = sum_adj.tocsc().sum(axis=0)  # in-edge
        ent_n_rels = ent_head_n_rels + ent_tail_n_rels.T
        plt.hist(ent_n_rels, log=True, bins=30)  # axis is scaled in the log space and not the values
        plt.title('Entity: Histogram of #edges\n Min: {}, Max: {}, Mean: {}'.format(np.min(ent_n_rels), np.max(ent_n_rels), round(np.mean(ent_n_rels))))
        plt.show()

        ent_rel_type = np.count_nonzero(ent_rel_type, axis=1)
        plt.hist(ent_rel_type, log=True, bins=20)
        plt.title('Entity: Histogram of #edge types\n Min: {}, Max: {}, Mean: {}'.format(np.min(ent_rel_type), np.max(ent_rel_type), round(np.mean(ent_rel_type))))
        plt.show()


