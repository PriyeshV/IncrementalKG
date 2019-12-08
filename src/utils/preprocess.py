import pickle
import numpy as np
from os import path
import pandas as pd
import networkx as nx
from tqdm import tqdm
import scipy.sparse as sps
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict, deque


def prepare_dataset(path_prefix, dataset, files, get_stats=False):

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
            entities = OrderedDict(zip(entities, list(range(n_entities))))
            relations = OrderedDict(zip(relations, list(range(n_relations))))

            adj = [None]*n_relations
            for i in tqdm(range(n_relations), ascii=True, desc='Creating Sparse Tensor'):
                adj[i] = sps.lil_matrix((n_entities, n_entities), dtype=np.int8)

            rel = []
            for i in tqdm(range(n_triples), ascii=True, desc='Computing Adjacency Tensor'):
                head, tail, relation = triples[i, 0], triples[i, 1], triples[i, 2]
                adj[relations[relation]][entities[head], entities[tail]] = 1
                rel.append(relations[relation])
            del triples

            if get_stats:
                # Compute Relation statistics
                rel_n_edges = dict()  # n_edges in each relation
                edge_n_rels = dict()   # edge/entity-pair statistics - n_relations for entity-pair
                edge_n_rels = defaultdict(lambda: 0, edge_n_rels)
                for n, rel in tqdm(enumerate(relations.keys()), ascii=True, desc='Edge count stats'):
                    (row, col) = adj[relations[rel]].nonzero()
                    nnz = len(row)
                    rel_n_edges[relations[rel]] = nnz
                    for k in range(nnz):
                        i, j = row[k], col[k]
                        edge_n_rels[(i, j)] += 1

            # Convert to CSR for better row-wise operations
            for i in tqdm(range(n_relations), ascii=True, desc='Converting to CSR matrix'):
                adj[i] = adj[i].tocsr()

            with open(path.join(*[path_prefix, dataset], file+'_adj_spslist_b.pkl'), 'wb') as out:  # dump as binary data
                pickle.dump(adj, out, pickle.HIGHEST_PROTOCOL)
            with open(path.join(*[path_prefix, dataset], file+'_map_ent-id_b.pkl'), 'wb') as out:
                pickle.dump(entities, out)
            with open(path.join(*[path_prefix, dataset], file+'_map_rel-id_b.pkl'), 'wb') as out:
                pickle.dump(relations, out)

            if get_stats:
                rel_types = np.full((n_relations, 4), False, dtype=np.bool)  # one-one, one-many, many-one, many-many
                for _, rel in tqdm(enumerate(relations.keys()), ascii=True, desc='Relation type stats'):
                    head_tail = adj[relations[rel]].sum(axis=1)
                    tail_head = adj[relations[rel]].sum(axis=0)

                    if (head_tail > 1).any():
                        if (tail_head > 1).any():
                            rel_types[relations[rel], 3] = True
                        else:
                            rel_types[relations[rel], 1] = True
                    elif (tail_head > 1).any():
                        rel_types[relations[rel], 2] = True
                    else:
                        rel_types[relations[rel], 0] = True

                rel_types = np.count_nonzero(rel_types, axis=0)
                headers = ['1-1', '1-N', 'N-1', 'N-N']
                print(pd.DataFrame([rel_types], columns=headers))

                plt.hist(rel_n_edges.values(), log=True, bins=30)  # axis is scaled in the log space and not the values
                plt.title('Relation: Histogram of #edges\n Min: {}, Max: {}, Mean: {}'.format(min(rel_n_edges.values()), max(rel_n_edges.values()), round(np.mean(list(rel_n_edges.values()))), 1))
                plt.show()

                plt.hist(edge_n_rels.values(), log=True, bins=10)  # axis is scaled in the log space and not the values
                plt.title('Entity-pair: Histogram of #relations\n Min: {}, Max: {}, Mean: {}'.format(min(edge_n_rels.values()), max(edge_n_rels.values()), round(np.mean(list(edge_n_rels.values()))), 1))
                plt.show()

                # Compute Entity statistics
                sum_adj = adj[0].copy()
                ent_rel_type = np.full((n_entities, n_relations), False, dtype=np.bool)
                for n, rel in tqdm(enumerate(relations.keys()), ascii=True, desc='Entity count stats'):
                    i = relations[rel]
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
                plt.show()  # Doesn't include Tail entity case

                # TODO: Relation Vs Mean(freq(Entities))

                # TODO: Entity Vs Mean(freq(Relations))


def drop_nodes(adj, n_nodes):
    Orig_G = nx.to_networkx_graph(adj)   # undirected graph
    n = adj.shape[0]
    n_nodes = 5000

    head_ent = adj.sum(axis=1)
    tail_ent = adj.sum(axis=0)
    ent_prob = np.max(head_ent.T + tail_ent, axis=0)
    # ent_prob = ent_prob / ent_prob.sum()

    nodes = []
    # print('Number of Connected Components: ', nx.number_connected_components(Orig_G))
    max_nodes = list(max(nx.connected_component_subgraphs(Orig_G), key=len).nodes)
    G = max(nx.connected_component_subgraphs(Orig_G), key=len)

    iter = 0
    neg = 0
    flag = 1
    last_removed_node = deque(maxlen=10)
    last_removed_edge = deque(maxlen=10)
    while (len(nodes) < n_nodes) or iter > 2*n_nodes:
        n_cands = 1
        iter += 1

        if flag == 0:
            cands = np.random.choice(max_nodes, n_cands, replace=False)
        else:
            max_ent_prob = ent_prob[0, max_nodes]
            # max_ent_prob = max_ent_prob.sum() - max_ent_prob
            max_ent_prob = max_ent_prob / max_ent_prob.sum()
            cands = np.random.choice(max_nodes, n_cands, replace=False, p=max_ent_prob.tolist()[0])

        added = 0
        for node in cands:
            edges = list(G.edges(node, data=False))
            G.remove_node(node)

            if nx.is_connected(G):
                nodes.append(node)
                added += 1
                last_removed_node.append(node)
                last_removed_edge.append(edges)
            else:
                G.add_node(node)
                G.add_edges_from(edges)

        print(iter, len(cands), len(nodes))
        if added > 0:
            max_nodes = G.nodes()
            neg = 0
        else:
            neg += 1
            if last_removed_node.__len__() == 0 and len(nodes) > 1:
                break
            if neg > 5:
                # print('Switching sampling strategy')
                G.add_node(last_removed_node.pop())
                G.add_edges_from(last_removed_edge.pop())
                flag = 1
            if neg > 10:
                # print('Switching sampling strategy')
                flag = 1
                G.add_node(last_removed_node.pop())
                G.add_edges_from(last_removed_edge.pop())
                neg = 0
        # n_cands = n_nodes - len(nodes)
    return nodes
    # exit()


def remove_nodes(adj, n_nodes):
    adj = adj - sps.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # edges_all = sparse_to_tuple(adj)[0].tolist()

    cands = drop_nodes(adj, n_nodes)








path_prefix = '../../Datasets'
dataset = 'FB20K'
# # files = ['train', 'val', 'test', 'test_zeroshot']
files = ['train']
prepare_dataset(path_prefix, dataset, files, get_stats=True)
exit()

#  Create Training Data
fname_suffixes = ['_adj_spslist_b', '_map_ent-id_b', '_map_rel-id_b']
file = 'train'
with open(path.join(*[path_prefix, dataset], file+fname_suffixes[0]+'.pkl'), 'rb') as inp:
    adj = pickle.load(inp)
with open(path.join(*[path_prefix, dataset], file+fname_suffixes[1]+'.pkl'), 'rb') as inp:
    entitites = pickle.load(inp)
with open(path.join(*[path_prefix, dataset], file+fname_suffixes[2]+'.pkl'), 'rb') as inp:
    relations = pickle.load(inp)
n_entities = len(entitites)
n_relations = len(relations)

# Collapsing to single graph
sing_adj = adj[0].copy()
ent_rel_type = np.full((n_entities, n_relations), False, dtype=np.bool)
for rel in tqdm(relations.keys(), ascii=True, desc='Preparing simple graph'):
    sing_adj += adj[relations[rel]]

set1 = remove_nodes(sing_adj, n_nodes=5000)
set2 = remove_nodes(sing_adj, n_nodes=5000)
set3 = remove_nodes(sing_adj, n_nodes=5000)

print(len(set1), len(set2), len(set3))
print(np.unique([set1, set2, set3]))


