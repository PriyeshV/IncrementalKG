import os
import sys
import importlib
import numpy as np
from os import path
from yaml import dump
import tensorflow as tf


class Config(object):
    def __init__(self, args):

        # SET UP PATHS
        self.paths = dict()
        self.paths['root'] = '../'

        self.retrain = False
        self.dataset_name = args.dataset
        self.n_data_augments = args.n_augments
        self.n_nodes_batch = args.n_nodes_batch

        self.gpu = args.gpu
        self.queue_capacity = 1
        self.max_epochs = 100
        self.learning_rate = args.lr
        self.dropout = args.dropout
        self.bias = args.bias

        self.max_depth = 2
        self.l2 = args.l2
        self.opt = tf.compat.v1.train.AdamOptimizer
        self.kernel_class = getattr(
            importlib.import_module("src.layers.graph_convolutions." + args.gcnKernel), "Kernel")

        self.paths['datasets'] = path.join(self.paths['root'], 'Datasets')
        self.paths['data'] = path.join(self.paths['datasets'], self.dataset_name)
        self.paths['experiments'] = path.join(self.paths['root'], 'Experiments')
        self.paths['experiment'] = path.join(self.paths['experiments'], args.timestamp, self.dataset_name)

        self.paths['logs'] = path.join(self.paths['experiment'], 'Logs/')
        self.paths['ckpt'] = path.join(self.paths['experiment'], 'Checkpoints/')
        self.paths['embed'] = path.join(self.paths['experiment'], 'Embeddings/')
        self.paths['results'] = path.join(self.paths['experiment'], 'Results/')

        # early stopping hyper parametrs
        self.patience = args.pat  # look as this many epochs regardless
        self.learning_rate = args.lr
        self.drop_lr = args.drop_lr




