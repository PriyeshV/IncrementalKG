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

        self.dataset_name = args.dataset
        self.n_data_augments = args.n_augments
        self.n_nodes_batch = args.n_nodes_batch

        self.gpu = args.gpu

        self.paths['datasets'] = path.join(self.paths['root'], 'Datasets')
        self.paths['data'] = path.join(self.paths['datasets'], self.dataset_name)
        self.paths['experiments'] = path.join(self.paths['root'], 'Experiments')
        self.paths['experiment'] = path.join(self.paths['experiments'], args.timestamp, self.dataset_name)





