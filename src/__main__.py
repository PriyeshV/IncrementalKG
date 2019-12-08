import os
import time
import threading
import numpy as np
from os import path
from copy import deepcopy
from tabulate import tabulate
import tensorflow as tf

from src.dataset import Dataset
from src.parser import Parser
from src.config import Config
from tensorflow.python.framework import ops
from src.utils.dummy_data import IncrementalDataset


class KG_GCN(object):

    def __init__(self, dataset):
        self.config = dataset.config
        self.dataset = dataset

        # Variable initialization
        self.placeholders = {}
        self.queue_placeholders_keys = {}
        self.Q = self.enqueue_op = self.dequeue_op = None
        self.coord = tf.train.Coordinator()

        # Setup Architecture
        self.update_predictions_op = None
        self.data = self.model = self.saver = self.summary = None
        self.setup_arch()

        # Setup initializers
        self.init = tf.global_variables_initializer()
        self.init2 = tf.no_op()

    def setup_arch(self):
        # Setup place_holders
        self.placeholders = {}
        self.queue_placeholders_keys = ['mask', 'degrees', 'n_conn_nodes', 'n_node_ids', 'batch_density',
                                        'f_indices', 'f_data', 'f_shape', 'nnz_features', 'targets', 'labeled_ids',
                                        'batch_ids', 'adj_indices', 'adj_data', 'adj_shape']
        self.get_placeholders()

        # Setup Input Queue
        self.Q, self.enqueue_op, self.dequeue_op = self.setup_data_queues()

        # Create model and data for the model
        self.data = self.create_tfgraph_data()

        self.model = self.config.prop_class(self.config, self.data, logging=True, wce=self.config.wce,
                                            multilabel=self.config.multilabel)
        self.saver = tf.train.Saver()
        self.summary = tf.summary.merge_all()

    def create_feed_dict(self, sources):
        keys = self.queue_placeholders_keys
        feed_dict = {}
        for i, key in enumerate(keys):
            feed_dict[self.placeholders[key]] = sources[i]
        return feed_dict

    def setup_data_queues(self):
        Q = tf.FIFOQueue(capacity=self.config.queue_capacity,
                         dtypes=[tf.int32, tf.float32, tf.int64, tf.int32, tf.float32, tf.int64, tf.float32, tf.int64,
                                 tf.int32, tf.float32, tf.int32, tf.int32, tf.int64, tf.float32, tf.int64])
        keys = self.queue_placeholders_keys
        enqueue_op = Q.enqueue([self.placeholders[key] for key in keys])
        dequeue_op = Q.dequeue()
        return Q, enqueue_op, dequeue_op



def init_model(config, dataset):
    ops.reset_default_graph()
    tf.random.set_seed(1234)
    np.random.seed(1234)

    with tf.compat.v1.variable_scope('Graph_Convolutional_Network', reuse=None):
        model = KG_GCN(dataset)

    # configure GPU usage
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)

    tf_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    tf_config.inter_op_parallelism_threads = 32  # #CPUs - how many ops to run in parallel [0 - Default]
    tf_config.intra_op_parallelism_threads = 1  # how many threads each op gets
    sm = tf.compat.v1.train.SessionManager()

    if config.retrain:
        print("Loading model from checkpoint")
        load_ckpt_dir = config.ckpt_dir
    else:
        # print("No model loaded from checkpoint")
        load_ckpt_dir = ''
    sess = sm.prepare_session("", init_op=model.init, saver=model.saver, checkpoint_dir=load_ckpt_dir, config=tf_config)
    return model, sess



def train_model(dataset):
    config = deepcopy(dataset.get_config())
    model, sess = init_model(config, dataset)
    summary_writers = model.add_summaries(sess)
    results = 0
    return results



def main():

    args = Parser().get_parser().parse_args()
    print("=====Configurations=====\n", args)

    # Load Configuration and data
    config = Config(args)
    dataset = Dataset(config)

    results = train_model(dataset)

if __name__ == "__main__":
    main()