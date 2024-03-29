import os
import time
import pickle
import threading
import numpy as np
from os import path
from copy import deepcopy
from tabulate import tabulate
import tensorflow as tf

import sys
sys.path.append('/content/IncrementalKG')
sys.path.append('/content/IncrementalKG/src')

from src.dataset import Dataset
from src.parser import Parser
from src.config import Config
from tensorflow.python.framework import ops
from src.utils.utils import get_tf_unnormalize_adj
from src.models.KG_GCN import KG_GCN
from src.utils.dummy_data import IncrementalDataset


tf.compat.v1.disable_eager_execution()
class Incremental_KG(object):

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
        self.init = tf.compat.v1.global_variables_initializer()
        self.init2 = tf.no_op()

    def setup_arch(self):
        # Setup place_holders
        self.placeholders = {}
        self.queue_placeholders_keys = ['new_ent_ids', 'old_ent_ids', 'mask_new', 'mask_old', 'mask_old_neigh', 'emb_rel', 'ip_ent_emb', 'op_ent_emb',\
                  'adj_ind', 'adj_data', 'adj_shape', 'rel_in_ind', 'rel_in_data', 'rel_out_ind', 'rel_out_data', 'rel_shape']

        self.get_placeholders()

        # Setup Input Queue
        self.Q, self.enqueue_op, self.dequeue_op = self.setup_data_queues()

        # Create model and data for the model
        self.data = self.create_tfgraph_data()

        self.model = KG_GCN(self.config, self.data, logging=True)

        self.saver = tf.compat.v1.train.Saver()
        self.summary = tf.compat.v1.summary.merge_all()

    def setup_data_queues(self):
        Q = tf.queue.FIFOQueue(capacity=self.config.queue_capacity,
                         dtypes=[tf.int64, tf.int64, tf.bool, tf.bool, tf.bool, tf.float32, tf.float32, tf.float32,
                                 tf.int64, tf.float32, tf.int64, tf.int64, tf.float32, tf.int64, tf.float32, tf.int64])
        keys = self.queue_placeholders_keys
        enqueue_op = Q.enqueue([self.placeholders[key] for key in keys])
        dequeue_op = Q.dequeue()
        return Q, enqueue_op, dequeue_op

    def get_placeholders(self):
        with tf.compat.v1.variable_scope('Placeholders'):
            self.placeholders['lr'] = tf.compat.v1.placeholder_with_default(0.01, name='learning_rate', shape=())
            self.placeholders['dropout'] = tf.compat.v1.placeholder_with_default(0., name='dropout', shape=())
            self.placeholders['is_training'] = tf.compat.v1.placeholder(tf.bool, name='is_training')
            self.get_queue_placeholders()

    def get_queue_placeholders(self):
        with tf.compat.v1.variable_scope('Queue_placeholders'):
            self.placeholders['new_ent_ids'] = tf.compat.v1.placeholder(tf.int64, name='new_ent_ids', shape=None)
            self.placeholders['old_ent_ids'] = tf.compat.v1.placeholder(tf.int64, name='old_ent_ids', shape=None)

            self.placeholders['mask_new'] = tf.compat.v1.placeholder(tf.bool, name='mask_new', shape=None)
            self.placeholders['mask_old'] = tf.compat.v1.placeholder(tf.bool, name='mask_old', shape=None)
            self.placeholders['mask_old_neigh'] = tf.compat.v1.placeholder(tf.bool, name='mask_old_neigh', shape=None)

            self.placeholders['emb_rel'] = tf.compat.v1.placeholder(tf.float32, name='emb_rel', shape=[None, None])
            self.placeholders['ip_ent_emb'] = tf.compat.v1.placeholder(tf.float32, name='ip_ent_emb', shape=[None, None])
            self.placeholders['op_ent_emb'] = tf.compat.v1.placeholder(tf.float32, name='op_ent_emb', shape=[None, None])

            self.placeholders['adj_ind'] = tf.compat.v1.placeholder(tf.int64, name='adj_indices', shape=None)
            self.placeholders['adj_data'] = tf.compat.v1.placeholder(tf.float32, name='adj_data', shape=None)
            self.placeholders['adj_shape'] = tf.compat.v1.placeholder(tf.int64, name='adj_shape', shape=None)

            self.placeholders['rel_in_ind'] = tf.compat.v1.placeholder(tf.int64, name='adj_indices', shape=None)
            self.placeholders['rel_in_data'] = tf.compat.v1.placeholder(tf.float32, name='adj_data', shape=None)
            self.placeholders['rel_out_ind'] = tf.compat.v1.placeholder(tf.int64, name='adj_indices', shape=None)
            self.placeholders['rel_out_data'] = tf.compat.v1.placeholder(tf.float32, name='adj_data', shape=None)
            self.placeholders['rel_shape'] = tf.compat.v1.placeholder(tf.int64, name='adj_shape', shape=None)

    def create_tfgraph_data(self):
        data = {}
        (
        data['new_ent_ids'], data['old_ent_ids'], data['mask_new'], data['mask_old'], data['mask_old_neigh'],
        data['emb_rel'], data['ip_ent_emb'], data['op_ent_emb'], adj_ind, adj_data, adj_shape,
        rel_in_ind, rel_in_data, rel_out_ind, rel_out_data, data['rel_shape']) = self.dequeue_op

        data['n_rel'] = tf.shape(data['emb_rel'])[0]
        data['n_dims'] = tf.shape(data['emb_rel'])[1]
        data['n_nodes'] = adj_shape[0]

        adjmat = tf.SparseTensor(indices=adj_ind, values=adj_data, dense_shape=adj_shape)
        data['adj_out_deg'] = tf.sparse.reduce_sum(adjmat, axis=1, keepdims=True)
        data['adj_in_deg'] = tf.sparse.reduce_sum(adjmat, axis=0, keepdims=True)
        data['adj_out_mat'] = get_tf_unnormalize_adj(adjmat, data['adj_out_deg'])
        data['adj_in_mat'] = get_tf_unnormalize_adj(tf.sparse.transpose(adjmat, [1, 0]), data['adj_in_deg'])

        data['rel_in_mat'] = tf.SparseTensor(indices=rel_in_ind, values=rel_in_data, dense_shape=data['rel_shape'])
        data['rel_in_deg'] = tf.sparse.reduce_sum(data['rel_in_mat'], axis=1, keepdims=True)
        data['rel_in_mat'] = get_tf_unnormalize_adj(data['rel_in_mat'], data['rel_in_deg'])

        data['rel_out_mat'] = tf.SparseTensor(indices=rel_out_ind, values=rel_out_data, dense_shape=data['rel_shape'])
        data['rel_out_deg'] = tf.sparse.reduce_sum(data['rel_out_mat'], axis=1, keepdims=True)
        data['rel_out_mat'] = get_tf_unnormalize_adj(data['rel_out_mat'], data['rel_out_deg'])

        data['dropout'] = self.placeholders['dropout']
        data['lr'] = self.placeholders['lr']
        data['is_training'] = self.placeholders['is_training']
        return data

    def add_summaries(self, sess):
        # Instantiate a SummaryWriter to output summaries and the Graph.
        suffix = ''
        summary_writer_train = tf.compat.v1.summary.FileWriter(self.config.paths['logs' + suffix] + "train", sess.graph)
        summary_writer_val = tf.compat.v1.summary.FileWriter(self.config.paths['logs' + suffix] + "validation", sess.graph)
        summary_writer_test = tf.compat.v1.summary.FileWriter(self.config.paths['logs' + suffix] + "test", sess.graph)
        summary_writers = {'train': summary_writer_train, 'val': summary_writer_val, 'test': summary_writer_test}
        return summary_writers

    def load_and_enqueue(self, sess, data):
        for idx, batch in enumerate(self.dataset.batch_generator(data)):
            feed_dict = self.create_feed_dict(batch)
            sess.run(self.enqueue_op, feed_dict=feed_dict)

    def create_feed_dict(self, sources):
        keys = self.queue_placeholders_keys
        feed_dict = {}
        for i, key in enumerate(keys):
            feed_dict[self.placeholders[key]] = sources[i]
        return feed_dict

    def get_embedding(self, sess, data='test'):
        feed_dict = {self.placeholders['is_training']: False}

        # Start Running Queue
        t = threading.Thread(target=self.load_and_enqueue, args=[sess, data])
        t.daemon = True
        t.start()

        embeddings = {}
        new_mse, old_mse, loss, embeddings['new_ent_ids'], embeddings['old_ent_ids'], \
        emb_ent_new, emb_ent_old, embeddings['emb_rel'] = \
            sess.run([self.model.new_ent_mse, self.model.old_ent_mse, self.model.loss,
                      self.data['new_ent_ids'], self.data['old_ent_ids'],
                      self.model.new_ent_predictions, self.model.old_ent_predictions, self.model.data['emb_rel']], feed_dict=feed_dict)
        with open(path.join(self.config.paths['data'], 'test_0_data_obj_b.pkl'), 'rb') as inp:
            new_emb = pickle.load(inp).emb_new_ent
            new_emb[embeddings['new_ent_ids'], :] = emb_ent_new
            new_emb[embeddings['old_ent_ids'], :] = emb_ent_old
            embeddings['new_emb'] = new_emb
            print('Embedding obtained')
        return (loss, new_mse, old_mse), embeddings

    def run_epoch(self, sess, data, learning_rate=0, summary_writers=None):
        if data == 'train':
            train_op = self.model.opt_op
            feed_dict = {self.placeholders['dropout']: self.config.dropout,
                         self.placeholders['lr']: learning_rate,
                         self.placeholders['is_training']: True}
        else:
            train_op = tf.no_op()
            feed_dict = {self.placeholders['is_training']: False}

        # Start Running Queue
        t = threading.Thread(target=self.load_and_enqueue, args=[sess, data])
        t.daemon = True
        t.start()

        loss = 0
        new_mse = 0
        old_mse = 0
        n_batches = self.dataset.n_batches[data]
        for step in range(n_batches):
            b_new_mse, b_old_mse, b_loss, _ = sess.run([self.model.new_ent_mse, self.model.old_ent_mse, self.model.loss, train_op], feed_dict=feed_dict)
            # print('Batch id: ', step, 'of Batches: ', self.dataset.n_batches[data], ' | Loss: ', b_loss, 'New MSE: ', b_new_mse, 'Old MSE: ', b_old_mse)
            loss += b_loss
            new_mse += b_new_mse
            old_mse += b_old_mse
        return (round(loss/n_batches, 3), round(new_mse/n_batches, 3), round(old_mse/n_batches, 3))

    def fit(self, sess, summary_writers):
        sess.run(tf.compat.v1.local_variables_initializer())
        threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=self.coord)

        epoch_id = 0
        lr = self.config.learning_rate

        tr_loss, val_loss = [], []
        for epoch_id in range(self.config.max_epochs):
            tr_op = self.run_epoch(sess, 'train', lr, summary_writers['train'])
            tr_loss.append(tr_op)

            if epoch_id % 2 == 0:
                te_loss, embeddings = self.get_embedding(sess, data='test')
                if not os.path.exists(self.config.paths['embed']):
                    os.makedirs(self.config.paths['embed'])
                with open(self.config.paths['embed']+'_'+str(epoch_id)+'_emb.pkl', 'wb') as out:
                    pickle.dump(embeddings, out, pickle.HIGHEST_PROTOCOL)
                print('Epoch: ', epoch_id, '######  Train New MSE: ', tr_loss[-1][1], '- Test New MSE:',
                      te_loss[1], '||  Train Old MSE: ', tr_loss[-1][2], '- Test Old MSE:', te_loss[2],
                      '||  Train loss: ', tr_loss[-1][0], '- Test loss:', te_loss[0])
            else:
                val_op = self.run_epoch(sess, 'val', lr, summary_writers['val'])
                val_loss.append(val_op)
                print('Epoch: ', epoch_id, '###  Train New MSE: ', tr_loss[-1][1], '- Val New MSE:',
                      val_loss[-1][1], '||  Train Old MSE: ', tr_loss[-1][2], '- Val Old MSE:',
                      val_loss[-1][2], '||  Train loss: ', tr_loss[-1][0], '- Val loss:',
                      val_loss[-1][0])

        self.coord.request_stop()
        self.coord.join(threads)
        return epoch_id, tr_loss, val_loss


def init_model(config, dataset):
    ops.reset_default_graph()
    tf.random.set_seed(1234)
    np.random.seed(1234)

    with tf.compat.v1.variable_scope('Graph_Convolutional_Network', reuse=None):
        model = Incremental_KG(dataset)

    # configure GPU usage
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
    n_epochs, tr_loss, val_loss = model.fit(sess, summary_writers)
    test_loss, embeddings = model.get_embedding(sess, data='test')
    print('N_epochs: ', n_epochs, '| Train loss: ', tr_loss[-1][0], '| Val loss:', val_loss[-1][0], '| Test loss:', test_loss[0])
    return test_loss, embeddings



def main():

    args = Parser().get_parser().parse_args()
    print("=====Configurations=====\n", args)

    # Load Configuration and data
    config = Config(args)
    dataset = Dataset(config)
    mse, embeddings = train_model(dataset)


if __name__ == "__main__":
    main()