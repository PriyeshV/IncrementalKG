import tensorflow as tf
from src.utils.metrics import *
from src.models.model import Model
from src.layers.dense import Dense


class KG_GCN(Model):

    def __init__(self, config, data, **kwargs):
        kwargs['name'] = 'KG_GCN'
        super(KG_GCN, self).__init__(**kwargs)

        self.data.update(data)
        self.inputs = data['ip_ent_emb']
        self.l2 = config.l2
        self.bias = config.bias
        self.n_nodes = data['n_nodes']
        self.n_rel = data['n_rel']

        # self.conv_layer = config.kernel_class
        self.n_layers = config.max_depth
        self.input_dims = config.emb_dim
        self.output_dims = config.emb_dim

        self.dims = [config.emb_dim] * self.n_layers
        self.act = [tf.nn.tanh] * (self.n_layers)
        self.act.append(lambda x: x)
        self.conv_layer = config.kernel_class

        self.dropouts = self.data['dropout']
        self.optimizer = config.opt(learning_rate=data['lr'])
        self.values = []
        self.build()
        self.new_ent_predictions, self.old_ent_predictions = self.predict()

    def _build(self):
        # TODO Featureless
        self.layers.append(
            Dense(input_dim=self.input_dims, output_dim=self.output_dims, dropout=self.dropouts,
                  act=self.act[0], bias=self.bias, logging=self.logging, model_name=self.name))

        self.data['h_rel_out'] = tf.compat.v1.sparse_tensor_dense_matmul(self.data['rel_out_mat'], self.data['emb_rel'])
        self.data['h_rel_in'] = tf.compat.v1.sparse_tensor_dense_matmul(self.data['rel_in_mat'], self.data['emb_rel'])

        for i in range(1):
        # for i in range(self.n_layers):
            self.layers.append(self.conv_layer(layer_id=i, x_names=['x', 'h'], adj_name='adj_out', dims=self.dims,
                                               dropout=self.dropouts, act=self.act[i], bias=self.bias,
                                               shared_weights=False, skip_connection=True,
                                               logging=self.logging, model_name=self.name))

        self.layers.append(
            Dense(input_dim=self.input_dims, output_dim=self.output_dims, dropout=self.dropouts,
                  act=self.act[-1], bias=self.bias, logging=self.logging, model_name=self.name))

    def predict(self):
        new_ent_predictions = tf.gather(self.outputs, tf.squeeze(tf.where(self.data['mask_new'])))
        old_ent_predictions = tf.gather(self.outputs, tf.squeeze(tf.where(self.data['mask_old'])))
        return new_ent_predictions, old_ent_predictions

    def _loss(self):
        self.loss = 0
        new_ent_predictions, old_ent_predictions = self.predict()
        target_ent = tf.gather(self.data['op_ent_emb'], tf.squeeze(tf.where(self.data['mask_new'])))
        target_old = tf.gather(self.data['op_ent_emb'], tf.squeeze(tf.where(self.data['mask_old'])))

        self.target_ent = target_ent
        self.target_old = target_old

        # Squared error loss
        new_ent_mse = tf.reduce_mean(tf.square(new_ent_predictions - target_ent))   # add entity weigthting
        old_ent_mse = tf.reduce_mean(tf.square(old_ent_predictions - target_old))
        self.mse_loss = new_ent_mse + old_ent_mse
        self.loss += self.mse_loss

        # L2 Loss
        for v in tf.compat.v1.trainable_variables():
            reject_cands = ['bias']
            if v.name not in reject_cands:  # and sub_names[2][:7] not in ['weights']:
                self.loss += self.l2 * tf.nn.l2_loss(v)
        # self.loss *= self.density
        tf.summary.scalar('loss', self.loss)

    def _accuracy(self):
        self.metric_values = {}
        # predictions = self.predict()