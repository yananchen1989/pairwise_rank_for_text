from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import nn_layers
import sys

class TextRankRnnModel(object):
  def __init__(self, tensor_dict, config, opt=None):
    self.learning_rate = config.learning_rate
    self.l2_reg_lambda = config.l2_reg_lambda
    self.batch_size = config.batch_size
    self.vocab_size = config.vocab_size
    self.embed_size = config.embedding_size
    self.rnn_size = config.rnn_size
    self.rnn_type = config.rnn_type
    self.num_rnn_layers = config.num_rnn_layers
    self.fc_sizes = config.fc_sizes
    self.pad_id = config.pad_id
    self.keep_prob_for_train = config.keep_prob
    self.global_step = tf.Variable(0, name='global_step', trainable=False)

    with tf.name_scope('input'):
      self.query_inputs = tensor_dict["query"]
      self.title_inputs_pos = tensor_dict["title_pos"]
      self.title_inputs_neg = tensor_dict["title_neg"]
      self.labels = tf.constant(1, tf.int32, [self.batch_size, 1], name="label")
      self.keep_prob = tf.placeholder_with_default(1.0, shape=None, name="keep_prob")

    with tf.variable_scope('sentence_embedding'):
      query_embed = nn_layers.rnn_text_embedding(self.query_inputs, self.vocab_size, self.embed_size, self.rnn_size, self.num_rnn_layers, self.rnn_type, self.pad_id, self.keep_prob)
      tf.get_variable_scope().reuse_variables()
      title_pos_embed = nn_layers.rnn_text_embedding(self.title_inputs_pos, self.vocab_size, self.embed_size, self.rnn_size, self.num_rnn_layers, self.rnn_type, self.pad_id, self.keep_prob)
      title_neg_embed = nn_layers.rnn_text_embedding(self.title_inputs_neg, self.vocab_size, self.embed_size, self.rnn_size, self.num_rnn_layers, self.rnn_type, self.pad_id, self.keep_prob)

    with tf.variable_scope('sentence_similarity'):
      pos_pair_sim = nn_layers.mlp_similarity(query_embed, title_pos_embed, self.fc_sizes, self.keep_prob)
      tf.get_variable_scope().reuse_variables()
      neg_pair_sim = nn_layers.mlp_similarity(query_embed, title_neg_embed, self.fc_sizes, self.keep_prob)

    with tf.name_scope('predictions'):
      sim_diff = pos_pair_sim - neg_pair_sim
      predictions = tf.sigmoid(sim_diff)
      self.infers = pos_pair_sim

    ## loss and optim
    with tf.name_scope('loss'):
      self.loss = nn_layers.cross_entropy_loss_with_reg(self.labels, predictions)
      tf.summary.scalar('loss', self.loss)

    with tf.name_scope('optim'):
      if not opt:
        optim = nn_layers.get_optimizer(config.optimizer, learning_rate=self.learning_rate, momentum=config.momentum)
      else:
        optim = opt
      self.train_op = optim.minimize(self.loss, global_step=self.global_step)

    with tf.name_scope('score'):
      self.correct_num = tf.reduce_sum(tf.cast(tf.greater(predictions, 0.5), tf.float32))
      self.wrong_num = tf.reduce_sum(tf.cast(tf.less(predictions, 0.5), tf.float32))
      self.score = self.correct_num / (self.wrong_num + 0.0001)
      tf.summary.scalar('score', self.score)

    self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=config.max_to_keep)
    self.merged = tf.summary.merge_all()

  def get_infer(self):
    return self.infers

  def train_step(self, session):
    _, loss, score, summary_pb = session.run([self.train_op, self.loss, self.score, self.merged], feed_dict={self.keep_prob: self.keep_prob_for_train})
    return loss, score, summary_pb

  def infer_step(self, session):
    values = session.run(self.infers)
    for i in range(values.shape[0]):
      print(values[i][0])
      sys.stdout.flush()
    return values

  def validate_steps(self, session, validate_batches):
    validate_score = 0.0
    validate_loss = 0.0
    correct_num_total = 0
    wrong_num_total = 0
    for _ in range(validate_batches):
      loss, correct_num, wrong_num = session.run([self.loss, self.correct_num, self.wrong_num])
      validate_loss += loss
      correct_num_total += correct_num
      wrong_num_total += wrong_num
    validate_loss = validate_loss / validate_batches
    validate_score = correct_num_total / (wrong_num_total + 0.0001)
    return validate_loss, validate_score
