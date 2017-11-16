from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import nn_layers
import sys

class TextDssmModel(object):
  def __init__(self, tensor_dict, config, opt=None):
    self.learning_rate = config.learning_rate
    self.l2_reg_lambda = config.l2_reg_lambda
    self.batch_size = config.batch_size
    self.vocab_size = config.vocab_size
    self.embed_size = config.embedding_size
    self.fc_sizes = config.fc_sizes
    self.keep_prob_for_train = config.keep_prob
    self.global_step = tf.Variable(0, name='global_step', trainable=False)

    with tf.name_scope('inputs'):
      query = tensor_dict["query"]
      titles = [tensor_dict["title"]]
      titles.append(tensor_dict["title_neg1"])
      titles.append(tensor_dict["title_neg2"])
      titles.append(tensor_dict["title_neg3"])
      titles.append(tensor_dict["title_neg4"])
      labels = tf.constant(0, tf.int64, [self.batch_size], name="label")
      self.keep_prob = tf.placeholder_with_default(1.0, shape=None, name="keep_prob")

    with tf.variable_scope('sentence_embedding'):
      tmp_embed = nn_layers.sparse_text_embedding(query, [self.vocab_size, self.embed_size])
      query_embed = nn_layers.multi_full_connect(tmp_embed, config.fc_sizes, activation='relu')
      tf.get_variable_scope().reuse_variables()
      titles_embed = []
      for i in range(len(titles)):
        tmp_embed = nn_layers.sparse_text_embedding(titles[i], [self.vocab_size, self.embed_size])
        titles_embed.append(nn_layers.multi_full_connect(tmp_embed, config.fc_sizes, activation='relu'))

    with tf.variable_scope('sentence_similarity'):
      pairs_sim = []
      for i in range(len(titles)):
        if i > 0:
          tf.get_variable_scope().reuse_variables()
        pairs_sim.append(nn_layers.cosine_similarity(query_embed, titles_embed[i]))
      logits = tf.concat(pairs_sim, 1)
        
    with tf.name_scope('score'):
      self.correct_num = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(logits,1), labels), tf.float32))
      self.wrong_num = self.batch_size - self.correct_num
      self.score = self.correct_num / (self.wrong_num + 0.0001)
      self.infers = pairs_sim[0]
      tf.summary.scalar('score', self.score)

    ## loss and optim
    with tf.name_scope('loss'): # logits: (?,4)  labels: (1000,)
      self.loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
      tf.summary.scalar('loss', self.loss)

    with tf.name_scope('optim'):
      if not opt:
        optim = nn_layers.get_optimizer(config.optimizer, learning_rate=self.learning_rate)
      else:
        optim = opt
      self.train_op = optim.minimize(self.loss, global_step=self.global_step)

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
