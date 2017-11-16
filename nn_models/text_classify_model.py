from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import nn_layers
import sys

class TextClassifyModel(object):
  def __init__(self, tensor_dict, config, opt=None):
    self.learning_rate = config.learning_rate
    self.l2_reg_lambda = config.l2_reg_lambda
    self.batch_size = config.batch_size
    self.vocab_size = config.vocab_size
    self.embed_size = config.embedding_size
    self.filter_sizes = config.filter_sizes
    self.num_filters = config.num_filters
    self.fc_sizes = config.fc_sizes
    self.num_classes = config.num_classes
    self.keep_prob_for_train = config.keep_prob
    self.global_step = tf.Variable(0, name='global_step', trainable=False)

    ## inputs & placeholder
    inputs = tensor_dict["text"]
    labels = tensor_dict["label"]
    self.keep_prob = tf.placeholder_with_default(1.0, shape=None, name="keep_prob")
     
    ## sentence embedding
    inputs_embed = nn_layers.cnn_text_embedding(inputs, config.vocab_size, config.embedding_size, config.filter_sizes, config.num_filters, self.keep_prob)
    inputs_embed = nn_layers.multi_full_connect(inputs_embed, config.fc_sizes, activation='relu', keep_prob=self.keep_prob)
    logits = nn_layers.full_connect(inputs_embed, config.num_classes, name='output_layer')

    ## loss and optim
    self.loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
    tf.summary.scalar('loss', self.loss)
    if not opt:
      optim = nn_layers.get_optimizer(config.optimizer, learning_rate=self.learning_rate)
    else:
      optim = opt
    self.train_op = optim.minimize(self.loss, global_step=self.global_step)

    ## score & infers
    self.infers = tf.argmax(logits, 1)
    self.score = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,1), labels), tf.float32))
    tf.summary.scalar('score', self.score)

    ## saver & summary
    self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=config.max_to_keep)
    self.merged_summary = tf.summary.merge_all()

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
    for _ in range(validate_batches):
      loss, score = session.run([self.loss, self.score])
      validate_loss += loss
      validate_score += score 
    validate_loss = validate_loss / validate_batches
    validate_score = validate_score / validate_batches
    return validate_loss, validate_score
