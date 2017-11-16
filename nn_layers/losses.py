import tensorflow as tf


def cross_entropy_loss_with_reg(labels, predictions):
  reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
  #cross_entropy = -tf.reduce_mean(tf.log(predictions))
  #cross_entropy = tf.losses.sigmoid_cross_entropy(self.labels, logits)
  cross_entropy = tf.losses.log_loss(labels, predictions)
  #self.loss = tf.contrib.losses.log_loss(predictions, self.labels)
  return cross_entropy + reg_loss
