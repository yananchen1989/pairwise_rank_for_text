import tensorflow as tf


def get_optimizer(optimizer='sgd', learning_rate=None, momentum=None):
  if optimizer == 'sgd':
    if not learning_rate: learning_rate = 0.01
    return tf.train.GradientDescentOptimizer(learning_rate)
  elif optimizer == 'momentum':
    if not learning_rate: learning_rate = 0.01
    if not momentum: momentum = 0.9
    return tf.train.MomentumOptimizer(learning_rate, momentum)
  elif optimizer == 'rmsprop':
    if not learning_rate: learning_rate = 0.001
    return tf.train.RMSPropOptimizer(learning_rate)
  elif optimizer == 'adam':
    if not learning_rate: learning_rate = 0.001
    return tf.train.AdamOptimizer(learning_rate)
  else:
    raise Exception("Invalid incoming optimizer name.")
