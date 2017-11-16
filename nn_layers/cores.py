import tensorflow as tf

def weight_variable(shape, l2_reg_lambda=None, l1_reg_lambda=None):
  regularizer = None
  if l2_reg_lambda:
      regularizer = tf.contrib.layers.l2_regularizer(l2_reg_lambda)
  elif l1_reg_lambda:
      regularizer = tf.contrib.layers.l1_regularizer(l1_reg_lambda)
  return tf.get_variable('weight', shape, initializer=tf.random_normal_initializer(stddev=0.1), regularizer=regularizer)

def bias_variable(shape):
  return tf.get_variable('bias', shape, initializer=tf.constant_initializer(0.1))

def full_connect_(inputs, num_units, activation=None, use_bn = None, keep_prob = 1.0, name='full_connect_'):
  with tf.variable_scope(name):
    shape = [inputs.get_shape()[-1], num_units]
    weight = weight_variable(shape)
    bias = bias_variable(shape[-1])
    outputs_ = tf.matmul(inputs, weight) + bias
    if use_bn:
        outputs_ = tf.contrib.layers.batch_norm(outputs_, center=True, scale=True, is_training=True,decay=0.9,epsilon=1e-5, scope='bn')
    if activation=="relu":
      outputs = tf.nn.relu(outputs_)
    elif activation == "tanh":
      outputs = tf.tanh(outputs_)
    elif activation == "sigmoid":
      outputs = tf.nn.sigmoid(outputs_)
    elif activation == "elu":
      outputs = tf.nn.elu(outputs_)
    else:
      outputs = outputs_
    #if use_bn:
    #    outputs = tf.contrib.layers.batch_norm(outputs, center=True, scale=True, is_training=True,decay=0.9,epsilon=1e-5, scope='bn')
    return  outputs

'''
def full_connect(inputs, num_units, activation=None, name='full_connect'):
  with tf.variable_scope(name):
    shape = [inputs.get_shape()[-1], num_units]
    weight = weight_variable(shape)
    bias = bias_variable(shape[-1])
    outputs = tf.matmul(inputs, weight) + bias
    if activation=="relu":
      outputs = tf.nn.relu(outputs)
    elif activation == "tanh":
      outputs = tf.tanh(outputs)
    elif activation == "sigmoid":
      outputs = tf.nn.sigmoid(outputs)
    return outputs
'''   

def cnn1d(inputs, filter_sizes, num_filters, stride=1, keep_prob=1.0, name='cnn'):
  with tf.variable_scope(name) as scope:
    inputs_expand = tf.expand_dims(inputs, 1)
    in_channel = inputs.get_shape()[-1]
    out_channel = num_filters
    outputs_list = [] 
    for i, ksize in enumerate(filter_sizes):
      with tf.variable_scope('conv-%d' % ksize):
        filter_shape = [1, ksize, in_channel, out_channel]
        w = cores.weight_variable(filter_shape)
        b = cores.bias_variable([filter_shape[-1]])
        conv = tf.nn.conv2d(inputs_expand, w, strides=[1,1,stride,1], padding='SAME', name='conv')
        relu = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
        outputs_list.append(relu)
    outputs = tf.squeeze(tf.concat(outputs_list, 3), 1)
    return tf.nn.dropout(outputs, keep_prob)


def inception_cnn1d(inputs, filter_sizes=[1,3,5], num_filters=32, incep_channel=16, stride=1, keep_prob=1.0, name='inception_cnn'):
  with tf.variable_scope(name) as scope:
    inputs_expand = tf.expand_dims(inputs, 1)
    in_channel = inputs.get_shape()[-1]
    out_channel = num_filters
    for i, ksize in enumerate(filter_sizes):
      if ksize == 1:
        with tf.variable_scope('conv-%d' % ksize):
          filter_shape = [1, ksize, in_channel, out_channel]
          w = cores.weight_variable(filter_shape)
          b = cores.bias_variable([filter_shape[-1]])
          conv = tf.nn.conv2d(inputs_expand, w, strides=[1,1,stride,1], padding='SAME', name='conv')
          output = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
          outputs_list.append(output)
      else:
        with tf.variable_scope('conv-1x1'):
          filter_shape = [1, 1, in_channel, incep_channel]
          w = cores.weight_variable(filter_shape)
          b = cores.bias_variable([filter_shape[-1]])
          conv = tf.nn.conv2d(inputs_expand, w, strides=[1,1,1,1], padding='SAME', name='conv')
          tmp_output = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
        with tf.variable_scope('conv-1x%d' % ksize):
          filter_shape = [1, ksize, incep_channel, out_channel]
          w = cores.weight_variable(filter_shape)
          b = cores.bias_variable([incep_channel[-1]])
          conv = tf.nn.conv2d(tmp_output, w, strides=[1,1,stride,1], padding='SAME', name='conv')
          output = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
          outputs_list.append(output)
    outputs = tf.squeeze(tf.concat(outputs_list, 3), 1)
    return tf.nn.dropout(outputs, keep_prob)


def rnn(inputs, sequence_length, num_units, num_layers=1, type='lstm', keep_prob=1.0, name='rnn'):
  with tf.variable_scope(name) as scope:
    if type=='lstm':
      def single_cell():
        return tf.contrib.rnn.BasicLSTMCell(num_units, reuse=tf.get_variable_scope().reuse)
    elif type=='gru':
      def single_cell():
        return tf.contrib.rnn.GRUCell(num_units, reuse=tf.get_variable_scope().reuse)
    else:
      def single_cell():
        return tf.nn.rnn_cell.BasicRNNCell(num_units, reuse=tf.get_variable_scope().reuse)
    cell = single_cell()
    if num_layers > 1:
      cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)])
    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, sequence_length, dtype=tf.float32, scope=scope)
    return outputs
# inputs size: shape: [batch_size, sequence_length, embed_size]
# outputs size: shape:[batch_size, sequence_length, embed_size]
# final_state: shape: [batch_size, embed_size]

def birnn(inputs, lengths, num_units, num_layers=1, type='lstm', keep_prob=1.0, combiner='concat', name='birnn'):
  with tf.variable_scope(name) as scope:
    if type=='lstm':
      def single_cell():
        return tf.contrib.rnn.BasicLSTMCell(num_units, reuse=tf.get_variable_scope().reuse)
    elif type=='gru':
      def single_cell():
        return tf.contrib.rnn.GRUCell(num_units, reuse=tf.get_variable_scope().reuse)
    else:
      def single_cell():
        return tf.nn.rnn_cell.BasicRNNCell(num_units, reuse=tf.get_variable_scope().reuse)
    fw_cell = single_cell()
    bw_cell = single_cell()
    if num_layers > 1:
      fw_cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)])
      bw_cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)])
    outputs, state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs, lengths, dtype=tf.float32, scope=scope)
    if combiner=='concat':
      combine_outputs = tf.concat(outputs, 2)
    elif combiner=='sum':
      combine_outputs = outputs[0] + outputs[1]
    else:
      combine_outputs = None
    return combine_outputs
      

def multi_full_connect(inputs, hidden_units, activation=None, keep_prob=1.0,  name='multi_full_connect'):
  with tf.variable_scope(name):
    layer = inputs
    for i, num_units in enumerate(hidden_units):
      layer = full_connect(layer, num_units, activation=activation, name=('fc%d' % (i+1)))
      layer = tf.nn.dropout(layer, keep_prob)
    return layer


def cosine_similarity(inputs1, inputs2, name='cosine_similarity'):
  with tf.name_scope(name):
    norm1 = tf.sqrt(tf.reduce_sum(tf.square(inputs1), 1, keep_dims=True))
    normlized_inputs1 = inputs1 / norm1
    norm2 = tf.sqrt(tf.reduce_sum(tf.square(inputs2), 1, keep_dims=True))
    normlized_inputs2 = inputs2 / norm2
    return tf.reduce_sum(tf.multiply(normlized_inputs1, normlized_inputs2), 1, keep_dims=True)


def mlp_similarity(inputs1, inputs2, hidden_units, keep_prob=1.0, name='mlp_similarity'):
  with tf.name_scope(name):
    layer = tf.concat([inputs1, inputs2], 1)
    for i, num_units in enumerate(hidden_units):
      if i+1 < len(hidden_units):
        layer = full_connect(layer, num_units, activation='relu', name='full_connect_%d' % (i+1))
        layer = tf.nn.dropout(layer, keep_prob)
      else:
        layer = full_connect(layer, num_units, name='full_connect_%d' % (i+1))
    return layer


def get_length(inputs, pad_id):
  return tf.reduce_sum(tf.cast(tf.not_equal(inputs, pad_id), tf.int32), 1)
