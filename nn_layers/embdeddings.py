import tensorflow as tf
import cores

def word_embedding(shape, dtype=tf.float32, name='word_embedding'):
  with tf.device('/cpu:0'), tf.variable_scope(name):
    return tf.get_variable('embedding', shape, dtype=dtype, initializer=tf.random_normal_initializer(stddev=0.1), trainable=True)


def sparse_text_embedding(inputs, shape, weights=None, name='sparse_text_embedding'):
  with tf.variable_scope(name):
    embedding = word_embedding(shape)
    #return tf.nn.embedding_lookup_sparse(embedding, inputs, weights, combiner='sum')
    return tf.nn.relu(tf.nn.embedding_lookup_sparse(embedding, inputs, weights, combiner='sum'))


def dense_text_embedding(inputs, shape, weights=None, name='dense_text_embedding'):
  with tf.variable_scope(name):
    embedding = word_embedding(shape)
    return tf.contrib.layers.embedding_lookup_unique(embedding, inputs)


def cnn_text_embedding(inputs, vocab_size, embed_size, filter_sizes, num_filters, keep_prob=1.0, name='cnn_text_embedding'):
  with tf.variable_scope(name):
    embedding = word_embedding([vocab_size, embed_size])
    #word_embedding = tf.nn.embedding_lookup(embedding, inputs)
    inputs_embedding = tf.contrib.layers.embedding_lookup_unique(embedding, inputs)
    inputs_embedding_expand = tf.expand_dims(inputs_embedding, -1)
    pooled_outputs = []
    for i, ksize in enumerate(filter_sizes):
      with tf.variable_scope('conv-maxpool-%d' % ksize):
        filter_shape = [ksize, embed_size, 1, num_filters]
        w = cores.weight_variable(filter_shape)
        b = cores.bias_variable([num_filters])
        conv = tf.nn.conv2d(inputs_embedding_expand, w, strides=[1,1,1,1], padding='VALID', name='conv')
        relu = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
        pooled = tf.nn.max_pool(relu, ksize=[1,relu.get_shape()[1],1,1], strides=[1,1,1,1], padding='VALID', name='maxpool')
        pooled_outputs.append(pooled)
  
    outputs = tf.concat(pooled_outputs, 3)
    outputs_flat = tf.reshape(outputs, [-1, int(outputs.get_shape()[3])])
    outputs_drop = tf.nn.dropout(outputs_flat, keep_prob)
    return outputs_drop 


def rnn_text_embedding(inputs, vocab_size, embed_size, num_units, num_layers=1, type='lstm', pad_id=0, keep_prob=1.0, name='rnn_text_embedding'):
  with tf.variable_scope(name) as scope:
    inputs_embedding = dense_text_embedding(inputs, [vocab_size, embed_size], name=scope)
    sequence_length = cores.get_length(inputs, pad_id)
    outputs = cores.rnn(inputs_embedding, sequence_length, num_units, num_layers, type=type)
    batch_size = tf.shape(outputs)[0]
    max_length = int(outputs.get_shape()[1])
    out_size = int(outputs.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (sequence_length - 1)
    return tf.gather(tf.reshape(outputs, [-1, out_size]), index)  ## last output: shape: (batch_size, out_size)
# tf.reshape(outputs, [-1, out_size]) ==>  shape: (batch_size*max_length, out_size)  
# for example:  (1000,20,128) ==> (20000, 128)

def birnn_text_embedding(inputs, vocab_size, embed_size, num_units, num_layers=1, type='lstm', pad_id=0, combiner='max', keep_prob=1.0, name='birnn_text_embedding'):
  with tf.variable_scope(name) as scope:
    inputs_embedding = dense_text_embedding(inputs, [vocab_size, embed_size], name=scope)
    lengths = cores.get_length(inputs, pad_id)
    outputs = cores.birnn(inputs_embedding, lengths, num_units, num_layers, type=type, combiner='concat')
    if combiner=='max':
      outputs_pool = tf.reduce_max(outputs, 1)
    elif combiner=='avg':
      outputs_pool = tf.reduce_sum(outputs, 1) / tf.cast(tf.reshape(lengths, [-1,1]), tf.float32)
    else:
      outputs_pool = None
    return outputs_pool
