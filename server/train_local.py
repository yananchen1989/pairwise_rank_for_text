from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import tensorflow as tf
import nn_model
import nn_config

flags = tf.app.flags
flags.DEFINE_string('conf', "config/qt_config.json", 'cluster config file')
FLAGS = flags.FLAGS


def main(_):
    config = nn_config.NNConfig(FLAGS.conf).get_config()
    model = nn_model.NNModel(config)

    if not tf.gfile.Exists(config.dump_path):
        tf.gfile.MakeDirs(config.dump_path)
    
    with tf.Session() as sess:
        model.set_session(sess)
        if config.train:
            model.train()
        else:
            model.inference()

if __name__ == "__main__": 
    tf.app.run()
