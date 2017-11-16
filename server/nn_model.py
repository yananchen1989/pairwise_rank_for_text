import tensorflow as tf
import time
import sys
import os
import nn_models
import tf_util.parser.sm_standard_kv_parser as lib_parser
from tf_util.ganglia import gmetric_writer
import logging


class NNModel(object):
    def __init__(self, config, num_workers=1, task_index=0, opt=None, metric_host=None, metric_group=None, distribute=False):
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(logging.StreamHandler())
        self.logger.setLevel(logging.INFO)
        self.sess=None
        self.task_index = task_index
        self.opt = opt
        self.config = config
        input_schema = self.config.input_schema
        parse_schema = self.config.parse_schema
        self.distribute = distribute
        if self.distribute:
            gmetric_writer.init(job_name="worker", task_index=task_index, host=metric_host, group=metric_group)

        parser = lib_parser.schema_parser.KvSchemaParser(input_schema)
        st = parser.parse(parse_schema)

        if self.distribute:
            if(task_index == 0):
                self.reader = lib_parser.standard_kv_reader.StandardKvHdfsReader(config.validation_path, config.batch_size, record_flag=st.record_flag, cycle=sys.maxint)
            else:
                self.reader = lib_parser.standard_kv_reader.StandardKvHdfsReader(config.train_path, config.batch_size, record_flag=st.record_flag, cycle=config.num_epoches, total_workers=(num_workers-1),worker_index=task_index-1)
        else:
            self.reader = lib_parser.standard_kv_reader.StandardKvHdfsReaderWithValidation(train_data_paths=self.config.train_path, batch_size=self.config.batch_size, record_flag=st.record_flag, cycle=self.config.num_epoches, total_workers=num_workers, worker_index=task_index, shuffle=True, validation_data_paths=self.config.validation_path, validation_step=self.config.validation_step, validation_batch=self.config.validation_batch)

        self.reader.init()
        self.raw_input = self.reader.get_raw_input_tensor()
        standardKvParser = lib_parser.StandardKvParser(self.raw_input, input_schema, parse_schema)
        tensor_dict = standardKvParser.get_tensor_dict()

        self.logger.info("train model is %s" %(self.config.nn))
        self.model = nn_models.nn_factory(self.config.nn, tensor_dict, self.config, opt)
        self.global_step = self.model.global_step
        tf.add_to_collection('raw_input', self.raw_input)
        tf.add_to_collection('infer', self.model.get_infer())

    def set_session(self, sess):
        self.sess = sess
        if (not self.distribute):
            self.load_model()

    def train(self):
        if self.config.summary_step:
            summary_writer = tf.summary.FileWriter(self.config.dump_path, self.sess.graph)
        self.reader.set_session(self.sess)
        self.reader.start()
        loss, score, step_time = 0.0, 0.0, 0.0
        current_step = 0
        while(True):
            ## training
            if self.task_index != 0 or not self.distribute:
                begin_time = time.time()
                step_loss, step_score, summary_pb = self.model.train_step(self.sess)
                step_time += (time.time() - begin_time) / self.config.log_step
                loss += step_loss / self.config.log_step
                score += step_score / self.config.log_step

            ## train log
            if (self.task_index != 0 or not self.distribute) and current_step % self.config.log_step == 0 :
                if self.config.gmetric_flag and self.distribute:
                    gmetric_writer.write('train_score', score, type='float', units='score')
                    gmetric_writer.write('train_loss', loss, type='float', units='loss')
                self.logger.info('training step_%d  learning_rate  %.3f  step-time %.2f  score %.3f  loss %.8f' 
                                 % (current_step, self.model.learning_rate, step_time, score, loss))
                loss, score, step_time = 0.0, 0.0, 0.0

            ## validation log
            global_step_val = self.global_step.eval(self.sess)
            if (self.task_index == 0 or not self.distribute) and self.config.validation_step and global_step_val % self.config.validation_step == 0:
                validate_loss, validate_score = self.model.validate_steps(self.sess, self.config.validation_batch)
                if self.config.gmetric_flag and self.distribute:
                    gmetric_writer.write('validate_score', validate_score, type='float', units='score')
                    gmetric_writer.write('validate_loss', validate_loss, type='float', units='loss')
                self.logger.info('validating global_step_%d  validate_score %.3f  validate_loss %.8f'
                                 % (global_step_val, validate_score, validate_loss))

            ## dump model (local)
            if (not self.distribute) and self.config.dump_step and global_step_val % self.config.dump_step == 0:
                checkpoint_path = os.path.join(self.config.dump_path, 'model')
                self.model.saver.save(self.sess, checkpoint_path, global_step=self.global_step)
                self.logger.info('dump model...%s/model-%d' % (checkpoint_path, global_step_val))

            ## dump summary (local)
            if (not self.distribute) and self.config.summary_step and global_step_val % self.config.summary_step == 0:
                summary_writer.add_summary(summary_pb, global_step_val)
                self.logger.info('summary writing...step-%d' % global_step_val)

            current_step += 1
        self.reader.stop()

    def inference(self):
        self.reader.set_session(self.sess)
        self.reader.start()
        begin_time = time.time()
        consume = 0
        while True:
            results = self.model.infer_step(self.sess)
            consume += 1
            if (consume % 10 == 0):
                self.logger.info("qps is %f" %((consume * self.config.batch_size) / ((time.time() - begin_time))))
        self.reader.stop()

    def load_model(self):
        if self.config.load_path:
            self.logger.info("Reading model parameters from %s" % self.config.load_path)
            self.model.saver.restore(self.sess, self.config.load_path)
        else:
            self.logger.info("Created model with fresh parameters.")
            self.sess.run(tf.global_variables_initializer())
