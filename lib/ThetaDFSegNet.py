import tensorflow as tf
import numpy as np
import scipy.io
from math import ceil
import cv2
from utils.BatchDatasetReader import BatchDatasetReader
from utils.DatasetReader import DatasetReader
from utils.DataPreprocessor import DataPreprocessor
from utils.DataPostprocessor import DataPostprocessor
from utils.OutsideDataFeeder import OutsideDataFeeder
from utils.CustomTestDataFeeder import CustomTestDataFeeder
from utils.Logger import Logger
from PIL import Image
import datetime
import os
import math

# Only use a single GPU when not testing
if os.name != 'nt': 
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'


class ThetaDFSegNet:
    def load_vgg_weights(self):
        """ Use the VGG model trained on
            imagent dataset as a starting point for training """
        vgg_path = "models/imagenet-vgg-verydeep-19.mat"
        vgg_mat = scipy.io.loadmat(vgg_path)

        self.vgg_params = np.squeeze(vgg_mat['layers'])
        self.layers = ('conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
                        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
                        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
                        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
                        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
                        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
                        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
                        'relu5_3', 'conv5_4', 'relu5_4')

    def __init__(self, dataset_directory, num_classes=11):
        self.dataset_directory = dataset_directory
        self.num_classes = num_classes
        self.load_vgg_weights()
        self.build()

        # Begin a TensorFlow session
        config = tf.ConfigProto(allow_soft_placement=True)
        self.session = tf.Session(config=config)
        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.local_variables_initializer())

        # Make saving trained weights and biases possible
        self.saver = tf.train.Saver(max_to_keep = 5, 
                                    keep_checkpoint_every_n_hours = 1)
        self.checkpoint_directory = './checkpoints/'

        self.logger = Logger()
        # Add summary logging capability
        """
        self.train_writer = tf.summary.FileWriter('./summaries' + '/train', 
                                                  self.session.graph)
        self.val_writer = tf.summary.FileWriter('./summaries' + '/val', 
                                                 self.session.graph)
        """


    def vgg_weight_and_bias(self, name, W_shape, b_shape):
        """ 
            Initializes weights and biases to the pre-trained VGG model.
            
            Args:
                name: name of the layer for which you want to initialize weights
                W_shape: shape of weights tensor exkpected
                b_shape: shape of bias tensor expected
            returns:
                w_var: Initialized weight variable
                b_var: Initialized bias variable
        """
        if name not in self.layers:
            return self.weight_variable(W_shape), self.weight_variable(b_shape)
        else:
            w, b = self.vgg_params[self.layers.index(name)][0][0][0][0]
            init_w = tf.constant(value=np.transpose(w, (1, 0, 2, 3)), 
                                 dtype=tf.float32, shape=W_shape)
            init_b = tf.constant(value=b.reshape(-1), dtype=tf.float32, 
                                 shape=b_shape)
            w_var = tf.Variable(init_w)
            b_var = tf.Variable(init_b)
            return w_var, b_var 

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def pool_layer(self, x):
        return tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1], 
                                          strides=[1, 2, 2, 1], padding='SAME')

    def unpool(self, pool, ind, ksize=[1, 2, 2, 1], scope='unpool'):
        """ Unpooling layer after max_pool_with_argmax.
        Args:
            pool: max pooled output tensor
            ind: argmax indices
            ksize: ksize is the same as for the pool
        Return:
            unpool: unpooling tensor
        """
        with tf.variable_scope(scope):
            input_shape =  tf.shape(pool)
            output_shape = [input_shape[0], 
                            input_shape[1] * ksize[1], 
                            input_shape[2] * ksize[2], 
                            input_shape[3]]

            flat_input_size = tf.cumprod(input_shape)[-1]
            flat_output_shape = tf.stack([output_shape[0], output_shape[1] 
                                                           * output_shape[2] 
                                                           * output_shape[3]])

            pool_ = tf.reshape(pool, tf.stack([flat_input_size]))
            batch_range = tf.range(tf.cast(output_shape[0], tf.int64), 
                                   dtype=ind.dtype)
            reshape_shape = tf.stack([input_shape[0], 1, 1, 1])
            reshaped_batch_range = tf.reshape(batch_range, 
                                              shape=reshape_shape)

            b = tf.ones_like(ind) * reshaped_batch_range
            b = tf.reshape(b, tf.stack([flat_input_size, 1]))
            ind_ = tf.reshape(ind, tf.stack([flat_input_size, 1]))
            ind_ = tf.concat([b, ind_], 1)

            ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, 
                                                           tf.int64))
            ret = tf.reshape(ret, tf.stack(output_shape))
            return ret

    def conv_layer_with_bn(self, x, W_shape, train_phase, name, padding='SAME'):
        b_shape = W_shape[3]
        W, b = self.vgg_weight_and_bias(name, W_shape, [b_shape])
        convolved_output = tf.nn.conv2d(x, W, strides=[1,1,1,1], 
                                        padding=padding) + b
        batch_norm = tf.contrib.layers.batch_norm(convolved_output, 
                                                  is_training=train_phase)
        return tf.nn.relu(batch_norm)

    def gen_dynamic_filter(self, pooled_layer, filter_shape):
        '''
            filter_shape=[3, 3, 512, 512]

            tf.reduce_mean(pool_5, axis=3)
            pool_5 shape = NUM_BATCHES * WIDTH * HEIGHT * 512
            feature_map shape = NUM_BATCHES * WIDTH * HEIGHT

        '''
        # feature_map shape = BATCH_SIZE * WIDTH * HEIGHT
        feature_map = tf.reduce_mean(pooled_layer, axis=3)
        # feature_map_shape = tf.shape(feature_map)
        length = feature_map.get_shape()[1] * feature_map.get_shape()[2]
        # TODO: Make this workable for any image shape
        # length = 480 * 320
        features = tf.reshape(feature_map, [-1, int(length)])
        # features_shape = tf.shape(features)
        # features.set_shape([features_shape[0], features_shape[1]])
        fc1 = tf.contrib.layers.fully_connected(features, 64)
        fc2 = tf.contrib.layers.fully_connected(fc1, 128)
        fc3 = tf.contrib.layers.fully_connected(fc2, 
                                                filter_shape[0]*filter_shape[1], 
                                                activation_fn=None)
        fc3 = tf.reduce_mean(fc3, axis=0)
        filt = tf.reshape(fc3, filter_shape[0:2])
        filt = tf.expand_dims(filt,2)
        filt = tf.expand_dims(filt,3)
        filt = tf.tile(filt,[1,1,filter_shape[2],filter_shape[3]])
        return filt 

    def dynamic_conv_layer(self, bottom, filter_shape, dynamic_filter, name, 
                           strides=[1,1,1,1], padding="SAME"):
        init_w = tf.truncated_normal(filter_shape, stddev=0.2)
        init_b = tf.constant_initializer(value=0.0, dtype=tf.float32)
        filt = tf.get_variable(name="%s_w"%name,
                               initializer=init_w,
                               dtype=tf.float32)
        filt = tf.add(filt, dynamic_filter)
        conv = tf.nn.conv2d(bottom,filter=filt,strides=strides,padding=padding,
                                   name=name)
        bias = tf.get_variable(name="%s_b"%name,initializer=init_b,
                               shape=[filter_shape[-1]],dtype=tf.float32)
        return tf.nn.bias_add(conv, bias)

    def build(self):
        with tf.device('/gpu:0'):
            # Declare placeholders
            # self.x dimensions = BATCH_SIZE * HEIGHT * WIDTH * NUM_CHANNELS
            # TODO: Make flexible for image sizes
            self.x = tf.placeholder(tf.float32, shape=[None, 320, 480, 3])
            # self.y dimensions = BATCH_SIZE * WIDTH * HEIGHT
            self.y = tf.placeholder(tf.int64, shape=[None, 320, 480])
            expected = tf.expand_dims(self.y, -1)
            self.train_phase = tf.placeholder(tf.bool, name='train_phase')
            self.rate = tf.placeholder(tf.float32, shape=[])

            # First encoder
            # conv_1_1 shape = BATCH_SIZE * HEIGHT * WIDTH * 64
            conv_1_1 = self.conv_layer_with_bn(self.x, [3, 3, 3, 64], 
                                               self.train_phase, 'conv1_1')
            conv_1_1_shape = tf.shape(conv_1_1)
            conv_1_2 = self.conv_layer_with_bn(conv_1_1, [3, 3, 64, 64], 
                                               self.train_phase, 'conv1_2')
            pool_1, pool_1_argmax = self.pool_layer(conv_1_2)

            # Second encoder
            conv_2_1 = self.conv_layer_with_bn(pool_1, [3, 3, 64, 128], 
                                               self.train_phase, 'conv2_1')
            conv_2_2 = self.conv_layer_with_bn(conv_2_1, [3, 3, 128, 128], 
                                               self.train_phase, 'conv2_2')
            pool_2, pool_2_argmax = self.pool_layer(conv_2_2)

            # Third encoder
            conv_3_1 = self.conv_layer_with_bn(pool_2, [3, 3, 128, 256], 
                                               self.train_phase, 'conv3_1')
            conv_3_2 = self.conv_layer_with_bn(conv_3_1, [3, 3, 256, 256], 
                                               self.train_phase, 'conv3_2')
            conv_3_3 = self.conv_layer_with_bn(conv_3_2, [3, 3, 256, 256], 
                                               self.train_phase, 'conv3_3')
            pool_3, pool_3_argmax = self.pool_layer(conv_3_3)

            # Fourth encoder
            conv_4_1 = self.conv_layer_with_bn(pool_3, [3, 3, 256, 512], 
                                               self.train_phase, 'conv4_1')
            conv_4_2 = self.conv_layer_with_bn(conv_4_1, [3, 3, 512, 512], 
                                               self.train_phase, 'conv4_2')
            conv_4_3 = self.conv_layer_with_bn(conv_4_2, [3, 3, 512, 512], 
                                               self.train_phase, 'conv4_3')
            pool_4, pool_4_argmax = self.pool_layer(conv_4_3)

            # Fifth encoder
            conv_5_1 = self.conv_layer_with_bn(pool_4, [3, 3, 512, 512], 
                                               self.train_phase, 'conv5_1')
            conv_5_2 = self.conv_layer_with_bn(conv_5_1, [3, 3, 512, 512], 
                                               self.train_phase, 'conv5_2')
            conv_5_3 = self.conv_layer_with_bn(conv_5_2, [3, 3, 512, 512], 
                                               self.train_phase, 'conv5_3')
            # pool_5 shape = BATCH_SIZE * HEIGHT * WIDTH * 512
            pool_5, pool_5_argmax = self.pool_layer(conv_5_3)

            # Dynamic Filtering
            df = self.gen_dynamic_filter(pool_5, 
                                         filter_shape=[3, 3, 512, 512])
            pool_5 = self.dynamic_conv_layer(pool_5, 
                                             filter_shape=[3, 3, 512, 512], 
                                             dynamic_filter=df, name="conv_d")

            # First decoder
            unpool_5 = self.unpool(pool_5, pool_5_argmax)
            deconv_5_3 = self.conv_layer_with_bn(unpool_5, [3, 3, 512, 512], 
                                                 self.train_phase, 'deconv5_3')
            deconv_5_2 = self.conv_layer_with_bn(deconv_5_3, [3, 3, 512, 512], 
                                                 self.train_phase, 'deconv5_2')
            deconv_5_1 = self.conv_layer_with_bn(deconv_5_2, [3, 3, 512, 512], 
                                                 self.train_phase, 'deconv5_1')

            # Second decoder
            unpool_4 = self.unpool(deconv_5_1, pool_4_argmax)
            deconv_4_3 = self.conv_layer_with_bn(unpool_4, [3, 3, 512, 512], 
                                                 self.train_phase, 'deconv4_3')
            deconv_4_2 = self.conv_layer_with_bn(deconv_4_3, [3, 3, 512, 512], 
                                                 self.train_phase, 'deconv4_2')
            deconv_4_1 = self.conv_layer_with_bn(deconv_4_2, [3, 3, 512, 256], 
                                                 self.train_phase, 'deconv4_1')

            # Third decoder
            unpool_3 = self.unpool(deconv_4_1, pool_3_argmax)
            deconv_3_3 = self.conv_layer_with_bn(unpool_3, [3, 3, 256, 256], 
                                                 self.train_phase, 'deconv3_3')
            deconv_3_2 = self.conv_layer_with_bn(deconv_3_3, [3, 3, 256, 256], 
                                                 self.train_phase, 'deconv3_2')
            deconv_3_1 = self.conv_layer_with_bn(deconv_3_2, [3, 3, 256, 128], 
                                                 self.train_phase, 'deconv3_1')

            # Fourth decoder
            unpool_2 = self.unpool(deconv_3_1, pool_2_argmax)
            deconv_2_2 = self.conv_layer_with_bn(unpool_2, [3, 3, 128, 128], 
                                                 self.train_phase, 'deconv2_2')
            deconv_2_1 = self.conv_layer_with_bn(deconv_2_2, [3, 3, 128, 64], 
                                                 self.train_phase, 'deconv2_1')

            # Fifth decoder
            unpool_1 = self.unpool(deconv_2_1, pool_1_argmax)
            deconv_1_2 = self.conv_layer_with_bn(unpool_1, [3, 3, 64, 64], 
                                                 self.train_phase, 'deconv1_2')
            deconv_1_1 = self.conv_layer_with_bn(deconv_1_2, [3, 3, 64, 32], 
                                                 self.train_phase, 'deconv1_1')

            # Produce class scores
            # score_1 dimensions: BATCH_SIZE * HEIGHT * WIDTH * NUM_CLASSES
            score_1 = self.conv_layer_with_bn(deconv_1_1, 
                                              [1, 1, 32, self.num_classes], 
                                              self.train_phase, 
                                              'score_1')
            logits = tf.reshape(score_1, (-1, self.num_classes))

            # Prepare network outputs
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.reshape(expected, [-1]), 
                logits=logits, 
                name='x_entropy')
            self.loss = tf.reduce_mean(cross_entropy, name='x_entropy_mean')
            optimizer = tf.train.AdamOptimizer(self.rate)
            self.train_step = optimizer.minimize(self.loss)
            
            # Metrics
            self.prediction = tf.argmax(score_1, axis=3, name="prediction")
            self.accuracy = tf.contrib.metrics.accuracy(self.prediction, 
                                                        self.y, 
                                                        name='accuracy')
            self.mean_IoU = tf.contrib.metrics.streaming_mean_iou(self.prediction, 
                                                        self.y,
                                                        self.num_classes, 
                                                        name='mean_IoU')

            # Prepare for summaries
            """
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            self.merged = tf.summary.merge_all()
            """


    def restore_session(self):
        global_step = 0

        if not os.path.exists(self.checkpoint_directory):
            raise IOError(self.checkpoint_directory + ' does not exist.')
        else:
            path = tf.train.get_checkpoint_state(self.checkpoint_directory)
            if path is None:
                pass
            else:
                self.saver.restore(self.session, path.model_checkpoint_path)
                global_step = int(path.model_checkpoint_path.split('-')[-1])

        return global_step

    
    def train(self, num_iterations=10000, learning_rate=0.1, batch_size=5):
        current_step = self.restore_session()

        bdr = BatchDatasetReader(self.dataset_directory, 480, 320, current_step, 
                                 batch_size)

        # Begin Training
        for i in range(current_step, num_iterations):

            # One training step
            images, ground_truths = bdr.next_training_batch()
            feed_dict = {self.x: images, self.y: ground_truths, 
                         self.train_phase: 1, self.rate: learning_rate}
            print('run train step: ' + str(i))
            self.train_step.run(session=self.session, feed_dict=feed_dict)

            # Print loss every 10 iterations
            if i % 10 == 0:
                train_loss = self.session.run(self.loss, feed_dict=feed_dict)
                print("Step: %d, Train_loss:%g" % (i, train_loss))

                """
                summary, _ = self.session.run([self.merged, self.train_step], 
                                              feed_dict=feed_dict)
                self.train_writer.add_summary(summary, i)
                """

            # Run against validation dataset for 100 iterations
            if i % 100 == 0:
                images, ground_truths = bdr.next_val_batch()
                feed_dict = {self.x: images, self.y: ground_truths, 
                             self.train_phase: 1, self.rate: learning_rate}
                val_loss = self.session.run(self.loss, feed_dict=feed_dict)
                val_accuracy = self.session.run(self.accuracy, 
                                                feed_dict=feed_dict)
                val_mean_IoU, update_op = self.session.run(self.mean_IoU, 
                                                feed_dict=feed_dict)
                print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), 
                                                       val_loss))
                print("%s ---> Validation_accuracy: %g" % 
                      (datetime.datetime.now(), val_accuracy))

                """
                summary, _ = self.session.run([self.merged, self.accuracy], 
                                              feed_dict=feed_dict)
                self.val_writer.add_summary(summary, i)
                """
                self.logger.log("%s ---> Number of epochs: %g\n" % 
                                (datetime.datetime.now(), 
                                 math.floor((i * batch_size)/bdr.num_train)))
                self.logger.log("%s ---> Number of iterations: %g\n" % 
                                 (datetime.datetime.now(), i))
                self.logger.log("%s ---> Validation_loss: %g\n" % 
                                 (datetime.datetime.now(), val_loss))
                self.logger.log("%s ---> Validation_accuracy: %g\n" % 
                                 (datetime.datetime.now(), val_accuracy))
                self.logger.log_for_graphing(i, val_loss, val_accuracy, 
                                             val_mean_IoU)

                # Save the model variables
                self.saver.save(self.session, 
                                self.checkpoint_directory + 'DFSegNet', 
                                global_step = i)

            # Print outputs every 1000 iterations
            if i % 1000 == 0:
                self.test(learning_rate, self.dataset_directory)
                self.logger.graph_training_stats()


    def test(self, learning_rate, dataset_directory):

        # Get trained weights and biases
        current_step = self.restore_session()

        dr = DatasetReader(480, 320, dataset_directory)

        for i in range(min(dr.test_data_size, 10)):
            image, ground_truth = dr.next_test_pair()
            # TODO: Fix self.train_phase = 0 not working issue
            feed_dict = {self.x: [image], self.y: [ground_truth], 
                         self.train_phase: 1, self.rate: learning_rate}
            segmentation = np.squeeze(self.session.run(self.prediction, 
                                                       feed_dict=feed_dict))
            dp = DataPostprocessor()
            dp.write_out(i, image, segmentation, ground_truth, current_step)

    def test_outside_dataset(self, learning_rate=0.1):

        # Get trained weights and biases
        current_step = self.restore_session()

        dr = OutsideDataFeeder(480, 320, './datasets/amfam_dataset/')

        for i in range(min(dr.test_data_size, 10)):
            image = dr.next_test_image()
            ground_truth = np.zeros((480, 320))
            # TODO: Fix self.train_phase = 0 not working issue
            feed_dict = {self.x: [image], self.y: [ground_truth],
                         self.train_phase: 1, self.rate: learning_rate}
            segmentation = np.squeeze(self.session.run(self.prediction, 
                                                       feed_dict=feed_dict))
            dp = DataPostprocessor()
            dp.write_out(i, image, segmentation, current_step)

    def test_individual_file(self, input_file, learning_rate=0.1):
        current_step = self.restore_session()
        dpp = DataPreprocessor()
        image = dpp.load_image(input_file, 480, 320)
        ground_truth = np.zeros((480, 320))
        feed_dict = {self.x: [image], self.y: [ground_truth], 
                     self.train_phase: 1, self.rate: learning_rate}
        segmentation = np.squeeze(self.session.run(self.prediction, 
                                  feed_dict=feed_dict))
        dp = DataPostprocessor()
        dp.write_out(0, image, segmentation, current_step)

    def test_sequence(self, learning_rate=0.1):
        # Get trained weights and biases
        current_step = self.restore_session()

        dr = CustomTestDataFeeder(480, 320, './datasets/UnrealNeighborhood-11Class-StreetPrimary-0.15/Sequence-20/')

        for i in range(min(dr.dataset_size, 20)):
            image, ground_truth = dr.next_test_image()
            # TODO: Fix self.train_phase = 0 not working issue
            feed_dict = {self.x: [image], self.y: [ground_truth],
                         self.train_phase: 1, self.rate: learning_rate}
            segmentation = np.squeeze(self.session.run(self.prediction, 
                                                       feed_dict=feed_dict))
            test_accuracy = self.session.run(self.accuracy, 
                                                feed_dict=feed_dict)
            self.logger.log_for_test_graphing(i, test_accuracy)
      
            dp = DataPostprocessor()
            dp.write_out(i, image, segmentation, current_step)

        self.logger.graph_test_stats()


