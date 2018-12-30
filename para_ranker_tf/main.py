import os
import json
import argparse

import tensorflow as tf
import numpy as np

from os.path import join as pjoin

import logging
from model import PassageRanking
from data_ops import PAD_ID

flags = tf.app.flags

flags.DEFINE_integer("word_vec_dim", 50, "word vector dimension [300]")
flags.DEFINE_integer("hdim", 50, "hidden layer size of neural network [200]")
flags.DEFINE_integer("batch_size", 2, "batch size [64]")
flags.DEFINE_integer("vocab_size", 115613, "Vocabulary size [100000]")
flags.DEFINE_integer("context_size", 5, "Number of context words")
flags.DEFINE_integer("ques_max_len", 15, "Maximum length of question")
flags.DEFINE_integer("para_max_len", 60, "Maximum length of paragraph")
flags.DEFINE_integer("num_classes", 2, "Number of classes")
flags.DEFINE_integer("num_epochs", 5, "Number of epochs")
flags.DEFINE_integer("pad_id", PAD_ID, "padding id")
flags.DEFINE_integer("buffer_size", 20, "buffer size for shuffle")
flags.DEFINE_integer("train_data_size", 10, "training data size")
flags.DEFINE_integer("valid_data_size", 10, "validation data size")
flags.DEFINE_float("init_lr", 0.01, "initial learning rate [0.01]")
flags.DEFINE_float("init_std", 0.05, "initial standard deviation [0.05]")

FLAGS = flags.FLAGS

with tf.Session() as sess:
    train_file = ["../data/train.ids.question","../data/train.ids.passage","../data/train.labels"]
    valid_file = ["../data/valid.ids.question","../data/valid.ids.passage","../data/valid.labels"]
    embeddings = np.load("../data/glove.trimmed.50.npz")['glove']
    model = PassageRanking(FLAGS, sess)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    model.train(sess, train_file, valid_file, embeddings)