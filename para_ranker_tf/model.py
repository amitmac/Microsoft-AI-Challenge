import tensorflow as tf
import numpy as np
import logging
import math
import pickle

"""
Ranking Paragraphs for Improving Answer Recall in Open-Domain Question Answering 
https://arxiv.org/pdf/1810.00494.pdf
"""

class PassageRanking:
    def __init__(self, config, sess):
        self.batch_size = config.batch_size
        self.hdim = config.hdim
        self.input_dim = config.word_vec_dim
        self.init_std = config.init_std
        self.vocab_size = config.vocab_size
        self.ques_max_len = config.ques_max_len
        self.para_max_len = config.para_max_len
        self.num_classes = config.num_classes
        self.num_epochs = config.num_epochs
        self.train_data_size = config.train_data_size
        self.valid_data_size = config.valid_data_size
        self.pad_id = config.pad_id
        self.buffer_size = config.buffer_size

        self.set_system()

    def set_system(self):
        self.ques = tf.placeholder(tf.int32, shape=(None, None), name='ques')
        self.para = tf.placeholder(tf.int32, shape=(None, None), name='para')
        self.label = tf.placeholder(tf.int32, shape=(None), name='label')

        self.embeddings = tf.placeholder(shape=(self.vocab_size, self.input_dim), dtype=tf.float32)
        self.ques_sentence_lengths = tf.placeholder(shape=(None), dtype=tf.int32)
        self.para_sentence_lengths = tf.placeholder(shape=(None), dtype=tf.int32)

        self.ques_embedded = tf.nn.embedding_lookup(self.embeddings, self.ques)
        self.para_embedded = tf.nn.embedding_lookup(self.embeddings, self.para)

        ques_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(self.hdim)
        ques_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(self.hdim)

        para_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(self.hdim)
        para_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(self.hdim)
        
        with tf.variable_scope("question"):
            (_, _), ques_output_states = tf.nn.bidirectional_dynamic_rnn(ques_cell_fw, ques_cell_bw, 
                                                                    self.ques_embedded, 
                                                                    sequence_length=self.ques_sentence_lengths,
                                                                    dtype=tf.float32)
        with tf.variable_scope("paragraph"):                                                                                
            (_, _), para_output_states = tf.nn.bidirectional_dynamic_rnn(para_cell_fw, para_cell_bw, 
                                                                    self.para_embedded,
                                                                    sequence_length=self.para_sentence_lengths,
                                                                    dtype=tf.float32)   
        
        ques_output = tf.concat((ques_output_states[0].h, ques_output_states[1].h), 1)
        # Add some variability to the ques representation
        ques_output = tf.layers.dense(ques_output, self.hdim*2, activation=tf.nn.tanh)

        para_output = tf.concat((para_output_states[0].h, para_output_states[1].h), 1)
        
        logits = self.similarity(ques_output, para_output)
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label, logits=logits)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train_op = optimizer.minimize(self.loss)
        
        predictions = tf.cast(tf.argmax(logits, axis=1), tf.int32)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, self.label), tf.float32))
        self.f1_score = tf.contrib.metrics.f1_score(self.label, predictions)

    def similarity(self, ques, para):
        #qp = tf.concat([ques, para], 1)
        qp = tf.matmul(ques, para)
        qp_norm = tf.layers.batch_normalization(qp)
        qp_act = tf.nn.elu(qp_norm)
        output = tf.layers.dense(qp_act, self.num_classes, activation=tf.nn.softmax)
        return output

    def parse_func(self, line):
        string_vals = tf.string_split([line]).values
        return tf.string_to_number(string_vals, tf.int32), tf.size(string_vals)

    def create_file_iterator(self, train_data, valid_data, batch_size):
        padded_shapes = ((tf.TensorShape([None]), tf.TensorShape([])), 
                         (tf.TensorShape([None]), tf.TensorShape([])),
                         (tf.TensorShape([None]), tf.TensorShape([])))
        padding_values = ((0,0), (0,0), (0,0))
        
        train_ques_para_label_dataset = tf.data.Dataset.zip(train_data)
        train_ques_para_label_dataset = train_ques_para_label_dataset.padded_batch(batch_size, 
                                                                                   padded_shapes=padded_shapes, 
                                                                                   padding_values=padding_values)
        train_ques_para_label_iterator = train_ques_para_label_dataset.make_initializable_iterator()
        train_ques_para_label_init_op  = train_ques_para_label_iterator.initializer

        valid_ques_para_label_dataset = tf.data.Dataset.zip(valid_data)
        valid_ques_para_label_dataset = valid_ques_para_label_dataset.padded_batch(batch_size, 
                                                                                   padded_shapes=padded_shapes, 
                                                                                   padding_values=padding_values)
        valid_ques_para_label_iterator = valid_ques_para_label_dataset.make_initializable_iterator()
        valid_ques_para_label_init_op  = valid_ques_para_label_iterator.initializer

        train_next_element = train_ques_para_label_iterator.get_next()
        valid_next_element = valid_ques_para_label_iterator.get_next()

        return (train_next_element, train_ques_para_label_init_op, valid_next_element, valid_ques_para_label_init_op)


    def train(self, sess, train_file, valid_file, embeddings):
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        logging.info("Number of params: %d" % (num_params))        

        
        train_ques_file, train_para_file, train_label_file = train_file
        valid_ques_file, valid_para_file, valid_label_file = valid_file

        train_ques = tf.data.TextLineDataset(train_ques_file)
        train_ques = train_ques.map(map_func=self.parse_func)
        
        train_para = tf.data.TextLineDataset(train_para_file)
        train_para = train_para.map(map_func=self.parse_func)

        train_label = tf.data.TextLineDataset(train_label_file)
        train_label = train_label.map(map_func=self.parse_func)

        valid_ques = tf.data.TextLineDataset(valid_ques_file)
        valid_ques = valid_ques.map(map_func=self.parse_func)

        valid_para = tf.data.TextLineDataset(valid_para_file)
        valid_para = valid_para.map(map_func=self.parse_func)
        
        valid_label = tf.data.TextLineDataset(valid_label_file)
        valid_label = valid_label.map(map_func=self.parse_func)

        train_data = (train_ques, train_para, train_label)
        valid_data = (valid_ques, valid_para, valid_label)
        
        training_loss, validation_loss = [], []

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        for i in range(self.num_epochs):
            batch_size = self.batch_size + i*2

            iterators = self.create_file_iterator(train_data, valid_data, batch_size)
            train_next_element, train_data_init_op, valid_next_element, valid_data_init_op = iterators

            sess.run(train_data_init_op) 
    
            train_num_batches = int(math.ceil(self.train_data_size / batch_size))
            valid_num_batches = int(math.ceil(self.valid_data_size / batch_size))

            train_loss, train_accuracy, train_f1_score = 0, 0, 0
            for j in range(train_num_batches):
                ((q_next_batch, q_seq_len), (p_next_batch, p_seq_len), (label_next_batch, _)) = sess.run(train_next_element)
                
                label_next_batch = np.reshape(label_next_batch, [-1])
                
                input_feed = {
                    self.ques: q_next_batch,
                    self.para: p_next_batch,
                    self.embeddings: embeddings,
                    self.label: label_next_batch,
                    self.ques_sentence_lengths: q_seq_len,
                    self.para_sentence_lengths: p_seq_len
                }

                output_feed = [self.train_op, self.loss, self.accuracy, self.f1_score]

                outputs = sess.run(output_feed, input_feed)
                print((outputs[2]))
                print((outputs[3][0]))
                train_loss += np.mean(outputs[1])
                train_accuracy += outputs[2]
                train_f1_score += outputs[3][0]
            
            training_loss.append(train_loss/train_num_batches)

            logging.info("Training average loss after {0} epochs, {1}".format(i+1, train_loss/train_num_batches))
            logging.info("Training accuracy after {0} epochs, {1}".format(i+1, train_accuracy/train_num_batches))
            logging.info("Training F1-Score after {0} epochs, {1}".format(i+1, train_f1_score/train_num_batches))

            save_path = saver.save(sess, "/workspace/ms-ai-challenge/paragraph-ranker-tf/model_pr_" + str(i) + ".ckpt")
            logging.info("Model saved in path: %s" % save_path)

            sess.run(valid_data_init_op)
            
            valid_loss, valid_accuracy, valid_f1_score = 0, 0, 0
            for j in range(valid_num_batches):
                ((q_next_batch, q_seq_len), (p_next_batch, p_seq_len), (label_next_batch, _)) = sess.run(valid_next_element)
                
                label_next_batch = np.reshape(label_next_batch, [-1])
                
                input_feed = {
                    self.ques: q_next_batch,
                    self.para: p_next_batch,
                    self.embeddings: embeddings,
                    self.label: label_next_batch,
                    self.ques_sentence_lengths: q_seq_len,
                    self.para_sentence_lengths: p_seq_len
                }

                output_feed = [self.loss, self.accuracy, self.f1_score]

                outputs = sess.run(output_feed, input_feed)
                valid_loss += np.mean(outputs[0])
                valid_accuracy += outputs[1]
                valid_f1_score += outputs[2][0]
            
            validation_loss.append(valid_loss/valid_num_batches)
            
            logging.info("Validation accuracy after {0} epochs, {1}".format(i+1,valid_accuracy/valid_num_batches))
            logging.info("Validation F1-Score after {0} epochs, {1}".format(i+1,valid_f1_score/valid_num_batches))

        with open('parrot.pkl', 'wb') as f:
            loss = zip(training_loss, validation_loss)
            pickle.dump(loss, f)