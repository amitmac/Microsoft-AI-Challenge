import tensorflow as tf
import numpy as np
import logging
import math
import pickle

"""
Attention and Negative Sampling
"""

logging.basicConfig(filename='passage_ranking_attention_training_logs.log',level=logging.INFO)

class PassageRanking:
    def __init__(self, config, sess):
        self.batch_size = config.batch_size
        self.hdim = config.hdim
        self.input_dim = config.word_vec_dim
        self.init_std = config.init_std
        self.init_lr = config.init_lr
        self.vocab_size = config.vocab_size
        self.ques_max_len = config.ques_max_len
        self.para_max_len = config.para_max_len
        self.num_classes = config.num_classes
        self.num_epochs = config.num_epochs
        self.train_data_size = config.train_data_size
        self.valid_data_size = config.valid_data_size
        self.pad_id = config.pad_id
        self.buffer_size = config.buffer_size
        self.valid_batch_size = config.valid_batch_size
        self.num_negative_samples = config.num_negative_samples
        
        self.set_system()

    def set_placeholders(self):
        self.ques = tf.placeholder(tf.int32, shape=(None, None), name='ques')
        self.para = tf.placeholder(tf.int32, shape=(None, None), name='para')
        
        # neg_ques and neg_para size would be (batch_size*negative_samples) x (sequence_length)
        # Its important to send the right negative samples
        self.neg_ques = tf.placeholder(tf.int32, shape=(None, None), name='neg_ques')
        self.neg_para = tf.placeholder(tf.int32, shape=(None, None), name='neg_para')
        
        self.embeddings = tf.placeholder(shape=(self.vocab_size, self.input_dim), dtype=tf.float32, name='embeddings')
        
        self.ques_sentence_lengths = tf.placeholder(shape=(None), dtype=tf.int32, name='ques_sent_lens')
        self.para_sentence_lengths = tf.placeholder(shape=(None), dtype=tf.int32, name='para_sent_lens')

        self.neg_ques_sentence_lengths = tf.placeholder(shape=(None), dtype=tf.int32, name='neg_ques_sent_lens')
        self.neg_para_sentence_lengths = tf.placeholder(shape=(None), dtype=tf.int32, name='neg_para_sent_lens')
        
    def set_embedded_inputs(self):
        self.ques_embedded = tf.nn.embedding_lookup(self.embeddings, self.ques)
        self.para_embedded = tf.nn.embedding_lookup(self.embeddings, self.para)

        self.neg_ques_embedded = tf.nn.embedding_lookup(self.embeddings, self.neg_ques)
        self.neg_para_embedded = tf.nn.embedding_lookup(self.embeddings, self.neg_para)
        
    def set_system(self):
        self.set_placeholders()
        self.set_embedded_inputs()
        
        # Set up LSTM cells
        ques_cell_fw = tf.nn.rnn_cell.LSTMCell(self.hdim, name="ques_fw")
        ques_cell_bw = tf.nn.rnn_cell.LSTMCell(self.hdim, name="ques_bw")

        para_cell_fw = tf.nn.rnn_cell.LSTMCell(self.hdim, name="para_fw")
        para_cell_bw = tf.nn.rnn_cell.LSTMCell(self.hdim, name="para_bw")
        
        with tf.variable_scope("question"):
            (ques_fw_hidden, ques_bw_hidden), ques_output_states = tf.nn.bidirectional_dynamic_rnn(ques_cell_fw, ques_cell_bw, 
                                                                    self.ques_embedded, 
                                                                    sequence_length=self.ques_sentence_lengths,
                                                                    dtype=tf.float32)
            
            ques_cell_fw = tf.nn.rnn_cell.LSTMCell(self.hdim, name="ques_fw", reuse=True)
            ques_cell_bw = tf.nn.rnn_cell.LSTMCell(self.hdim, name="ques_bw", reuse=True)
            
            (neg_ques_fw_hidden, neg_ques_bw_hidden), neg_ques_output_states = tf.nn.bidirectional_dynamic_rnn(ques_cell_fw, 
                                                                    ques_cell_bw, 
                                                                    self.neg_ques_embedded, 
                                                                    sequence_length=self.neg_ques_sentence_lengths,
                                                                    dtype=tf.float32)
        with tf.variable_scope("paragraph"):                                                                                
            (para_fw_hidden, para_bw_hidden), para_output_states = tf.nn.bidirectional_dynamic_rnn(para_cell_fw, para_cell_bw, 
                                                                    self.para_embedded,
                                                                    sequence_length=self.para_sentence_lengths,
                                                                    dtype=tf.float32)   
            
            para_cell_fw = tf.nn.rnn_cell.LSTMCell(self.hdim, name="para_fw", reuse=True)
            para_cell_bw = tf.nn.rnn_cell.LSTMCell(self.hdim, name="para_bw", reuse=True)
            
            (neg_para_fw_hidden, neg_para_bw_hidden), neg_para_output_states = tf.nn.bidirectional_dynamic_rnn(para_cell_fw, para_cell_bw, 
                                                                    self.neg_para_embedded,
                                                                    sequence_length=self.neg_para_sentence_lengths,
                                                                    dtype=tf.float32)
            
        
        para_hidden = tf.concat((para_fw_hidden, para_bw_hidden), 2) # B x T x 2D
        neg_para_hidden = tf.concat((neg_para_fw_hidden, neg_para_bw_hidden), 2) # (B*NEG_SAMPLE) x T x 2D
        
        ques_output = tf.concat((ques_output_states[0].h, ques_output_states[1].h), 1) # B x 2D
        neg_ques_output = tf.concat((neg_ques_output_states[0].h, neg_ques_output_states[1].h), 1) # (B*NEG_SAMPLE) x 2D
        
        para_output = tf.concat((para_output_states[0].h, para_output_states[1].h), 1) # B x 2D
        neg_para_output = tf.concat((neg_para_output_states[0].h, neg_para_output_states[1].h), 1) # (B*NEG_SAMPLE) x 2D
        
        # Learn final representations for question and negative sampled questions
        ques_rep = tf.layers.dense(ques_output, self.hdim*2, activation=tf.nn.tanh, name="final_embedding")
        ques_rep= tf.reshape(ques_rep, [-1, self.hdim*2, 1]) # B x 2D x 1
        
        neg_ques_rep = tf.layers.dense(neg_ques_output, self.hdim*2, activation=tf.nn.tanh, name="final_embedding", reuse=True)
        neg_ques_rep = tf.reshape(neg_ques_rep, [-1, self.hdim*2, 1]) # (B*NEG_SAMPLE) x 2D x 1
        
        # Attention Pooling for paragraphs
        context_vector = tf.Variable(shape=[self.hdim*2, 1], initializer=tf.initializers.random_normal, trainable=True, name="context_vector")
        
        para_alignment_vectors = tf.tensordot(para_hidden, context_vector, axes=[[2]]) # B x T x 1
        neg_para_alignment_vectors = tf.tensordot(neg_para_hidden, context_vector, axes=[[2]]) # B*NEG_SAMPLE) x T x 1
        
        para_hidden_softmax = tf.nn.softmax(para_alignment_vectors, axis=1) # B x T x 1
        neg_para_hidden_softmax = tf.nn.softmax(neg_para_alignment_vectors, axis=1) # B*NEG_SAMPLE) x T x 1
        
        W_attention = tf.Variable(shape=[self.hdim*2, self.hdim*2], initializer=tf.initializers.random_normal, trainable=True, name="att_weight")
        
        max_time_steps = para_hidden.shape[1]
        
        attention_vectors = tf.reshape(tf.matmul(tf.reshape(para_hidden, [-1, self.hdim*2]), W_attention),[-1, max_time_steps, self.hdim*2])
        
        para_rep = tf.reduce_mean(tf.multiply(para_hidden_softmax, attention_vectors), 1) # B x 2D x 1
        
        neg_attention_vectors = tf.reshape(tf.matmul(tf.reshape(neg_para_hidden, [-1, self.hdim*2]), W_attention),[-1, max_time_steps, self.hdim*2])
        
        neg_para_rep = tf.reduce_mean(tf.multiply(para_hidden_softmax, attention_vectors), 1) # (B*NEG_SAMPLE) x 2D x 1
        
        ques_para_score = tf.tensordot(para_rep, ques_rep, axes=[[1],[1]]) # B x 1 x 1
        ques_para_log_loss = tf.log_sigmoid(ques_para_score)
        
        neg_ques_para_score = tf.tensordot(neg_para_rep, neg_ques_rep, axes=[[2],[2]]) # (B*NEG_SAMPLE) x 1 x 1
        neg_ques_para_loss = tf.log_sigmoid(neg_ques_para_score)
        
        neg_ques_para_loss = tf.reduce_sum(tf.reshape(neg_ques_para_loss, [-1, self.num_negative_samples, 1]), 1)
        
        self.loss = ques_para_log_loss + neg_ques_para_loss

        optimizer = tf.train.AdamOptimizer(learning_rate=self.init_lr, epsilon=0.005)
        #optimizer = tf.train.AdagradOptimizer(self.init_lr)
        #optimizer = tf.train.RMSPropOptimizer(self.init_lr)
        
        self.train_op = optimizer.minimize(self.loss)
        
        predictions = tf.cast(tf.argmax(logits, axis=1), tf.int32)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, self.label), tf.float32))
        self.precision = tf.metrics.precision(self.label, predictions)
        self.recall = tf.metrics.recall(self.label, predictions)

    def similarity(self, ques, para):
        qp = tf.concat([ques, para], 1)
        qp_act = tf.layers.dense(qp, 20, activation=tf.nn.leaky_relu)
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
        train_ques_para_label_dataset = train_ques_para_label_dataset.shuffle(buffer_size=1000)
        train_ques_para_label_dataset = train_ques_para_label_dataset.padded_batch(batch_size, 
                                                                                   padded_shapes=padded_shapes, 
                                                                                   padding_values=padding_values)
        #train_ques_para_label_dataset = train_ques_para_label_dataset.prefetch(1)
        train_ques_para_label_iterator = train_ques_para_label_dataset.make_initializable_iterator()
        train_ques_para_label_init_op  = train_ques_para_label_iterator.initializer

        valid_ques_para_label_dataset = tf.data.Dataset.zip(valid_data)
        valid_ques_para_label_dataset = valid_ques_para_label_dataset.padded_batch(self.valid_batch_size, 
                                                                                   padded_shapes=padded_shapes, 
                                                                                   padding_values=padding_values)
        #valid_ques_para_label_dataset = valid_ques_para_label_dataset.prefetch(1)
        valid_ques_para_label_iterator = valid_ques_para_label_dataset.make_initializable_iterator()
        valid_ques_para_label_init_op  = valid_ques_para_label_iterator.initializer

        train_next_element = train_ques_para_label_iterator.get_next()
        valid_next_element = valid_ques_para_label_iterator.get_next()

        return (train_next_element, train_ques_para_label_init_op, valid_next_element, valid_ques_para_label_init_op)


    def train(self, sess, train_file, valid_file, embeddings):
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        logging.info("Number of params: %d" % (num_params))        
        logging.info("#################################################################################")
        
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
            batch_size = self.batch_size

            iterators = self.create_file_iterator(train_data, valid_data, batch_size)
            train_next_element, train_data_init_op, valid_next_element, valid_data_init_op = iterators

            sess.run(train_data_init_op) 
    
            train_num_batches = int(math.ceil(self.train_data_size / batch_size))
            valid_num_batches = int(math.ceil(self.valid_data_size / self.valid_batch_size))

            train_loss, train_accuracy, train_precision, train_recall = 0, 0, 0, 0
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

                output_feed = [self.train_op, self.loss, self.accuracy, self.precision, self.recall]

                outputs = sess.run(output_feed, input_feed)
                train_loss += np.mean(outputs[1])
                train_accuracy += outputs[2]
                train_precision += outputs[3][0]
                train_recall += outputs[4][0]
                
                if j in [int(train_num_batches/3), 2*int(train_num_batches/3)]:
                    training_loss.append(train_loss/j)
                    precision = train_precision/j
                    recall = train_recall/j
                    logging.info("Training average loss after {0} epochs, {1}".format(i+1, train_loss/j))
                    logging.info("Training accuracy after {0} epochs, {1}".format(i+1, train_accuracy/j))
                    logging.info("Training Precision after {0} epochs, {1}".format(i+1, precision))
                    logging.info("Training Recall after {0} epochs, {1}".format(i+1, recall))
                    logging.info("Training F1-score after {0} epochs, {1}".format(i+1, (2*recall*precision)/(recall+precision)))

                    logging.info("---------------------------------------------------------------------------------------")
                    
            
            training_loss.append(train_loss/train_num_batches)

            precision = train_precision/train_num_batches
            recall = train_recall/train_num_batches
            logging.info("Training average loss after {0} epochs, {1}".format(i+1, train_loss/train_num_batches))
            logging.info("Training accuracy after {0} epochs, {1}".format(i+1, train_accuracy/train_num_batches))
            logging.info("Training Precision after {0} epochs, {1}".format(i+1, precision))
            logging.info("Training Recall after {0} epochs, {1}".format(i+1, recall))
            logging.info("Training F1-score after {0} epochs, {1}".format(i+1, (2*recall*precision)/(recall+precision)))
            logging.info("---------------------------------------------------------------------------------------")
            save_path = saver.save(sess, "/home/amitmac4/ms-ai-challenge/paragraph_ranker_tf/models/model_attpr_" + str(i) + ".ckpt")
            logging.info("Model saved in path: %s" % save_path)

            sess.run(valid_data_init_op)
            
            valid_loss, valid_accuracy, valid_precision, valid_recall = 0, 0, 0, 0
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

                output_feed = [self.loss, self.accuracy, self.precision, self.recall]

                outputs = sess.run(output_feed, input_feed)
                valid_loss += np.mean(outputs[0])
                valid_accuracy += outputs[1]
                valid_precision += outputs[2][0]
                valid_recall += outputs[3][0]
                if j in [int(valid_num_batches/3), 2*int(valid_num_batches/3)]:
                    validation_loss.append(valid_loss/j)
                    precision = valid_precision/j
                    recall = valid_recall/j
                    logging.info("Validation average loss after {0} epochs, {1}".format(i+1, valid_loss/j))
                    logging.info("Validation accuracy after {0} epochs, {1}".format(i+1, valid_accuracy/j))
                    logging.info("Validation Precision after {0} epochs, {1}".format(i+1, precision))
                    logging.info("Validation Recall after {0} epochs, {1}".format(i+1, recall))
                    logging.info("Validation F1-score after {0} epochs, {1}".format(i+1, (2*recall*precision)/(recall+precision)))

                    logging.info("---------------------------------------------------------------------------------------")

            
            validation_loss.append(valid_loss/valid_num_batches)
            
            precision = valid_precision/valid_num_batches
            recall = valid_recall/valid_num_batches
            logging.info("Validation accuracy after {0} epochs, {1}".format(i+1,valid_accuracy/valid_num_batches))
            logging.info("Validation Precision after {0} epochs, {1}".format(i+1,precision))
            logging.info("Validation Recall after {0} epochs, {1}".format(i+1,recall))
            logging.info("Validation F1-score after {0} epochs, {1}".format(i+1, (2*recall*precision)/(recall+precision)))
            logging.info("###############################################################################################")
        with open('parrot.pkl', 'wb') as f:
            loss = zip(training_loss, validation_loss)
            pickle.dump(loss, f)
