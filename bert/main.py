import tensorflow as tf
from tensorflow.contrib.tpu.python.tpu.tpu_estimator import TPUEstimator, TPUEstimatorSpec

import data_ops
import modeling
import optimization
import tokenization
import logging
import os
import numpy as np

logging.basicConfig(filename="data/training_logs.log", level=logging.INFO)

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", "data/","data storage")
flags.DEFINE_string("bert_init_checkpoint", "pretrained_models/bert_model.ckpt", "path to pretrained model")
flags.DEFINE_string("bert_config_file", "pretrained_models/bert_config.json", "path to model config file")
flags.DEFINE_string("vocab_file", "pretrained_models/vocab.txt", "path to vocabulary tokens file")
flags.DEFINE_string("output_dir", "models/", "save model path")
flags.DEFINE_bool("do_train", True, "training the model?")
flags.DEFINE_bool("do_eval", True, "evaluate the model?")
flags.DEFINE_bool("do_predict", True, "predict the model?")
flags.DEFINE_bool("use_tpu", False, "True if using TPUs")
flags.DEFINE_integer("max_sequence_length", 176, "maximum sequence length of text")
flags.DEFINE_integer("max_ques_length", 16, "maximum sequence length of text")
flags.DEFINE_integer("save_checkpoints_steps", 5000, "How often to save the model checkpoint.")
flags.DEFINE_integer("iterations_per_loop", 1000, "How many steps to make in each estimator call.")
flags.DEFINE_integer("train_batch_size", 32, "training batch size")
flags.DEFINE_integer("eval_batch_size", 8, "validation batch size")
flags.DEFINE_integer("predict_batch_size", 8, "prediction batch size")
flags.DEFINE_integer("num_labels", 6, "number of classes")
flags.DEFINE_float("learning_rate", 2e-5, "The initial learning rate for Adam")
flags.DEFINE_float("num_epochs", 20.0, "number of epochs")
flags.DEFINE_float("warmup_proportion", 0.1, "use warmup for optimization")
tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")
flags.DEFINE_integer("num_tpu_cores", 8, "Only used if `use_tpu` is True. Total number of TPU cores to use.")

def create_simple_model(bert_config, is_training, input_ids, input_mask, segment_ids, labels,
                 num_labels, use_one_hot_embeddings):
    model = modeling.BertModel(config=bert_config,
                               is_training=is_training,
                               input_ids=input_ids,
                               input_mask=input_mask,
                               token_type_ids=segment_ids,
                               use_one_hot_embeddings=use_one_hot_embeddings)
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable("output_weights", [num_labels, hidden_size],
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable("output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_data_loss = -tf.reduce_sum(one_hot_labels*log_probs, axis=-1)
        loss = tf.reduce_mean(per_data_loss)

    return loss, per_data_loss, logits, probabilities

def create_attention_model(bert_config, is_training, input_ids, input_mask, segment_ids, labels,
                 num_labels, use_one_hot_embeddings):
    model = modeling.BertModel(config=bert_config,
                               is_training=is_training,
                               input_ids=input_ids,
                               input_mask=input_mask,
                               token_type_ids=segment_ids,
                               use_one_hot_embeddings=use_one_hot_embeddings)

    all_encoder_layer = model.get_all_encoder_layers()
    
    logging.info("all encoder layer %s, each shape %s" % (len(all_encoder_layer), all_encoder_layer[0].shape))
    
    outputs = tf.concat([all_encoder_layer[9], 
                         all_encoder_layer[10], 
                         all_encoder_layer[11]], axis=2)   # B x T x 2H
    
    hidden_size = outputs.shape[-1].value
    para_num_tokens = FLAGS.max_sequence_length - FLAGS.max_ques_length
    
    #-----------------------------------Transformer Layer------------------------------------------#
    attention_mask = modeling.create_attention_mask_from_input_mask(input_ids, input_mask)

    transformer_layers = modeling.transformer_model(outputs, 
                               attention_mask=attention_mask,
                               hidden_size=hidden_size,
                               num_hidden_layers=1,
                               num_attention_heads=12,
                               intermediate_size=2048,
                               do_return_all_layers=True)

    sequence_output = transformer_layers[-1]

    first_token_tensor = tf.squeeze(sequence_output[:, 0:1, :], axis=1)
    cls_output_2 = tf.layers.dense(
            first_token_tensor,
            hidden_size,
            activation=tf.tanh,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

    ques_outputs = sequence_output[:,:FLAGS.max_ques_length,:]
    para_outputs = sequence_output[:,FLAGS.max_ques_length:,:]

    #-----------------------------------Paragraph Attention-----------------------------------------#
    attention_weights_para = tf.get_variable("att_weights_para", [hidden_size, para_num_tokens],
                                        initializer=tf.truncated_normal_initializer(stddev=0.02))

    para_ques_context = tf.reshape(tf.matmul(tf.reshape(ques_outputs, [-1, hidden_size]), attention_weights_para), 
                                   [-1, FLAGS.max_ques_length, para_num_tokens]) # B x QT x PT

    para_attention_vector = tf.nn.softmax(tf.reduce_mean(para_ques_context, axis=1), axis=1) # B x PT
    para_attention_vector = tf.reshape(para_attention_vector, [-1, para_num_tokens, 1])

    para_output_layer = tf.multiply(para_attention_vector, para_outputs) # B x PT x 2H

    para_mask = tf.reshape(tf.cast(input_mask[:,FLAGS.max_ques_length:], tf.float32), 
                           [-1, para_num_tokens, 1])
    para_output_layer = tf.reduce_sum(tf.multiply(para_output_layer, para_mask), axis=1) # B x 2H
    
    #-----------------------------------Question Attention-----------------------------------------#
    attention_weights_ques = tf.get_variable("att_weights_ques", [hidden_size, FLAGS.max_ques_length],
                                        initializer=tf.truncated_normal_initializer(stddev=0.02))

    ques_para_context = tf.reshape(tf.matmul(tf.reshape(para_outputs, [-1, hidden_size]), attention_weights_ques), 
                                   [-1, para_num_tokens, FLAGS.max_ques_length]) # B x PT x QT

    ques_attention_vector = tf.nn.softmax(tf.reduce_mean(ques_para_context, axis=1), axis=1) # B x QT
    ques_attention_vector = tf.reshape(ques_attention_vector, [-1, FLAGS.max_ques_length, 1])

    ques_output_layer = tf.multiply(ques_attention_vector, ques_outputs) # B x QT x 2H

    ques_mask = tf.reshape(tf.cast(input_mask[:,:FLAGS.max_ques_length], tf.float32), 
                           [-1, FLAGS.max_ques_length, 1])
    ques_output_layer = tf.reduce_sum(tf.multiply(ques_output_layer, ques_mask), axis=1) # B x 2H
    
    #----------------------------------------------------------------------------------------------#    

    ques_para_output_layer = tf.concat([para_output_layer, ques_output_layer], axis=1)

    ques_para_rep = tf.layers.dense(ques_para_output_layer, hidden_size/2, activation=modeling.gelu,
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_layer = tf.concat([cls_output_2, ques_para_rep], axis=1)

    output_weights = tf.get_variable("output_weights", [num_labels, hidden_size + hidden_size/2],
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable("output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_data_loss = -tf.reduce_sum(one_hot_labels*log_probs, axis=-1)
        loss = tf.reduce_mean(per_data_loss)

    return loss, per_data_loss, logits, probabilities

def cnn_network(coembedding):
    logging.info("coembedding shape %s" % (coembedding.shape))
    convA1 = tf.layers.conv2d(coembedding, filters=7, kernel_size=[5, 13], activation=modeling.gelu)
    poolA1 = tf.layers.average_pooling2d(convA1, pool_size=[2, 5], strides=2)
    logging.info("pool a1 shape %s" % (poolA1.shape))
    convA2 = tf.layers.conv2d(poolA1, filters=5, kernel_size=[3, 3], activation=modeling.gelu)
    poolA2 = tf.layers.average_pooling2d(convA2, pool_size=[2, 2], strides=(1, 2))
    logging.info("pool a2 shape %s" % (poolA2.shape))
    denseA = tf.layers.dense(poolA2, 10, activation=modeling.gelu)
    logging.info("denseA shape %s" % (denseA.shape))
    return denseA

def create_cnn_attention_model(bert_config, is_training, input_ids, input_mask, segment_ids, labels,
                 num_labels, use_one_hot_embeddings):
    model = modeling.BertModel(config=bert_config,
                               is_training=is_training,
                               input_ids=input_ids,
                               input_mask=input_mask,
                               token_type_ids=segment_ids,
                               use_one_hot_embeddings=use_one_hot_embeddings)

    all_encoder_layer = model.get_all_encoder_layers()
    
    logging.info("all encoder layer %s, each shape %s" % (len(all_encoder_layer), all_encoder_layer[0].shape))
    
    ques_outputs = tf.concat([all_encoder_layer[8][:,:FLAGS.max_ques_length,:],
                              all_encoder_layer[9][:,:FLAGS.max_ques_length,:],
                              all_encoder_layer[10][:,:FLAGS.max_ques_length,:],
                              all_encoder_layer[11][:,:FLAGS.max_ques_length,:]], axis=2)   # B x QT x 4H
    
    para_outputs = tf.concat([all_encoder_layer[8][:,FLAGS.max_ques_length:,:],
                              all_encoder_layer[9][:,FLAGS.max_ques_length:,:],
                              all_encoder_layer[10][:,FLAGS.max_ques_length:,:],
                              all_encoder_layer[11][:,FLAGS.max_ques_length:,:]], axis=2)   # B x PT x 4H
    
    hidden_size = ques_outputs.shape[-1].value

    para_num_tokens = FLAGS.max_sequence_length - FLAGS.max_ques_length

    weights_from_ques = tf.layers.dense(ques_outputs, para_num_tokens, activation=modeling.gelu,
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))


    ques_para_coembedding = tf.matmul(ques_outputs, para_outputs, transpose_b=True) # B x QT x PT

    ques_para_coembedding = tf.layers.batch_normalization(ques_para_coembedding) # B x QT x PT

    para_mask = tf.reshape(tf.cast(input_mask[:,FLAGS.max_ques_length:], tf.float32), 
                           [-1, 1, para_num_tokens])
    ques_mask = tf.reshape(tf.cast(input_mask[:,:FLAGS.max_ques_length], tf.float32), 
                           [-1, FLAGS.max_ques_length, 1])

    ques_para_coembedding = tf.multiply(ques_para_coembedding, para_mask)
    ques_para_coembedding = tf.multiply(ques_para_coembedding, ques_mask)

    logging.info("ques_para_shape %s" % (ques_para_coembedding.shape))

    ques_para_coembedding = tf.reshape(ques_para_coembedding, [-1, FLAGS.max_ques_length,
                                                               para_num_tokens, 1])

    cnn_output = cnn_network(ques_para_coembedding)

    output_layer = tf.reshape(cnn_output, [-1, np.prod(cnn_output.shape[1:])])
    
    logging.info("output layer size %s" % (output_layer.shape))
    
    final_layer_size = output_layer.shape[1]

    output_weights = tf.get_variable("output_weights", [num_labels, final_layer_size],
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable("output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_data_loss = -tf.reduce_sum(one_hot_labels*log_probs, axis=-1)
        loss = tf.reduce_mean(per_data_loss)

    return loss, per_data_loss, logits, probabilities


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate, num_train_steps,
                     num_warmpup_steps, use_tpu, use_one_hot_embeddings):
    def model_fn(features, labels, mode, params):
        tf.logging.info("Features")
        for name in sorted(features.keys()):
            tf.logging.info(" name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_data_loss, logits, probabilities) = create_attention_model(bert_config, 
                                is_training, input_ids, input_mask, segment_ids, label_ids,
                                num_labels, use_one_hot_embeddings)
        
        tf.logging.info("probabilities shape = %s" % (probabilities.shape))

        tvars = tf.trainable_variables()
        initialized_variables = {}
        scaffold_fn = None

        if init_checkpoint:
            assignment_map, initialized_variable_names = modeling.get_assignment_map_from_checkpoint(
                                                                        tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("Training Variables...")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info(" name = %s, shape = %s%s", var.name, var.shape, init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(total_loss, learning_rate, num_train_steps,
                                                     num_warmpup_steps, use_tpu)
            output_spec = TPUEstimatorSpec(mode=mode, loss=total_loss, 
                                           train_op=train_op, scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(per_data_loss, label_ids, logits):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(label_ids, predictions)
                loss = tf.metrics.mean(per_data_loss)
                return {"eval_accuracy": accuracy, "eval_loss": loss}

            eval_metrics = (metric_fn, [per_data_loss, label_ids, logits])
            output_spec = TPUEstimatorSpec(mode=mode, loss=total_loss, 
                                           eval_metrics=eval_metrics, scaffold_fn=scaffold_fn)
        else:
            output_spec = TPUEstimatorSpec(mode=mode, predictions=probabilities, 
                                           scaffold_fn=scaffold_fn)
        return output_spec
    
    return model_fn

def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder, test=False):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64)
    }

    name_to_features["label_ids"] = tf.FixedLenFeature([], tf.int64)

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)

        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]

        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat().shuffle(buffer_size=100)
        d = d.apply(tf.contrib.data.map_and_batch(
                        lambda record: _decode_record(record, name_to_features),
                        batch_size=batch_size,
                        drop_remainder=drop_remainder))
        return d

    return input_fn

def main(_):
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file)
    train_data = data_ops.get_data(FLAGS.data_dir + "traindata_main.tsv")
    num_train_steps = int(len(train_data) / FLAGS.train_batch_size * FLAGS.num_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(bert_config=bert_config,
                                num_labels=FLAGS.num_labels,
                                init_checkpoint=FLAGS.bert_init_checkpoint,
                                learning_rate=FLAGS.learning_rate,
                                num_train_steps=num_train_steps,
                                num_warmpup_steps=num_warmup_steps,
                                use_tpu=False,
                                use_one_hot_embeddings=True)
            
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
                                                                FLAGS.tpu_name, 
                                                                zone=FLAGS.tpu_zone, 
                                                                project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(cluster=tpu_cluster_resolver,
                                          master=FLAGS.master,
                                          model_dir=FLAGS.output_dir,
                                          save_checkpoints_steps=FLAGS.save_checkpoints_steps,
                                          tpu_config=tf.contrib.tpu.TPUConfig(
                                                    iterations_per_loop=FLAGS.iterations_per_loop,
                                                    num_shards=FLAGS.num_tpu_cores,
                                                    per_host_input_for_training=is_per_host))
    
    estimator = TPUEstimator(use_tpu=FLAGS.use_tpu,
                             model_fn=model_fn,
                             config=run_config,
                             train_batch_size=FLAGS.train_batch_size,
                             eval_batch_size=FLAGS.eval_batch_size,
                             predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        train_file = FLAGS.data_dir + "train_data.tf_record"
        if not os.path.exists(train_file):
            data_ops.file_based_convert_data_to_features(train_data, FLAGS.max_sequence_length, 
                                            FLAGS.max_ques_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_data))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)

        train_input_fn = file_based_input_fn_builder(input_file=train_file,
                                                     seq_length=FLAGS.max_sequence_length,
                                                     is_training=True,
                                                     drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
        
    if FLAGS.do_predict:
        predict_data = data_ops.get_data(FLAGS.data_dir + "sample_eval1_unlabelled.tsv", test=True)
        predict_file = FLAGS.data_dir + "predict.tf_record"
        if not os.path.exists(predict_file):
            data_ops.file_based_convert_data_to_features(predict_data, FLAGS.max_sequence_length,
                                                     FLAGS.max_ques_length, tokenizer, predict_file,
                                                     test=True)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d", len(predict_data))
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        if FLAGS.use_tpu:
            raise ValueError("Prediction in TPU not supported.")

        predict_drop_remainder = True if FLAGS.use_tpu else False

        predict_input_fn = file_based_input_fn_builder(input_file=predict_file,
                                                       seq_length=FLAGS.max_sequence_length,
                                                       is_training=False,
                                                       drop_remainder=predict_drop_remainder,
                                                       test=True)
        result = estimator.predict(input_fn=predict_input_fn)
        
        output_predict_file = FLAGS.output_dir + "test_results.txt"

        with tf.gfile.GFile(output_predict_file, "w") as writer:
            tf.logging.info("**** Predict results ****")
            for prediction in result:
                output_line = "\t".join(str(class_probability) for class_probability in prediction) + "\n"
                writer.write(output_line)

if __name__ == "__main__":
    tf.app.run()
