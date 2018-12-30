import tensorflow as tf
import data_ops
import numpy as np
from collections import defaultdict
import math

sess = tf.Session()
saver = tf.train.import_meta_graph("/workspace/ms-ai-challenge/paragraph-ranker-tf/model_pr_1.ckpt.meta")        
saver.restore(sess, tf.train.latest_checkpoint('./'))
ts = sess.graph.get_tensor_by_name('dense/Softmax:0')

eval_data = "../data/eval1_unlabelled.tsv"
eval_test_ques = "../data/test.ids.question"
eval_test_para = "../data/test.ids.passage"
eval_query_id = "../data/test.para.id"
vocab_path = "../data/vocab.dat"

data_ops.data_to_token_ids(eval_data, [eval_test_ques, eval_test_para, eval_query_id], vocab_path)

embeddings = np.load("../data/glove.trimmed.100.npz")['glove']
import math

all_scores = defaultdict(list)

filenames = [eval_test_ques, eval_test_para, eval_query_id]

def parse_func(line):
    string_vals = tf.string_split([line]).values
    return tf.string_to_number(string_vals, tf.int32), tf.size(string_vals)

def parse_func_for_id(line):
    string_vals = tf.string_split([line]).values
    return tf.string_to_number(string_vals, tf.int32)

test_ques = tf.data.TextLineDataset(eval_test_ques)
test_ques = test_ques.map(map_func=parse_func)

test_para = tf.data.TextLineDataset(eval_test_para)
test_para = test_para.map(map_func=parse_func)

query_ids = tf.data.TextLineDataset(eval_query_id)
query_ids = query_ids.map(map_func=parse_func)

padded_shapes = ((tf.TensorShape([None]), tf.TensorShape([])), 
                 (tf.TensorShape([None]), tf.TensorShape([])),
                 (tf.TensorShape([None]), tf.TensorShape([])))
padding_values = ((0,0), (0,0), (0,0))

batch_size = 500
test_data_size = 104170

test_ques_para_dataset = tf.data.Dataset.zip((test_ques, test_para, query_ids))
test_ques_para_dataset = test_ques_para_dataset.padded_batch(batch_size, 
                                                             padded_shapes=padded_shapes, 
                                                             padding_values=padding_values)
test_ques_para_iterator = test_ques_para_dataset.make_initializable_iterator()
test_ques_para_init_op  = test_ques_para_iterator.initializer

test_num_batches = int(math.ceil(test_data_size / batch_size))

test_next_element = test_ques_para_iterator.get_next()

sess.run(test_ques_para_init_op)

for j in range(test_num_batches+1):
    ((ques_next_batch, ques_seq_lengths), (para_next_batch, para_seq_lengths), (query_ids, _)) = sess.run(test_next_element)

    input_feed = {
        'ques:0': ques_next_batch,
        'para:0': para_next_batch,
        'Placeholder:0': embeddings,
        'Placeholder_1:0': ques_seq_lengths,
        'Placeholder_2:0': para_seq_lengths
    }
    output = sess.run([ts], input_feed)[0]
    for k in range(len(query_ids)):
        all_scores[str(query_ids[k][0])].append(output[k][1])
        
    if (j+1)%20 == 0:
        print("{} done".format(j+1))
    
with open("answer.tsv","w") as fw:
    for query_id in all_scores:
        scores = all_scores[query_id]
        scores_str = "\t".join([str(sc) for sc in scores])
        fw.write(query_id+"\t"+scores_str+"\n")