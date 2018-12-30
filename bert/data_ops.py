# -*- coding: utf-8 -*-
import os
import sys
import re
import argparse

sys.path.append("/workspace/ms-ai-challenge/")

import collections
import json
import numpy as np
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from os.path import join as pjoin
from tensorflow.python.platform import gfile
import csv
import tqdm

import tokenization


_PAD = b"<pad>"
_SOS = b"<sos>"
_UNK = b"<unk>"
_START_VOCAB = [_PAD, _SOS, _UNK]

PAD_ID = 0
SOS_ID = 1
UNK_ID = 2

wordnet_lemmatizer = WordNetLemmatizer()

def basic_tokenizer(sentence):
    words = []
    for w in sentence.strip().split():
        if not w.isupper():
            w = w.lower()
        w = re.sub("<.*?>","",w)
        w = re.sub(r"\b(\d+\.?\d*%?)\b"," NUMBER ",w)
        w = re.sub(r"([\(\),/\?\\$;\[\]<>&%\-!\"'}{:“”])", r" \1 ", w)
        w = re.sub(r"[\(\),/\\\;\[\]<>\-!\"'}{:“”’]", "", w)
        w = re.sub(r'\.{2,}',' . ',w)
        try:
            words.extend([wordnet_lemmatizer.lemmatize(x) for x in re.split(" ", w)])
        except:
            words.extend(re.split(" ", w))
    return [w for w in words if w]

def initialize_vocabulary(vocab_path):
    # map vocabulary to word embeddings
    if gfile.Exists(vocab_path):
        rev_vocab = []
        with gfile.GFile(vocab_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x,y) for (y,x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file not found at %s",vocab_path)

def process_glove(args, vocab_list, save_path, size=4e5, random_init=True):
    if not gfile.Exists(save_path + ".npz"):
        glove_path = pjoin(args.glove_dir, "glove.6B.{}d.txt".format(args.glove_dim))
        
        if random_init:
            glove = np.random.randn(len(vocab_list), args.glove_dim)
        else:
            glove = np.zeros((len(vocab_list), args.glove_dim))
        
        found = 0

        with open(glove_path, 'r') as f:
            for line in tqdm(f, total=size):
                array = line.strip().split(" ")
                word = array[0]
                vector = list(map(float, array[1:]))
                if word in vocab_list:
                    idx = vocab_list.index(word)
                    glove[idx, :] = vector
                    found += 1
                if word.upper() in vocab_list:
                    idx = vocab_list.index(word.upper())
                    glove[idx, :] = vector
                    found += 1
        print("{}/{} of word vocab have corresponding vectors in {}".format(found, len(vocab_list), glove_path))
        np.savez_compressed(save_path, glove=glove)
        print("saved trimmed glove matrix at: {}".format(save_path))

def read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
        return lines

class InputData(object):
    def __init__(self, ques, para, label):
        self.ques = ques
        self.para = para
        self.label = label

class TestInputData(object):
    def __init__(self, ques, para, qid):
        self.ques = ques
        self.para = para
        self.label = 1
        self.id = qid

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class TestInputFeatures(object):
    """A single set of features of final test data."""
    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = 0

def get_data(data_path, test=False):
    lines = read_tsv(data_path)

    data = []
    for (i, line) in enumerate(lines):
        ques = tokenization.convert_to_unicode(line[1])
        para = tokenization.convert_to_unicode(line[2])
        if not test:
            label = tokenization.convert_to_unicode(line[3])
            data.append(InputData(ques, para, label))
        else:
            data.append(InputData(ques, para, u'1'))
    return data

def convert_single_data(ind, row, max_seq_length, max_ques_length, tokenizer, test=False):
    ques_tokens = tokenizer.tokenize(row.ques)
    para_tokens = tokenizer.tokenize(row.para)

    ques_tokens = ques_tokens[:max_ques_length]
    para_tokens = para_tokens[:(max_seq_length - max_ques_length - 3)]

    """
    BERT input format:
    tokens:     [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    segment_ids: 0    0   0    0    0     0      0  0     1  1  1  1  1  1

    segment_ids indicate whether its question or paragraph and embeddings for segment 0 and 1
    were learned while training BERT.
    """
    tokens, segment_ids = [], []
    
    tokens.append("[CLS]")
    segment_ids.append(0)

    tokens += ques_tokens
    segment_ids += [0] * len(ques_tokens)
    input_mask = [1] * len(tokens)

    # pad question tokens
    ques_padding_tokens = ["[PAD]"] * (max_ques_length - len(ques_tokens))
    ques_padding = [0] * (max_ques_length - len(ques_tokens))

    tokens += ques_padding_tokens
    segment_ids += ques_padding
    input_mask += ques_padding

    tokens.append("[SEP]")
    segment_ids.append(0)
    input_mask.append(1)
    
    # pad paragraph tokens
    tokens += para_tokens
    segment_ids += [1] * len(para_tokens)
    input_mask += [1] * len(para_tokens)

    tokens.append("[SEP]")
    segment_ids.append(1)
    input_mask.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding
    
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    if ind < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("tokens: %s" % " ".join(
                            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s" % (row.label))
        print(type(row.label), row.label)
    feature = InputFeatures(input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            label_id=row.label)
    
    return feature


def file_based_convert_data_to_features(data, max_seq_length, max_ques_length, tokenizer, output_file, test=False):
    writer = tf.python_io.TFRecordWriter(output_file)

    def create_int_feature(values):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))

    for (ind, row) in enumerate(data):
        if ind % 100000 == 0:
            tf.logging.info("Writing data %d of %d" % (ind, len(data)))
        feature = convert_single_data(ind, row, max_seq_length, max_ques_length, tokenizer, test)

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature([int(feature.label_id)])

        tf_data = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_data.SerializeToString())


def create_vocabulary(vocabulary_path, data_paths, start_ind, end_ind, tokenizer=None):
    fpath = vocabulary_path.split(".")
    fpath = fpath[:-1] + "_" + str(start_ind) + "_" + str(end_ind) + fpath[-1]
    if not gfile.Exists(fpath):
        print("Creating vocabulary %s from data %s" % (fpath, str(data_paths)))
        vocab = {}
        for path in data_paths:
            with open(path, mode="rb") as f:
                if end_ind:
                    data = f.readlines()[start_ind: end_ind]
                else:
                    data = f.readlines()
                counter = 0
                for line in data:
                    line = line.decode()
                    counter += 1
                    if counter % 100000 == 0:
                        print("processing line %d" % counter)
                    content = line.split("\t")
                    tokens = tokenizer(line) if tokenizer else basic_tokenizer(content[1])
                    tokens += tokenizer(line) if tokenizer else basic_tokenizer(content[2])
                    for w in tokens:
                        if w in vocab:
                            vocab[w] += 1
                        else:
                            vocab[w] = 1
            print(path + " done!")
            end_ind = None
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        print("Vocabulary size: %d" % len(vocab_list))
        with gfile.GFile(fpath, mode="wb") as vocab_file:
            for w in vocab_list:
                vocab_file.write(w + b"\n")

def sentence_to_token_ids(sentence, vocabulary, tokenizer=None):
    """
    Convert each word to its token in a sentence
    """
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    return [vocabulary.get(w, UNK_ID) for w in words]

def data_to_token_ids(data_path, target_paths, vocabulary_path,
                      tokenizer=None, label=False):
    
    ques_target_path, para_target_path, label_target_path = target_paths
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="rb") as data_file:
        with gfile.GFile(ques_target_path, mode="w") as q_tokens_file:
            with gfile.GFile(para_target_path, mode="w") as p_tokens_file:
                with gfile.GFile(label_target_path, mode="w") as labels_file:
                    counter = 0
                    for line in data_file:
                        content = line.split("\t")
                        counter += 1
                        if counter % 5000 == 0:
                            print("tokenizing line %d" % counter)
                        token_ids = sentence_to_token_ids(content[1], vocab, tokenizer)
                        q_tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")

                        token_ids = sentence_to_token_ids(content[2], vocab, tokenizer)
                        p_tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")

                        labels_file.write(content[0] + "\n")

def get_data_old(data_path, target_paths, vocab_path, num_neg_samples=5):
    """
    Get positive and negative samples
    """
    pos_para_path, pos_ques_path, neg_para_path, neg_ques_path = target_paths
    vocab, _ = initialize_vocabulary(vocab_path)

    query_neg_samples_count = {}
    
    with open(data_path, "r") as f:
        with open(pos_para_path, "w") as fpos_para:
            with open(neg_para_path, "w") as fneg_para:
                with open(pos_ques_path, "w") as fpos_ques:
                    with open(neg_ques_path, "w") as fneg_ques:
                        for count, line in enumerate(f):
                            tokens = line.strip().lower().split("\t")
                            query_id, query, passage, label = tokens[0], basic_tokenizer(tokens[1]), basic_tokenizer(tokens[2]), tokens[3]
                            
                            query = [vocab.get(w, UNK_ID) for w in query]
                            passage = [vocab.get(w, UNK_ID) for w in passage]

                            if label == "1":
                                fpos_ques.write(" ".join([str(tok) for tok in query]) + "\n")
                                fpos_para.write(" ".join([str(tok) for tok in passage]) + "\n")
                            elif label == "0":
                                if query_id not in query_neg_samples_count or query_neg_samples_count[query_id] < num_neg_samples:
                                    fneg_ques.write(" ".join([str(tok) for tok in query]) + "\n")
                                    fneg_para.write(" ".join([str(tok) for tok in passage]) + "\n")
                                    
                                    if query_id not in query_neg_samples_count:
                                        query_neg_samples_count[query_id] = 1
                                    else:
                                        query_neg_samples_count[query_id] += 1
                            
                            if (count+1)%100000 == 0:
                                print("{} data processed".format(count))
    
def separate_data(data_path, target_paths, num_neg_samples):
    ques_path, pos_data_path, neg_data_path, labels_path = target_paths

    with open(data_path, "r") as f:
        for count, line in enumerate(f):
            tokens = line.strip().lower().split("\t")
            query_id, query, passage, label = tokens[0], tokens[1], tokens[2], tokens[3]

def get_raw_data(data_path):
    """
    Get raw data in form of list of strings
    """
    if tf.gfile.Exists(data_path):
        with tf.gfile.GFile(data_path, mode="rb") as f:
            data = f.readlines()
        return data
    else:
        raise ValueError("Data file %s not found.", data_path)

def remove_less_significant_words(words_list, max_length, doc_IDF_dict):
    words_with_IDF = [(w, doc_IDF_dict[w]) for w in words_list]
    words_with_IDF = sorted(words_with_IDF, key=lambda x: x[1])

    
    return [w for w,_ in words_with_IDF[:max_length]]


def pad_sequences(data, max_length):
    """
    Pad the data with PAD_ID which has length less than max_length and 
    discard the extra words.
        
    args:
        -   data: list of sentences
        -   max_length: sequence of length max_length to be maintained
    
    return:
        -   padded_data: data padded with PAD_ID
    """
    with open("docIDFDict.pickle","rb") as f:
        doc_IDF_dict = json.load(f)
    
    new_data, mask_data = [], []
    for d in data:
        word_count = len(d)
        remaining = max_length - word_count
        if remaining >= 0:
            d += [PAD_ID]*remaining
            md = [True]*word_count + [False]*remaining
        else:
            d = remove_less_significant_words(d, max_length, doc_IDF_dict)
            md = [True]*max_length
        new_data.append(d)
        mask_data.append(md)

    return (new_data, mask_data)

if __name__ == '__main__':
    args = setup_args()

    vocab_path = pjoin(args.vocab_dir, "vocab.dat")

    train_path = pjoin(args.source_dir, "train")
    valid_path = pjoin(args.source_dir, "valid")
    dev_path = pjoin(args.source_dir, "dev")

    start_ind, end_ind = int(sys.argv[1]), int(sys.argv[2])

    create_vocabulary(vocab_path,
                      [pjoin(args.source_dir, "traindata_main.tsv"),
                       pjoin(args.source_dir, "validationdata.tsv"),
                       pjoin(args.source_dir, "sample_eval.tsv")], 
                      start_ind, end_ind)
    print("Vocabulary created")
    # vocab, rev_vocab = initialize_vocabulary(pjoin(args.vocab_dir, "vocab.dat"))
    # print("Vocabulary initialized!")
    # process_glove(args, rev_vocab, args.source_dir + "/glove.trimmed.{}".format(args.glove_dim),
    #               random_init=args.random_init)
    # print("Glove processing done!")
    # ques_train_dis_path = train_path + ".ids.question"
    # para_train_dis_path = train_path + ".ids.passage"
    # label_train_dis_path = train_path + ".labels"
    # train_dis_path = [ques_train_dis_path, para_train_dis_path, label_train_dis_path]
    # data_to_token_ids(args.source_dir + "sample_train.tsv", train_dis_path, vocab_path)
    # print("Train data to token done!")
    # ques_valid_dis_path = valid_path + ".ids.question"
    # para_valid_dis_path = valid_path + ".ids.passage"
    # label_valid_dis_path = valid_path + ".labels"
    # valid_dis_path = [ques_valid_dis_path, para_valid_dis_path, label_valid_dis_path]
    # data_to_token_ids(args.source_dir + "sample_train.tsv", valid_dis_path, vocab_path)
    # print("Valid data to token done!")
    # print("All done!")
