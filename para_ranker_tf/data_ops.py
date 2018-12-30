# -*- coding: utf-8 -*-
import os
import sys
import re
import argparse

from tensorflow.python.platform import gfile
from nltk.stem import WordNetLemmatizer
from tqdm import *
import numpy as np
from os.path import join as pjoin
import json

_PAD = b"<pad>"
_SOS = b"<sos>"
_UNK = b"<unk>"
_START_VOCAB = [_PAD, _SOS, _UNK]

PAD_ID = 0
SOS_ID = 1
UNK_ID = 2

wordnet_lemmatizer = WordNetLemmatizer()

def setup_args():
    parser = argparse.ArgumentParser()
    code_dir = "/workspace/ms-ai-challenge/"
    vocab_dir = os.path.join(code_dir, "data/")
    glove_dir = os.path.join(code_dir, "data/")
    source_dir = os.path.join(code_dir, "data/")
    parser.add_argument("--source_dir", default=source_dir)
    parser.add_argument("--glove_dir", default=glove_dir)
    parser.add_argument("--vocab_dir", default=vocab_dir)
    parser.add_argument("--glove_dim", default=50, type=int)
    parser.add_argument("--random_init", default=True, type=bool)
    return parser.parse_args()

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

def get_data(data_path):
    """
    Get tokenized data in form of list of lists
    """
    if tf.gfile.Exists(data_path):
        data = []
        with tf.gfile.GFile(data_path, mode="rb") as f:
            for line in f.readlines():
                data.append([int(x) for x in line.strip().split()])
        return data
    else:
        raise ValueError("Data file %s not found.", data_path)

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