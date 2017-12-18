#!/usr/bin/env python
'''
author: cheng wang, 09.09.2017
'''

from hashlib import sha1
import os
import random

random.seed(3)
import re, json,os
import sys,math
import numpy as np
from os import listdir
from os.path import isfile, join
import pdb,pickle
#from nltk.tokenize import word_tokenize

# UNK is the word used to identify unknown words
UNK_IDENTIFIER = '<UNK>'
WORD_START = '<START>'
WORD_END = '<END>'
OUTPUT_DIR = './'

SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')

class SequenceGenerator():
    def __init__(self, trainset, testset, valset, batch_size, n_timesteps):
        self.trainset = trainset
        self.testset = testset
        self.valset = valset
        
        self.batch_size = batch_size
        self.n_timesteps = n_timesteps
        
        self.init_vocabulary()
        self.one_hot_word_encoding()


    def init_vocabulary(self, min_count=1):
        words_to_count = {}
        for word in self.trainset:
            #pdb.set_trace()
            #for word in sentence:
            if word not in words_to_count:
                words_to_count[word] = 0  # key-vaule pair, get the count of each word
            words_to_count[word] += 1
                # print words_to_count
                # Sort words by count, then alphabetically
        words_by_count = sorted(words_to_count.keys(), key=lambda w: (-words_to_count[w], w))
        print 'Built vocabulary with %d words; \n The top 10 words:' % len(words_by_count)
        for word in words_by_count[:10]:
            print '\t%s (%d)' % (word, words_to_count[word])
            # Add words to vocabulary
        self.vocabulary = { }
        self.vocabulary_inverted = []
        #pdb.set_trace()
        for index, word in enumerate(words_by_count):
            word = word.strip()
            if words_to_count[word] < min_count:
                break
            self.vocabulary_inverted.append(word)
            self.vocabulary[word] = index
        #pdb.set_trace()
        print 'Remove the words with counts less than %d' % (min_count)
        print 'Final vocabulary has %d words' % (len(self.vocabulary))
        vocab_out_path = '%s/vocabulary_penntree.txt' % OUTPUT_DIR
        self.dump_vocabulary(vocab_out_path)
        #pdb.set_trace()

    def dump_vocabulary(self,vocab_filename):
        print 'Dumping vocabulary to file: %s' % vocab_filename
        with open(vocab_filename, 'wb') as reviewer_vocab_file:
            for reviewer in self.vocabulary_inverted:
                reviewer_vocab_file.write('%s\n' % reviewer)
        print 'dump vocabulary done.'

    def encoding_word(self):
        train_one_hot_vec=[]
        test_one_hot_vec=[]
        val_one_hot_vec=[]
        for word in self.trainset:
            train_one_hot_word = [self.vocabulary[word]]
            train_one_hot_vec.append(train_one_hot_word)
        for word in self.testset:    
            test_one_hot_word = [self.vocabulary[word]]
            test_one_hot_vec.append(test_one_hot_word)
        for word in self.valset:    
            val_one_hot_word = [self.vocabulary[word]]
            val_one_hot_vec.append(val_one_hot_word)
        return train_one_hot_vec, test_one_hot_vec, val_one_hot_vec,
 
 
    def prepare_input(self, data):
        batch_size=self.batch_size
        num_steps=self.n_timesteps
        data_len = len(data)
        batch_len = data_len // batch_size
        data = np.resize(data, batch_size * batch_len)
        data = data.reshape(batch_size, -1)
        x = data[:, :-1].T
        y = data[:, 1:].T 
 
        return x, y 
       
    def one_hot_word_encoding(self):
        
        trainset, testset, valset=self.encoding_word()
        print 'trainset has %d words' % (len(trainset))
        print 'valset has %d words' % (len(valset))
        print 'testset has %d words' %(len(testset))
        self.train_x, self.train_y = self.prepare_input(np.asarray(trainset))
        self.test_x, self.test_y = self.prepare_input(np.asarray(testset))
        self.val_x, self.val_y = self.prepare_input(np.asarray(valset))
        

        print 'one-hot encoding is done!'

    def save_dataset(self, savename, dataset):
        with open(savename, 'wb') as f:
              pickle.dump(dataset, f)
  
    def get_dataset(self):
        return self.train_x, self.train_y, self.test_x, self.test_y,self.val_x, self.val_y

def split_sentence(sentence):
    sentence = [s.lower().strip() for s in SENTENCE_SPLIT_REGEX.split(sentence.strip()) if len(s.strip()) > 0]
    return sentence

def read_text(path):
    text = open(path, 'r').read()
    words = text.decode("utf-8").replace("\n", "<eos>").split()
    return words
        

def load_dataset(batch_size, n_timesteps):
    print 'load peentree dataset...'

    trainset=read_text('penntree/ptb.train.txt')
    testset=read_text('penntree/ptb.test.txt')
    valset=read_text('penntree/ptb.valid.txt')
    f_seq = SequenceGenerator(trainset, testset, valset, batch_size, n_timesteps)
    train_x, train_y, test_x, test_y, val_x, val_y = f_seq.get_dataset()
    return train_x, train_y, test_x, test_y, val_x, val_y 


if __name__ == "__main__":
    trainse, testset, valset=load_dataset()
    SequenceGenerator(trainse, testset, valset)
