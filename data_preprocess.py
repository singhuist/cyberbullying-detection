import csv
import tensorflow as tf
from tensorflow import keras
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.model_selection import ShuffleSplit
import numpy as np
import math
from segment import segment
import pickle

def cleanString(sentence):
    '''
    Remove stop words, special symbols and punctuation from a sentence
    :param sentence: Sentence to be cleaned
    :return: cleaned sentence
    '''

    #print(sentence)
    stop_words = set(stopwords.words('english'))
    punctuations = set(string.punctuation)
    #specialsym = set(['~', '`', '@', '$', '#', '%', '^', '&', '*', '``', "''", '..', '...', 'n/a', 'na'])

    sentence = sentence.lower()
    words = sentence.split()

    for w in range(len(words)):
        words[w] = word_tokenize(words[w])

        if words[w][0] == '#' and len(words[w])==2:
            words[w] = ' '.join(segment(words[w][1]))
        if words[w][0] == '@' or words[w][0] == 'rt' or words[w][0] == 'http' or words[w][0] == 'https':
            words[w] = [] #remove rt, usernames and hyperlinks
        #print(words[w])
        words[w] = ''.join(words[w])

    sentence = ' '.join(words)

    word_tokens = word_tokenize(sentence)
    #print(words)

    no_punc = [w for w in word_tokens if not w in punctuations]  # remove punctuation
    #no_sym = [w for w in no_punc if not w in specialsym]  # remove special symbols
    filtered_sentence = [w for w in no_punc if not w in stop_words]  # remove stop words

    sentence = ' '.join(filtered_sentence)

    #print(sentence)

    return sentence 

def pickleData(filename):
    data = pickle.load(open(filename, 'rb'))
    x_text = []
    labels = [] 
    ids = []
    for i in range(len(data)):
        text = "".join(l for l in data[i]['text'] if l not in string.punctuation)
        x_text.append(cleanString((data[i]['text']).encode('utf-8').decode('utf-8')))
        if data[i]['label'] == 'none':
            labels.append(0)
        else:
            labels.append(1)
    return x_text,labels

def pickleData_staggered(filename):
    data = pickle.load(open(filename, 'rb'))
    x_text = []
    bi_labels = []
    multi_labels = [] 
    ids = []
    for i in range(len(data)):
        text = "".join(l for l in data[i]['text'] if l not in string.punctuation)
        x_text.append(cleanString((data[i]['text']).encode('utf-8').decode('utf-8')))
        if data[i]['label'] == 'none':
            labels.append(0)
        else:
            labels.append(1)
    return x_text,labels


def pickleData_multi(filename):
    data = pickle.load(open(filename, 'rb'))
    x_text = []
    labels = [] 
    ids = []
    for i in range(len(data)):
        text = "".join(l for l in data[i]['text'] if l not in string.punctuation)
        x_text.append(cleanString((data[i]['text']).encode('utf-8').decode('utf-8')))
        if data[i]['label'] == 'none':
            labels.append(0)
        elif data[i]['label'] == 'racism':
            labels.append(1)
        elif data[i]['label'] == 'sexism':
            labels.append(2)
    return x_text,labels