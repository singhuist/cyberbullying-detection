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

def cleanString(sentence):
    '''
    Remove stop words, special symbols and punctuation from a sentence
    :param sentence: Sentence to be cleaned
    :return: cleaned sentence
    '''

    print(sentence)
    stop_words = set(stopwords.words('english'))
    punctuations = set(string.punctuation)
    #specialsym = set(['~', '`', '@', '$', '#', '%', '^', '&', '*', '``', "''", '..', '...', 'n/a', 'na'])

    sentence = sentence.lower()
    words = sentence.split()

    for w in range(len(words)):
        words[w] = word_tokenize(words[w])

        if words[w][0] == '#' and len(words[w])==2:
            words[w] = ' '.join(segment(words[w][1]))
        if words[w][0] == '@' or words[w][0] == 'rt' or words[w][0] == 'http':
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

    print(sentence)

    return sentence


def parseFile(datafile):
    '''
    Reads the data file and extracts necessary information in a consistent format
    :param datafile: File containing the data
    :return:
    '''

    tweets = [] #stores the tweets
    bully_class = [] #stores the class: 0 = hate_speech, 1 = offensive language, 2 = neither
    bully_severity = [] #score of the severity of bullying

    with open(datafile) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        for row in csv_reader:
            tweets.append(cleanString(row[6]))
            if (int(row[5]) == 0 or int(row[5]) ==1):
                bully_class.append(1) # 1 for no cyberbullying
            elif (int(row[5])==2):
                bully_class.append(0) # 0 for no cyberbullying
            bully_severity.append(max(int(row[2]),int(row[3]),int(row[4])))

    return tweets, bully_class, bully_severity
