import numpy as np
import random
#from spellchecker import SpellChecker
import tensorflow as tf

#spell = SpellChecker()

def simpleEncode(sentenceList):
    '''
    Encodes sentences into numbers, simply adding a new entry for a new word
    :param sentenceList: List of sentences to be vectorised
    :return: encoded list of sentences
    '''

    dictionary = {}
    counter = 0
    encodedData = []
    padlength = 0
    for s in sentenceList:
        encodedsent = []
        s = s.split()
        for w in range(len(s)):
            word = s[w]
            if word not in dictionary:
                dictionary[word] = counter + 1
                counter = counter + 1
            s[w] = dictionary[word]
            encodedsent.append(s[w])
        encodedData.append(encodedsent)

        if len(encodedsent)>padlength:
            padlength = len(encodedsent)

    #pad data to ensure equal size
    ##encodedData = keras.preprocessing.sequence.pad_sequences(encodedData, value=0, padding='post', maxlen=padlength)

    return encodedData, padlength


def gloveEmbed(docs, padlength):
    '''
    Encode sentences with pre-trained GloVe embeddings
    :param sentenceList: list of sentences to be encoded
    :return: encoded list of sentences
    '''

    # load the whole embedding into memory
    embeddings_index = dict()
    f = open('GloVe/glove.6B.50d.txt', encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Loaded %s word vectors.' % len(embeddings_index))

    # prepare tokenizer
    t = tf.keras.preprocessing.text.Tokenizer()
    t.fit_on_texts(docs)
    vocab_size = len(t.word_index) + 1
    # integer encode the documents
    encoded_docs = t.texts_to_sequences(docs)
    # pad documents to a max length of longest sentence
    padded_docs = tf.keras.preprocessing.sequence.pad_sequences(encoded_docs, maxlen=padlength, padding='post')

    # create a weight matrix for words in training docs
    embedding_matrix = np.zeros((vocab_size, 100))
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    emb_layer = tf.keras.layers.Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=padlength, trainable=False)
    return embedding_matrix, emb_layer


