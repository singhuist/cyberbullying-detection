from numpy import array
from numpy import asarray
from numpy import zeros
import tensorflow as tf
import data_preprocess
import embeddings
import nn_models


data_source = 'data/hatebase_labeled_data.csv'
data, classification, severity = data_preprocess.parseFile(data_source)
docs = data[:50]
labels = array(classification[:50])
sequences, padlength, wordIndex = embeddings.simpleEncode(data)

# prepare tokenizer
t = tf.keras.preprocessing.text.Tokenizer()
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1
# integer encode the documents
encoded_docs = t.texts_to_sequences(docs)
# pad documents to a max length of 4 words
padded_docs = tf.keras.preprocessing.sequence.pad_sequences(encoded_docs, maxlen=padlength, padding='post')

# load the whole embedding into memory
embeddings_index = dict()
f = open('GloVe/glove.6B.100d.txt',encoding="utf8")
for line in f:
	values = line.split()
	word = values[0]
	coefs = asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()

print('Loaded %s word vectors.' % len(embeddings_index))

# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, 100))
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

e = tf.keras.layers.Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=padlength, trainable=False)

trainX = padded_docs[:35]
trainY = labels[:35]
valX = padded_docs[35:40]
valY = labels[35:40]
testX = padded_docs[40:]
testY = labels[40:]

##history = nn_models.MultiLayerPerceptron(trainX,trainY,valX,valY,testX,testY,e)
history = nn_models.GatedRecurrentUnit(trainX,trainY,valX,valY,testX,testY,e)

'''history_dict = history.history
history_dict.keys()

# PLOT GRAPHS

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

print("ACCURACY IS: ",acc)'''
