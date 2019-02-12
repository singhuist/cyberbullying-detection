from numpy import array
from numpy import asarray
from numpy import zeros
import tensorflow as tf
import data_preprocess
import embeddings
import nn_models
from sklearn.model_selection import ShuffleSplit
import math
import matplotlib.pyplot as plt


print("Loading Data ...")
data_source = 'data/hatebase_labeled_data.csv'
no_bully_source = 'data/formspring_data.csv'

print("preprocessing Data ...")
data_bi, classification_bi = data_preprocess.biClass(data_source)
data_mult, classification_mult = data_preprocess.mult_only(data_source)
data_nb, class_nb = data_preprocess.noBully(no_bully_source)
docs_bi = data_bi + data_nb
docs_mult = data_mult
labels_bi = array(classification_bi + class_nb)
labels_mult = array(classification_mult)
sequences_bi, padlength_bi = embeddings.simpleEncode(data_bi)
sequences_mult, padlength_mult = embeddings.simpleEncode(data_mult)

# prepare tokenizer
t = tf.keras.preprocessing.text.Tokenizer()
t.fit_on_texts(docs_bi)
t.fit_on_texts(docs_mult)
vocab_size = len(t.word_index) + 1
# integer encode the documents
encoded_docs_bi = t.texts_to_sequences(docs_bi)
encoded_docs_mult = t.texts_to_sequences(docs_mult)
# pad documents
padded_docs_bi = tf.keras.preprocessing.sequence.pad_sequences(encoded_docs_bi, maxlen=padlength_bi, padding='post')
padded_docs_mult = tf.keras.preprocessing.sequence.pad_sequences(encoded_docs_mult, maxlen=padlength_mult, padding='post')

print("Prepare Embeddings ...")
# load the whole embedding into memory
embeddings_index = dict()
f = open('GloVe/glove.6B.50d.txt',encoding="utf8")
for line in f:
	values = line.split()
	word = values[0]
	coefs = asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()

print('Loaded %s word vectors.' % len(embeddings_index))

print("preparing embedding matrices ...")
# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, 50))
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector
	else:
		embedding_matrix[i] = array([0 for x in range(50)])


e_bi = tf.keras.layers.Embedding(vocab_size, 50, weights=[embedding_matrix], input_length=padlength_bi, trainable=False)
e_mult = tf.keras.layers.Embedding(vocab_size, 50, weights=[embedding_matrix], input_length=padlength_mult, trainable=False)

print("Shuffling sets....")
# SHUFFLE DATA AND SET UP CROSS-VALIDATION
shuffdata = ShuffleSplit(n_splits=10, test_size=0.2, train_size=0.8)

train_accuracies = [] #stores the average of different cross validations
val_accuracies = [] #stores the average of different cross validations
train_losses = [] #stores the average of different cross validations
val_losses = [] #stores the average of different cross validations

for train_index, test_index in shuffdata.split(padded_docs_bi,labels_bi):

	trainX = []
	trainY = []

	testX = []
	testY = []

	for tr in train_index:
		trainX.append(padded_docs_bi[tr])
		trainY.append(labels_bi[tr])

	for ts in test_index:
		testX.append(padded_docs_bi[ts])
		testY.append(labels_bi[ts])

	# VALIDATION DATA
	valsize = len(trainX) - math.floor(0.2 * len(trainX))
	valX = array(trainX[valsize:len(trainX)])
	valY = array(trainY[valsize:len(trainX)])
	trainX = array(trainX[:valsize])
	trainY = array(trainY[:valsize])

	testX = array(testX)
	testY = array(testY)

	print("Creating and Training Model ...")
	bi_history,bi_prediction = nn_models.GatedRecurrentUnit(trainX,trainY,valX,valY,testX,testY,e_bi)

	mult_testX = []
	mult_testY = []

	for p in range(len(bi_prediction)):
		if bi_prediction[p] == 1:
			mult_testX.append(testX[p])
			mult_testY.append(testY[p])

	mult_testX = array(mult_testX)
	mult_testY = array(mult_testY)

	print(len(mult_testX),":",len(mult_testY))

	mult_trainX = padded_docs_mult
	mult_trainY = labels_mult

	# VALIDATION DATA
	valsize = len(mult_trainX) - math.floor(0.2 * len(mult_trainX))
	mult_valX = array(mult_trainX[valsize:len(mult_trainX)])
	mult_valY = array(mult_trainY[valsize:len(mult_trainX)])
	mult_trainX = array(mult_trainX[:valsize])
	mult_trainY = array(mult_trainY[:valsize])

	print(len(mult_trainX),":",len(mult_trainY))
	print(len(mult_valX),":",len(mult_valY))


	print("Creating and Training Model ...")
	history,mult_prediction = nn_models.GatedRecurrentUnit(mult_trainX,mult_trainY,mult_valX,mult_valY,mult_testX,mult_testY,e_mult)


	history_dict = history.history
	history_dict.keys()

	# PLOT GRAPHS
	print("Plotting graphs ...")
	acc = history.history['acc']
	val_acc = history.history['val_acc']
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	epochs = range(1, len(acc) + 1)

	plt.plot(epochs, acc, 'bo', label='Training acc')
	plt.plot(epochs, val_acc, 'b', label='Validation acc')
	plt.title('Training and validation accuracy')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.show()

	train_accuracies.append(sum(acc) / len(acc))
	val_accuracies.append(sum(val_acc) / len(val_acc))


#OVERALL AVG MODEL ACCURACIES
kfold = range(1, len(train_accuracies) + 1)
plt.plot(kfold, train_accuracies, 'bo', label='Training acc')
plt.plot(kfold, val_accuracies, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('K-Fold')
plt.ylabel('Accuracy')
plt.legend()

plt.show()



