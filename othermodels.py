from numpy import array
from numpy import asarray
from numpy import zeros
import tensorflow as tf
import data_preprocess
import embeddings
from awekar_models import lstm,blstm,blstm_atten
from sklearn.model_selection import ShuffleSplit
import math
import matplotlib.pyplot as plt


print("Loading Data ...")
#data_source = 'data/hatebase_labeled_data.csv'
#no_bully_source = 'data/formspring_data.csv'

data_source = 'data/twitter_data.pkl'

print("preprocessing Data ...")
'''data_hb, class_hb = data_preprocess.biClass(data_source)
data_nb, class_nb = data_preprocess.noBully(no_bully_source)
#data, classification, severity = data_preprocess.multiClass(data_source)
data = data_hb + data_nb
classification = class_hb + class_nb

del data_hb
del class_hb
del data_nb
del class_nb'''

data, classification = data_preprocess.pickleData(data_source)


docs = data
labels = array(classification)

sequences, padlength = embeddings.simpleEncode(data)

del data
del classification

# prepare tokenizer
t = tf.keras.preprocessing.text.Tokenizer()
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1
# integer encode the documents
encoded_docs = t.texts_to_sequences(docs)
# pad documents
padded_docs = tf.keras.preprocessing.sequence.pad_sequences(encoded_docs, maxlen=padlength, padding='post')

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


e = tf.keras.layers.Embedding(vocab_size, 50, weights=[embedding_matrix], input_length=padlength, trainable=True)

print("Shuffling sets....")
# SHUFFLE DATA AND SET UP CROSS-VALIDATION
shuffdata = ShuffleSplit(n_splits=10, test_size=0.2, train_size=0.8)

train_accuracies = [] #stores the average of different cross validations
val_accuracies = [] #stores the average of different cross validations
train_losses = [] #stores the average of different cross validations
val_losses = [] #stores the average of different cross validations

for train_index, test_index in shuffdata.split(padded_docs,labels):

	trainX = []
	trainY = []

	testX = []
	testY = []

	for tr in train_index:
		trainX.append(padded_docs[tr])
		trainY.append(labels[tr])

	for ts in test_index:
		testX.append(padded_docs[ts])
		testY.append(labels[ts])

	# VALIDATION DATA
	valsize = len(trainX) - math.floor(0.2 * len(trainX))
	valX = array(trainX[valsize:len(trainX)])
	valY = array(trainY[valsize:len(trainX)])
	trainX = array(trainX[:valsize])
	trainY = array(trainY[:valsize])

	testX = array(testX)
	testY = array(testY)

	print("Creating and Training Model ...")
	#history, pred = lstm(trainX,trainY,valX,valY,testX,testY,e)
	#history, pred = blstm(trainX,trainY,valX,valY,testX,testY,e)
	history, pred = blstm_atten(trainX,trainY,valX,valY,testX,testY,e)
	#del pred

	bcount=0
	bully_count = 0
	nbcount=0
	non_bully_count = 0
	for x in range(len(testX)):
		if testY[x] == 1:
			bully_count = bully_count + 1
			if pred[x] == 1:
				bcount = bcount + 1
		if testY[x] == 0:
			non_bully_count = non_bully_count + 1
			if pred[x] == 0:
				nbcount = nbcount + 1
	print ("Bullying in-category accuracy: ", str(bcount/bully_count))
	print ("Non-Bullying in-category accuracy: ", str(nbcount/non_bully_count))



	history_dict = history.history
	history_dict.keys()

	# PLOT GRAPHS
	print("Plotting graphs ...")
	acc = history.history['acc']
	val_acc = history.history['val_acc']
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	epochs = range(1, len(acc) + 1)

	'''plt.plot(epochs, acc, 'bo', label='Training acc')
	plt.plot(epochs, val_acc, 'b', label='Validation acc')
	plt.title('Training and validation accuracy')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.show()'''

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



