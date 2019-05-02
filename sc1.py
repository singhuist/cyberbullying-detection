from numpy import array, asarray, zeros
import tensorflow as tf
import data_preprocess, embeddings
import nn_models, multi_models, awekar_models
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import math
import matplotlib.pyplot as plt
import random


print("Loading Data ...")

data_source = 'data/twitter_data.pkl'

print("preprocessing Data ...")

data, classification = data_preprocess.pickleData(data_source) #bi-class data
mult_data, mult_class = data_preprocess.pickleData_multi(data_source) #multi-class data
#bully_data, bully_class = data_preprocess.pickleData_multi(data_source)
del mult_data

ostext = []
oslabel = []
oslabelmult = []
for l in range(len(classification)):
	if classification[l] != 0:
		ostext.append(data[l])
		oslabel.append(classification[l])
		oslabelmult.append(mult_class[l])

data = data + ostext
classification = classification + oslabel
mult_class = mult_class + oslabelmult

oversampled = list(zip(data, classification, mult_class))
random.shuffle(oversampled)
data, classification, mult_class = zip(*oversampled)


docs = data
labels = array(classification) #binary class labels
mult_labels = array(mult_class) #multi-class labels
#bully_labels = array(bully_class)

sequences, padlength = embeddings.simpleEncode(data)

del data
del classification
del mult_class

# prepare tokeanizer
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


e = tf.keras.layers.Embedding(vocab_size, 50, weights=[embedding_matrix], input_length=padlength, trainable=False)

print("Shuffling sets....")
# SHUFFLE DATA AND SET UP CROSS-VALIDATION
shuffdata = ShuffleSplit(n_splits=10, test_size=0.2, train_size=0.8)

train_accuracies = [] #stores the average of different cross validations
val_accuracies = [] #stores the average of different cross validations
train_losses = [] #stores the average of different cross validations
val_losses = [] #stores the average of different cross validations

allres = 'results/staggered/sc1/sc1.txt'
grures = 'results/staggered/sc1/gru.txt'
lstmres = 'results/staggered/sc1/lstm.txt'
blstmres = 'results/staggered/sc1/blstm.txt'

afile = open(allres,'w')
gfile = open(grures,'w')
lfile = open(lstmres,'w')
bfile = open(blstmres,'w')

for train_index, test_index in shuffdata.split(padded_docs,labels):

	trainX = []
	trainY = []

	trainX_mult = []
	trainY_mult = []

	testX = []
	testY = []

	for tr in train_index:

		trainX.append(padded_docs[tr])
		trainY.append(labels[tr])

		## append the ones marked as bullying into other dataset for training only bullying sentences 
		if labels[tr] == 1:
			trainX_mult.append(padded_docs[tr])
			trainY_mult.append(mult_labels[tr])

			
	multtestY = []
	for ts in test_index:
		testX.append(padded_docs[ts])
		testY.append(labels[ts])
		multtestY.append(mult_labels[ts])

	# VALIDATION DATA
	valsize = len(trainX) - math.floor(0.2 * len(trainX))
	valX = array(trainX[valsize:len(trainX)])
	valY = array(trainY[valsize:len(trainX)])

	valsize_mult = len(trainX_mult) - math.floor(0.2 * len(trainX_mult))
	valX_mult = array(trainX_mult[valsize_mult:len(trainX_mult)])
	valY_mult = array(trainY_mult[valsize_mult:len(trainX_mult)])

	trainX = array(trainX[:valsize])
	trainY = array(trainY[:valsize])
	
	trainX_mult = array(trainX_mult[:valsize_mult])
	trainY_mult = array(trainY_mult[:valsize_mult])

	testX = array(testX)
	testY = array(testY)

	
	print("Creating and Training Bi-Classification Model ...")
	#bi_history,bi_prediction = nn_models.GatedRecurrentUnit(trainX,trainY,valX,valY,testX,testY,e)
	bi_history, bi_prediction = nn_models.GatedRecurrentUnit(trainX,trainY,valX,valY,testX,testY,e)

	testX_mult = []
	testY_mult = []

	for p in range(len(bi_prediction)):
		if bi_prediction[p] == 1:
			testX_mult.append(testX[p])
			testY_mult.append(multtestY[p])

	print("Obtained multi-class training data")

	testX_mult = array(testX_mult)
	testY_mult = array(testY_mult)

	print("Creating and Training Multi-Classification Model ...")
	history,gru_mult_prediction = multi_models.gru_multi(trainX_mult,trainY_mult,valX_mult,valY_mult,testX_mult,testY_mult,e)

	f_mes = f1_score(testY_mult,gru_mult_prediction,average='weighted')
	a_mes = accuracy_score(testY_mult,gru_mult_prediction)
	p_mes = precision_score(testY_mult,gru_mult_prediction,average='weighted')
	r_mes = recall_score(testY_mult,gru_mult_prediction,average='weighted')

	gfile.write("Accuracy: "+str(a_mes)+'\n')
	gfile.write("Precision: "+str(p_mes)+'\n')
	gfile.write("Recall: "+str(r_mes)+'\n')
	gfile.write("F-Score: "+str(f_mes)+'\n')

	afile.write("GRU F-Score: "+str(f_mes)+'\n')
	afile.write('\n')


	####################################################################################
	

	print("Creating and Training Multi-Classification Model ...")
	history,lstm_mult_prediction = multi_models.lstm_multi(trainX_mult,trainY_mult,valX_mult,valY_mult,testX_mult,testY_mult,e)

	f_mes = f1_score(testY_mult,lstm_mult_prediction,average='weighted')
	a_mes = accuracy_score(testY_mult,lstm_mult_prediction)
	p_mes = precision_score(testY_mult,lstm_mult_prediction,average='weighted')
	r_mes = recall_score(testY_mult,lstm_mult_prediction,average='weighted')

	lfile.write("Accuracy: "+str(a_mes)+'\n')
	lfile.write("Precision: "+str(p_mes)+'\n')
	lfile.write("Recall: "+str(r_mes)+'\n')
	lfile.write("F-Score: "+str(f_mes)+'\n')

	afile.write("LSTM F-Score: "+str(f_mes)+'\n')
	afile.write('\n')

	####################################################################################

	
	print("Creating and Training Multi-Classification Model ...")
	history,bl_mult_prediction = multi_models.blstm_multi(trainX_mult,trainY_mult,valX_mult,valY_mult,testX_mult,testY_mult,e)

	f_mes = f1_score(testY_mult,bl_mult_prediction,average='weighted')
	a_mes = accuracy_score(testY_mult,bl_mult_prediction)
	p_mes = precision_score(testY_mult,bl_mult_prediction,average='weighted')
	r_mes = recall_score(testY_mult,bl_mult_prediction,average='weighted')

	bfile.write("Accuracy: "+str(a_mes)+'\n')
	bfile.write("Precision: "+str(p_mes)+'\n')
	bfile.write("Recall: "+str(r_mes)+'\n')
	bfile.write("F-Score: "+str(f_mes)+'\n')

	afile.write("BLSTM F-Score: "+str(f_mes)+'\n')
	afile.write('\n')


	############# IMPLEMENTING VOTING ALGORITHM #############

	cons_dict = []
	for g,l,b in zip(gru_mult_prediction,lstm_mult_prediction,bl_mult_prediction):
		cons = {}
		cons[0] = 0
		cons[1] = 0
		cons[2] = 0

		if g == 1:
			cons[1] += 1
		elif g == 2:
			cons[2] += 1

		if l == 1:
			cons[1] += 1
		elif l == 2:
			cons[2] += 1

		if b == 1:
			cons[1] += 1
		elif b == 2:
			cons[2] += 1


		cons_dict.append(cons)

	#print(cons_dict)

	cons_pred = []
	for x in range(len(cons_dict)):
		d = cons_dict[x]
		if d[1]>d[2]:
			cons_pred.append(1)
		else:
			cons_pred.append(2)


	allpred = bi_prediction

	idx = 0
	for a in range(len(allpred)):
		if allpred[a] == 1:
			allpred[a] = cons_pred[idx]
			idx += 1

	print(set(allpred))

	
	f_mes = f1_score(multtestY,array(allpred),average='weighted')
	a_mes = accuracy_score(multtestY,array(allpred))
	p_mes = precision_score(multtestY,array(allpred),average='weighted')
	r_mes = recall_score(multtestY,array(allpred),average='weighted')

	afile.write("Accuracy: "+str(a_mes)+'\n')
	afile.write("Precision: "+str(p_mes)+'\n')
	afile.write("Recall: "+str(r_mes)+'\n')
	afile.write("F-Score: "+str(f_mes)+'\n')


	'''history_dict = history.history
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
	val_accuracies.append(sum(val_acc) / len(val_acc))'''


#OVERALL AVG MODEL ACCURACIES
'''kfold = range(1, len(train_accuracies) + 1)
plt.plot(kfold, train_accuracies, 'bo', label='Training acc')
plt.plot(kfold, val_accuracies, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('K-Fold')
plt.ylabel('Accuracy')
plt.legend()

plt.show()'''



