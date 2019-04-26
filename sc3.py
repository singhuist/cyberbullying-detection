from numpy import array, asarray, zeros
import tensorflow as tf
import data_preprocess, embeddings
import nn_models, multi_models, awekar_models
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import f1_score
import math
import matplotlib.pyplot as plt


print("Loading Data ...")

data_source = 'data/twitter_data.pkl'

print("preprocessing Data ...")

data, classification = data_preprocess.pickleData(data_source) #bi-class data
mult_data, mult_class = data_preprocess.pickleData_multi(data_source) #multi-class data
#bully_data, bully_class = data_preprocess.pickleData_multi(data_source)
del mult_data

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
shuffdata = ShuffleSplit(n_splits=25, test_size=0.2, train_size=0.8)

train_accuracies = [] #stores the average of different cross validations
val_accuracies = [] #stores the average of different cross validations
train_losses = [] #stores the average of different cross validations
val_losses = [] #stores the average of different cross validations

allres = 'results/staggered/sc3.txt'
grures = 'results/staggered/gru.txt'
lstmres = 'results/staggered/lstm.txt'
blstmres = 'results/staggered/blstm.txt'

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

	
	print("Creating and Training GRU Bi-Classification Model ...")
	bi_history, gru_bi_prediction = nn_models.GatedRecurrentUnit(trainX,trainY,valX,valY,testX,testY,e)
	
	print("Creating and Training LSTM Bi-Classification Model ...")
	bi_history, lstm_bi_prediction = awekar_models.lstm(trainX,trainY,valX,valY,testX,testY,e)

	print("Creating and Training Bi-Classification Model ...")
	bi_history, bl_bi_prediction = awekar_models.blstm(trainX,trainY,valX,valY,testX,testY,e)


	### INSERT CONSENSUS ###

	cons_dict = []
	for g,l,b in zip(gru_bi_prediction,lstm_bi_prediction,bl_bi_prediction):

		cons = {}
		cons[0] = 0
		cons[1] = 0

		if g == 0:
			cons[0] += 1
		elif g == 1:
			cons[1] += 1

		if l == 0:
			cons[0] += 1
		elif l == 1:
			cons[1] += 1

		if b == 0:
			cons[0] += 1
		elif b == 1:
			cons[1] += 1


		cons_dict.append(cons)

	#print(cons_dict[:20])

	cons_pred = []
	for x in range(len(cons_dict)):
		d = cons_dict[x]
		if d[0]>d[1]:
			cons_pred.append(0)
		else:
			cons_pred.append(1)

	#print(cons_pred[:20])


	### FINAL CLASSIFICATION GRU MODEL ###

	testX_mult = []
	testY_mult = []

	for p in range(len(cons_pred)):
		if cons_pred[p] == 1:
			testX_mult.append(testX[p])
			testY_mult.append(multtestY[p])

	print("Obtained multi-class training data")

	testX_mult = array(testX_mult)
	testY_mult = array(testY_mult)

	print("Creating and Training Multi-Classification Models Now ...")
	history, g_mult_prediction = multi_models.gru_multi(trainX_mult,trainY_mult,valX_mult,valY_mult,testX_mult,testY_mult,e)
	history, l_mult_prediction = multi_models.lstm_multi(trainX_mult,trainY_mult,valX_mult,valY_mult,testX_mult,testY_mult,e)
	history, b_mult_prediction = multi_models.blstm_multi(trainX_mult,trainY_mult,valX_mult,valY_mult,testX_mult,testY_mult,e)

	f_mes = f1_score(testY_mult,g_mult_prediction,average='weighted')
	#gfile.write("F-Score: "+str(f_mes)+'\n')
	afile.write("GRU F-Score: "+str(f_mes)+'\n')

	f_mes = f1_score(testY_mult,l_mult_prediction,average='weighted')
	print(f_mes)
	#lfile.write("F-Score: "+str(f_mes)+'\n')
	afile.write("LSTM F-Score: "+str(f_mes)+'\n')

	f_mes = f1_score(testY_mult,b_mult_prediction,average='weighted')
	#bfile.write("F-Score: "+str(f_mes)+'\n')
	afile.write("BLSTM F-Score: "+str(f_mes)+'\n')


	#### CONSENSUS AGAIN ####

	cons_mult = []
	for g,l,b in zip(g_mult_prediction,l_mult_prediction,b_mult_prediction):
		cons = {}
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


		cons_mult.append(cons)

	#print(cons_dict)

	cons_pred_mult = []
	for x in range(len(cons_mult)):
		d = cons_mult[x]
		if d[1]>d[2]:
			cons_pred_mult.append(1)
		else:
			cons_pred_mult.append(2)


	allpred = cons_pred

	idx = 0
	for a in range(len(allpred)):
		if allpred[a] == 1:
			allpred[a] = cons_pred_mult[idx]
			idx += 1

	print(set(allpred))


	f_mes = f1_score(multtestY,allpred,average='weighted')

	afile.write("F-Score ALL: "+str(f_mes)+'\n')
	afile.write('\n')


	



