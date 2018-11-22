from numpy import array
from numpy import asarray
from numpy import zeros
import tensorflow as tf
import data_preprocess
import embeddings


data_source = 'data/hatebase_labeled_data.csv'
data, classification, severity = data_preprocess.parseFile(data_source)
docs = data[:10]
#classification = classification[:10]
labels = array(classification[:10])
sequences, padlength, wordIndex = embeddings.simpleEncode(data)

# prepare tokenizer
t = tf.keras.preprocessing.text.Tokenizer()
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1
# integer encode the documents
encoded_docs = t.texts_to_sequences(docs)
print(encoded_docs)
# pad documents to a max length of 4 words
padded_docs = tf.keras.preprocessing.sequence.pad_sequences(encoded_docs, maxlen=padlength, padding='post')
print(padded_docs)

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

# define model
model = tf.keras.models.Sequential()
e = tf.keras.layers.Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=padlength, trainable=False)
model.add(e)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())
# fit the model
model.fit(padded_docs, labels, epochs=50, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))