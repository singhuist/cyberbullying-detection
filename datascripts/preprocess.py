import csv
import tensorflow as tf
from tensorflow import keras
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.model_selection import ShuffleSplit
import numpy as np
#from gensim.models import Word2Vec
import math

#ques = []
ans1 = []
isbully1 = []
severity1 = []
bully1 = []
ans2 = []
isbully2 = []
severity2 = []
bully2 = []
ans3 = []
isbully3 = []
severity3 = []
bully3 = []

with open('formspring_data.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            if len(row)>=1:
                r = row[0].split('\t')

                if len(r)==14:
                    #print(r)
                    #ques.append(r[2])
                    ans1.append(r[1])
                    isbully1.append(r[5])
                    severity1.append(r[6])
                    bully1.append(r[7])
                    ans2.append(r[2])
                    isbully2.append(r[8])
                    severity2.append(r[9])
                    bully2.append(r[10])
                    ans3.append(r[3])
                    isbully3.append(r[11])
                    severity3.append(r[12])
                    bully3.append(r[13])

                    line_count += 1

ans = ans1+ans2+ans3
isbully = isbully1+isbully2+isbully3
severity = severity1+severity2+severity3
bully = bully1+bully2+bully3

#######################################################################################################
#preprocessing and cleanup

stop_words = set(stopwords.words('english'))
punctuations = set(string.punctuation)
specialsym = set(['~','`','@','$','#','%','^','&','*','``', "''", '..', '...', 'n/a', 'na'])

for a in range(len(ans)):
    ans[a] = ans[a].lower() #convert to lowercase

    word_tokens = word_tokenize(ans[a])
    no_punc = [w for w in word_tokens if not w in punctuations] #remove punctuation
    no_sym = [w for w in no_punc if not w in specialsym] #remove special symbols
    filtered_sentence = [w for w in no_sym if not w in stop_words] #remove stop words

    #ans[a] = ' '.join(filtered_sentence)
    ans[a] = filtered_sentence

    #print(ans[a])

for i in range(len(isbully)):
    if isbully[i] == 'Yes':
        isbully[i] = 1
    else:
        isbully[i] = 0

"""model = Word2Vec(ans, size=100, window=5, min_count=3, workers=3)
vectors = model.wv

print(vectors.similarity('music','face'))
print(vectors.most_similar('book'))
print(vectors['sing'])
print(model['book'])

#del model"""

#vectorising the sentences and words
dictionary = {}
counter = 0
for a in ans:
    for w in range(len(a)):
        word = a[w]
        if a[w] not in dictionary:
            dictionary[word] = counter + 1
            counter = counter + 1
        a[w] = dictionary[word]

### building the ANN model ###
# https://www.tensorflow.org/tutorials/keras/basic_text_classification#top_of_page

#shuffling data and setting up cross-validation
shuffdata = ShuffleSplit(n_splits=5, test_size=0.25, train_size=0.75) #shuffled data

'''train_data = ans[:30000]
train_label = isbully[:30000]

test_data = ans[30000:]
test_label = isbully[30000:]'''


for train_index, test_index in shuffdata.split(ans,isbully):

    train_data = []
    train_label = []

    test_data = []
    test_label = []

    for tr in train_index:
        train_data.append(ans[tr])
        train_label.append(isbully[tr])

    for ts in test_index:
        test_data.append(ans[ts])
        test_label.append(isbully[ts])

    #padding
    train_data = keras.preprocessing.sequence.pad_sequences(train_data,value=0,padding='post',maxlen=100)
    test_data = keras.preprocessing.sequence.pad_sequences(test_data,value=0,padding='post',maxlen=100)

    ## build the model
    # input shape is the vocabulary count used (10,000 words)
    vocab_size = 100000

    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

    #model.summary()

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    ## validation data
    valsize = len(train_data)-math.floor(0.2*len(train_data))
    x_val = train_data[valsize:len(train_data)]
    partial_x_train = train_data[:valsize]

    y_val = train_label[valsize:len(train_data)]
    partial_y_train = train_label[:valsize]

    ##train the model
    history = model.fit(partial_x_train,partial_y_train,epochs=2,batch_size=100,validation_data=(x_val, y_val),verbose=1)

    #evaluate the model
    results = model.evaluate(test_data, test_label)

    print(results)

    history_dict = history.history
    history_dict.keys()

    import matplotlib.pyplot as plt

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    plt.clf()   # clear figure
    acc_values = history_dict['acc']
    val_acc_values = history_dict['val_acc']

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()