#Author: Soumya Singh

import data_preprocess
import nn_models
import embeddings
from sklearn.model_selection import ShuffleSplit
import math
import matplotlib.pyplot as plt
from numpy import array

# GET CYBERBULLYING DATA
data_source = 'data/hatebase_labeled_data.csv'
data, classification, severity = data_preprocess.parseFile(data_source)

docs = data[:10]
labels = array(classification[:10])

# ENCODE DATA TO SIMPLE ENCODING
sequences, padlength, wordIndex = embeddings.simpleEncode(data)
code_data, embed_layer = embeddings.gloveEmbed(docs,padlength)

# BUILDING THE NEURAL NETWORK MODEL
# REFERENCE: https://www.tensorflow.org/tutorials/keras/basic_text_classification#top_of_page

train_accuracies = [] #stores the average of different cross validations
val_accuracies = [] #stores the average of different cross validations

train_losses = [] #stores the average of different cross validations
val_losses = [] #stores the average of different cross validations

# SHUFFLE DATA AND SET UP CROSS-VALIDATION
#shuffdata = ShuffleSplit(n_splits=5, test_size=0.25, train_size=0.75)

'''for train_index, test_index in shuffdata.split(code_data,classification):

    train_data = []
    train_label = []

    test_data = []
    test_label = []

    for tr in train_index:
        train_data.append(code_data[tr])
        train_label.append(classification[tr])

    for ts in test_index:
        test_data.append(code_data[ts])
        test_label.append(classification[ts])

    train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=0, padding='post', maxlen=20)
    test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=0, padding='post', maxlen=20)

    print("Train Data: ",train_data)
    print("Test Data: ",test_data)'''

train_data = data[:8]
train_label = classification[:8]

test_data = data[8:]
test_label = classification[8:]

# VALIDATION DATA
valsize = len(train_data) - math.floor(0.2 * len(train_data))
x_val = train_data[valsize:len(train_data)]
partial_x_train = train_data[:valsize]
y_val = train_label[valsize:len(train_data)]
partial_y_train = train_label[:valsize]

# BUILD THE MODEL AND SAVE THE RESULTS HISTORY
history = nn_models.MultiLayerPerceptron(partial_x_train,partial_y_train,x_val,y_val,test_data,test_label, embed_layer)
##history = nn_models.GatedRecurrentUnit(partial_x_train, partial_y_train, x_val, y_val, test_data, test_label, embed_layer)

history_dict = history.history
history_dict.keys()

# PLOT GRAPHS

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

'''epochs = range(1, len(acc) + 1)


   # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    plt.clf()'''  # clear figure

acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

'''plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()'''

train_accuracies.append(sum(acc_values) / len(acc_values))
val_accuracies.append(sum(val_acc_values) / len(val_acc_values))

kfold = range(1, len(train_accuracies) + 1)
plt.plot(kfold, train_accuracies, 'bo', label='Training acc')
plt.plot(kfold, val_accuracies, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('K-Fold')
plt.ylabel('Accuracy')
plt.legend()

plt.show()



