import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

vocab_size = 100000

def MultiLayerPerceptron(Xtrain,Ytrain,XVal,YVal,testdata,testlabel):

    '''
    Creates a multilayer perceptron model
    :param Xtrain: Training Data Sentences
    :param Ytrain: Training Data Labels
    :param XVal: Validation Data Sentences
    :param YVal: Validation Data Labels
    :param testdata: Test Data Sentences
    :param testlabel: Test Data Labels
    :return:
    '''

    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
    #model.summary()

    model.compile(optimizer=tf.train.AdamOptimizer(), loss='binary_crossentropy', metrics=['accuracy'])

    # TRAIN THE MODEL
    history = model.fit(Xtrain, Ytrain, epochs=10, batch_size=100, validation_data=(XVal,YVal), verbose=1)

    # EVALUATE THE MODEL
    results = model.evaluate(testdata, testlabel)

    print(results)

    return history



