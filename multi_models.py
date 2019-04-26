import tensorflow as tf
from tensorflow import keras
import numpy as np

vocab_size = 100000


def gru_multi(Xtrain,Ytrain,XVal,YVal,testdata,testlabel,embedlayer):
    '''
    Creates a gated recurrent unit model
    :param Xtrain: Training Data Sentences
    :param Ytrain: Training Data Labels
    :param XVal: Validation Data Sentences
    :param YVal: Validation Data Labels
    :param testdata: Test Data Sentences
    :param testlabel: Test Data Labels
    :return:
    '''

    model = keras.Sequential()
    model.add(embedlayer)
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.GRU(units=50))
    model.add(keras.layers.Dropout(0.50))
    model.add(keras.layers.Dense(3, activation="softmax"))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # TRAIN THE MODEL
    history = model.fit(Xtrain, Ytrain, epochs=15, batch_size=100, validation_data=(XVal,YVal), verbose=1)

    print("model fitted....making predictions")

    print(testlabel.shape)
    print(len(testlabel))
    print(testlabel[:10])

    prediction = np.argmax(model.predict(testdata),axis=1) #output of the model

    print("predictions made .... now evaluating")

    # EVALUATE THE MODEL
    results = model.evaluate(testdata, testlabel)

    print(results)

    return history, prediction


def lstm_multi(Xtrain,Ytrain,XVal,YVal,testdata,testlabel,embedlayer):
    # K.clear_session()
    model = keras.Sequential()
    model.add(embedlayer)
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.LSTM(50)) #size of embedding
    model.add(keras.layers.Dropout(0.50))
    model.add(keras.layers.Dense(3, activation='softmax')) #1 must be no. of classes
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(Xtrain, Ytrain, epochs=15, batch_size=100, validation_data=(XVal,YVal), verbose=1)

    prediction = np.argmax(model.predict(testdata),axis=1) #output of the model
    
     # EVALUATE THE MODEL
    results = model.evaluate(testdata, testlabel)

    print(results)

    return history, prediction


def blstm_multi(Xtrain,Ytrain,XVal,YVal,testdata,testlabel,embedlayer):
    # K.clear_session()
    model = keras.Sequential()
    model.add(embedlayer)
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(50))) #size of embedding
    model.add(keras.layers.Dropout(0.50))
    model.add(keras.layers.Dense(3, activation='softmax')) #1 must be no. of classes
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(Xtrain, Ytrain, epochs=15, batch_size=100, validation_data=(XVal,YVal), verbose=1)
    prediction = np.argmax(model.predict(testdata),axis=1) #output of the model

     # EVALUATE THE MODEL
    results = model.evaluate(testdata, testlabel)

    print(results)

    return history, prediction
