import tensorflow as tf
from tensorflow import keras
import tflearn
import numpy as np

def lstm(Xtrain,Ytrain,XVal,YVal,testdata,testlabel,embedlayer):
    # K.clear_session()
    model = keras.Sequential()
    model.add(embedlayer)
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.LSTM(50)) #size of embedding
    model.add(keras.layers.Dropout(0.50))
    model.add(keras.layers.Dense(2, activation='softmax')) #1 must be no. of classes
    #model.add(keras.layers.Dense(1, activation='softmax')) #1 must be no. of classes
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(Xtrain, Ytrain, epochs=50, batch_size=250, validation_data=(XVal,YVal), verbose=1)

    #prediction = model.predict(testdata) #output of the model
    prediction = np.argmax(model.predict(testdata),axis=1) #output of the model
    print (model.summary())
    return history, prediction


def blstm(Xtrain,Ytrain,XVal,YVal,testdata,testlabel,embedlayer):
    # K.clear_session()
    model = keras.Sequential()
    model.add(embedlayer)
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(50))) #size of embedding
    model.add(keras.layers.Dropout(0.50))
    model.add(keras.layers.Dense(2, activation='softmax')) #1 must be no. of classes
    #model.add(keras.layers.Dense(1, activation='softmax')) #1 must be no. of classes
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(Xtrain, Ytrain, epochs=50, batch_size=250, validation_data=(XVal,YVal), verbose=1)

    #prediction = model.predict(testdata) #output of the model
    prediction = np.argmax(model.predict(testdata),axis=1) #output of the model
    print (model.summary())
    return history, prediction

