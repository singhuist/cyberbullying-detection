import tensorflow as tf
from tensorflow import keras
import numpy as np

vocab_size = 100000

def GatedRecurrentUnit(Xtrain,Ytrain,XVal,YVal,testdata,testlabel,embedlayer):
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
    model.add(keras.layers.Dense(2, activation="softmax"))
    
    #print(model.summary())

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # TRAIN THE MODEL
    history = model.fit(Xtrain, Ytrain, epochs=50, batch_size=250, validation_data=(XVal,YVal), verbose=1)

    #prediction = model.predict_class(testdata) #output of the model
    prediction = np.argmax(model.predict(testdata),axis=1) #output of the model

    # EVALUATE THE MODEL
    results = model.evaluate(testdata, testlabel)

    print(results)

    return history, prediction



