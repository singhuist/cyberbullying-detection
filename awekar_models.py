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
    history = model.fit(Xtrain, Ytrain, epochs=2, batch_size=500, validation_data=(XVal,YVal), verbose=1)

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
    history = model.fit(Xtrain, Ytrain, epochs=2, batch_size=500, validation_data=(XVal,YVal), verbose=1)

    #prediction = model.predict(testdata) #output of the model
    prediction = np.argmax(model.predict(testdata),axis=1) #output of the model
    print (model.summary())
    return history, prediction


"""def cnn(Xtrain, Ytrain, XVal, YVal, testdata, testlabel, embedlayer):

    model = keras.Sequential()
    model.add(embedlayer)
    model.add(keras.layers.Dropout(0.25))

    branch1 = keras.Sequential()
    branch1.add(keras.layers.Conv1D(50, 3, padding='valid', activation='relu', kernel_regularizer=keras.regularizers.l2))

    branch2 = keras.Sequential()
    branch2.add(keras.layers.Conv1D(50, 4, padding='valid', activation='relu', kernel_regularizer=keras.regularizers.l2))

    branch3 = keras.Sequential()
    branch3.add(keras.layers.Conv1D(50, 5, padding='valid', activation='relu', kernel_regularizer=keras.regularizers.l2))

    merged = keras.layers.concatenate([branch1, branch2, branch3], axis=1)

    #model.add(keras.layers.Conv1D(50, 3, padding='valid', activation='relu', kernel_regularizer=keras.regularizers.l2))
    model.add(keras.backend.expand_dims(merged))
    #model.add(keras.backend.expand_dims())
    model.add(keras.layers.GlobalMaxPooling1D())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(2, activation='softmax'))
    ##regression??
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        

    history = model.fit(Xtrain, Ytrain, epochs=2, batch_size=500, validation_data=(XVal,YVal), verbose=1)

    prediction = np.argmax(model.predict(testdata),axis=1) #output of the model
    print (model.summary())
    return history, prediction"""

def cnn(Xtrain, Ytrain, XVal, YVal, testdata, testlabel, embedlayer, embedding_matrix):

    inputs = tf.convert_to_tensor(embedding_matrix, dtype=tf.float32)

    x = keras.layers.Dropout(0.25)(inputs)

    b1 = keras.layers.Conv1D(50, 3, padding='valid', activation='relu', kernel_regularizer=keras.regularizers.l2)(x)
    b2 = keras.layers.Conv1D(50, 4, padding='valid', activation='relu', kernel_regularizer=keras.regularizers.l2)(x)
    b3 = keras.layers.Conv1D(50, 5, padding='valid', activation='relu', kernel_regularizer=keras.regularizers.l2)(x)

    merged = keras.layers.concatenate([b1,b2,b3], axis=1)

    predictions = keras.backend.expand_dims()(merged)
    predictions = keras.layers.GlobalMaxPooling1D()(predictions)
    predictions = keras.layers.Dropout(0.5)(predictions)
    predictions = Dense(2, activation='softmax')(predictions)
    predictions = Dense(1, activation='softmax')(predictions)

    model = keras.models.Model(inputs=inputs, outputs=predictions)

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        

    history = model.fit(Xtrain, Ytrain, epochs=2, batch_size=500, validation_data=(XVal,YVal), verbose=1)

    prediction = np.argmax(model.predict(testdata),axis=1) #output of the model
    print (model.summary())
    return history, prediction




'''
def cnn(inp_dim, vocab_size, embed_size, num_classes, learn_rate):
    tf.reset_default_graph()
    network = input_data(shape=[None, inp_dim], name='input')
    network = tflearn.embedding(network, input_dim=vocab_size, output_dim=embed_size, name="EmbeddingLayer")
    network = dropout(network, 0.25)
    branch1 = conv_1d(network, embed_size, 3, padding='valid', activation='relu', regularizer="L2", name="layer_1")
    branch2 = conv_1d(network, embed_size, 4, padding='valid', activation='relu', regularizer="L2", name="layer_2")
    branch3 = conv_1d(network, embed_size, 5, padding='valid', activation='relu', regularizer="L2", name="layer_3")
    #network = merge([branch1, branch2, branch3], mode='concat', axis=1)
    network = Concatenate([branch1, branch2, branch3], axis=1)
    network = tf.expand_dims(network, 2)
    network = global_max_pool(network)
    network = dropout(network, 0.50)
    network = fully_connected(network, num_classes, activation='softmax', name="fc")
    network = regression(network, optimizer='adam', learning_rate=learn_rate,
                         loss='categorical_crossentropy', name='target')
    
    model = tflearn.DNN(network, tensorboard_verbose=0)
    return model'''

'''
def blstm_atten(Xtrain,Ytrain,XVal,YVal,testdata,testlabel,embedlayer):
    # K.clear_session()
    model = keras.Sequential()
    model.add(embedlayer)
    model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(50))) #size of embedding
    model.add(keras.layers.AttLayer())
    model.add(keras.layers.Dropout(0.50))
    model.add(keras.layers.Dense(2, activation='softmax')) #1 must be no. of classes
    #model.add(keras.layers.Dense(1, activation='softmax')) #1 must be no. of classes
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(Xtrain, Ytrain, epochs=10, batch_size=500, validation_data=(XVal,YVal), verbose=1)

    #prediction = model.predict(testdata) #output of the model
    prediction = np.argmax(model.predict(testdata),axis=1) #output of the model
    print (model.summary())
    return history, prediction
'''
