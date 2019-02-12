import tensorflow as tf
from tensorflow import keras
import numpy as np

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
    return model

def get_model(m_type,inp_dim, vocab_size, embed_size, num_classes, learn_rate):
    if m_type == 'cnn':
        model = cnn(inp_dim, vocab_size, embed_size, num_classes, learn_rate)
    elif m_type == 'lstm':
        model = lstm_keras(inp_dim, vocab_size, embed_size, num_classes, learn_rate)
    elif m_type == "blstm":
        model = blstm(inp_dim)
    elif m_type == "blstm_attention":
        model = blstm_atten(inp_dim, vocab_size, embed_size, num_classes, learn_rate)
    else:
        print ("ERROR: Please specify a correst model")
        return None
    return model'''

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
    history = model.fit(Xtrain, Ytrain, epochs=10, batch_size=500, validation_data=(XVal,YVal), verbose=1)

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
    history = model.fit(Xtrain, Ytrain, epochs=10, batch_size=500, validation_data=(XVal,YVal), verbose=1)

    #prediction = model.predict(testdata) #output of the model
    prediction = np.argmax(model.predict(testdata),axis=1) #output of the model
    print (model.summary())
    return history, prediction

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