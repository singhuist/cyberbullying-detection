##CODE ADOPTED FROM: https://github.com/enry12/adversarial_training_methods

import tensorflow as tf
from tensorflow import keras as K
import numpy as np
import argparse
#from progressbar import ProgressBar
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn as sk

class Network:
    def __init__(self, session, dict_weight, dropout=0.2, gru_units=1024, dense_units=50):
        self.sess = session
        K.backend.set_session(self.sess)
        #defining layers
        dict_shape = dict_weight.shape
        self.emb = K.layers.Embedding(dict_shape[0], dict_shape[1], weights=[dict_weight], trainable=True, name='embedding')
        self.drop = K.layers.Dropout(rate=dropout, name='dropout')
        self.gru = K.layers.GRU(gru_units, name='gru')
        self.drop = K.layers.Dropout(rate=dropout, name='dropout')
        self.dense = K.layers.Dense(dense_units, activation='softmax', name='dense')
        self.p = K.layers.Dense(1, activation='sigmoid', name='p')
        #defining optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)
        # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)

    def __call__(self, batch, perturbation=None):
        embedding = self.emb(batch) 
        drop = self.drop(embedding)
        if (perturbation is not None):
            drop += perturbation
        gru = self.gru(drop)
        dense = self.dense(gru)
        return self.p(dense), embedding
    
    def get_minibatch(self, x, y, batch_shape=(64, 400)):
        #x = K.preprocessing.sequence.pad_sequences(x, maxlen=batch_shape[1])
        permutations = np.random.permutation( len(y) )
        len_ratio = None
        for s in range(0, len(y), batch_shape[0]):
            perm = permutations[s:s+batch_shape[0]]
            minibatch = {'x': x[perm], 'y': y[perm]}
            yield minibatch
            
    def get_loss(self, batch, labels):
        pred, emb = self(batch)
        loss = K.losses.binary_crossentropy(labels, pred)
        return tf.reduce_mean( loss ), emb
    
    def get_adv_loss(self, batch, labels, loss, emb, p_mult):
        gradient = tf.gradients(loss, emb, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)[0]
        p_adv = p_mult * tf.nn.l2_normalize(tf.stop_gradient(gradient), dim=1)
        adv_loss = K.losses.binary_crossentropy(labels, self(batch, p_adv)[0])
        return tf.reduce_mean( adv_loss )
    
    def validation(self, f, x, y, batch_shape=(64, 400)):
        print( 'Validation...' )
        
        labels = tf.placeholder(tf.float32, shape=(None, 1), name='validation_labels')
        batch = tf.placeholder(tf.float32, shape=(None, batch_shape[1]), name='validation_batch')

        accuracy = tf.reduce_mean( K.metrics.binary_accuracy(labels, self(batch)[0]) )
        
        accuracies = list()
        minibatch = self.get_minibatch(x, y, batch_shape=batch_shape)
        for val_batch in minibatch:
            fd = {batch: val_batch['x'], labels: val_batch['y'], K.backend.learning_phase(): 0} #test mode
            accuracies.append( self.sess.run(accuracy, feed_dict=fd) )
        
        print( "Average accuracy on validation is {:.3f}".format(np.asarray(accuracies).mean()) )
        f.write("Average Val Acc: "+str(np.asarray(accuracies).mean())+"\n")

    
    def train(self, f, xtrain, ytrain, xval, yval, batch_shape=(64, 400), epochs=10, loss_type='none', p_mult=0.02, init=None, save=None):

        print( 'Training...' )
        
        # defining validation set
        print( '{} elements in validation set'.format(len(yval)) )
        # ---
        yval = np.reshape(yval, newshape=(yval.shape[0], 1))
        ytrain = np.reshape(ytrain, newshape=(ytrain.shape[0], 1))
        
        labels = tf.placeholder(tf.float32, shape=(None, 1), name='train_labels')
        batch = tf.placeholder(tf.float32, shape=(None, batch_shape[1]), name='train_batch')
        
        accuracy = tf.reduce_mean( K.metrics.binary_accuracy(labels, self(batch)[0]) )
        loss, emb = self.get_loss(batch, labels)
        if (loss_type == 'adv'):
            loss += self.get_adv_loss(batch, labels, loss, emb, p_mult)
        

        opt = self.optimizer.minimize( loss )
        #initializing parameters
        if (init is None):
            self.sess.run( [var.initializer for var in tf.global_variables() if not('embedding' in var.name)] )
            print( 'Random initialization' )
        else:
            saver = tf.train.Saver()
            saver.restore(self.sess, init)
            print( 'Restored value' )
        
        _losses = list()
        _accuracies = list()
       
        for epoch in range(epochs):
            losses = list()
            accuracies = list()
            validation = list()
            
            #bar = ProgressBar(max_value=np.floor(len(ytrain)/batch_shape[0]).astype('i'))
            minibatch = enumerate(self.get_minibatch(xtrain, ytrain, batch_shape=batch_shape))
            for i, train_batch in minibatch:
                fd = {batch: train_batch['x'], labels: train_batch['y'], K.backend.learning_phase(): 1} #training mode
                
                _, acc_val, loss_val = self.sess.run([opt, accuracy, loss], feed_dict=fd)
                
                accuracies.append( acc_val )
                losses.append( loss_val )
                #bar.update(i)
            
            #saving accuracies and losses
            _accuracies.append( accuracies )
            _losses.append(losses)
            
            log_msg = "\nEpoch {} of {} -- average accuracy is {:.3f} (train) -- average loss is {:.3f}"
            print( log_msg.format(epoch+1, epochs, np.asarray(accuracies).mean(), np.asarray(losses).mean()) )
            
            # validation log
            self.validation(f, xval, yval, batch_shape=batch_shape)
            
            #saving model
            if (save is not None) and (epoch == (epochs-1)):
                saver = tf.train.Saver()
                saver.save(self.sess, save)
                print( 'model saved' )
        
        '''#plotting value
        #plt.plot([l for loss in _losses for l in loss], color='magenta', linestyle='dashed', marker='s', linewidth=1)
        plt.plot([np.asarray(l).mean() for l in _losses], color='red', linestyle='solid', marker='o', linewidth=2)
        #plt.plot([a for acc in _accuracies for a in acc], color='cyan', linestyle='dashed', marker='s', linewidth=1)
        plt.plot([np.asarray(a).mean() for a in _accuracies], color='blue', linestyle='solid', marker='o', linewidth=2)
        plt.savefig('./train_{}_e{}_m{}_l{}.png'.format(loss_type, epochs, batch_shape[0], batch_shape[1]))'''
        
    def test(self, f, xtest, ytest, batch_shape=(64, 400)):
        print( 'Test...' )
        ytest = np.reshape(ytest, newshape=(ytest.shape[0], 1))
        
        labels = tf.placeholder(tf.float32, shape=(None, 1), name='test_labels')
        batch = tf.placeholder(tf.float32, shape=(None, batch_shape[1]), name='test_batch')

        #print("labels:",labels[:10])
        #print("pred:",self(batch)[0][:10])

        accuracy = tf.reduce_mean( K.metrics.binary_accuracy(labels, self(batch)[0]) )
        f_score = tf.reduce_mean( tf.contrib.metrics.f1_score(labels, self(batch)[0], weights=None, num_thresholds=2, metrics_collections=None, updates_collections=None, name=None) )
        #f_score = tf.reduce_mean( sk.metrics.f1_score(labels, self(batch)[0] ))
        
        accuracies = list()
        f_scores = list()
        #bar = ProgressBar(max_value=np.floor(len(ytest)/batch_shape[0]).astype('i'))
        minibatch = enumerate(self.get_minibatch(xtest, ytest, batch_shape=batch_shape))
        for i, test_batch in minibatch:
            fd = {batch: test_batch['x'], labels: test_batch['y'], K.backend.learning_phase(): 0} #test modesess.run(tf.local_variables_initializer())
            accuracies.append( self.sess.run(accuracy, feed_dict=fd) )
            self.sess.run(tf.local_variables_initializer())
            f_scores.append(self.sess.run(f_score, feed_dict=fd))
            
            #bar.update(i)
        
        print( "\nAverage accuracy is {:.3f}".format(np.asarray(accuracies).mean()) )
        print( "\nAverage f-score is {:.3f}".format(np.asarray(f_scores).mean()))
        f.write("Average Test Acc: "+str(np.asarray(accuracies).mean())+"\n")
        f.write("Average f-Score: "+str(np.asarray(f_scores).mean())+"\n")
        f.write("___________________________________________________________"+"\n")



def main(xtrain, ytrain, xval, yval, xtest, ytest, emb_mat, n_epochs, n_ex, ex_len, lt, pm):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    embedding_weights = emb_mat

    resfile = 'results/adversarial/gadv.txt'
    f = open(resfile, 'w')

    net = Network(session, embedding_weights)
    net.train(f, xtrain, ytrain, xval, yval, batch_shape=(n_ex, ex_len), epochs=n_epochs, loss_type=lt, p_mult=pm, init=None, save=None)
    net.test(f, xtest, ytest, batch_shape=(n_ex, ex_len))
    
    K.backend.clear_session()
