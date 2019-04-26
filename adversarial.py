import tensorflow as tf
from tensorflow import keras as K
import numpy as np
import argparse
#from progressbar import ProgressBar
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Network:
    def __init__(self, session, dict_weight, dropout=0.2, lstm_units=1024, dense_units=30):
        self.sess = session
        K.backend.set_session(self.sess)
        #defining layers
        dict_shape = dict_weight.shape
        self.emb = K.layers.Embedding(dict_shape[0], dict_shape[1], weights=[dict_weight], trainable=False, name='embedding')
        self.drop = K.layers.Dropout(rate=dropout, seed=91, name='dropout')
        self.lstm = K.layers.LSTM(lstm_units, stateful=False, return_sequences=False, name='lstm')
        self.dense = K.layers.Dense(dense_units, activation='relu', name='dense')
        self.p = K.layers.Dense(1, activation='sigmoid', name='p')
        #defining optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)
        # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)

    def __call__(self, batch, perturbation=None):
        embedding = self.emb(batch) 
        drop = self.drop(embedding)
        if (perturbation is not None):
            drop += perturbation
        lstm = self.lstm(drop)
        dense = self.dense(lstm)
        return self.p(dense), embedding
    
    def get_minibatch(self, x, y, ul, batch_shape=(64, 400)):
        #x = K.preprocessing.sequence.pad_sequences(x, maxlen=batch_shape[1])
        permutations = np.random.permutation( len(y) )
        ul_permutations = None
        len_ratio = None
        if (ul is not None):
            #ul = K.preprocessing.sequence.pad_sequences(ul, maxlen=batch_shape[1])
            ul_permutations = np.random.permutation( len(ul) )
            len_ratio = len(ul)/len(y)
        for s in range(0, len(y), batch_shape[0]):
            perm = permutations[s:s+batch_shape[0]]
            minibatch = {'x': x[perm], 'y': y[perm]}
            if (ul is not None):
                ul_perm = ul_permutations[int(np.floor(len_ratio*s)):int(np.floor(len_ratio*(s+batch_shape[0])))]
                minibatch.update( {'ul': np.concatenate((ul[ul_perm], x[perm]), axis=0)} )
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
    
    def get_v_adv_loss(self, ul_batch, p_mult, power_iterations=1):
        bernoulli = tf.distributions.Bernoulli
        prob, emb = self(ul_batch)
        prob = tf.clip_by_value(prob, 1e-7, 1.-1e-7)
        prob_dist = bernoulli(probs=prob)
        #generate virtual adversarial perturbation
        d = tf.random_uniform(shape=tf.shape(emb), dtype=tf.float32)
        for _ in range( power_iterations ):
            d = (0.02) * tf.nn.l2_normalize(d, dim=1)
            p_prob = tf.clip_by_value(self(ul_batch, d)[0], 1e-7, 1.-1e-7)
            kl = tf.distributions.kl_divergence(prob_dist, bernoulli(probs=p_prob), allow_nan_stats=False)
            gradient = tf.gradients(kl, [d], aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)[0]
            d = tf.stop_gradient(gradient)
        d = p_mult * tf.nn.l2_normalize(d, dim=1)
        tf.stop_gradient(prob)
        #virtual adversarial loss
        p_prob = tf.clip_by_value(self(ul_batch, d)[0], 1e-7, 1.-1e-7)
        v_adv_loss = tf.distributions.kl_divergence(prob_dist, bernoulli(probs=p_prob), allow_nan_stats=False)
        return tf.reduce_mean( v_adv_loss )

    def validation(self, f, x, y, batch_shape=(64, 400)):
        print( 'Validation...' )
        
        labels = tf.placeholder(tf.float32, shape=(None, 1), name='validation_labels')
        batch = tf.placeholder(tf.float32, shape=(None, batch_shape[1]), name='validation_batch')

        accuracy = tf.reduce_mean( K.metrics.binary_accuracy(labels, self(batch)[0]) )
        
        accuracies = list()
        minibatch = self.get_minibatch(x, y, ul=None, batch_shape=batch_shape)
        for val_batch in minibatch:
            fd = {batch: val_batch['x'], labels: val_batch['y'], K.backend.learning_phase(): 0} #test mode
            accuracies.append( self.sess.run(accuracy, feed_dict=fd) )
        
        print( "Average accuracy on validation is {:.3f}".format(np.asarray(accuracies).mean()) )
        f.write("Average Val Acc: "+str(np.asarray(accuracies).mean())+"\n")

    
    def train(self, f, xtrain, ytrain, xval, yval, batch_shape=(64, 400), epochs=10, loss_type='none', p_mult=0.02, init=None, save=None):
        
        print( 'Training...' )
        
        ultrain = xtrain if (loss_type == 'v_adv') else None
        # defining validation set
        print( '{} elements in validation set'.format(len(yval)) )
        # ---
        yval = np.reshape(yval, newshape=(yval.shape[0], 1))
        ytrain = np.reshape(ytrain, newshape=(ytrain.shape[0], 1))
        
        labels = tf.placeholder(tf.float32, shape=(None, 1), name='train_labels')
        batch = tf.placeholder(tf.float32, shape=(None, batch_shape[1]), name='train_batch')
        ul_batch = tf.placeholder(tf.float32, shape=(None, batch_shape[1]), name='ul_batch')
        
        accuracy = tf.reduce_mean( K.metrics.binary_accuracy(labels, self(batch)[0]) )
        loss, emb = self.get_loss(batch, labels)
        if (loss_type == 'adv'):
            loss += self.get_adv_loss(batch, labels, loss, emb, p_mult)
        elif (loss_type == 'v_adv'):
            loss += self.get_v_adv_loss(ul_batch, p_mult)

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
        list_ratio = (len(ultrain)/len(ytrain)) if (ultrain is not None) else None
        for epoch in range(epochs):
            losses = list()
            accuracies = list()
            validation = list()
            
            #bar = ProgressBar(max_value=np.floor(len(ytrain)/batch_shape[0]).astype('i'))
            minibatch = enumerate(self.get_minibatch(xtrain, ytrain, ultrain, batch_shape=batch_shape))
            for i, train_batch in minibatch:
                fd = {batch: train_batch['x'], labels: train_batch['y'], K.backend.learning_phase(): 1} #training mode
                if (loss_type == 'v_adv'):
                    fd.update( {ul_batch: train_batch['ul']} )
                
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

        accuracy = tf.reduce_mean( K.metrics.binary_accuracy(labels, self(batch)[0]) )
        
        accuracies = list()
        #bar = ProgressBar(max_value=np.floor(len(ytest)/batch_shape[0]).astype('i'))
        minibatch = enumerate(self.get_minibatch(xtest, ytest, ul=None, batch_shape=batch_shape))
        for i, test_batch in minibatch:
            fd = {batch: test_batch['x'], labels: test_batch['y'], K.backend.learning_phase(): 0} #test mode
            accuracies.append( self.sess.run(accuracy, feed_dict=fd) )
            
            #bar.update(i)
        
        print( "\nAverage accuracy is {:.3f}".format(np.asarray(accuracies).mean()) )
        f.write("Average Test Acc: "+str(np.asarray(accuracies).mean())+"\n")
        f.write("___________________________________________________________"+"\n")


def main(xtrain, ytrain, xval, yval, xtest, ytest, emb_mat, n_epochs, n_ex, ex_len, lt, pm):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    embedding_weights = emb_mat

    resfile = 'results/adversarial/vadv.txt'
    f = open(resfile, 'w')

    net = Network(session, embedding_weights)
    net.train(f,xtrain, ytrain, xval, yval, batch_shape=(n_ex, ex_len), epochs=n_epochs, loss_type=lt, p_mult=pm, init=None, save=None)
    net.test(f,xtest, ytest, batch_shape=(n_ex, ex_len))
    
    K.backend.clear_session()

