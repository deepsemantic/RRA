
#!/usr/bin/env python
'''
author: cheng wang, 09.09.2017
'''
from __future__ import print_function
import six.moves.cPickle as pickle

from collections import OrderedDict
import sys
import time

import numpy
numpy.set_printoptions(threshold=numpy.nan)
import theano
import theano.typed_list
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from peentree import *
import pdb

# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed(SEED)

def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)


def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.items():
        tparams[kk].set_value(vv)

def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params

def _p(pp, name):
    return '%s_%s' % (pp, name)


def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params

def grad_clipping(grads, clip_c):
    g2 = 0.
    for g in grads:
        g2 += (g**2).sum()
    new_grads = []
    for g in grads:
        new_grads.append(tensor.switch(g2 > (clip_c**2), g/tensor.sqrt(g2) * clip_c, g))
    grads = new_grads
    return grads

class ParameterInitializator:

    def __init__(self, options):
     
        self.options = options
        
    def global_init_params(self):
        """
        Global (not rra) parameter. 
        """
        options = self.options
        params = OrderedDict()

        weight_filler = FillWeight()
        params['Wemb'] = weight_filler.uniform_weight(options['n_words'], options['dim_hidden']).astype(config.floatX)
        params = self.rra_init_params(options, params, weight_filler, prefix='rra_1')
        params = self.rra_init_params(options, params, weight_filler, prefix='rra_2')
        
        params['W_s'] = weight_filler.uniform_weight(options['dim_hidden'], options['n_words']).astype(config.floatX)
        params['b_s'] = numpy.zeros((options['n_words'],)).astype(config.floatX)
        
        return params
        
    def rra_init_params(self, options, params, weight_filler, prefix='rra'):
        '''
        Init the rra parameter:
        '''
        W = numpy.concatenate([weight_filler.uniform_weight(options['dim_hidden']),
                               weight_filler.uniform_weight(options['dim_hidden']),
                               weight_filler.uniform_weight(options['dim_hidden']),
                               weight_filler.uniform_weight(options['dim_hidden'])], axis=1)
        params[_p(prefix, 'W')] = W
        U = numpy.concatenate([weight_filler.uniform_weight(options['dim_hidden']),
                               weight_filler.uniform_weight(options['dim_hidden']),
                               weight_filler.uniform_weight(options['dim_hidden']),
                               weight_filler.uniform_weight(options['dim_hidden'])], axis=1)
        params[_p(prefix, 'U')] = U
        
        b = numpy.zeros((4 * options['dim_hidden'],))
        params[_p(prefix, 'b')] = b.astype(config.floatX)
        
        W_a = numpy.random.rand(1, options['attention_window']).astype('float32')
        params[_p(prefix, 'W_a')] = W_a

        return params

    def tensor_params(self, params):
        tparams = OrderedDict()
        for kk, pp in params.items():
            tparams[kk] = theano.shared(params[kk], name=kk)
        return tparams
   
    
class FillWeight:
    def uniform_weight(self, n_in, n_out=None, fix=True):
        if n_out is None:
            n_out = n_in
        W = numpy.random.uniform(low=-numpy.sqrt(6. / (n_in + n_out)),
        high=numpy.sqrt(6. / (n_in + n_out)), size=(n_in, n_out))   
        #pdb.set_trace()        
        if fix:    
            W = numpy.random.uniform(low=-0.1, high=0.1, size=(n_in, n_out))
        return W.astype('float32')

    def ortho_weight(self, ndim):
        W = numpy.random.randn(ndim, ndim)
        u, s, v = numpy.linalg.svd(W)
        return u.astype(config.floatX)

class Optimizer:
    '''
    original code from "http://deeplearning.net/tutorial/lstm.html"
    '''
    def sgd(self, lr, tparams, grads, x, y, h1_init, c1_init, h2_init, c2_init, h1_last, c1_last, h2_last, c2_last, cost):
        """ Stochastic Gradient Descent

        :note: A more complicated version of sgd then needed.  This is
            done like that for adadelta and rmsprop.

        """
        # New set of shared variable that will contain the gradient
        # for a mini-batch.
        gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
                   for k, p in tparams.items()]
        gsup = [(gs, g) for gs, g in zip(gshared, grads)]

        # Function that computes gradients for a mini-batch, but do not
        # updates the weights.
        f_grad_shared = theano.function([x, y, h1_init, c1_init, h2_init, c2_init], [h1_last, c1_last, h2_last, c2_last, cost], updates=gsup,
                                        name='sgd_f_grad_shared')

        pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

        # Function that updates the weights from the previously computed
        # gradient.
        f_update = theano.function([lr], [], updates=pup, name='sgd_f_update')

        return f_grad_shared, f_update
    
class Layers:

    def embedding_layer(self, tparams, x_in):
        x_out = tparams['Wemb'][x_in.flatten()]
        return x_out
        
    def mlp_layer(self, tparams, x_in):
        x_out = tensor.dot(x_in, tparams['W_s']) + tparams['b_s']
        return x_out
        
    def rra_layer(self, tparams, x_in, h_0, c_0, options, prefix='rra'):
        
        nsteps = x_in.shape[0]
        attention_window = options['attention_window']

        if x_in.ndim == 3:
            n_samples = x_in.shape[1]
        else:
            n_samples = 1
            
        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n * dim:(n + 1) * dim]
            return _x[:, n * dim:(n + 1) * dim]

        def _rra_unit(x_, h_, c_, h_att):
            '''
            @param x_: x(t) input
            @param h_: h(t-1) previous recurrent hidden state
            @param c_: cell states
            @param h_att: hidden states from h(t-2) to h(t-k)
            @return: h(t), c(t), h_att_up(t)
            '''
           
            preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
            preact += x_

            # gates
            i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_hidden']))
            
            f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_hidden']))
            o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_hidden']))
            c = tensor.tanh(_slice(preact, 3, options['dim_hidden']))

            # cell state
            c = f * c_ + i * c
            
            # attention gate
            W_norm = tparams[_p(prefix, 'W_a')]/tparams[_p(prefix, 'W_a')].sum()
            
            a = tensor.tensordot(W_norm, h_att,1)
            reshaped_a = tensor.reshape(a,(a.shape[1],a.shape[2]))
            
            h = o * tensor.tanh(c+reshaped_a)
            
            # update attention window
            h_att_up=tensor.concatenate([h_att[1:,:], h.dimshuffle('x', 0, 1)], axis=0)
	
            return h, c, h_att_up


        x_in = (tensor.dot(x_in, tparams[_p(prefix, 'W')]) +
                       tparams[_p(prefix, 'b')])
        
        dim_hidden = options['dim_hidden']

        h_att_list =tensor.alloc(numpy_floatX(0.),attention_window,n_samples, options['dim_hidden'])

        [h, c, h_att_list], updates = theano.scan(_rra_unit,
                                    sequences=[x_in],
                                    outputs_info=[h_0, c_0, h_att_list],
                                    name=_p(prefix, 'layers'),
                                    n_steps=nsteps,
                                    truncate_gradient=options['truncate_grad'])
        return h, c


def build_model(tparams, options):
  
    x = tensor.matrix('x', dtype='int64')
    y = tensor.matrix('y', dtype='int64')
    
    h1_init = tensor.matrix('h1_init', dtype=config.floatX) 
    c1_init = tensor.matrix('c1_init', dtype=config.floatX)
     
    h2_init = tensor.matrix('h2_init', dtype=config.floatX) 
    c2_init = tensor.matrix('c2_init', dtype=config.floatX) 

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]
    
    layer = Layers()
    emb = layer.embedding_layer(tparams, x).reshape([n_timesteps, n_samples, options['dim_hidden']])
          
    h_1, c_1 = layer.rra_layer(tparams, emb, h1_init, c1_init, options, prefix='rra_1') 
    h_2, c_2 = layer.rra_layer(tparams, h_1, h2_init, c2_init, options, prefix='rra_2')  
     
    fc = layer.mlp_layer(tparams, h_2)
    
    fc=tensor.reshape(fc, (-1, options['n_words']))
    
    reshaped_s = tensor.nnet.softmax(fc)
    reshaped_y=tensor.flatten(y)   
    
    cost = tensor.nnet.categorical_crossentropy(reshaped_s, reshaped_y).sum()/n_samples

    loss = cost

    return x, y, h1_init, c1_init, h2_init, c2_init, h_1[-1], c_1[-1], h_2[-1], c_2[-1], cost, loss

def pred_ppl(f_ppl, set_x, set_y, batch_size, dim_hidden, n_timesteps):

    total_cost=0
    iters=0
    batch_len = set_x.shape[0]//n_timesteps
    h1_init = numpy.zeros((batch_size, dim_hidden)).astype(numpy.float32)
    c1_init = numpy.zeros((batch_size, dim_hidden)).astype(numpy.float32)
    
    h2_init = numpy.zeros((batch_size, dim_hidden)).astype(numpy.float32)
    c2_init = numpy.zeros((batch_size, dim_hidden)).astype(numpy.float32)
    for batch_index in range(batch_len):
                
        batch_start = batch_index * n_timesteps
        batch_end = (batch_index+1) * n_timesteps

        x = set_x[batch_start:batch_end]
        y = set_y[batch_start:batch_end]

        x=x.astype(numpy.int64)
        y=y.astype(numpy.int64)
        
        (h1_last, c1_last, h2_last, c2_last, cost) = f_ppl(x, y, h1_init, c1_init, h2_init, c2_init)
        
        h1_init = h1_last
        c1_init = c1_last
        
        h2_init = h2_last
        c2_init = c2_last
        
        
        iters += n_timesteps
        total_cost+=cost
    
    ppl = numpy.exp(total_cost/iters)
    return ppl


def train_rra(
    dim_hidden=200,  # hidden units.
    patience=10,  # Number of epoch to wait before early stop if no progress
    max_epochs=500,  # The maximum number of epoch to run
    dispFreq=100,  # Display to stdout the training progress every N updates
    decay_c=0.00025,  # Weight decay 
    lrate=1,  
    step_size=[20, 30, 50, 100], # step size to descrease learning rate
    n_words=10000,  # Vocabulary size
    encoder='rra',  
    model_path='models/',  # path to save model
    model_name='ptb_1127_5', #logs
    validFreq=2000,  # Iterations for validation.
    saveFreq=10000,  # Iterations for save
    n_timesteps =35, # Length of sequence 
    batch_size=20,  # Training batch size
    valid_batch_size=20,  # Val/Test batch size
    attention_window=10, #  attention window size to cover previous states
    clip_c=10., # gradient clipping threshold
    reload_model=None,  
    truncate_grad=-1,
    init_state=True,
    log_path='log/rra.log'
):

    # Model options
    model_options = locals().copy()
    #pdb.set_trace()
    print("model options", model_options)
    logger = open(log_path, "a")
    logger.write("model options:\n")
    if init_state:
       init_state=False
       for kk, vv in model_options.iteritems():
          logger.write("\t"+kk+":\t"+str(vv)+"\n")

      
    print('Loading data') 
    train_x, train_y, test_x, test_y, val_x, val_y = load_dataset(batch_size, n_timesteps)
    

    ydim =n_words 
    model_options['ydim'] = ydim

    print('Building model')

    Parameter = ParameterInitializator(model_options)
    params = Parameter.global_init_params()
    
    tparams = Parameter.tensor_params(params)

    (x, y, h1_init, c1_init, h2_init, c2_init, h1_last, c1_last, h2_last, c2_last, cost,loss) = build_model(tparams, model_options)#, h_, c_
    
    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        for p in tparams:
            weight_decay += (tparams[p] ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay
    
    f_cost = theano.function([x, y, h1_init, c1_init, h2_init, c2_init], [h1_last, c1_last, h2_last, c2_last, loss], name='f_cost')

    grads = tensor.grad(cost, wrt=list(tparams.values()))
    f_grad = theano.function([x, y, h1_init, c1_init, h2_init, c2_init], grads, name='f_grad')
     
    #f_ppl =  theano.function([x, mask, y], ppl, name='f_ppl')

    if clip_c > 0.:
            print('Applying gradient clipping (clip_c:'+str(clip_c)+')...')
            grads = grad_clipping(grads, clip_c)

    lr = tensor.scalar(name='lr')
    
    slover = Optimizer()        
    f_grad_shared, f_update = slover.sgd(lr, tparams, grads, x, y, h1_init, c1_init, h2_init, c2_init, h1_last, c1_last, h2_last, c2_last, cost)
    
    print('Start Training...')
   

    history_errs = []
    best_p = None
    bad_count = 0

    uidx = 0  # the number of update done
    estop = False  # early stop
    
    batch_len = train_x.shape[0]//n_timesteps
    start_time = time.time()
    h1_init = numpy.zeros((batch_size, dim_hidden)).astype(numpy.float32)
    c1_init = numpy.zeros((batch_size, dim_hidden)).astype(numpy.float32)
    
    h2_init = numpy.zeros((batch_size, dim_hidden)).astype(numpy.float32)
    c2_init = numpy.zeros((batch_size, dim_hidden)).astype(numpy.float32)
    try:
        for eidx in range(max_epochs):
            print('learning rate ',str(lrate) )
            if eidx in step_size:
                    lrate=0.1*lrate
                    print('descrease learn rate to ', str(lrate))            
            num_timesteps = 0
            all_cost = 0
            
            for batch_index in range(batch_len):
                uidx += 1
                batch_start = batch_index * n_timesteps
                batch_end = (batch_index+1) * n_timesteps
                
                x = train_x[batch_start:batch_end]
                y = train_y[batch_start:batch_end]

                x=x.astype(numpy.int64)

                y=y.astype(numpy.int64)
   
                
                (h1_last, c1_last, h2_last, c2_last, cost) = f_grad_shared(x, y, h1_init, c1_init, h2_init, c2_init)
                (h1_last, c1_last, h2_last, c2_last, loss) = f_cost(x, y, h1_init, c1_init, h2_init, c2_init)
                all_cost += loss
                num_timesteps +=n_timesteps
                
                h1_init = h1_last
                c1_init = c1_last
                
                h2_init = h2_last
                c2_init = c2_last
                
                ppl = numpy.exp(all_cost/num_timesteps)         
               
                f_update(lrate)

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print('bad cost detected: ', cost)
                    return 1., 1., 1.

                if numpy.mod(uidx, dispFreq) == 0:
                    print('epoch ', eidx, 'iters ', uidx, 'cost ', cost, 'ppl ', ppl)
                    logger.write('epoch '+str(eidx)+'\titers '+str(uidx)+'\t cost '+str(cost)+'\tppl '+str(ppl)+ '\n')
            
                if model_path and numpy.mod(uidx, saveFreq) == 0:
                    print('Saving Model...')

                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    numpy.savez(model_path+model_name+'_'+str(uidx)+'.npz', history_errs=history_errs, **params)
                    pickle.dump(model_options, open('%s.pkl' % model_path, 'wb'), -1)
                    print('Done')

            #print ('Test...')
            train_ppl = pred_ppl(f_cost, train_x, train_y, batch_size, dim_hidden, n_timesteps)
            valid_ppl = pred_ppl(f_cost, val_x, val_y, batch_size, dim_hidden, n_timesteps)
            test_ppl =  pred_ppl(f_cost, test_x, test_y, batch_size, dim_hidden, n_timesteps)

            history_errs.append([valid_ppl, test_ppl])

            if (best_p is None or valid_ppl <= numpy.array(history_errs)[:,0].min()):
                best_p = unzip(tparams)
                bad_counter = 0

            print ('------- PPL ---------')
            print('Train ', train_ppl, ' Val ', valid_ppl, 'Test ', test_ppl)
            logger.write('Train PPL '+str(train_ppl)+'\t Val PPL '+str(valid_ppl)+ '\t Test PPL '+str(test_ppl)+ '\n')

            if (len(history_errs) > patience and
                test_ppl >= numpy.array(history_errs)[:-patience,0].min()):
                bad_counter += 1
                if bad_counter > patience:
                    print('Early Stop!')
                    estop = True
                    break

            if estop:
                break

    except KeyboardInterrupt:
        print("Training interupted")

    end_time = time.time()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)
        
    print('...Finished Training...\n')
    logger.close()
    return train_ppl, test_ppl


if __name__ == '__main__':
    train_rra()
