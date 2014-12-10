'''
Author : Hu Yuhuang
Date   : 2014-12-10

This script is to classify HIGGS and SUSY dataset by using
Multiple Preceptron Network
'''

import os;
import sys;
import numpy;
import theano;
import theano.tensor as T;
import pylearn2;
import pylearn2.models.mlp as mlp;
import pylearn2.training_algorithms.sgd
import pylearn2.termination_criteria
import pylearn2.costs.mlp.dropout
import pylearn2.train

from physics import PHYSICS;


###### Load dataset ######

idpath = os.path.splitext(os.path.abspath(__file__))[0]; # ID for output files.
save_path = idpath + '.pkl';
momentum_saturate=200;

## HIGGS
benchmark=2; ## higgs dataset
derived_feat=True;
nvis=18;
train_set = PHYSICS(which_set='train',
                    benchmark=benchmark,
                    derived_feat=derived_feat);
train_monitor = PHYSICS(which_set='train',
                        benchmark=benchmark,
                        derived_feat=derived_feat,
                        start=0,
                        stop=100000);
valid_set = PHYSICS(which_set='valid',
                    benchmark=benchmark,
                    derived_feat=derived_feat);
test_set  = PHYSICS(which_set='test',
                    benchmark=benchmark,
                    derived_feat=derived_feat);


## BUILD MODEL

model=pylearn2.models.mlp.MLP(layers=[mlp.Tanh(layer_name='hidden_0',
                                               dim=300,
                                               istdev=0.1),
                                      mlp.Sigmoid(layer_name='y',
                                                 dim=1,
                                                 istdev=0.01)],
                              nvis=nvis);

print '[MESSAGE] The model is built'

## TRAINING ALGORITHM

algorithm=pylearn2.training_algorithms.sgd.SGD(batch_size=100,
                                               learning_rate=0.05,
                                               monitoring_dataset={'train':train_monitor,
                                                                   'valid':valid_set,
                                                                   'test':test_set},
                                               
                                               termination_criterion=pylearn2.termination_criteria.Or(criteria=[pylearn2.termination_criteria.MonitorBased(channel_name="valid_objective",
                                                                                                                                                           prop_decrease=0.00001,
                                                                                                                                                           N=40),
                                                                                                                pylearn2.termination_criteria.EpochCounter(max_epochs=momentum_saturate)]),
                                               cost = pylearn2.costs.mlp.dropout.Dropout(
                                                   input_include_probs={'hidden_0':1., 'y':0.5},
                                                   input_scales={ 'hidden_0': 1., 'y':2.}),
                                               
                                               update_callbacks=pylearn2.training_algorithms.sgd.ExponentialDecay(
                                                   decay_factor=1.0000003, # Decreases by this factor every batch. (1/(1.000001^8000)^100 
                                                   min_lr=.000001
                                               ));

## EXTENSION

extensions=[pylearn2.training_algorithms.learning_rule.MomentumAdjustor(start=0,
                                                                        saturate=momentum_saturate,
                                                                        final_momentum=.99)];

## TRAINING MODEL

train=pylearn2.train.Train(dataset=train_set,
                           model=model,
                           algorithm=algorithm,
                           save_path=save_path,
                           save_freq=100);

debug = False
logfile = os.path.splitext(train.save_path)[0] + '.log'
print 'Using=%s' % theano.config.device # Can use gpus. 
print 'Writing to %s' % logfile
print 'Writing to %s' % train.save_path
sys.stdout = open(logfile, 'w')

train.main_loop();