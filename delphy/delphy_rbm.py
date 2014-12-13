'''
Author : Hu Yuhuang
Date   : 2014-12-13

This script is to classify HIGGS and SUSY dataset by using
Deep Restricted Boltzmann Machine
'''

import os;
import sys;
import numpy;
import theano;
import theano.tensor as T;
import pylearn2;
import pylearn2.models.mlp as mlp;
from pylearn2.models.dbm.dbm import DBM;
from pylearn2.model.dbm.layer import GaussianVisLayer;
from pylearn2.model.dbm.layer import BinaryLayer;
from pylearn2.model.dbm.layer import BinaryVectorMaxPool;
import pylearn2.training_algorithms.sgd
from pylearn2.training_algorithms.sgd import SGD;
import pylearn2.termination_criteria
from pylearn2.termination_criteria import EpochCounter;
from pylearn2.training_algorithms.sgd import ExponentialDecay;
from pylearn2.costs.cost import SumOfCosts;
from pylearn2.costs.dbm import VariationalPCD;
from pylearn2.costs.dbm import WeightDecay;
from pylearn2.costs.dbm import TorontoSparsity;
from pylearn2.training_algorithms.learning_rule import MomentumAdjustor;
import pylearn2.train

from physics import PHYSICS;


###### Load dataset ######

idpath = os.path.splitext(os.path.abspath(__file__))[0]; # ID for output files.
save_path = idpath + '.pkl';
momentum_saturate=200;

## HIGGS
benchmark=2; ## SUSY dataset
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

## Layers

visLayer=BinaryLayer(nvis=nvis,
                     bias_from_marginals=train_set);

hidLayer_1=BinaryVectorMaxPool(layer_name="hidden_1",
                               detector_layer_dim=300,
                               pool_size=1,
                               irange=0.05,
                               init_bias=-2.0);

hidLayer_2=BinaryVectorMaxPool(layer_name="hidden_2",
                               detector_layer_dim=300,
                               pool_size=1,
                               irange=0.05,
                               init_bias=-2.0);

## Model

model=DBM(batch_size=100,
          visible_layer=visLayer,
          hidden_layer=[hidLayer_1, hidLayer_2],
          niter=1);

## ALGORITHM

max_epochs=300;

algorithm=SGD(learning_rate=1e-3,
              monitoring_dataset={'train':train_monitor},
              termination_criterion=EpochCounter(max_epochs=max_epochs),
              cost = SumOfCosts(costs=[VariationalPCD(num_chains=100,
                                                      num_gibbs_steps=5),
                                       WeightDecay(coeffs=[0.0001,
                                                           0.0001]),
                                       TorontoSparsity(targets=[0.2,
                                                                0.2],
                                                       coeffs=[0.001,
                                                               0.001])]),
              update_callbacks=ExponentialDecay(decay_factor=1.0000003,
                                                min_lr=.000001));


## EXTENSION

extensions=[MomentumAdjustor(start=0,
                             saturate=max_epochs,
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