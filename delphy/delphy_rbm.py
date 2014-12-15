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
from pylearn2.models.rbm import RBM;

import pylearn2.training_algorithms.sgd
from pylearn2.training_algorithms.sgd import SGD;
from pylearn2.training_algorithms.sgd import ExponentialDecay;
from pylearn2.training_algorithms.learning_rule import MomentumAdjustor;

import pylearn2.termination_criteria
from pylearn2.termination_criteria import EpochCounter;

from pylearn2.costs.cost import SumOfCosts;
from pylearn2.costs.autoencoder import MeanSquaredReconstructionError;

import pylearn2.train
from pylearn2.train import Train;

from pylearn2.datasets.transformer_dataset import TransformerDataset;

from physics import PHYSICS;


###### Load dataset ######

idpath = os.path.splitext(os.path.abspath(__file__))[0]; # ID for output files.
save_path = idpath + '.pkl';
momentum_saturate=200;
max_epochs=200;

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

rbm_layer_1=RBM(nvis=nvis,
                nhid=300,
                monitor_reconstruction=True);

rbm_l1_algo=SGD(batch_size=100,
                learning_rate=1e-3,
                monitoring_dataset=train_monitor,
                cost=SumOfCosts([MeanSquaredReconstructionError()]),
                termination_criterion=EpochCounter(max_epochs=max_epochs));

print '[MESSAGE] Layer 1 model is built';

rbm_l1_savepath='delphy_rbm_l1.pkl';

rbm_l1_train=Train(dataset=train_set,
                   model=rbm_layer_1,
                   algorithm=rbm_l1_algo,
                   save_path=rbm_l1_savepath,
                   save_freq=50);

debug = False
rbm_l1_logfile = 'delphy_rbm_l1.log';
print 'Using=%s' % theano.config.device # Can use gpus. 
print 'Writing to %s' % rbm_l1_savepath
#sys.stdout = open(l1_logfile, 'w')

rbm_l1_train.main_loop();

print '[MESSAGE] The Layer 1 is trained';


l1_train_set=TransformerDataset(raw=train_set,
                                transformer=l1_model);
l1_train_monitor=TransformerDataset(raw=train_monitor,
                                transformer=l1_model);

print '[MESSAGE] Layer 1 Data is transformed'

rbm_layer_2=RBM(nvis=300,
                nhid=300,
                monitor_reconstruction=True);

rbm_l2_algo=SGD(batch_size=100,
                learning_rate=1e-3,
                monitoring_dataset=l1_train_monitor,
                cost=SumOfCosts([MeanSquaredReconstructionError()]),
                termination_criterion=EpochCounter(max_epochs=max_epochs));

print '[MESSAGE] Layer 1 model is built';

rbm_l2_savepath='delphy_rbm_l2.pkl';

rbm_l2_train=Train(dataset=l1_train_set,
                   model=rbm_layer_2,
                   algorithm=rbm_l2_algo,
                   save_path=rbm_l2_savepath,
                   save_freq=50);

debug = False
rbm_l2_logfile = 'delphy_rbm_l2.log';
print 'Using=%s' % theano.config.device # Can use gpus. 
print 'Writing to %s' % rbm_l2_savepath
#sys.stdout = open(l1_logfile, 'w')

rbm_l2_train.main_loop();

print '[MESSAGE] The Layer 2 is trained';

