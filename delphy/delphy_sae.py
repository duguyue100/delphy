'''
Author : Hu Yuhuang
Date   : 2014-12-13

This script is to classify HIGGS and SUSY dataset by using
Stacked Auto-encoders
'''

import os;
import sys;
import numpy;
import theano;
import theano.tensor as T;

# pylearn2
import pylearn2;

import pylearn2.models.mlp as mlp;
from pylearn2.models.autoencoder import Autoencoder;
from pylearn2.models.autoencoder import DeepComposedAutoencoder;

import pylearn2.training_algorithms.sgd;
from pylearn2.training_algorithms.sgd import SGD;
from pylearn2.training_algorithms.sgd import ExponentialDecay;
from pylearn2.training_algorithms.learning_rule import MomentumAdjustor;

import pylearn2.train;
from pylearn2.train import Train;

import pylearn2.termination_criteria;
from pylearn2.termination_criteria import EpochCounter;
from pylearn2.termination_criteria import Or;
from pylearn2.termination_criteria import MonitorBased;

from pylearn2.costs.cost import SumOfCosts;
from pylearn2.costs.autoencoder import MeanSquaredReconstructionError;
from pylearn2.costs.mlp.dropout import Dropout;

from pylearn2.datasets.transformer_dataset import TransformerDataset;

from physics import PHYSICS;

###### UTILITY FUNCTIONS ######

def get_autoencoder(structure):
    nvis, nhid = structure;
    conf = {'nhid': nhid,
            'nvis': nvis,
            'tied_weights': False,
            'act_enc': 'tanh',
            'act_dec': 'linear',
            'irange': 0.0001,};

    return Autoencoder(**conf);


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


###### TRAINING MODEL ######
# LAYER 1

l1_model=Autoencoder(nvis=nvis,
                     nhid=300,
                     act_enc="tanh",
                     act_dec="linear");

l1_algo=SGD(batch_size=100,
            learning_rate=1e-3,
            monitoring_dataset=train_monitor,
            cost=SumOfCosts([MeanSquaredReconstructionError()]),
            termination_criterion=EpochCounter(max_epochs=max_epochs));

print '[MESSAGE] The Layer 1 model is built.';

l1_save_path="./delphy_sae_l1.pkl";

l1_train=Train(dataset=train_set,
               model=l1_model,
               algorithm=l1_algo,
               save_path=l1_save_path,
               save_freq=50);

debug = False
l1_logfile = 'delphy_sae_l1.log';
print 'Using=%s' % theano.config.device # Can use gpus. 
print 'Writing to %s' % l1_save_path
#sys.stdout = open(l1_logfile, 'w')

l1_train.main_loop();

print '[MESSAGE] The Layer 1 is trained';

# LAYER 2

l1_train_set=TransformerDataset(raw=train_set,
                                transformer=l1_model);
l1_train_monitor=TransformerDataset(raw=train_monitor,
                                transformer=l1_model);


print '[MESSAGE] L1 Trained dataset transformed';

l2_model=Autoencoder(nvis=300,
                     nhid=300,
                     act_enc="tanh",
                     act_dec="linear");

l2_algo=SGD(batch_size=100,
            learning_rate=1e-3,
            monitoring_dataset=l1_train_monitor,
            cost=SumOfCosts([MeanSquaredReconstructionError()]),
            termination_criterion=EpochCounter(max_epochs=max_epochs));

print '[MESSAGE] The Layer 2 model is built.';

l2_save_path="./delphy_sae_l2.pkl";

l2_train=Train(dataset=l1_train_set,
               model=l2_model,
               algorithm=l2_algo,
               save_path=l2_save_path,
               save_freq=50);

debug = False
l2_logfile = 'delphy_sae_l1.log';
print 'Using=%s' % theano.config.device # Can use gpus. 
print 'Writing to %s' % l2_save_path
#sys.stdout = open(l2_logfile, 'w')

l2_train.main_loop();

print '[MESSAGE] The Layer 2 is trained';

# MLP network

sigLayer=mlp.Sigmoid(layer_name='y',
                     dim=1,
                     istdev=0.01,
                     monitor_style='bit_vector_class');

model=mlp.MLP(layers=[mlp.PretrainedLayer(layer_name='hidden_0',
                                          layer_content=l1_model),
                      mlp.PretrainedLayer(layer_name='hidden_1',
                                          layer_content=l2_model),
                      sigLayer],
              nvis=nvis);

algo=SGD(batch_size=100,
         learning_rate=0.05,
         monitoring_dataset={'train': train_monitor,
                             'valid': valid_set,
                             'test' : test_set},
         termination_criterion=Or(criteria=[MonitorBased(channel_name="valid_objective",
                                                         prop_decrease=0.00001,
                                                         N=40),
                                            EpochCounter(max_epochs=momentum_saturate)]),
         cost = Dropout(input_include_probs={'hidden_0':1., 'hidden_1':1., 'y':0.5},
                        input_scales={'hidden_0': 1., 'hidden_1':1.,'y':2.}),
         
         update_callbacks=ExponentialDecay(decay_factor=1.0000003,
                                           min_lr=.000001));

train=Train(dataset=train_set,
            model=model,
            algorithm=algo,
            save_path=save_path,
            save_freq=50);

print 'Using=%s' % theano.config.device # Can use gpus. 
print 'Writing to %s' % save_path

train.main_loop();