'''
Author : Hu Yuhuang
Date   : 2014-12-10

This script is to classify HIGGS and SUSY dataset by using
Multiple Preceptron Network
'''

import numpy;
import theano;
import theano.tensor as T;
from physics import PHYSICS;


###### Load dataset ######

## HIGGS
benchmark=2; ## higgs dataset
derived_feat=True;
nvis=28;
train_set = PHYSICS(which_set='train', benchmark=benchmark, derived_feat=derived_feat);
valid_set = PHYSICS(which_set='valid', benchmark=benchmark, derived_feat=derived_feat);
test_set  = PHYSICS(which_set='set', benchmark=benchmark, derived_feat=derived_feat);

print "[MESSAGE] DATASET IS LOADED";

## BUILD MODEL