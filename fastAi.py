# This file contains all the main external libs we'll use
from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
import os
# Let's see what the directories are like
print(os.listdir("../input/"))


# After some listdir fun we've determined the proper path
PATH = '../dataset'

from os.path import expanduser, join, exists
from os import makedirs
cache_dir = expanduser(join('~', '.torch'))
if not exists(cache_dir):
    makedirs(cache_dir)
models_dir = join(cache_dir, 'models')
if not exists(models_dir):
    makedirs(models_dir)


arch='resnet101/resnet101.pth'
workers=8 # This number should match your number of processor cores for best results
sz=240 # image size 240x240
bs=64 # batch size
learnrate = 5e-3 #0.005
dropout = [0.3,0.6]


# TESTING added lighting changes | July 13, 2018 Version 21
#tfms = tfms_from_model(arch, sz, aug_tfms = [RandomLighting(b=0.5, c=0.5, tfm_y=TfmType.NO)], max_zoom=1.1)
#data = ImageClassifierData.from_paths(PATH, tfms=tfms, bs=bs, num_workers=workers)
data = ImageClassifierData.from_paths(PATH, tfms=tfms, bs=bs, num_workers=workers)
import pathlib
data.path = pathlib.Path('.')
learn = ConvLearner.pretrained(arch, data, precompute=False, ps=dropout)

# Finding the learning rate
lrf=learn.lr_find()
# Plotting the learning rate
learn.sched.plot()

learn.fit(learnrate, 1)

learn.fit(learnrate, 2, cycle_len=1)
lr = learnrate # just in case the next code block is skipped

learn.save("240_resnet101_all")