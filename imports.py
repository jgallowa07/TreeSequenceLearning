
import pickle
import sys
import msprime as msp
import numpy as np
import subprocess
import pyslim
import os,io,shutil
import multiprocessing as mp
import threading
import subprocess as sp
import shlex
import shutil
import random
import copy
import argparse
import math

import keras
import keras.backend as K
from keras import optimizers
from keras.optimizers import RMSprop
from keras.models import Model,Sequential
from keras import layers
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Lambda
from keras.layers import Conv2D, Conv1D, MaxPooling2D, AveragePooling2D,concatenate, MaxPooling1D, AveragePooling1D
from keras.utils import np_utils
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.utils.layer_utils import convert_all_kernels_in_model
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from sklearn import preprocessing
from keras import regularizers

import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
from scipy import stats
from prettytable import PrettyTable as pt
from sklearn.neighbors import NearestNeighbors



