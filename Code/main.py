#%%
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, matthews_corrcoef
from tensorflow.keras.utils import to_categorical