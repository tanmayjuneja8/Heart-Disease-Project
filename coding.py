import sys

from numpy.lib.arraysetops import unique
import torch

sys.path.append("/Users/tanmayjuneja/Documents/ML Project/code/utils_and_models")
import utils
import os
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import multiprocessing
from itertools import repeat
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
import torch.nn.functional as F
from models.fastai_model import fastai_model

sns.set(style="whitegrid", palette="muted", font_scale=1.2)

HAPPY_COLORS_PALETTE = [
    "#01BEFE",
    "#FFDD00",
    "#FF7D00",
    "#FF006D",
    "#ADFF02",
    "#8F00FF",
]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams["figure.figsize"] = 12, 8

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


sampling_frequency = 100
datafolder = "/Users/tanmayjuneja/Documents/ML Project/ptbxl/"
task = "superdiagnostic"
outputfolder = "/Users/tanmayjuneja/Documents/ML Project/output"

# Load PTB-XL data
data, raw_labels = utils.load_dataset(datafolder, sampling_frequency)

# Preprocess label data
labels = utils.compute_label_aggregations(raw_labels, datafolder, task)

# Select relevant data and convert to one-hot
data, labels, Y, _ = utils.select_data(
    data, labels, task, min_samples=0, outputfolder=outputfolder
)

# 1-9 for training
X_train = data[labels.strat_fold < 10]
y_train = Y[labels.strat_fold < 10]
# 10 for validation
X_val = data[labels.strat_fold == 10]
y_val = Y[labels.strat_fold == 10]

num_classes = 5  # <=== number of classes in the finetuning dataset
input_shape = [
    1000,
    12,
]  # <=== shape of samples, [None, 12] in case of different lengths
# print("X_train shape : ", X_train.shape, "X_val.shape : ", X_val.shape)


experiment = "exp0"
modelname = "fastai_xresnet1d34_deep"
# pretrainedfolder = "../output/" + experiment + "/models/" + modelname + "/"
mpath = "../output/"  # <=== path where the finetuned model will be stored
n_classes_pretrained = 71  # <=== because we load the model from exp0, this should be fixed because this depends the experiment

model = fastai_model(
    modelname,
    num_classes,
    sampling_frequency,
    mpath,
    input_shape=input_shape,
)

# standardize the input using pickle.
standard_scaler = pickle.load(
    open(
        "/Users/tanmayjuneja/Documents/ML Project/code/output/"
        + experiment
        + "/data/standard_scaler.pkl",
        "rb",
    )
)

X_train = utils.apply_standardizer(X_train, standard_scaler)
X_val = utils.apply_standardizer(X_val, standard_scaler)

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(
#     X_train.shape
# )
# X_val = scaler.fit_transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)


if __name__ == "__main__":
    model.fit(X_train, y_train, X_val, y_val)
    y_val_pred = model.predict(X_val)
    utils.evaluate_experiment(y_val, y_val_pred)
