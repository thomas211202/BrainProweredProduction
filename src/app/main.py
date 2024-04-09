import os
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import xmltodict
import xmljson
import json
import random
import pickle

from xml.dom import minidom
import numpy as np
from numpy import multiply
from scipy import signal
from collections import defaultdict
from mne.decoding import CSP
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from bs4 import BeautifulSoup
from braindecode.models import ShallowFBCSPNet, EEGNetv4, Deep4Net, HybridNet
from braindecode.models.util import models_dict
from braindecode import EEGClassifier
from braindecode.util import set_random_seeds
# from skorch.dataset import ValidSplit
from skorch.callbacks import LRScheduler
import mne
import torch

from braindecode.preprocessing import (
    Preprocessor,
    exponential_moving_standardize,
    preprocess
)


from braindecode.datasets import (
    BaseDataset,
    BaseConcatDataset
)

from classes.transformData import (
    Data,
)

# matplotlib.use('TkAgg')

SAMPLING_FREQUENCY = 500
LENGTH_SAMPLE = 5
data_directory_mat = './app/BP_EEG_data/mat/'
data_directory_xml = './app/BP_EEG_data/xml/'
f_notch = 50
Q = 30

data = Data(data_directory_matlab = data_directory_mat,
            data_directory_xml = data_directory_xml,
            sampling_frequency = SAMPLING_FREQUENCY,
            sample_length = 5,
            f_notch = f_notch,
            q = Q)

for xml, mat in data.get_files():
    raw = data.get_mne_raw(xml, mat)
    print(raw)

    raw = data.get_mne_raw(xml, mat)
    markers = data.get_markers(xml, mat)

    break

# print("one_hot_df.columns: ", one_hot_df.columns)

raw.filter(1, 40)

# Get a 5 second subset of the data
start = int(raw.info['sfreq'] * 5)  # Start 5 seconds in
stop = start + int(raw.info['sfreq'] * 5)  # End 5 seconds later
subset = raw.get_data(start=start, stop=stop)

plt.figure(figsize=(15, 5))
plt.plot(subset.T)
plt.show()

marker_starts = data.find_marker_starts(xml, mat)
events = np.insert(marker_starts, 1, 0, axis=1)
# epochs = mne.Epochs(raw, events=events, event_id=None, tmin=0, tmax=5, baseline=None, preload=True)
epochs = data.get_epochs(xml, mat)
# print(epochs)

X = epochs.get_data()
y = epochs.events[:, 2] - 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print("report: ")
print("shape: ", X_test.shape)
df = epochs.to_data_frame()
print(df.describe())
print(df.columns)
print(df['condition'].unique())
print(df)
# epochs.plot_projs_topomap(vlim="joint")
# plt.savefig('/data/initial_plot.png')

models_available = list(models_dict.keys())
print(*models_available, sep='\n')

cuda = torch.cuda.is_available()  # check if GPU is available
mps = torch.backends.mps.is_available()


device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = True

seed = 20200220  # random seed to make results reproducible
set_random_seeds(seed=seed, cuda=cuda)

print(f"Device: {device}\nSeed: {seed}")

n_classes = len(np.unique(y))
classes = list(range(n_classes))
n_chans = X.shape[1]
input_window_samples = X.shape[2]

print(f"Classes: {classes}\nNumber of classes: {n_classes}\nNumber of channels: {n_chans}\nInput window samples: {input_window_samples}")

model = ShallowFBCSPNet(
    in_chans=n_chans,
    n_classes=n_classes,
    input_window_samples=input_window_samples,
    final_conv_length='auto',
)

print(model)

if cuda:
    model.cuda()
    print("Model on GPU")
elif mps:
    model.to(mps_device)
    print("Model on mps")


lr = 0.0625 * 0.01
weight_decay = 0.8 * 0.001
batch_size = 64
n_epochs = 5

clf = EEGClassifier(
    model,
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.AdamW,
    train_split=None,
    optimizer__lr=lr,
    optimizer__weight_decay=weight_decay,
    batch_size=batch_size,
    callbacks=[
        "accuracy",
        ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
    ],
    device=device,
    classes=classes,
    max_epochs=n_epochs,
)

clf.fit(X_train, y_train)

# Path to save the model file within the container
model_file_path = '/data/model.pkl'

# Ensure the directory structure exists
os.makedirs(os.path.dirname(model_file_path), exist_ok=True)

with open(model_file_path,'wb') as f:
    pickle.dump(clf,f)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
disp.plot()
# plt.show()
plt.savefig('/data/confusion_matrix.png')



# y_pred_1 = clf.predict(X_test[0])
# accuracy = accuracy_score(y_test[0], y_pred)
# print(f"Accuracy: {accuracy * 100:.2f}%")
# disp = ConfusionMatrixDisplay(confusion_matrix(y_test[0], y_pred_1))
# disp.plot()
# plt.show()
# plt.savefig('confusion_matrix_single.png')

# print(X_test)
# print(X_test.shape)
# print(y_test)
# print(y_test.shape)


# sample = test[np.random.choice(test.shape[0], 1, replace=False)]
# print(sample)
# print(sample.shape)
# X = sample[:, :-1, :25]
# y = sample[:, -1, :25]
# # clf.predict(X)
# print("test actual: ", y)


