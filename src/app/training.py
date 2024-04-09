import copy

import numpy as np
import sklearn
from mne import set_log_level

# from braindecode.datasets import BCICompetitionIVDataset4
from braindecode.datasets import create_from_mne_raw

from classes.transformData import (
    Data,
)

with open('config.yaml') as file:
  config = yaml.safe_load(file)




data = Data(data_directory_matlab = config['data_directory_mat'],
            data_directory_xml = config['data_directory_xml'],
            sampling_frequency = config['sampling_frequency'],
            sample_length = config['length_sample'],
            f_notch = config['f_notch'],
            q = config['q'])

xml, mat = data.get_files()[0]
raw = data.get_mne_raw(xml, mat)
print(raw)

raw.filter(1, 40)

# # Get a 5 second subset of the data
# start = int(raw.info['sfreq'] * 5)  # Start 5 seconds in
# stop = start + int(raw.info['sfreq'] * 5)  # End 5 seconds later
# subset = raw.get_data(start=start, stop=stop)
#
# plt.figure(figsize=(15, 5))
# plt.plot(subset.T)
# plt.show()

marker_starts = data.find_marker_starts(xml, mat)
events = np.insert(marker_starts, 1, 0, axis=1)
epochs = mne.Epochs(raw, events=events, event_id=None, tmin=0, tmax=5, baseline=None, preload=True)
# print(epochs)

X = epochs.get_data()
y = epochs.events[:, 2] - 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


dataset = BCICompetitionIVDataset4(subject_ids=[subject_id])
