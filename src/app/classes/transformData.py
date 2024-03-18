import os
import scipy.io
import pandas as pd
import numpy as np
from scipy import signal
from bs4 import BeautifulSoup
import mne


class Data():
    def __init__(self,
                 data_directory_matlab='./app/BP_EEG_data/mat/',
                 data_directory_xml='./app/BP_EEG_data/xml/',
                 sampling_frequency=512,
                 sample_length=5,
                 f_notch=50,
                 q=50
                 ):

        self.SAMPLING_FREQUENCY = sampling_frequency
        self.LENGTH_SAMPLE = sample_length
        self.data_directory_mat = data_directory_matlab
        self.data_directory_xml = data_directory_xml
        self.f_notch = f_notch
        self.Q = q

        self.b, self.a = signal.iirnotch(self.f_notch, self.Q, self.SAMPLING_FREQUENCY)
        self.sos = signal.butter(N=10, Wn=[8, 15], btype="bandpass", output="sos", fs=self.SAMPLING_FREQUENCY)

    def get_files(self):
        """
        Returns tuples (xml, mat) with the mat files, each with their corresponding xml file from the data directories.
        """

        subject_files_mat = os.listdir(self.data_directory_mat)
        subject_files_mat.sort()

        subject_files_xml = os.listdir(self.data_directory_xml)
        subject_files_xml.sort()

        return zip(subject_files_xml, subject_files_mat)

    def get_df_with_marker(self, file_xml, file_mat):
        """
        Returns pandas DataFrame containg the raw EEG-data, with a single marker-column, containing a float for every
        unique marker.
        """
        with open(self.data_directory_xml + file_xml, "r") as contents:
            content = contents.read()
            try:
                soup = BeautifulSoup(content, 'xml')
            except:
                soup = BeautifulSoup(content, 'lxml')

            titles = soup.find_all('name')
            labels = [x.text for x in titles][:-1]
            return pd.DataFrame(scipy.io.loadmat(self.data_directory_mat + file_mat)["data"], columns=labels)

    def get_df_without_marker(self, file_xml, file_mat):
        """
        Returns a pandas DataFrame, containing only the raw EEG data.
        """
        return self.get_df_with_marker(file_xml, file_mat).drop("marker", axis='columns')

    def get_df_with_onehot_encoded_marker(self, file_xml, file_mat):
        """
        Returns a pandas DataFrame, containing extra columns with one-hot encoded markers.
        """
        return pd.get_dummies(self.get_df_with_marker(file_xml, file_mat), columns=['marker', ])

    def get_filtered_df(self, file_xml, file_mat):
        """
        Returns a pandas DataFrame, containing the filtered EEG data and the one-hot-encoded markers
        """
        df = self.get_df_without_marker(file_xml, file_mat)
        df_filter = pd.DataFrame(signal.sosfilt(self.sos, df))
        df_filter.set_axis(df.columns.values, axis=1, inplace=True)
        df_encoded = self.get_df_with_onehot_encoded_marker(file_xml, file_mat)
        marker_column_names = list(set(df_encoded.columns.values) - set(df.columns.values))
        df_filter[marker_column_names] = df_encoded[marker_column_names]
        return df_filter

    def get_mne_raw(self, file_xml, file_mat):
        data = self.get_df_without_marker(file_xml, file_mat)
        ch_names = list(data.columns.values)
        ch_types = ['eeg' for _ in range(len(ch_names))]
        info = mne.create_info(ch_names=ch_names, sfreq=self.SAMPLING_FREQUENCY, ch_types=ch_types)
        raw = mne.io.RawArray(data.to_numpy().T, info)
        return raw

    def get_markers(self, file_xml, file_mat):
        return self.get_df_with_marker(file_xml, file_mat)['marker']

    def get_channels(self, file_xml, file_mat):
        return self.get_df_without_marker(file_xml, file_mat).columns.values

    def find_marker_starts(self, file_xml, file_mat):
        """
        Find the indices where markers start in a numpy array.

        Args:
          series: A pandas Series objects marker column

        Returns:
          A numpy array of shape (n_events, 2) where the first column is the index
          at which the marker starts, and the second column the marker it is about.
        """
        # Find the indices where the data changes from non-marker to marker.
        data = self.get_markers(file_xml, file_mat).to_numpy().astype(int)
        marker_starts = np.flatnonzero(np.diff(data) > 0) + 1

        # Get the markers at the start indices.
        markers = data[marker_starts]

        # Combine the start indices and markers into a single array.
        return np.vstack((marker_starts, markers)).T

    def get_events_matrix(self, file_xml, file_mat):
        return np.insert(self.find_marker_starts(file_xml, file_mat), 1, 0, axis=1)

    def get_epochs(self, file_xml, file_mat):
        return mne.Epochs(self.get_mne_raw(file_xml, file_mat), events=self.get_events_matrix(file_xml, file_mat), event_id=None, tmin=0, tmax=5, baseline=None, preload=True)
