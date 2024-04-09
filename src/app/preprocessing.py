import matplotlib.pyplot as plt
import numpy as np
import copy
import sklearn
import torch

import mne
from mne import io
from mne.datasets import sample
from mne import set_log_level

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split

from braindecode.datasets import BCICompetitionIVDataset4
from braindecode import EEGRegressor
from braindecode.training import CroppedTimeSeriesEpochScoring, TimeSeriesLoss
from braindecode.preprocessing import create_fixed_length_windows
from braindecode.models import ShallowFBCSPNet
from braindecode.util import set_random_seeds
from braindecode.models import get_output_shape, to_dense_prediction_model
from braindecode.preprocessing import (Preprocessor,
                                       exponential_moving_standardize,
                                       preprocess)

from mne.decoding import (
    CSP,
    GeneralizingEstimator,
    LinearModel,
    Scaler,
    SlidingEstimator,
    Vectorizer,
    cross_val_multiscore,
    get_coef,
)

from classes.transformData import (
    Data,
)

print(__doc__)

def pad_and_select_predictions(preds, y):
    preds = np.pad(preds,
                   ((0, 0), (0, 0), (y.shape[2] - preds.shape[2], 0)),
                   'constant',
                   constant_values=0)

    mask = ~np.isnan(y[0, 0, :])
    preds = np.squeeze(preds[..., mask], 0)
    y = np.squeeze(y[..., mask], 0)
    return y.T, preds.T

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
    markers = data.get_markers(xml, mat)
    events = data.get_events_matrix(xml, mat)

    # mapping = {
    # 1: "auditory/left",
    # 2: "auditory/right",
    # 3: "visual/left",
    # 4: "visual/right",
    # }
    # annot_from_events = mne.annotations_from_events(
    #     events=events,
    #     event_desc=mapping,
    #     sfreq=raw.info["sfreq"],
    #     orig_time=raw.info["meas_date"],
    # )

    epochs = data.get_epochs(xml, mat)



    subject_id = 1
    dataset = BCICompetitionIVDataset4(subject_ids=[subject_id])
    print("\n check \n")

    print(type(dataset))

    dataset = dataset.split('session')
    train_set = dataset['train']
    test_set = dataset['test']


    low_cut_hz = 1.  # low cut frequency for filtering
    high_cut_hz = 200.  # high cut frequency for filtering, for ECoG higher than for EEG
    # Parameters for exponential moving standardization
    factor_new = 1e-3
    init_block_size = 1000

    valid_set = preprocess(copy.deepcopy(train_set),
                       [Preprocessor('crop', tmin=24, tmax=30)], n_jobs=-1)

    preprocess(train_set, [Preprocessor('crop', tmin=0, tmax=24)], n_jobs=-1)
    preprocess(test_set, [Preprocessor('crop', tmin=0, tmax=24)], n_jobs=-1)
    preprocessors = [
        # TODO: ensure that misc is not removed
        Preprocessor('pick_types', ecog=True, misc=True),
        Preprocessor(lambda x: x / 1e6, picks='ecog'),  # Convert from V to uV
        Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
        Preprocessor(exponential_moving_standardize,  # Exponential moving standardization
                     factor_new=factor_new, init_block_size=init_block_size, picks='ecog')
    ]
    # Transform the data
    preprocess(train_set, preprocessors)
    preprocess(valid_set, preprocessors)
    preprocess(test_set, preprocessors)

    # Extract sampling frequency, check that they are same in all datasets
    sfreq = train_set.datasets[0].raw.info['sfreq']
    assert all([ds.raw.info['sfreq'] == sfreq for ds in train_set.datasets])
    # Extract target sampling frequency
    target_sfreq = train_set.datasets[0].raw.info['temp']['target_sfreq']


    input_window_samples = 1000

    cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
    device = 'cuda' if cuda else 'cpu'
    if cuda:
        torch.backends.cudnn.benchmark = True
    # Set random seed to be able to roughly reproduce results
    # Note that with cudnn benchmark set to True, GPU indeterminism
    # may still make results substantially different between runs.
    # To obtain more consistent results at the cost of increased computation time,
    # you can set `cudnn_benchmark=False` in `set_random_seeds`
    # or remove `torch.backends.cudnn.benchmark = True`
    seed = 20200220
    set_random_seeds(seed=seed, cuda=cuda)

    n_classes = 4
    # Extract number of chans and time steps from dataset
    n_chans = train_set[0][0].shape[0] - 5

    model = ShallowFBCSPNet(
        n_chans,
        n_classes,
        final_conv_length=2,
        add_log_softmax=False,
    )

    # Send model to GPU
    if cuda:
        model.cuda()

    to_dense_prediction_model(model)

    n_preds_per_input = get_output_shape(model, n_chans, input_window_samples)[2]

    # Create windows using braindecode function for this. It needs parameters to define how
    # trials should be used.

    train_set = create_fixed_length_windows(
        train_set,
        start_offset_samples=0,
        stop_offset_samples=None,
        window_size_samples=input_window_samples,
        window_stride_samples=n_preds_per_input,
        drop_last_window=False,
        targets_from='channels',
        last_target_only=False,
        preload=False
    )

    valid_set = create_fixed_length_windows(
        valid_set,
        start_offset_samples=0,
        stop_offset_samples=None,
        window_size_samples=input_window_samples,
        window_stride_samples=n_preds_per_input,
        drop_last_window=False,
        targets_from='channels',
        last_target_only=False,
        preload=False
    )

    test_set = create_fixed_length_windows(
        test_set,
        start_offset_samples=0,
        stop_offset_samples=None,
        window_size_samples=input_window_samples,
        window_stride_samples=n_preds_per_input,
        drop_last_window=False,
        targets_from='channels',
        last_target_only=False,
        preload=False
    )

    train_set.target_transform = lambda x: x[0: 1]
    valid_set.target_transform = lambda x: x[0: 1]
    test_set.target_transform = lambda x: x[0: 1]


    # These values we found good for shallow network for EEG MI decoding:
    lr = 0.0625 * 0.01
    weight_decay = 0
    batch_size = 27  # only 27 examples in train set, otherwise set to 64
    n_epochs = 8

    regressor = EEGRegressor(
        model,
        cropped=True,
        aggregate_predictions=False,
        criterion=TimeSeriesLoss,
        criterion__loss_function=torch.nn.functional.mse_loss,
        optimizer=torch.optim.AdamW,
        train_split=predefined_split(valid_set),
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        iterator_train__shuffle=True,
        batch_size=batch_size,
        callbacks=[
            ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
            ('r2_train', CroppedTimeSeriesEpochScoring(sklearn.metrics.r2_score,
                                                       lower_is_better=False,
                                                       on_train=True,
                                                       name='r2_train')
             ),
            ('r2_valid', CroppedTimeSeriesEpochScoring(sklearn.metrics.r2_score,
                                                       lower_is_better=False,
                                                       on_train=False,
                                                       name='r2_valid')
             )
        ],
        device=device,
    )
    set_log_level(verbose='WARNING')

    preds_train, y_train = regressor.predict_trials(train_set, return_targets=True)
    preds_train, y_train = pad_and_select_predictions(preds_train, y_train)

    preds_valid, y_valid = regressor.predict_trials(valid_set, return_targets=True)
    preds_valid, y_valid = pad_and_select_predictions(preds_valid, y_valid)

    preds_test, y_test = regressor.predict_trials(test_set, return_targets=True)
    preds_test, y_test = pad_and_select_predictions(preds_test, y_test)


    # X = epochs.get_data(copy=False)  # MEG signals: n_epochs, n_meg_channels, n_times
    # y = epochs.events[:, 2]  # target: auditory left vs visual left
    #
    # clf = make_pipeline(
    #     Scaler(epochs.info),
    #     Vectorizer(),
    #     LogisticRegression(solver="liblinear"),  # liblinear is faster than lbfgs
    # )
    #
    # scores = cross_val_multiscore(clf, X, y, cv=5, n_jobs=None)
    #
    # # Mean scores across cross-validation splits
    # score = np.mean(scores, axis=0)
    # print("Spatio-temporal: %0.1f%%" % (100 * score,))


    break




