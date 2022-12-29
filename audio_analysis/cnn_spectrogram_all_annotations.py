import numpy as np
import matplotlib.pyplot as plt
import pickle, os, sys
from tqdm import tqdm
import tensorflow as tf
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import librosa.display
import librosa
import torch
import tensorflow_io as tfio
from torch_audiomentations import Gain, AddColoredNoise, PitchShift, Compose
import gc
import scipy.signal as sps
import resampy
repo_base_directory = os.path.dirname(os.getcwd())
sys.path.append(repo_base_directory)
from common.annotation_utils import discretize_all
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, confusion_matrix
import pandas as pd

# ########################## PARAMETER DEFINITION #############################################################
# Annotation parameters:
combined_annotations_filepath = '../audio_annotation/outputs/compiled_annotations_df.parquet'
audio_directory = '../data/audio'
SEGMENT_LENGTH = 2.5
HOP_LENGTH = 0.5
OVERLAP_THRESH = 0.5

# Augmentation parameters:
AUG_PER_SAMPLE = 0 # number of augmentations to produce per training sample
SPECTROGRAM_AUGMENT = True # apply augmentation on top of spectrogram
SR = 22050 # Desired sample rate (if native sampling rate is different, will be resampled with resampy)

# Spectrogram parameters:
# See: https://stackoverflow.com/questions/62584184/understanding-the-shape-of-spectrograms-and-n-mels
# FFT_SIZE: frequency resolution of the window
# in speech processing, the recommended value is 512, corresponding to 23 milliseconds at SR of 22050 Hz
# at SR of 44100, this equals 1024; see https://librosa.org/doc/main/generated/librosa.stft.html#librosa.stft
# mel filterbank: matrix to warp linear scale spectrogram to mel scale
MEL_BINS = 50 # Number of mel-frequency bands in mel-spectrogram
F_MIN = 0 # Lowest frequency (in Hz) to include in mel-scale
F_MAX = 8000 # SR // 2
FFT_SIZE = 1024
FFT_HOP_LENGTH = FFT_SIZE // 4
MEL_FILTERBANK = tf.signal.linear_to_mel_weight_matrix(num_mel_bins = MEL_BINS,
                                                       num_spectrogram_bins = FFT_SIZE // 2 + 1,
                                                       sample_rate = SR,
                                                       lower_edge_hertz = F_MIN,
                                                       upper_edge_hertz = F_MAX)

# Plotting parameters:
# from https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#setup
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams["figure.figsize"] = (20,10)
OUTPUT_DIRECTORY = "./CNN_spectrogram_model_outputs"

# Model parameters:
EPOCHS = 30
DROPOUT = 0.3
BATCH_SIZE = 1024 # High batch size is important for imbalanced classes
LR = 1e-3
CLASS_WEIGHTS = True
WEIGHT_INIT = True

# Evaluation:
# Australian teenager disagreement: '1XgTQnRlfJ0zpDdg2DccbR'
aus = ['1XgTQnRlfJ0zpDdg2DccbR']
# "Hot Take" podcasts: ['7r367wUYs1EvyBbeyOcq39', '0pIwpmg5oPcMWJXVSyrx4E']
hottake = ['7r367wUYs1EvyBbeyOcq39', '0pIwpmg5oPcMWJXVSyrx4E']
TEST_EP_IDS = hottake

# Save config
config = {
    'segment_length': SEGMENT_LENGTH,
    'hop_length': HOP_LENGTH,
    'overlap_thresh': OVERLAP_THRESH,
    'dropout': DROPOUT,
    'batch_size': BATCH_SIZE,
    'lr': LR,
    'class_weights': CLASS_WEIGHTS,
    'weight_init': WEIGHT_INIT,
    'test_epids': TEST_EP_IDS,
    'sr': SR
}
f = open(f"{OUTPUT_DIRECTORY}/{'_'.join(TEST_EP_IDS)}_config.pkl", "wb")
pickle.dump(config, f)
f.close()

# ##############################################################################################################

def episode_id_to_audio(ep_id, ep_data, audio_directory):
    episode_filepath = os.path.join(audio_directory, ep_id + ep_data['filetype'])
    # fast: if prioritizing speed (over memory)
    if ep_data['filetype'] == '.ogg': #.ogg read is faster in soundfile
        audio, sr = sf.read(episode_filepath)
        audio = np.mean(audio, axis=1)
    elif ep_data['filetype'] == '.mp3':
        #.mp3 not supported in soundfile, use audioread instead, used in backend of librosa:
        # https://librosa.org/doc/main/_modules/librosa/core/audio.html#load
        audio, sr = librosa.core.audio.__audioread_load(episode_filepath,
                                                        offset = 0, duration = None, dtype=np.float32)
        audio = np.mean(audio, axis=0)
    if sr != SR:
        audio = resampy.resample(audio, sr, SR)
    return audio, SR

def episode_audio_to_audio_chunks(audio, audio_duration, segment_length, hop_length):
    audio_chunks =  []
    start_times = np.arange(0, audio_duration - segment_length, hop_length)
    for s in start_times:
        start_sample = librosa.time_to_samples(s, sr = SR)
        end_sample = librosa.time_to_samples(s + segment_length, sr = SR)
        sub_audio = audio[start_sample:end_sample]
        audio_chunks.append(sub_audio)
    audio_chunks = np.vstack(audio_chunks)
    #audio_chunks = pad_sequences(audio_chunks, value = 0, padding = 'post')
    return audio_chunks

def discretized_dict_to_dataset(discretized_dict, audio_directory, segment_length, hop_length):
    labels = []
    audio_chunks = []
    ep_ids = []; id2ep = {}; sample_rates = []
    prints = [] # store print statements, so print all episode durations at once

    for idx, ep_id in enumerate(tqdm(discretized_dict)):
        ep_data = discretized_dict[ep_id]
        audio_duration = ep_data['audio_duration']
        ep_audio, ep_sr = episode_id_to_audio(ep_id, ep_data, audio_directory)
        ep_audio_chunks = episode_audio_to_audio_chunks(ep_audio, audio_duration,
                                                        segment_length, hop_length)
        labels.append(ep_data['y'])
        audio_chunks.append(ep_audio_chunks)
        ep_ids.extend([idx] * len(ep_audio_chunks))
        id2ep[idx] = ep_id
        sample_rates.append(ep_sr)
        prints.append(f"{ep_id} duration: {(len(ep_audio)/ep_sr)/60:.2f} mins \n")
    labels = np.hstack(labels).reshape(-1,1)
    audio_chunks = np.vstack(audio_chunks)
    ep_ids = np.hstack(ep_ids).reshape(-1,1)
    print(f"Sample rates found: {np.unique(sample_rates)}")
    for p in prints:
        print(p)
    return audio_chunks, labels, ep_ids, id2ep

def chunks_and_labels_to_dataset(audio_chunks, labels):
    audio_ds = tf.data.Dataset.from_tensor_slices(audio_chunks)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((audio_ds, label_ds))

def audio_chunk_to_spectrogram(audio_chunk, mel_filterbank, frame_length, frame_step, sr = SR):
    '''
    References:
    * https://towardsdatascience.com/how-to-easily-process-audio-on-your-gpu-with-tensorflow-2d9d91360f06
    * https://gist.github.com/keunwoochoi/c9592922a17d71b745d47dc8eb7f0538
    '''
    # Cast waveform to float32
    audio_chunk = tf.cast(audio_chunk, dtype=tf.float32)
    # Convert waveform to spectrogram via STFT, and obtain magnitude
    spectrograms = tf.signal.stft(audio_chunk, frame_length = frame_length, frame_step = frame_step)
    magnitude_spectrograms = tf.abs(spectrograms)
    mel_spectrograms = tf.matmul(tf.square(magnitude_spectrograms), mel_filterbank)
    # Add channels axis:
    # mel_spectrograms = mel_spectrograms[..., tf.newaxis]
    return mel_spectrograms

def augment(audio_chunk):
    '''Augment single chunk w Gaussian noise and pitch shift'''
    white_noise = np.random.randn(len(audio_chunk))
    audio_chunk = audio_chunk + 0.01 * white_noise
    pitch_shift_amount = np.random.randn()
    audio_chunk = librosa.effects.pitch_shift(audio_chunk, SR, n_steps=pitch_shift_amount)
    return audio_chunk

def plot_metrics(history):
    metrics = ['loss', 'prc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
                 color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])
        plt.legend()
    plt.savefig(f"{OUTPUT_DIRECTORY}/CNN_model_training_history.png", dpi = 300)

def load_waveform(dict, filename, audio_directory = '../data/audio'):
    if os.path.exists(filename):
        print("Loading from local file...")
        with open(filename, 'rb') as f:
            audio_chunks = np.load(f)
            labels = np.load(f)
            ep_ids = np.load(f)
    else:
        print("Loading from raw audio...")
        audio_chunks, labels, ep_ids, id2ep = discretized_dict_to_dataset(dict, audio_directory, SEGMENT_LENGTH, HOP_LENGTH)
        with open(filename, 'wb') as f:
            np.save(f, audio_chunks)
            np.save(f, labels)
            np.save(f, ep_ids)
    return audio_chunks, labels, ep_ids

def get_augmented_data(chunks, labels, verbose = True):
    audio_augs = []; label_augs = []
    original_chunks_shape = chunks.shape; original_labels_shape = labels.shape
    for idx in tqdm(range(len(chunks))):
        label = labels[idx]
        chunk = chunks[idx]
        for n in range(AUG_PER_SAMPLE):
            aug_chunk = augment(chunk)
            audio_augs.append(aug_chunk)
            label_augs.append(label)
    audio_augs = np.vstack(audio_augs)
    label_augs = np.vstack(label_augs)
    chunks = np.vstack([chunks, audio_augs])
    labels = np.vstack([labels, label_augs])
    if verbose:
        print(f"Original X shape: {original_chunks_shape}")
        print(f"Original labels shape: {original_labels_shape}")
        print(f"Augmented train audio shape: {chunks.shape}")
        print(f"Augmented train labels shape: {labels.shape}")
    return chunks, labels

# from: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#evaluate_metrics_2
def plot_roc(name, labels, predictions, ax, logit_roc_auc, **kwargs):
    fp, tp, _ = roc_curve(labels, predictions)
    ax.plot(fp, tp, label=f'{name} (area = {logit_roc_auc:.2f})', linewidth=2, **kwargs)

if __name__ == '__main__':
    print("###################  Loading metadata... ###################")
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    # Load episode metadata (duration), and define labels based on overlap threshold
    _, discretized_dict = discretize_all(combined_annotations_filepath,
                                         audio_filepath = audio_directory,
                                         segment_length = SEGMENT_LENGTH,
                                         hop_length = HOP_LENGTH,
                                         overlap_thresh = OVERLAP_THRESH)
    discretized_dict = discretized_dict['data']

    # Define train / test episode IDs
    all_ep_ids = list(discretized_dict.keys())
    train_ep_ids = [i for i in all_ep_ids if i not in TEST_EP_IDS]
    test_ep_ids = TEST_EP_IDS

    train_dict = { ep_id: discretized_dict[ep_id] for ep_id in train_ep_ids }
    test_dict = { ep_id: discretized_dict[ep_id] for ep_id in test_ep_ids }

    print("###################  Loading waveforms... ###################")
    train_audio_chunks, train_labels, train_ep_ids = load_waveform(train_dict, f"train_{'_'.join(TEST_EP_IDS)}.npy", audio_directory)
    test_audio_chunks, test_labels, test_ep_ids = load_waveform(test_dict, f"test_{'_'.join(TEST_EP_IDS)}.npy", audio_directory)
    print(f"Train episode IDs: {train_ep_ids}")
    print(f"Test episode IDs: {test_ep_ids}")

    if AUG_PER_SAMPLE > 0:
        # Augment training data
        print("###################  Augmenting waveforms... ###################")
        train_audio_chunks, train_labels = get_augmented_data(train_audio_chunks, train_labels)

    print("################### Initializing datasets...  ###################")
    with tf.device('/cpu:0'):
        X_train, X_val, y_train, y_val = train_test_split(train_audio_chunks, train_labels,
                                                          test_size = 0.2,
                                                          stratify = np.hstack([train_ep_ids, train_labels]),
                                                          shuffle = True,
                                                          random_state = 0)
        X_test, y_test = test_audio_chunks, test_labels

        print(f"X_train: {X_train.shape}\n y_train: {y_train.shape}\n")
        print(f"X_val: {X_val.shape}\n y_val: {y_val.shape}\n")
        print(f"X_test: {X_test.shape}\n y_test: {y_test.shape}\n")

        train_ds = chunks_and_labels_to_dataset(X_train, y_train)
        train_ds = train_ds.map( #Transform audio wave to spectrograms
            lambda audio, label: (audio_chunk_to_spectrogram(audio, MEL_FILTERBANK, FFT_SIZE, FFT_HOP_LENGTH), label),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        if SPECTROGRAM_AUGMENT:
            train_ds = train_ds.map(
                lambda audio, label: (tfio.audio.freq_mask(audio, param = 10), label),
                num_parallel_calls=tf.data.AUTOTUNE
            ).map(
                lambda audio, label: (tfio.audio.time_mask(audio, param = 10), label),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        train_ds = train_ds.map(
            lambda audio, label: (audio[..., tf.newaxis], label),
            num_parallel_calls=tf.data.AUTOTUNE
        ).batch(
            BATCH_SIZE
        ).prefetch(
            tf.data.AUTOTUNE
        )

        val_ds = chunks_and_labels_to_dataset(X_val, y_val)
        val_ds = val_ds.map(
            lambda audio, label: (audio_chunk_to_spectrogram(audio, MEL_FILTERBANK, FFT_SIZE, FFT_HOP_LENGTH), label),
            num_parallel_calls=tf.data.AUTOTUNE
        ).map(
            lambda audio, label: (audio[..., tf.newaxis], label),
            num_parallel_calls=tf.data.AUTOTUNE
        ).batch(
            BATCH_SIZE
        ).prefetch(
            tf.data.AUTOTUNE
        )

        # Plot some example spectrograms from a random batch, to ensure working properly
        n_examples = 3
        fig, axes = plt.subplots(1, n_examples, figsize = (10,3))
        single_batch_examples = train_ds.take(1)
        for sample in single_batch_examples:
            _melspec, _label = sample[0], sample[1]
            _examples = _melspec[:3]
            for i, ax in enumerate(axes.flat):
                print(tf.squeeze(_examples[i]).shape)
                s_dB = librosa.power_to_db(tf.squeeze(_examples[i]), ref=np.max)
                img = librosa.display.specshow(s_dB, x_axis='time', y_axis='mel', sr = SR, ax = ax)
                ax.set_xticks([]);
        fig.tight_layout()
        plt.savefig(f"{OUTPUT_DIRECTORY}/CNN_example_spectrograms.png", dpi = 300)

    print("################### Initializing model...  ###################")
    # Based on: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#setup
    # Class imbalance check:
    neg, pos = np.bincount(y_train.flatten())
    total = neg + pos
    print('Training Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
        total, pos, 100 * pos / total))

    if WEIGHT_INIT:
        initial_bias = np.log([pos/neg])
        print(f"Initial bias of classification layer: {initial_bias}\n")

    if CLASS_WEIGHTS:
        class_weights = dict(enumerate(class_weight.compute_class_weight('balanced',
                                                                         classes = np.unique(y_train.flatten()),
                                                                         y = y_train.flatten())))
        print(f"Class weights: {class_weights}")

    model = keras.Sequential(
        [
            layers.Conv2D(filters = 5, kernel_size = (4,4), strides = (1,1), activation="relu", padding = 'same'),
            layers.MaxPooling2D(pool_size = (4,4), strides = (2,2)),
            layers.Dropout(DROPOUT),
            layers.Conv2D(filters = 5, kernel_size = (4,4), strides = (1,1), activation="relu", padding = 'same'),
            layers.MaxPooling2D(pool_size = (4,4), strides = (2,2)),
            layers.Dropout(DROPOUT),
            layers.Conv2D(filters = 5, kernel_size = (4,4), strides = (1,1), activation="relu", padding = 'same'),
            layers.MaxPooling2D(pool_size = (4,4), strides = (2,2)),
            layers.Dropout(DROPOUT),
            layers.Conv2D(filters = 5, kernel_size = (4,4), strides = (1,1), activation="relu", padding = 'same'),
            layers.MaxPooling2D(pool_size = (4,4), strides = (2,2)),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dense(units = 50, activation = "relu"),
            layers.Dense(units = 25, activation = "relu"),
            layers.Dense(units = 1, activation = "sigmoid")
                         #bias_initializer = tf.keras.initializers.Constant(initial_bias)) if WEIGHT_INIT else "zeros"
        ]
    )

    metrics = ['accuracy',
               tf.keras.metrics.Precision(name = 'precision'),
               tf.keras.metrics.Recall(name = 'recall'),
               tf.keras.metrics.AUC(name = 'prc', curve = 'PR')]

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = LR),
        loss = tf.keras.losses.BinaryCrossentropy(),
        metrics = metrics
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor = 'val_precision',
        verbose = 1,
        patience = 10,
        mode = 'max',
        restore_best_weights = True)

    model.predict(train_ds.take(1))
    print(model.summary())

    print("################### Fitting model...  ###################")
    history = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs = EPOCHS,
        class_weight = class_weights if CLASS_WEIGHTS else None,
        callbacks=[early_stopping]
    )

    plot_metrics(history)

    print("################### Evaluating model...  ###################")
    test_ds = chunks_and_labels_to_dataset(X_test, y_test)
    test_ds = test_ds.batch(BATCH_SIZE)

    test_ds = test_ds.map(
        lambda x, y: (audio_chunk_to_spectrogram(x, MEL_FILTERBANK, FFT_SIZE, FFT_HOP_LENGTH), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    results = model.evaluate(test_ds, batch_size = BATCH_SIZE, verbose=1)
    print(results)

    y_test_pred = (model.predict(test_ds) > 0.5).astype("int32").flatten()
    y_test_pred_proba = model.predict(test_ds).flatten()

    print("Test set classification report:")
    report = classification_report(y_test, y_test_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(f"{OUTPUT_DIRECTORY}/cnn_classification_report_{'_'.join(TEST_EP_IDS)}.csv")
    print(classification_report(y_test, y_test_pred))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    cmatrix = confusion_matrix(y_test, y_test_pred)
    df = pd.DataFrame(cmatrix)
    df.to_csv(f"{OUTPUT_DIRECTORY}/cnn_confusion_matrix_{'_'.join(TEST_EP_IDS)}.csv")

    test_logit_roc_auc = roc_auc_score(y_test, y_test_pred)
    fig, ax = plt.subplots()
    plot_roc("Test", y_test, y_test_pred_proba, color=colors[0], linestyle='--', ax = ax, logit_roc_auc = test_logit_roc_auc)
    plt.xlabel('False Positives [%]')
    plt.ylabel('True Positives [%]')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.plot([0, 1], [0, 1],'r--')
    plt.grid(True)
    ax.set_aspect('equal')
    plt.legend(loc='lower right')
    plt.title(f'ROC Curve \n (CNN spectrogram model)')
    plt.savefig(f"{OUTPUT_DIRECTORY}/cnn_roc_curve_logreg_{'_'.join(TEST_EP_IDS)}.png", dpi = 300)
