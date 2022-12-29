import numpy as np
import matplotlib.pyplot as plt
import pickle, os, sys, json
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
repo_base_directory = os.path.dirname(os.getcwd())
sys.path.append(repo_base_directory)
from common.annotation_utils import discretize_all, tp
import pandas as pd
import seaborn as sns

# ########################## PARAMETER DEFINITION #############################################################
# Annotation parameters:
combined_annotations_filepath = '../audio_annotation/outputs/compiled_annotations_df.parquet'
audio_directory = '../data/audio'
transcript_directory = '../data/transcripts'
SEGMENT_LENGTH = 2.5
HOP_LENGTH = 0.5
OVERLAP_THRESH = 0.5
WORD_TIME_OVERLAP_THRESH = 1.0 # What proportion of a word has to be within a SEGMENT_LENGTH chunk to be considered in that chunk?

# Plotting parameters:
# from https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#setup
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams["figure.figsize"] = (20,10)
OUTPUT_DIRECTORY = "./word_count_model_outputs"
#sns.set(rc = {'figure.figsize':(8,5)})
sns.set_style("white", {'font.family':'serif', 'font.serif':'Times Roman'})
sns.set_context("talk")
sns.set_palette("crest")

# Model parameters:
CLASS_WEIGHTS = True
PENALTY = 'none' #'elasticnet' #'l2'
SOLVER = 'saga' #'lbfgs'
L1RATIO = 0.5

# Evaluation:
# Australian teenager disagreement: '1XgTQnRlfJ0zpDdg2DccbR'
aus = ['1XgTQnRlfJ0zpDdg2DccbR']
# "Hot Take" podcasts: ['7r367wUYs1EvyBbeyOcq39', '0pIwpmg5oPcMWJXVSyrx4E']
hottake = ['7r367wUYs1EvyBbeyOcq39', '0pIwpmg5oPcMWJXVSyrx4E']
TEST_EP_IDS = aus

# Save config
config = {
    'segment_length': SEGMENT_LENGTH,
    'hop_length': HOP_LENGTH,
    'overlap_thresh': OVERLAP_THRESH,
    'word_time_overlap_thresh': WORD_TIME_OVERLAP_THRESH,
    'class_weights': CLASS_WEIGHTS,
    'penalty': PENALTY,
    'solver': SOLVER,
    'test_epids': TEST_EP_IDS
}
f = open(f"{OUTPUT_DIRECTORY}/{'_'.join(TEST_EP_IDS)}_config.pkl", "wb")
pickle.dump(config, f)
f.close()


# ##############################################################################################################

def episode_id_to_word_info(ep_id, ep_data, text_directory):
    episode_json_filepath = os.path.join(text_directory, ep_id + '.json')
    with open(episode_json_filepath, 'r') as j:
        transcript_file = json.loads(j.read())
    words_data = transcript_file['results'][-1]['alternatives'][0]['words']

    word_times = []; words = []
    for i in range(len(words_data)):
        start = float(words_data[i]['startTime'][:-1])
        end = float(words_data[i]['endTime'][:-1])
        word_times.append(tp(start, end))
        words.append(words_data[i]['word'])
    return word_times, words

def episode_transcript_to_word_chunks(word_times, words, audio_duration, segment_length, hop_length):
    assert(len(word_times)==len(words))
    text_chunks =  []
    start_times = np.arange(0, audio_duration - segment_length, hop_length)
    for s in start_times:
        text_subchunks = []
        end_time = s + segment_length
        chunk_tp = tp(s, end_time)
        for widx, w_tp in enumerate(word_times):
            if (w_tp.start == w_tp.end) and (w_tp.start >= s and w_tp.start <= end_time):
                text_subchunks.append(words[widx])
            elif (w_tp.start != w_tp.end):
                if w_tp.overlap(chunk_tp) >= WORD_TIME_OVERLAP_THRESH:
                    text_subchunks.append(words[widx])
        text_chunks.append(text_subchunks)
    return text_chunks

def discretized_dict_to_dataset(discretized_dict, text_directory, segment_length, hop_length):
    labels = []
    text_chunks = []
    ep_ids = []; id2ep = {}
    prints = [] # store print statements, so print all episode durations at once

    for idx, ep_id in enumerate(tqdm(discretized_dict)):
        ep_data = discretized_dict[ep_id]
        audio_duration = ep_data['audio_duration']
        word_times, words = episode_id_to_word_info(ep_id, ep_data, text_directory)
        ep_text_chunks = episode_transcript_to_word_chunks(word_times, words, audio_duration, segment_length, hop_length)
        labels.append(ep_data['y'])
        text_chunks.extend(ep_text_chunks)
        ep_ids.extend([idx] * len(ep_text_chunks))
        id2ep[idx] = ep_id
    labels = np.hstack(labels).reshape(-1,1)
    ep_ids = np.hstack(ep_ids).reshape(-1,1)
    return text_chunks, labels, ep_ids, id2ep

def load_transcripts(dict, filename, text_directory = '../data/transcripts'):
    if os.path.exists(filename):
        print("Loading from local file...")
        with open(filename, 'rb') as f:
            text_chunk_dict = pickle.load(f)
            text_chunks = text_chunk_dict['text_chunks']
            labels = text_chunk_dict['labels']
            ep_ids = text_chunk_dict['ep_ids']
    else:
        print("Loading from raw transcripts...")
        text_chunks, labels, ep_ids, id2ep = discretized_dict_to_dataset(dict, text_directory, SEGMENT_LENGTH, HOP_LENGTH)
        text_chunk_dict = {}
        text_chunk_dict['text_chunks'] = text_chunks
        text_chunk_dict['labels'] = labels
        text_chunk_dict['ep_ids'] = ep_ids
        with open(filename, 'wb') as f:
            pickle.dump(text_chunk_dict, f)
    return text_chunks, labels, ep_ids

def word2count(text_chunks, lexicon):
    # each index of count corresponds to lexicon
    all_count_vectors = []
    for chunk in tqdm(text_chunks):
        count_vector = np.zeros(len(lexicon))
        for word in chunk:
            for idx, negation_category in enumerate(lexicon):
                if word in lexicon[negation_category]:
                    count_vector[idx] += 1
        all_count_vectors.append(count_vector)
    output = np.vstack(all_count_vectors)
    return output

# from: https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#evaluate_metrics_2
def plot_roc(name, labels, predictions, ax, logit_roc_auc, **kwargs):
    fp, tp, _ = roc_curve(labels, predictions)
    ax.plot(fp, tp, label=f'{name} (area = {logit_roc_auc:.2f})', linewidth=2, **kwargs)

if __name__ == '__main__':
    print("###################  Loading metadata... ###################")
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

    print("###################  Loading transcripts... ###################")

    # Get text chunks (list of words in each window), and corresponding labels
    train_text_chunks, train_labels, train_ep_ids = load_transcripts(train_dict, f"preprocessed_data/train_text_chunk_dict_{'_'.join(TEST_EP_IDS)}.pkl", transcript_directory)
    test_text_chunks, test_labels, test_ep_ids = load_transcripts(test_dict, f"preprocessed_data/test_text_chunk_dict_{'_'.join(TEST_EP_IDS)}.pkl", transcript_directory)

    print("###################  Counting negation words... ###################")

    # Define lexicon (here, a dictionary of disagreement-related words)
    lexicon = {}
    lexicon['analytic negation'] = set(['no', 'not'])
    lexicon['contraction negation'] = set(['ain’t', 'aren’t', 'arent', 'aren’t', 'can’t', 'cannot', 'cant', 'couldn’t',
                                      'couldnt', 'couldn’t', 'didn’t', 'didnt', 'didn’t', 'doesn’t', 'doesnt', 'doesn’t',
                                      'don’t', 'don´t',
                                      'dont', 'hadn’t', 'hadnt', 'hasn’t', 'hasnt', 'hasn’t', 'haven’t', 'havent',
                                      'haven’t', 'havnt', 'isn’t', 'isnt', 'isn’t', 'mightnt', 'mightn’t', 'mustnt',
                                      'mustn’t', 'n’t', 'n’t', 'shouldn’t', 'shouldn’t', 'shouldnt', 'wasn’t',
                                      'wasnt', 'wasn’t', 'weren’t', 'werent', 'weren’t',
                                      'won’t', 'wont', 'won’t', 'wouldn’t', 'wouldnt', 'wouldn’t'])
    lexicon['synthetic negation'] = set(['neither', 'never', 'nor', 'none', 'nobody', 'noone', 'no-one'])
    lexicon['synonyms for without'] = set(['sans', 'without', 'w/o'])
    lexicon['other'] = set(["disagree", "incorrect", "wrong", "ridiculous", "absurd"])

    X_train = word2count(train_text_chunks, lexicon)
    y_train = train_labels.flatten()

    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
    #                                                   test_size = 0.2,
    #                                                   stratify = np.hstack([train_ep_ids, train_labels]),
    #                                                   shuffle = True,
    #                                                   random_state = 0)

    X_test = word2count(test_text_chunks, lexicon)
    y_test = test_labels.flatten()

    print(f"X_train: {X_train.shape}\n y_train: {y_train.shape}\n")
    print(f"X_test: {X_test.shape}\n y_test: {y_test.shape}\n")

    print("################### Initializing model...  ###################")
    clf = LogisticRegression(random_state=0, max_iter = 1000, l1_ratio = L1RATIO,
                            penalty = PENALTY,
                            solver = SOLVER,
                            class_weight = 'balanced' if CLASS_WEIGHTS else None)

    print("################### Fitting model...  ###################")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    clf.fit(X_train, y_train)

    print("################### Evaluating model...  ###################")
    y_train_pred = clf.predict(X_train)
    y_train_pred_proba = clf.predict_proba(X_train)[::,1]

    y_test_pred = clf.predict(X_test)
    y_test_pred_proba = clf.predict_proba(X_test)[::,1]

    print("Test set classification report:")
    print(classification_report(y_test, y_test_pred))
    train_logit_roc_auc = roc_auc_score(y_train, y_train_pred)
    test_logit_roc_auc = roc_auc_score(y_test, y_test_pred)

    fig, ax = plt.subplots()
    plot_roc("Train", train_labels, y_train_pred_proba, color=colors[0], ax = ax, logit_roc_auc = train_logit_roc_auc)
    plot_roc("Test", test_labels, y_test_pred_proba, color=colors[0], linestyle='--', ax = ax, logit_roc_auc = test_logit_roc_auc)
    plt.xlabel('False Positives [%]')
    plt.ylabel('True Positives [%]')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.plot([0, 1], [0, 1],'r--')
    plt.grid(True)
    ax.set_aspect('equal')
    plt.legend(loc='lower right')
    plt.title('ROC Curve \n (Logistic Regression on Negation Word Counts)')
    plt.savefig(f"{OUTPUT_DIRECTORY}/roc_curve_logreg.png", dpi = 300)
