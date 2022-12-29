import json
import pandas as pd
import numpy as np
# pip install nltk
# nltk.download('wordnet')
import nltk
from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import stopwords
# pip install -U scikit-learn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def process_nondiarized_transcript(nondiarized_transcript):
    """
    Given nondiarized_transcript, return:
        text_segments: list of strings corresponding to each audio segment (based on Google speech to text)
        text_segments_confidences: list of confidence scores per audio segment
        text_segments_times: list of (start, end) tuples corresponding to each text segment
    """
    text_segments = []; text_segments_confidences = []; text_segments_times = []
    for s in nondiarized_transcript:
        # assumption is that there is only a single transcription per time segment; raise error otherwise
        assert(len(s['alternatives'])==1)
        segment_info = s['alternatives'][0]

        if 'transcript' in segment_info:
            text_segments.append(segment_info['transcript'])
            text_segments_confidences.append(segment_info['confidence'])
            text_start = float(segment_info['words'][0]['startTime'][:-1])
            text_end = float(segment_info['words'][-1]['endTime'][:-1])
            text_segments_times.append((text_start, text_end))
    return text_segments, text_segments_confidences, text_segments_times

def process_diarized_info(diarized_info):
    """
    Given diarized_info, return:
        text_speakers: list of strings corresponding to each continuous speakerTag segment
        speaker_turn_times: list of (speakerTag, start, end) tuples corresponding to turns for each speaker
    """
    text_speakers = []; text_speakers_times = []
    prev_speaker = -1; curr_speaker_words = []
    diarized_info = diarized_info['alternatives'][0]['words']

    for idx, diarized_word_info in enumerate(diarized_info):
        current_speaker = int(diarized_word_info['speakerTag'])
        current_end = float(diarized_word_info['endTime'][:-1])
        # if current speaker is a new speaker...
        if (current_speaker != prev_speaker):
            current_start = float(diarized_word_info['startTime'][:-1]) # store start time
            if prev_speaker != -1: # if not the first speaker in the episode...
                text_speakers.append(' '.join(curr_speaker_words))
                text_speakers_times.append((prev_speaker, prev_start, prev_end)) # record previous speaker block
            curr_speaker_words = []
        if idx == (len(diarized_info)-1): # if the last speaker in the episode...
            text_speakers_times.append((current_speaker, current_start, current_end))
            text_speakers.append(' '.join(curr_speaker_words))
        prev_start = current_start
        prev_end = current_end
        prev_speaker = current_speaker
        curr_speaker_words.append(diarized_word_info['word'])
    return text_speakers, text_speakers_times

def process_transcript(json_filepath, segment_definition = 'default'):
    """
    Given str filepath of transcript json and string segment_definition specifying how to process it, 
    return dictionary with:
        if 'default' is specified as segment_definition:
            text_segments: list of strings corresponding to each audio segment (based on Google speech to text)
            text_segments_confidences: list of confidence scores per audio segment
            text_segments_times: list of (start, end) tuples corresponding to each text segment
        if 'utterance' is specified as segment_definition:
            text_speakers: list of strings corresponding to each continuous speakerTag segment
            speaker_turn_times: list of (speakerTag, start, end) tuples corresponding to turns for each speaker
    """
    output = {}
    
    with open(json_filepath, 'r') as j:
        transcript_file = json.loads(j.read())
        segments = transcript_file['results']

    if segments:
        # the last entry contains speaker diarization info
        # see: https://cloud.google.com/speech-to-text/docs/samples/speech-transcribe-diarization-beta
        if segment_definition == 'default':
            nondiarized_transcript = segments[:-1]
            output['text_chunks'], output['text_chunks_confidences'], output['text_chunks_times'] = process_nondiarized_transcript(nondiarized_transcript)
        elif segment_definition == 'utterance':
            diarized_info = segments[-1]
            output['text_chunks'], output['text_chunks_times'] = process_diarized_info(diarized_info)
        else:
            raise NotImplementedError
    return output

def get_tokens(text, lemmatize = False, remove_stopwords = False):
    """
    Given a string 'text', return list of lowercase, alphabetic tokens 
    (optionally lemmatize and remove stopwords)
    """
    text_tokens = [i.lower() for i in word_tokenize(text)
                   if (i.isalpha() and ((not remove_stopwords) or (i not in set(stopwords.words('english')))))]
    if lemmatize:
        lem = WordNetLemmatizer()
        return [lem.lemmatize(i) for i in text_tokens]
    else:
        return text_tokens     
    
def get_avg_word_vector(text_tokens, embedding_dict, embedding_dim = 300):
    """
    Given list of tokens (for single 'document') and lookup dictionary embedding_dict of dim embedding_dim
    return:
        averaged word vector to represent the 'document'
        number of tokens used in computing the average
    """
    avg_vector = np.zeros(embedding_dim)
    n_tokens = 0
    for t in text_tokens:
        if t in embedding_dict:
            avg_vector += embedding_dict[t]
            n_tokens += 1
    if n_tokens != 0:
        avg_vector /= n_tokens
    return avg_vector, n_tokens

def get_cosine_similarities(q, vectors):
    """
    Given a query vector q (to serve as the lookup)
    return the cosine similarities
    """
    cosine_similarities = np.zeros(len(vectors))
    for idx, v in enumerate(vectors):
        cosine_similarities[idx] = np.dot(v, q)/(np.sqrt(v.T @ v) * np.sqrt(q.T @ q))
    return cosine_similarities
