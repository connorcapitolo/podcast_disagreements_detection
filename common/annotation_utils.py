import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import librosa
import subprocess
import os

# Only used for default behavior of calling "in" on tp() e.g. tp(0,3) in tp(2.1, 4.5)
# Define minimum required proportion of overlap to return "True"
REQUIRED_OVERLAP = 0.2

class tp():
    """Class for abstracting common operations on timestamped windows"""
    def __init__(self,start,end):
        self.start=start
        self.end=end
    def __repr__(self):
        return '(%.2f,%.2f)' % (self.start, self.end)
    def __contains__(self, key):
        if isinstance(key, tp):
            # if overlap with self > 0.2 of self's length?
            return get_overlap_proportion(self, key) >= REQUIRED_OVERLAP
        elif isinstance(key, (int, float)):
            # is key within self?
            return (self.start <= key) and (key <= self.end)
        else:
            raise NotImplementedError
    def overlap(self, tp2):
        # return overlap between time window tp2 and self, relative to self's length
        return get_overlap_proportion(self, tp2)

def get_union(annot_df, episode_id):
    # Tuple union algorithm, modified from https://stackoverflow.com/questions/1034802/union-of-intervals
    """
    Return list of tp objects containing unioned disagreement annotation timestamps from pandas DataFrame annot_df, filtering to episode episode_id
    """
    annot_df = annot_df[annot_df['text']==episode_id]
    ranges = [tp(s,e) for (s,e) in zip(annot_df['start'], annot_df['end'])]

    s = ranges
    s.sort(key=lambda self: self.start)
    union = [s[0]]
    for x in s[1:]:
        if union[-1].end < x.start:
            union.append(x)
        elif union[-1].end == x.start:
            union[-1].end = x.end
        if x.end > union[-1].end:
            union[-1].end = x.end
    return union

def get_overlap_proportion(tp1, tp2):
    """Compute overlap between tp instances tp2 and tp1, relative to tp1 (i.e. the denominator is the length of tp1)"""
    if (tp1.start <= tp2.end) and (tp2.start <= tp1.end):
        latest_start = max(tp1.start, tp2.start)
        earliest_end = min(tp1.end, tp2.end)
        delta = (earliest_end - latest_start)
        return delta / (tp1.end - tp1.start)
    else:
        return 0

def discretize(disagreement_annotations, audio_filename, segment_length = 2.5, hop_length = 0.5, overlap_thresh = 0.2):
    """
    Returns:
        audio_duration: duration (seconds) of audio_filename
        y: np.array with discretized labels based on provided params
    """
    audio_duration = librosa.get_duration(filename = audio_filename)
    if audio_duration > 60 * 60 * 10:
        raise ValueError("Detected audio duration is greater than 10 hours for ogg file. Will convert to mp3 file for processing")
    segments = np.arange(0, audio_duration - segment_length, hop_length)

    y = []
    for i in range(len(segments)):
        start_time = segments[i]
        end_time = start_time + segment_length
        t = tp(start_time, end_time)
        y.append(1*any([t.overlap(d) >= overlap_thresh for d in disagreement_annotations]))
    return audio_duration, np.array(y)

def plot_annotations(compiled_annotations_df, episode_id, annotator_colors, font_size = '12'):
    """Plot timeline, filtering df to episode_id; plot params are annotator_colors, y_step"""
    df = compiled_annotations_df[compiled_annotations_df['text'] == episode_id]
    plt.figure(figsize=(20,5))
    plt.rcParams['font.size'] = font_size
    for i, annotator in enumerate(annotator_colors):
        i_df = df[df['annotator'] == annotator]
        for (start,end) in i_df[['start', 'end']].values:
            plt.plot([start/60, end/60], [i, i],
                     marker = '|',
                     color = annotator_colors[annotator],
                     linewidth = 30, markersize = 50, solid_capstyle="butt")
    plt.xlabel("Minute")
    plt.title(f"Annotations for episode: {episode_id}")
    plt.ylim(-1, len(annotator_colors))
    plt.yticks(range(len(annotator_colors)), annotator_colors)

def convert_to_mp3(audio_filename_ogg):
    subprocess.call([
        'ffmpeg',
        '-i',
        audio_filename_ogg,
        audio_filename_ogg[:-4] + '.mp3'
    ], shell = False)

def discretize_all(compiled_annotations_df_filepath, audio_filepath, segment_length = 2.5, hop_length = 0.5, overlap_thresh = 0.2):
    """
    Apply discretize function all episodes in compiled_annotation_df, returning two dictionaries:
        unioned_annotation: dictionary mapping episode_id to union of annotated disagreement time ranges.
            this dictionary looks like:
            {'episode_id': [tp(), tp(), ...]}
        discretized_dict: dictionary of the form:
            {'segment_length': 2.5,
            'hop_length': 0.5,
            'overlap_thresh': 0.2,
            'data':
                {
                    'episode_id': {
                        'audio_duration': 1000,
                        'filetype': '.ogg',
                        'y': np.array([1, 0, 0, 1, 0])
                    }
                }
            }
    """
    compiled_annotations_df = pd.read_parquet(compiled_annotations_df_filepath)
    unique_episodes = compiled_annotations_df['text'].unique()
    unioned_annotations = {}; discretized_dict = {}
    discretized_dict['data'] = {}
    discretized_dict['segment_length'] = segment_length
    discretized_dict['hop_length'] = hop_length
    discretized_dict['overlap_thresh'] = overlap_thresh

    for e in unique_episodes:
        disagreement_annotations = get_union(compiled_annotations_df, e)
        unioned_annotations[e] = disagreement_annotations

        audio_filename_ogg = os.path.join(audio_filepath, f"{e}.ogg")

        if os.path.isfile(audio_filename_ogg):
            try:
                audio_duration, y = discretize(disagreement_annotations = disagreement_annotations,
                                        audio_filename = audio_filename_ogg,
                                        segment_length = segment_length,
                                        hop_length = hop_length,
                                        overlap_thresh = overlap_thresh)
                discretized_dict['data'][e] = {}
                discretized_dict['data'][e]['audio_duration'] = audio_duration
                discretized_dict['data'][e]['filetype'] = '.ogg'
                discretized_dict['data'][e]['y'] = y
                print(f'Ogg worked for {audio_filename_ogg}')
            except ValueError as err:
                print(f"Error with ogg file: {err}")
                # print(f"Trying to use mp3 file")
                audio_filename_mp3 = audio_filename_ogg[:-4] + '.mp3'
                if not os.path.isfile(audio_filename_mp3):
                    print(f"No mp3 file found. Performing mp3 conversion for processing")
                    convert_to_mp3(audio_filename_ogg)
                audio_duration, y = discretize(disagreement_annotations = disagreement_annotations,
                                        audio_filename = audio_filename_mp3,
                                        segment_length = segment_length,
                                        hop_length = hop_length,
                                        overlap_thresh = overlap_thresh)
                discretized_dict['data'][e] = {}
                discretized_dict['data'][e]['audio_duration'] = audio_duration
                discretized_dict['data'][e]['filetype'] = '.mp3'
                discretized_dict['data'][e]['y'] = y
                # print(f"mp3 load successful")
        else:
            raise ValueError(f"Attempting to discretize {audio_filename_ogg} but file does not exist. Place .ogg file to avoid unused annotations.")

    return unioned_annotations, discretized_dict
