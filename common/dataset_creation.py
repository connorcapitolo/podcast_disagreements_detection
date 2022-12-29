import time
import argparse
import logging

import pywt
import librosa

from common.annotation_utils import *


def process_entire_episode(podcast_id: str, segment_length: float, hop_length: float, y, audio_duration: float, parquet_directory: str) -> None:
    '''Processes the entire podcast episode
    
    Args:
        podcast_id: the podcast that needs to be processed
        segment_length: how long each chunk of audio should be considered a single observation
        hop_length: how far forward each chunk should be moved in the audio file
        y: the disagreement labels based on segment length, hop, and overlap threshold
        audio_duration: how long the podcast episode is in seconds
        parquet_directory: directory where the parquet files should be saved

    
    Returns:
        a parquet file that has the DWT coefficients which is saved in data/dwt_df_parquet that can be used later
    
    Example of how to run as script:
        >>> python -m common.dataset_creation -seg 0.5 -hop 0.5 -ovlap 0.5
    '''

    # find the appropriate path (use mp3 if it's available)
    prev_directory = 'data/audio/'
    if os.path.exists(f'{prev_directory}{podcast_id}.mp3'):
        audio_filepath = f'{prev_directory}{podcast_id}.mp3'
    else:
        audio_filepath = f'{prev_directory}{podcast_id}.ogg'


    logging.info(f'Found audio file at location {audio_filepath}\n')
    start_load = time.time()
    waveform, sr = librosa.load(audio_filepath)
    logging.info(
        f'{podcast_id} took {((time.time() - start_load) / 60):.2f} minutes to load\n')

    
    # Convert sample indices to time (in seconds)
    time_stamp = librosa.samples_to_time(np.arange(0, len(waveform)), sr=sr)
    audio_df = pd.DataFrame()
    audio_df['time'] = time_stamp
    audio_df['original'] = waveform

    # necessary information gained about the podcast
    chunks = np.arange(0, audio_duration - segment_length, hop_length)
    logging.info(f'Number of chunks in podcast id {podcast_id}: {len(chunks)}')

    # getting segments of the waveform data based on the chunks
    audio_time_seg = []
    # getting segments of the time based on the chunks
    time_seg = []
    for chunk in chunks:

        # audio segments
        start_sr = librosa.time_to_samples(chunk, sr=sr)
        end_sr = librosa.time_to_samples(chunk + segment_length, sr=sr)

        sub_audio = audio_df.original[start_sr:end_sr + 1]
        audio_time_seg.append(sub_audio)

        sub_time = audio_df.time[start_sr:end_sr + 1]
        time_seg.append(sub_time)

    logging.info(f'Number of audio time segments: {len(audio_time_seg)}')
    logging.info(
        f'Length of first audio time segment: {len(audio_time_seg[0])/sr} seconds')

    dwt_time = time.time()
    cA_lst = []
    for sub_episode in audio_time_seg:
        data = sub_episode / max(sub_episode)  # normalize data
        cA, _ = pywt.dwt(data, 'db1', 'symmetric')
        cA_lst.append(cA)

    logging.info(
        f'Took {(time.time() - dwt_time):.4f} seconds to complete dwt over each audio time segment')

    logging.info(f"length of approximation coefficients cA_lst: {len(cA_lst)}")
    logging.info(
        f"length of each chunk's approximation coefficients in cA_lst: {len(cA_lst[0])}\n")

    dwt_df = pd.DataFrame(cA_lst)
    dwt_df["y"] = y
    # need to set the column names to string, or get this error ()
    # https://github.com/pandas-dev/pandas/issues/25043
    dwt_df.columns = dwt_df.columns.astype(str)

    logging.info(
        f'Size of final dataframe for podcast id {podcast_id}: {dwt_df.shape}')


    dwt_df.to_parquet(
        f"{parquet_directory}/{podcast_id}.parquet")


def process_all_episodes(seg_len, hop_len, overlap) -> None:
    '''Processes each episode one-by-one

    Args:
        seg_len: how long each chunk of audio should be considered a single observation
        hop_len: how far forward each chunk should be moved in the audio file
        overlap: overlap threshold (how much of the particular chunk needs to contain disagreement for it to be labeled as disagreement)

    Returns:
        a new directory of parquet files in data/dwt_df_parquet
    

    '''

    combined_annotations_filepath = 'audio_annotation/outputs/compiled_annotations_df.parquet'
    _, discretized_dict = discretize_all(combined_annotations_filepath,
                                        audio_filepath='data/audio',
                                        segment_length=seg_len,
                                        hop_length=hop_len,
                                        overlap_thresh=overlap)

    start_entire_podcast_loop = time.time()


    segment_length = discretized_dict['segment_length']
    hop_length = discretized_dict['hop_length']
    overlap_thresh = discretized_dict['overlap_thresh']

    parquet_directory = f'data/dwt_df_parquet/seg_{segment_length}_hop_{hop_length}_ovlap_{overlap_thresh}'
    if os.path.exists(parquet_directory):
        print('\nDirectory already exists; please be aware you may be overwriting already created files')
    else:
        os.mkdir(parquet_directory)

    logging_location = f'{parquet_directory}/runtime_info.log'
    print(f'\nCreating logging information at {logging_location}\n')
    # https://www.machinelearningplus.com/python/python-logging-guide/
    # only needing a root log here
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s :: %(message)s', filename=logging_location)
    logging.info(f'Initializing log\n\n')

    # check = ['0pIwpmg5oPcMWJXVSyrx4E']
    # for i in check:
    for i in discretized_dict['data'].keys():

        start_individual_podcast_time = time.time()

        y = discretized_dict['data'][i]['y']

        audio_duration = discretized_dict['data'][i]['audio_duration']
        logging.info(
            f'Length of audio file for podcast id {i}: {audio_duration} seconds')

        process_entire_episode(i, segment_length, hop_length, y, audio_duration, parquet_directory)

        time_entire_individual_podcast = time.time() - start_individual_podcast_time
        logging.info(
            f'Podcast id {i} took {time_entire_individual_podcast / 60: .3f} minutes to complete\n\n')

    # TO DO: add in some logging so that it's writing to a file or something


    time_entire_podcast_loop = time.time() - start_entire_podcast_loop
    total_num_episodes = len(discretized_dict['data'].keys())
    print(
        f'\n\nFinished: took {time_entire_podcast_loop / 60: .3f} minutes to loop through all {total_num_episodes} podcast episodes\n')
    logging.info(
        f'Finished: took {time_entire_podcast_loop / 60: .3f} minutes to loop through all {total_num_episodes} podcast episodes')

if __name__=="__main__":

    parser = argparse.ArgumentParser(
        description="Looping through each of the podcast episodes, and creating dataframes (saved as parquet files) based on segment length (how long each chunk of audio should be considered a single observation), hop length (how far forward each chunk should be moved in the audio file), and overlap threshold (how much of the particular chunk needs to contain disagreement for it to be labeled as disagreement). Example use from the command line: 'python -m common.dataset_creation -seg 0.5 -hop 0.5 -ovlap 0.5'",
        epilog="The annotations that are created here can found found in the data/dwt_df_parquet folder as a parquet file")

    parser.add_argument(
        "-seg", "--segmentlength", type = float, help="segment length (how long each chunk of audio should be considered a single observation)")

    parser.add_argument(
        "-hop", "--hoplength", type = float, help="hop length (how far forward each chunk should be moved in the audio file)"
    )

    parser.add_argument(
        "-ovlap", "--overlap", type = float, help="overlap threshold (how much of the particular chunk needs to contain disagreement for it to be labeled as disagreement)"
    )

    args = parser.parse_args()

    process_all_episodes(args.segmentlength, args.hoplength, args.overlap)
