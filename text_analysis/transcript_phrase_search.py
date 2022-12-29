#!/usr/env/bin python

import os
import argparse
import pandas as pd
import numpy as np
import time
# could use relative import if there's a subdirectory within common that calling from
from common.transcript_utils import process_transcript


def get_lexicon_matrix(json_filepath, lexicon):
    """Return matrix with counts for each set of words defined in the lexicon dictionary"""
    # view the process_transcript for more information on this
    transcript = process_transcript(json_filepath)
    # gives the text chunks from the transcript (e.g. ["I'm Bill Simmons here...)
    # the transcript data is a list of strings based on Google's Speech-to-Text API
    text_segments = transcript['text_chunks']
    # print(len(text_segments)) # prints number of elements in the list

    # creating an array of the number of segments and the lexicon size
    X_counts = np.zeros((len(text_segments), len(lexicon)))

    # loop through each of the segments
    for i, segment in enumerate(text_segments):
        # each segment is just a long string, and making it case-insensitive
        segment = segment.lower()
        # print(segment) # i'm bill simmons here...

        for j, phrase_to_check in enumerate(lexicon):

            # checking if the phrase is within the segmenet
            X_counts[(i, j)] += segment.count(phrase_to_check)

    # getting the sum column-wise for each of the phrases over the podcast transcript
    return list(X_counts.sum(axis=0))


def transcript_phrase_search(root_directory: str = 'text_analysis', text_file: str = 'text_analysis/transcript-search.txt', output: str = 'data', parquet: str = False):
    '''Function that returns how many times the phrase appears in each podcast episode

    Args:
    
    Returns:
        df: 
            pandas DataFrame that contains podcast episode id, each of the phrases and their counts, as well as their total phrase count; sorted by total phrase count

    Example of how to run as function:
        >>> import text_analysis
        >>> check = text_analysis.transcript_phrase_search(parquet=True)
        Transcript search file location: data/transcript_search/transcript_search_1639262199.parquet
        >>> check.head()
                                    i disagree  nope  phrase_count
        7C5qwcOZVhmS1TpOvc53d9.json         2.0   0.0           2.0
        2HlZHW5dbKgmasTzq9Tw19.json         1.0   0.0           1.0
        6e0oE3YKXfFfwhAGYt6sTu.json         0.0   0.0           0.0
        4azJHZtgPnwJWofY5uv5BR.json         0.0   0.0           0.0
        3p1Ktxuqsqq90wrfGyuZm0.json         0.0   0.0           0.0
    
    Example of how to run as script:
        >>> python -m text_analysis.transcript_phrase_search --parquet
    '''

    phrases_to_find = []
    with open(text_file, 'r') as file:
        lines = file.readlines()
        phrases_to_find = [line.rstrip().lower() for line in lines]

    # print(phrases_to_find)  # ['bill simmons', 'jayson']
    # print(type(phrases_to_find))  # <class 'list'>

    # show = 'show_55R0CZoqgY3hcmCmVkmBig/0AUxUB5ukD9c0APU6l2Gcv.json'

    # current_dir = ''
    corpus_info = {}

    # this will recursively scan through all potential subdirectories for this file
    for dirName, _, fileList in os.walk(root_directory):
        for fname in fileList:
            if fname.endswith('.json'):
                # num_files += 1
                file_path = os.path.join(dirName, fname)
                phrase_counts = get_lexicon_matrix(file_path, phrases_to_find)
                corpus_info[fname] = phrase_counts
                # num_success_files += 1

    # phrase_counts = get_lexicon_matrix(show, phrases_to_find)
    # corpus_info[show] = phrase_counts

    df = pd.DataFrame(corpus_info.values(),
                    columns=phrases_to_find, index=corpus_info.keys())
    df["phrase_count"] = df.sum(axis=1)
    df.sort_values(by="phrase_count", ascending=False, inplace=True)


    if parquet:

        folder_name = os.path.join(output, 'transcript_search')
        
        # create folder if it doesn't exist
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)

        # output 
        output_file = os.path.join(
            folder_name, f'transcript_search_{round(time.time())}.parquet')
        df.to_parquet(output_file)

        print(f'Transcript search file location: {output_file}')

    return df

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=". An example of how to run this (on the command line) from top level directory is: 'python -m text_analysis.transcript_phrase_search'",
        epilog="The annotations that are created here in the pickle file are then used for audio analysis")

    # Print usage notes:
    # parser.print_help()

    parser.add_argument(
        "-rd", "--rootdir", default="text_analysis", help="Specify the directory where your .json transcript files are located")

    parser.add_argument(
        "-ti", "--transcriptinfo", default="text_analysis/transcript-search.txt", help="Specify the relative directory to the .txt file mapping annotators to .jsonl files"
    )

    parser.add_argument(
        "-dest", "--destinationfolder", default="data", help="Specify the directory for the output (pandas DataFrame with all annotations)")

    parser.add_argument("--parquet", action="store_true",
                        help="Whether or not to pickle the returned dataframe")
    
    args = parser.parse_args()

    transcript_phrase_search(
        args.rootdir, args.transcriptinfo, args.destinationfolder, args.parquet)
