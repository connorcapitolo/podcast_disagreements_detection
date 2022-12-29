#!/usr/bin/env python3

import time
import re
# from typing import List
import argparse

import pandas as pd
# default='warn' # https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
pd.options.mode.chained_assignment = None


def episode_description_phrases(text_file: str, metadata_path: str, *, parquet: bool = False, csv: bool = False) -> pd.DataFrame:
    '''Reads in a txt file where each line is a different phrase should should be searched for in the episode_description column of metadata.tsv

    Args:
        text_file: the txt file with the phrases on each line
        metadata_path: path to the metadata.tsv file
        parquet: if the pandas dataframe being returned should be saved as a parquet file
        csv: if the pandas dataframe being returned should be saved as a csv file

    Returns:
        metadata_phrases_to_find: the metadata.tsv file, along with a 'phrase_count' column that is sorted by the number of phrases that are contained for the corresponding episode_description

    Example of how to run as function:
        >>> import audio_annotation
        >>> check = audio_annotation.episode_description_phrases('audio_annotation/metadata-search.txt', 'data/metadata/metadata.tsv', csv = True)
        >>> check.shape
            (41, 13)

    Example of how to run as script:
        >>> python -m audio_annotation.metadata_descriptions --parquet

    '''

    phrases_to_find = []
    with open(text_file, 'r') as file:
        lines = file.readlines()
        phrases_to_find = [line.rstrip().lower() for line in lines]

    # reading in csv to pandas dataframe
    metadata_df = pd.read_csv(metadata_path, sep='\t')
    
    # remove rows that are null in the episode duration
    metadata_df = metadata_df.loc[metadata_df['episode_description'].notnull() , :]
    
    # contains an indicator vector of booleans for if it contains the 
    # note that if the word argument is 'disagree', then phrases_to_find like 'disagreement' will also appear in this
    # https://stackoverflow.com/questions/26577516/how-to-test-if-a-string-contains-one-of-the-substrings-in-a-list-in-pandas
    indicator = metadata_df['episode_description'].str.contains('|'.join(phrases_to_find), case = False)
    metadata_with_phrases_to_find = metadata_df.loc[indicator, :]

    # https://stackoverflow.com/questions/17573814/count-occurrences-of-each-of-certain-words-in-pandas-dataframe
    # ignoring case
    metadata_with_phrases_to_find["phrase_count"] = metadata_with_phrases_to_find['episode_description'].str.count(
        '|'.join(phrases_to_find), flags=re.IGNORECASE)

    # sorting by count
    metadata_with_phrases_to_find.sort_values(by="phrase_count", ascending=False, inplace=True)

    current_time = round(time.time())
    if parquet:
        saving_location = f'data/metadata/metadata_with_phrases_to_find_{current_time}.parquet'
        metadata_with_phrases_to_find.to_parquet(saving_location)
        print(f'Saved parquet file to ')
    if csv:
        saving_location = f'data/metadata/metadata_with_phrases_to_find_{current_time}.csv'
        metadata_with_phrases_to_find.to_csv(saving_location)
    # print(metadata_with_word.head())
    # print(metadata_with_phrases_to_find.shape[0])
    return metadata_with_phrases_to_find
    # return metadata_with_word.head()

if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Based on the metadata.tsv, returns a dataframe that includes any matches on the list of strings argument regarding each of the 106k episode descriptions. From the top level directory, can run as a script with 'python -m audio_annotation.metadata_descriptions'")

    # accepting a string that is then being parsed based on the comma
    # parser.add_argument("phrases", type=lambda phrases: [phrase for phrase in phrases.split(',')], help="A list of strings that should be searched for in the episode description for each of the podcasts in metadata.tsv")

    parser.add_argument(
        "-mi", "--metadatainfo", default="audio_annotation/metadata-search.txt", help="Specify the relative directory to the .txt file mapping annotators to .jsonl files"
    )

    parser.add_argument("--tsvpath", default='data/metadata/metadata.tsv', help="Location of the metadata.tsv file")

    # will automatically store "False" if this is not passed i.e. in the Terminal, does not contain "--parquet" as an argument
    parser.add_argument("--parquet", action="store_true", help="Whether or not to pickle the returned dataframe")

    parser.add_argument("--csv", action="store_true",
                        help="Whether or not to create a csv of the returned dataframe")



    args = parser.parse_args()

    episode_description_phrases(
        args.metadatainfo, args.tsvpath, parquet=args.parquet, csv=args.csv)
