import json
import pandas as pd
import sys, os
import argparse
import ast
import pickle
import subprocess
import time


def convert_to_mp3(audio_filename_ogg):
    subprocess.call([
        'ffmpeg',
        '-i',
        audio_filename_ogg,
        audio_filename_ogg[:-4] + '.mp3'
    ], shell=False)


def get_audio_spans(example, annotator_label):
    '''
    Returns list of lists with text, label, start, and end
    example: single row of the .jsonl file, corresponding to a single .ogg file annotation (there will most likely be multiple rows corresponding to a single .ogg file performed by the annotator where disagreement has been recognized)
    annotator_label: name of the annotator
    '''
    audio_spans = example['audio_spans']
    return [[annotator_label, example['text'], a['label'], a['start'], a['end']] for a in audio_spans]

def get_annotations(jsonl_filepath, annotator_label):
    '''
    Returns DataFrame of annotations
    jsonl_filepath: filepath of .jsonl file (containing annotations), exported via e.g. db-out
    annotator_label: name of the annotator (corresponding to the jsonl file)
    '''
    dataset = [json.loads(episode) for episode in open(jsonl_filepath, 'r')]
    annotations = []
    for example in dataset:
        if example['answer']=='accept':
            annotations.extend(get_audio_spans(example, annotator_label))
    return pd.DataFrame(annotations, columns = ['annotator', 'text', 'label', 'start', 'end'])

def combine_annotations(annotations_directory, annotation_info):
    '''
    Returns DataFrame with all annotations in the annotations_directory
    annotation_info: dictionary with keys (annotator_label) and values (jsonl_filepath)

    Example dataframe output:
      annotator                    text     label      start        end
    0    connor  2KRFxcWULTtVe0LXBjXduT  DISAGREE   2.374966   4.124941
    1    connor  2KRFxcWULTtVe0LXBjXduT  DISAGREE  27.674606  29.224584
    2    connor  2KRFxcWULTtVe0LXBjXduT  DISAGREE   9.324867  11.274840
    3    connor  73qaKP1ZaRmlOwhAwBHrug  DISAGREE   6.674372   8.774175
    4    connor  73qaKP1ZaRmlOwhAwBHrug  DISAGREE  15.473545  18.073300
    '''


    annotation_dfs = []
    for a in annotation_info:
        annotation_dfs.append(get_annotations(os.path.join(annotations_directory, annotation_info[a]), a))
    combined_annotations = pd.concat(annotation_dfs)
    return combined_annotations


def compile_annotations(rootdir: str = "audio_annotation/annotations_jsonl", annotationinfo: str = "audio_annotation/annotation_info.txt", discretize: bool = False, destination: str = "audio_annotation/outputs", filename: str = "compiled_annotations_df", parquet: bool = False):
    '''Reads in all your annotations.jsonls files and combines them the information together for use in a Pandas dataframe

    Args:
        rootdir: 
            Specify the directory where your .jsonl files are located
        annotationinfo: 
            Specify the relative directory to the .txt file mapping annotators to .jsonl files
        discretize: 
            whether or not you want to test out the discretization
        destination: 
            Specify the relative directory for the output (pandas DataFrame with all annotations
        filename: 
            Name of the parquet file
        parquet: 
            If the pandas dataframe being returned should be saved as a parquet file

    Returns:
        combined_annotations: 
            combinations of all the annotations of the jsonl files from Prodigy in a Pandas Datframe

    Example of how to run as function:
        >>> import audio_annotation
        >>> check = audio_annotation.compile_annotations()
        >>> check.shape
            (193, 5)

    Example of how to run as script:
        python -m audio_annotation.compile_annotations --parquet

    '''

    with open(annotationinfo, "r") as f:
        contents = f.read()
        annotation_info_dict = ast.literal_eval(contents)

    combined_annotations = combine_annotations(rootdir, annotation_info_dict)
    combined_annotations_df_filepath = os.path.join(
        destination, filename) + f'_{round(time.time())}.parquet'


    if parquet:
        combined_annotations.to_parquet(combined_annotations_df_filepath)
        print(f'Parquet file location: {combined_annotations_df_filepath}')

    # Optionally also produce a dictionary mapping episode_id to a list of tp instances (tuples of unioned disagreement times)
    if discretize:
        SEGMENT_LENGTH = 2.5
        HOP_LENGTH = 0.5
        OVERLAP_THRESH = 0.2

        parent_directory = os.path.dirname(os.getcwd())
        sys.path.append(parent_directory)
        from common.annotation_utils import discretize_all

        unioned_annotations, discretized_dict = discretize_all(combined_annotations_df_filepath,
                                                                audio_filepath = '../data/audio',
                                                                segment_length = SEGMENT_LENGTH,
                                                                hop_length = HOP_LENGTH,
                                                                overlap_thresh = OVERLAP_THRESH) 

        f = open(os.path.join(args.destination, "unioned_annotations_dict.pkl"), "wb")
        pickle.dump(unioned_annotations, f)
        f.close()
        print('unioned_annotations_dict saved')

        f = open(os.path.join(args.destination, "discretized_dict.pkl"), "wb")
        pickle.dump(discretized_dict, f)
        f.close()
        print("discretized_dict saved")

    return combined_annotations


if __name__ == "__main__":


    parser = argparse.ArgumentParser(
        description="Extracting all the annotations from the .jsonl files output by Prodigy into a  dataframe saved as a parquet file. An example of how to run this (on the command line) from the top-level directory is: 'python -m audio_annotation.compile_annotations'",
        epilog="The annotations that are created here in the parquet file are then used for audio analysis")

    # Print usage notes:
    # parser.print_help()

    parser.add_argument(
        "-rd", "--rootdir", default = "audio_annotation/annotations_jsonl", help="Specify the directory where your .jsonl files are located")

    parser.add_argument(
        "-ai", "--annotationinfo", default="audio_annotation/annotation_info.txt", help="Specify the relative directory to the .txt file mapping annotators to .jsonl files"
    )

    parser.add_argument(
        "-di", "--discretize", action = "store_true", default = False,
    )

    parser.add_argument(
        "-dest", "--destination", default="audio_annotation/outputs", help="Specify the relative directory for the output (pandas DataFrame with all annotations")

    parser.add_argument(
        "-fn", "--filename", default="compiled_annotations_df", help="Name of the parquet file")

    parser.add_argument("--parquet", action="store_true",
                        help="Whether or not to pickle the returned dataframe")

    args = parser.parse_args()

    compile_annotations(args.rootdir, args.annotationinfo, args.discretize, args.destination, args.filename, args.parquet)
