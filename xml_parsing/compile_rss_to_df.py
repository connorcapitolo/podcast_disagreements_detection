import os
import pandas as pd
import xml.etree.ElementTree as ET
import argparse

def get_show_info(show_level_tree):
    """Returns list with metadata for the given show's xml tree"""
    show_itunes_author = show_level_tree.find(f"{ITUNES_PREFIX}author").text
    show_itunes_category = show_level_tree.find(f"{ITUNES_PREFIX}category")
    show_itunes_category_text = show_itunes_category.get("text")
    show_itunes_subcategory = show_itunes_category.find(f"{ITUNES_PREFIX}category")
    if show_itunes_subcategory is None:
        show_itunes_subcategory_text = ''
    else:
        show_itunes_subcategory_text = show_itunes_subcategory.get("text")
    return [show_itunes_author, show_itunes_category_text, show_itunes_subcategory_text]

def get_episode_info(show_level_tree, fname):
    """Returns list with metadata for each episode in the given show; each element of the list is one episode's metadata"""
    episode_data = []
    for episode in show_level_tree.findall("item"):
        episode_attributes = []
        episode_title = episode.find("title")
        duration = episode.find(f"{ITUNES_PREFIX}duration")
        season = episode.find(f"{ITUNES_PREFIX}season")
        episode_number = episode.find(f"{ITUNES_PREFIX}episode")
        episode_type = episode.find(f"{ITUNES_PREFIX}episodeType")
    
        for attribute in [episode_title, duration, season, episode_number, episode_type]:
            if attribute is None:
                episode_attributes.append("")
            else:
                episode_attributes.append(attribute.text)
        episode_data.append([fname] + episode_attributes)
    return episode_data

def compile_xml(rootDir):
    """Returns list with metadata for all xml files in rootDir"""
    num_files = 0
    num_success_files = 0
    num_fail_files = 0
    
    success_show_info = []; success_episode_info = []
    fail_shows = []
    
    for i in os.listdir(rootDir):
        current_dir = rootDir + i + '/' 
        for dirName, subdirList, fileList in os.walk(current_dir):
            for fname in fileList:
                if '.xml' in fname: 
                    num_files += 1
                    file_path = os.path.join(rootDir, dirName, fname)
                    try:
                        tree = ET.parse(file_path)
                        show_level_tree = tree.getroot().find("channel")
                        show_info = [fname] + get_show_info(show_level_tree)
                        episode_info = get_episode_info(show_level_tree, fname)
                        success_show_info.append(show_info)
                        success_episode_info.extend(episode_info)
                        num_success_files += 1
                    except:
                        fail_shows.append(file_path)
                        num_fail_files += 1
            
    print('total files: ',num_files)
    print('failed files: ',num_fail_files)
    return success_show_info, success_episode_info, fail_shows

if __name__=='__main__':   
    # Define argument parser (using docstring at top of file):
    parser = argparse.ArgumentParser(description="Parse RSS data to extract the show id, creator, genre, etc. at the show and episode level from each of the XML files. An example of how to run this (please make sure to use the command line) is 'python compile_rss_to_df.py -rd spotify-podcasts-2020/show-rss/'")
    # Define arguments (action="store_true" means True if specified and false otherwise):
    parser.add_argument(
        "-rd", "--rootdir", help="Specify the root directory, which should have the innermost sub directory as show-rss/ e.g. spotify-podcasts-2020/show-rss/")
    # Print usage notes:
    parser.print_help()
    # Parse arguments:
    args = parser.parse_args()
    # Extract arguments from parser:
    root_dir = args.rootdir
    # Set the directory you want to start from
    # rootDir = 'spotify-podcasts-2020/show-rss/'
    #'spotify-podcasts-2020/show-rss/'

    # itunes:category in the xml is actually an xml element with tag name "ITUNES_PREFIX:category"
    ITUNES_PREFIX = "{http://www.itunes.com/dtds/podcast-1.0.dtd}"

    success_show_info, success_episode_info, fail_shows = compile_xml(root_dir)
    show_df = pd.DataFrame(success_show_info, columns = ["show_id", "author", "category", "subcategory"])
    show_df.to_pickle("show_rss_data.pkl")

    episode_df = pd.DataFrame(success_episode_info, columns = ["show_id", "episode_title", "duration", "season", "episode_number", "episode_type"])
    episode_df.to_pickle("episode_rss_data.pkl")
