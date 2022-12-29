# Podcast Disagreement Detection
#### Capstone project for AC297R in partnership with Spotify

*Please note that due to the sensitive nature of our data, we are not able to publicly share data sets or data files. If you would like to run our code, please visit the [Spotify English-Language Podcast Dataset](https://podcastsdataset.byspotify.com/) and request access to the data. When you are able to obtain the Spotify data, please place them in the appropriate location within the data/ folder in order to properly run our code.*

[Pipenv](https://pipenv.pypa.io/en/latest/) automatically creates and manages the virtual environment for this project, meaning that it includes all packages necessary to run our code*. To activate the virtual environment, from the top-level directory, simply type

```
pipenv shell
```

### Directories:

```
Project
│   README.md
│   Pipfile
│   Pipfile.lock
│   .gitignore
|
└─── deliverables (deliverables such as slides and report)
│
└───audio_annotation (setup instructions for audio labeling; scripts for audio annotation data processing)
│
└───audio_analysis (audio analysis and modeling performed on Spotify's .ogg files)
|
└───xml_parsing (compiling rss metadata provided by Spotify into a single podcasts metadata dataset for further exploration)
|
└───text_analysis (text analysis on transcript data)
|
└───metadata (general exploration of the metadata.tsv file)
|    
└───common (utility functions commonly used across analyses, e.g. finding the union of annotations, parsing transcripts, etc.)
|
└───data (folder to place the audio, metadata, and text files that are downloaded from Spotify as well as output from working with each of these data types)
```


### Common Folders Across Directories
Within each of these top-level folders, it is typical to find *notebooks/* (containing .ipynb files used for modeling and analysis) and *outputs/* (displaying metrics or saving datasets for future use; the different data types are pkl, pq, pdf, or png files)

### Python Scripts
You will find a number of Python scripts within each of these top-level folders. Most of them can be executed either from the command line (utilizing [Argparse](https://docs.python.org/3/library/argparse.html) for its arguments) or as standalone functions. For more information on what each of the scripts do, please view their docstrings. 

These scripts were utilized to build our pipeline for  modeling. Here is a typical structure for how to produce a dataset that can be used for modeling:
1. Run *text_analysis/transcript_phrase_search* and/or *metadata/metadata_descriptions* to find potential podcasts that contain disagreement
2. Run *audio_annotation/custom_recipe* to perform disagreement labeling on your .ogg file(s) that contain disagreement
3. Run *audio_annotation/compile_annotations* to compile your annotations into a file that can be used to create your labels for modeling
4. Run *common/dataset_creation_to_aws* to perform the necessary preprocessing and create your dataframe of observations and labels that is saved to AWS as a parquet file
5. Run *audio_analysis/iid_modeling_loocv* with a model of your choice to obtain performance results on your dataset (comes with multitudes of model metrics to examine)
  - we recommend running any sort of modeling on a cloud provider (such as AWS) as these datasets can get very large and exensive

**Please note that in order to use [Prodigy](https://prodi.gy/), you must first have the appropriate license properly installed.*
