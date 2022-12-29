# Audio annotation setup

This directory contains setup instructions and scripts for loading and combining audio annotations.

### Overview:
1. Annotate data in Prodigy using the instructions below
2. Run `compile_annotations.py` (note the arguments it accepts) to produce a pandas DataFrame (by default, to the directory `outputs/compiled_annotation_df.pkl`)
3. (Optional) Visualize audio annotations using `notebooks/visualize_compiled_annotations_df.ipynb`

### Prodigy annotation setup instructions:
1. (Optional) Create and activate a virtual environment via conda:
```
conda create -n capstone python=3.8
conda activate capstone
```
2. Install Prodigy via pip (replace XXXX with license key):
```
pip install prodigy -f https://XXXX-XXXX-XXXX-XXXX@download.prodi.gy
```

Instead of installing with https, you can also install from a *.whl* file (basically just a pre-compiled Python package)

To do this, just point to `pip` to the directory **directory** using the `-f` option:

```
pip install prodigy -f /path/to/wheels
```

See [this link](https://prodi.gy/docs/install) for further instructions if you're still confused.

3. Download custom_recipe.py from this repo (e.g. via git cloning this repo)

4. Start the annotation server, using the custom function "custom-audio-manual" from python file "custom_recipe.py", saving to a dataset named "disagree_data" (this name can be changed as desired), loading in .ogg files from "./audio_folder" with label "DISAGREE"
* *Note: The below assumes that the folder w/ the audio files, and custom_recipe are in the current directory*
* *Note: All files in audio_folder must be .ogg files, i.e. currently nested directory search not yet supported*
* *Note: The default location of the Prodigy dataset (called "disagree_data" in this case) can be found by running "prodigy stats" on the command line; note that the same dataset name can be used for multiple podcast episode annotations, in which case the annotations will be appended to the dataset*
```
prodigy custom-audio-manual disagree_data ./audio_folder --label DISAGREE -F custom_recipe.py
```

5. As prompted, visit the localhost link in a web browser

6. Highlight portions with disagreement, and click the green check button at bottom when ready to 'submit'; the next audio file in the audio folder will automatically load

7. Once finished (or tired), click save button on the top left to save annotations

8. Close browser window, ctrl+c in terminal to stop the server. The annotations will be saved in the destination defined in part 4 (e.g. "disagree_data").

9. (Optional) To view annotations and save to json file, can run:
```
prodigy db-out disagree_data > ./annotations.jsonl
```
