# based on https://support.prodi.gy/t/webm/3319/4
from prodigy.recipes.audio import manual as audio_manual
from prodigy.components.loaders import Audio
from prodigy.util import get_labels
import prodigy

@prodigy.recipe(
    "custom-audio-manual",
    dataset=("Dataset to save annotations to", "positional", None, str),
    source=("Data to annotate (file path or '-' to read from standard input)", "positional", None, str),
    label=("Comma-separated label(s) to annotate or text file with one label per line", "option", "l", get_labels),
)
def custom_audio_manual(dataset, source, label):
    stream = Audio(source, file_ext=(".ogg"))
    components = audio_manual(dataset = dataset, source = stream, label = label)
    #components["config"]["audio_rate"] = 2.0
    return components
