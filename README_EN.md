# ComfyUI-File-Converter

This is a custom node extension designed for ComfyUI, providing JSON merging and text line conversion functionalities.

### Overview

ComfyUI-File-Converter includes two main nodes:

- Json Combiner: Merges two JSON files or strings.
- Line Converter: Performs regex replacement or string processing on text lines.
- File Dict Converter: Replaces keys in a file that exist in a dict with their corresponding values.

### Installation

1.Clone or download this repository to your local machine.
2.Place the ComfyUI-File-Converter folder into the custom_nodes directory of your ComfyUI project.
3.Restart ComfyUI to load the new nodes.

### Usage

- Locate the Json Combiner and Line Converter nodes in the ComfyUI node list and drag them onto the canvas.
- Configure the node parameters and run the workflow to ensure everything works as expected.
- Check the workflow.json file:
![workflow.json](workflow.png)
