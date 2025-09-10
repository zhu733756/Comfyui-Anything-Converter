# ComfyUI-Anything-Converter

This is a custom node extension designed for ComfyUI, providing JSON merging and text line conversion functionalities.

### Overview

ComfyUI-Anything-Converter includes the following core nodes:

JsonConverter

- **Json Combiner**: merges two JSON files or strings.
- **Json Parser**: parses any JSON string into a JSON object.

FileConverter

- **Line Converter**: performs per-line regex replacement or string processing.
- **File Dict Converter**: replaces every key found in a dictionary with its corresponding value inside a file.
- **File Splitter**: splits a file into two files by inserting a newline before or after each regex-matched content.

ImageConverter

- **Save Image**: based on [ComfyUI-KJNodes: Save Image KJ](https://github.com/kijai/ComfyUI-KJNodes), retrieves paths of all generated images.
- **Load Image**: based on [ImageStitch](https://github.com/comfyanonymous/ComfyUI/blob/e08ecfbd8a9deda8939b14d7f1ff7d7139f1a4ed/comfy_extras/nodes_images.py#L235), using metadata by character2prompt.json and character2img.json to load image.

### Installation

1.Clone or download this repository to your local machine.
2.Place the ComfyUI-File-Converter folder into the custom_nodes directory of your ComfyUI project.
3.Restart ComfyUI to load the new nodes.

### Usage

- Locate the Json Combiner and Line Converter nodes in the ComfyUI node list and drag them onto the canvas.
- Configure the node parameters and run the workflow to ensure everything works as expected.
- Check the workflow.json file:
  ![workflow.json](workflow.png)
