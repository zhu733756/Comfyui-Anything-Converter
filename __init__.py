from .FileConverter import LineConverter, FileDictConverter, FileSplitter
from .JsonCoverter import JsonCombiner, JsonParser, JsonPromptProcessor
from .ImageCoverter import SaveImage


# ---------- 节点映射 ----------
NODE_CLASS_MAPPINGS = {
    "JsonCoverter.JsonCombiner": JsonCombiner,
    "JsonCoverter.JsonParser": JsonParser,
    "JsonCoverter.JsonPromptProcessor": JsonPromptProcessor,
    "FileConverter.LineConverter": LineConverter,
    "FileConverter.FileDictConverter": FileDictConverter,
    "FileConverter.FileSplitter": FileSplitter,
    "ImageCoverter.SaveImage": SaveImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JsonCoverter.JsonCombiner": "JsonCombiner",
    "JsonCoverter.JsonParser": "JsonParser",
    "JsonCoverter.JsonPromptProcessor": "JsonPromptProcessor",
    "FileConverter.LineConverter": "LineConverter",
    "FileConverter.FileDictConverter": "FileDictConverter",
    "FileConverter.FileSplitter": "FileSplitter",
    "ImageCoverter.SaveImage": "SaveImage",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]