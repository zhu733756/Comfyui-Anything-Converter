from .FileConverter import LineConverter, FileDictConverter, FileSplitter
from .JsonConverter import JsonCombiner, JsonParser, JsonPromptProcessor
from .ImageConverter import SaveImage, LoadImageTextSetFromMetadata
from .TextConverter import PromptTemplateText


# ---------- 节点映射 ----------
NODE_CLASS_MAPPINGS = {
    "JsonCoverter.JsonCombiner": JsonCombiner,
    "JsonCoverter.JsonParser": JsonParser,
    "JsonCoverter.JsonPromptProcessor": JsonPromptProcessor,
    "FileConverter.LineConverter": LineConverter,
    "FileConverter.FileDictConverter": FileDictConverter,
    "FileConverter.FileSplitter": FileSplitter,
    "ImageCoverter.SaveImage": SaveImage,
    "ImageCoverter.LoadImageTextSetFromMetadata": LoadImageTextSetFromMetadata,
    "TextConverter.PromptTemplateText": PromptTemplateText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JsonCoverter.JsonCombiner": "JsonCombiner",
    "JsonCoverter.JsonParser": "JsonParser",
    "JsonCoverter.JsonPromptProcessor": "JsonPromptProcessor",
    "FileConverter.LineConverter": "LineConverter",
    "FileConverter.FileDictConverter": "FileDictConverter",
    "FileConverter.FileSplitter": "FileSplitter",
    "ImageCoverter.SaveImage": "SaveImage",
    "ImageCoverter.LoadImageTextSetFromMetadata": "LoadImageTextSetFromMetadata",
    "TextConverter.PromptTemplateText": "PromptTemplateText",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]