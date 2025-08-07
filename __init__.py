from .FileConverter import JsonCombiner, LineConverter, FileDictConverter,JsonParser


# ---------- 节点映射 ----------
NODE_CLASS_MAPPINGS = {
    "FileConverter.JsonCombiner": JsonCombiner,
    "FileConverter.LineConverter": LineConverter,
    "FileConverter.FileDictConverter": FileDictConverter,
    "FileConverter.JsonParser": JsonParser,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FileConverter.JsonCombiner": "Json Combiner",
    "FileConverter.LineConverter": "Line Converter",
    "FileConverter.FileDictConverter": "File Dict Converter",
    "FileConverter.JsonParser": "Json Converter",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]