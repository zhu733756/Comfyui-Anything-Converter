from .FileConverter import JsonCombiner, LineConverter, FileDictConverter


# ---------- 节点映射 ----------
NODE_CLASS_MAPPINGS = {
    "FileConverter.JsonCombiner": JsonCombiner,
    "FileConverter.LineConverter": LineConverter,
    "FileConverter.FileDictConverter": FileDictConverter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FileConverter.JsonCombiner": "Json Combiner",
    "FileConverter.LineConverter": "Line Converter",
    "FileConverter.FileDictConverter": "File Dict Converter",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]