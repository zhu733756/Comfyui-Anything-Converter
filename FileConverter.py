import os
import re
import json
import tempfile
from pathlib import Path

class JsonCombiner:
    """
    合并两个 JSON（文件或 JSON 字符串），输出为新的 JSON 文件或字符串
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_a": ("STRING", {"multiline": True, "default": ""}),
                "json_b": ("STRING", {"multiline": True, "default": ""}),
                "json_a_mode": (["file", "string"], {"default": "string"}),
                "json_b_mode": (["file", "string"], {"default": "string"}),
                "merge_strategy": (["deep_merge", "a_first", "b_first"], {"default": "deep_merge"}),
            },
            "optional": {
                "indent": ("INT", {"default": 2}),
                "output_mode": (["file", "string"], {"default": "string"}),
                "output_path": ("STRING", {"default": ""}),  # ����空则使用临时文件
            }
        }

    RETURN_TYPES = ("STRING",)  # 返回最终生成的 JSON 文件路径或字符串
    RETURN_NAMES = ("output",)
    FUNCTION = "combine"
    CATEGORY = "FileConverter"

    def combine(self, json_a, json_b, json_a_mode, json_b_mode, merge_strategy, indent=None, output_mode="", output_path=""):
        def _load(j, mode):
            if mode == "file" and os.path.isfile(j.strip()):
                with open(j.strip(), "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                return json.loads(j)

        a = _load(json_a, json_a_mode)
        b = _load(json_b, json_b_mode)

        if merge_strategy == "deep_merge":
            merged = self._deep_merge(a, b)
        elif merge_strategy == "b_first":
            merged = {**a, **b}
        else:  # a_first
            merged = {**b, **a}

        if indent <= 0 or indent is None:
            indent==None
        if output_mode == "file":
            if not output_path.strip():
                fd, output_path = tempfile.mkstemp(suffix=".json", text=True)
                os.close(fd)
            else:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(merged, f, ensure_ascii=False, separators=(',', ':'))
            return (output_path,)
        else: 
            return (json.dumps(merged, ensure_ascii=False, indent=indent),)

    # ---------- 工具 ----------
    def _deep_merge(self, a: dict, b: dict) -> dict:
        result = a.copy()
        for k, v in b.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = self._deep_merge(result[k], v)
            elif k in result and isinstance(result[k], list) and isinstance(v, list):
                result[k] = list(set(result[k] + v))
            else:
                result[k] = v
        return result


class LineConverter:
    """
    按行正则替换或者字符串处理，并输出到新文件或字符串
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("STRING", {"multiline": True, "default": ""}),
                "input_mode": (["file", "string"], {"default": "file"}),
                "covert_mode": (["regex", "plain"], {"default": "regex"}),
                "pattern": ("STRING", {"default": ""}),
                "repl": ("STRING", {"default": ""}),
            },
            "optional": {
                "output_mode": (["file", "string"], {"default": "string"}),
                "output_file": ("STRING", {"default": ""}), 
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "convert"
    CATEGORY = "FileConverter"

    def convert(self, input, input_mode, covert_mode, pattern, repl, output_mode="", output_file=""):
        if input_mode == "file":
            if not os.path.isfile(input):
                raise FileNotFoundError(input)

            with open(input, "r", encoding="utf-8") as f:
                lines = f.readlines()
        else:
            lines = input.splitlines()

        flags = re.IGNORECASE if covert_mode == "regex" else 0
        if covert_mode == "regex":
            def _sub(line):
                return re.sub(pattern, repl, line, flags=flags)
        else:  # plain
            def _sub(line):
                return line.replace(pattern, repl)

        new_lines = [_sub(line) for line in lines]

        if output_mode == "file":
            if not output_file.strip():
                output_file = input  
            else:
                Path(output_file).parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w", encoding="utf-8") as f:
                f.writelines(new_lines)

            return (output_file,)
        else:  
            return ("".join(new_lines),)