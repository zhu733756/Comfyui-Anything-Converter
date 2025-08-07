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
                "output_path": ("STRING", {"default": ""}),  # 留空则使用临时文件
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
            indent = None
        if output_mode == "file":
            if not output_path.strip():
                fd, output_path = tempfile.mkstemp(suffix=".json", text=True)
                os.close(fd)
            else:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(merged, f, ensure_ascii=False, separators=(',', ':') if indent is None else (',', ': '))
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
                "convert_mode": (["regex", "plain", "dedup", "merge"], {"default": "regex"}),
                "pattern": ("STRING", {"default": ""}),
                "repl": ("STRING", {"default": ""}),
                "delimiter": ("STRING", {"default": "[,|，]"}),  # 新增分隔符参数
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

    def convert(self, input, input_mode, convert_mode, pattern, repl, delimiter, output_mode="", output_file=""):
        if input_mode == "file":
            if not os.path.isfile(input):
                raise FileNotFoundError(input)

            with open(input, "r", encoding="utf-8") as f:
                lines = f.readlines()
        else:
            lines = input.splitlines()

        if convert_mode == "regex":
            flags = re.IGNORECASE
            new_lines = [re.sub(pattern, repl, line, flags=flags) for line in lines]
        elif convert_mode == "plain":
            new_lines = [line.replace(pattern, repl) for line in lines]
        elif convert_mode in ["dedup", "merge"]:
            new_lines = self._process_lines(lines, delimiter, convert_mode)
        else:
            raise ValueError(f"Unsupported convert_mode: {convert_mode}")

        if output_mode == "file":
            if not output_file.strip():
                output_file = input
            else:
                Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, "w", encoding="utf-8") as f:
                for line in new_lines:
                    f.writelines(line)

            return (output_file,)
        else:
            return ("\n".join(new_lines),)
          
    def _process_lines(self, lines, delimiter, mode):
      seen = {}
      new_lines = []

      for line in lines:
          fields = re.split(delimiter, line.strip())
          if fields:
              key = fields[0]
              if mode == "dedup":
                  if key not in seen:
                      seen[key] = True
                      new_lines.append(line)
              elif mode == "merge":
                  if key not in seen:
                      seen[key] = fields[1:]  # 存储后续字段
                  else:
                      seen[key] = list(set(seen[key] + fields[1:]))  # 合并并去重

      # 生成最终的合并后行
      if mode == "merge":
          for key, values in seen.items():
              merged_line = ",".join([key] + values)
              new_lines.append(merged_line)

      return new_lines
    
class FileDictConverter:
    """
    遍历给定的字符串或文件内容，当其中有 key 存在于给定的 dict 中时，把 key 指定的内容替换成 dict 对应的 value
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("STRING", {"multiline": True, "default": ""}),
                "dict_input": ("STRING", {"multiline": True, "default": ""}),
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

    def convert(self, input, dict_input, output_mode="string", output_file=""):
        def _load_content(content):
            """加载输入内容"""
            if  os.path.isfile(content.strip()):
                with open(content.strip(), "r", encoding="utf-8") as f:
                    return f.read()
            else:
                return content

        def _load_dict(d):
            """加载替换字典"""
            if isinstance(d, dict):
                return d
            
            if isinstance(d, str):
                if os.path.isfile(d.strip()):
                    with open(d.strip(), "r", encoding="utf-8") as f:
                        return json.load(f)
                else:
                    return json.loads(d)

        # 加载输入内容
        input_content = _load_content(input)
        # 加载替换字典
        replace_dict = _load_dict(dict_input)

        # 替换逻辑
        for key, value in replace_dict.items():
            input_content = input_content.replace(key, value)

        # 输出处理
        if output_mode == "file":
            if not output_file.strip():
                output_file = tempfile.mkstemp(suffix=".txt", text=True)[1]
            else:
                Path(output_file).parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(input_content)
            return (output_file,)
        else:
            return (input_content,)


# 示例用法
if __name__ == "__main__":
    converter = LineConverter()
    input_text = '''大主宰，林动
斗破苍穹，小薰儿
斗破苍穹，消炎'''
    result = converter.convert(input_text, "string", "merge", "", "", "[,|，]", "string")
    print(result)
    result = converter.convert(input_text, "string", "merge", "", ",", "，", "string")
    print(result)
    
    converter = FileDictConverter()
    input_text = "Hello, world! This is a test."
    replace_dict = {"world": "Earth", "test": "example"}
    output = converter.convert(input_text, replace_dict, "string")
    print(output)
    
    replace_dict_str = '{"world": "Earth", "test": "example"}'
    output = converter.convert(input_text,  replace_dict_str, "string")
    print(output)