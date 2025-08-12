import os
import re
import json
import tempfile
from pathlib import Path
import logging



logger = logging.getLogger(__name__)


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
        

class FileSplitter:
    """
    根据给定的正则匹配，按匹配的内容前一行或者后一行换行符分割文件，分割成两个文件，并支持文件的名称自定义
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_file": ("STRING", {"default": ""}),
                "pattern": ("STRING", {"default": ""}),
                "split_mode": (["before", "after"], {"default": "after"}),
                "output_file_1": ("STRING", {"default": ""}),
                "output_file_2": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("output_file_1", "output_file_2",)
    FUNCTION = "split"
    CATEGORY = "FileConverter"

    def split(self, input, pattern, split_mode, output_file_1, output_file_2):
        if os.path.isfile(input):
          with open(input, "r", encoding="utf-8") as f:
            lines = f.readlines()
        else:
            lines = str(input)

        split_index = None
        for i, line in enumerate(lines):
            if re.search(pattern, line):
                split_index = i + (1 if split_mode == "after" else 0)
                break

        if split_index is None:
            raise ValueError(f"Pattern not found in file: {pattern}")

        part1 = lines[:split_index]
        part2 = lines[split_index:]

        output_dir = Path(output_file_1).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_file_1, "w", encoding="utf-8") as f:
            f.writelines(part1)

        with open(output_file_2, "w", encoding="utf-8") as f:
            f.writelines(part2)

        return (output_file_1, output_file_2)

def run_all_tests():
    # 1. LineConverter.merge
    lc = LineConverter()
    src = "大主宰，林动\n斗破苍穹，小薰儿\n斗破苍穹，消炎"
    result = lc.convert(src, "string", "merge", "", "", "[,|，]", "string")[0]
    assert "大主宰,林动" in result and "斗破苍穹,小薰儿,消炎" in result or "斗破苍穹,消炎,小薰儿" in result, f"LineConverter merge test failed: {result}"

    # 2. FileDictConverter
    fdc = FileDictConverter()
    txt = "Hello, world! This is a test."
    repl = {"world": "Earth", "test": "example"}
    out = fdc.convert(txt, repl, "string")[0]

    # 3. FileSplitter
    fs = FileSplitter()
    src_file = "test_split.txt"
    with open(src_file, "w", encoding="utf-8") as f:
        f.write("line1\nline2\nline3\nline4\n")
    output_file_1, output_file_2 = fs.split(src_file, "line3", "after", "output1.txt", "output2.txt")
    with open(output_file_1, "r", encoding="utf-8") as f:
        content = str(f.read())
        assert  content == "line1\nline2\nline3\n", f"got {content}, expect line1\nline2\nline3\n"
    with open(output_file_2, "r", encoding="utf-8") as f:
        content = f.read()
        assert content == "line4\n", f"got {content}, expect line4\n"
    os.remove(src_file)
    os.remove(output_file_1)
    os.remove(output_file_2)
    
    # 5. JsonPromptProcessor
    print("✅ 所有测试通过！")

if __name__ == "__main__":
    run_all_tests()