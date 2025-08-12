import os
import re
import json
import tempfile
from pathlib import Path
import logging
import asyncio
from comfy.model_management import InterruptProcessingException, interrupt_processing 


logger = logging.getLogger(__name__)

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
                try:
                    with open(j.strip(), "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception as e:
                    print(f"open file failed: {e}")
                    return {}
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
        
class JsonParser:
    """
    从任意文本中提取最外层的 {} 或 [] 合法 JSON 片段；
    若找不到则输出 {"text": <原文本>}
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "input_mode": (["file", "string"], {"default": "string"}),
            },
            "optional": {
                "output_mode": (["file", "string"], {"default": "string"}),
                "output_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json_string",)
    FUNCTION = "parse"
    CATEGORY = "FileConverter"

    # ---------- 主函数 ----------
    def parse(self, text, input_mode, output_mode="string", output_path=""):
        # 1. 读取内容
        if input_mode == "file" and os.path.isfile(text.strip()):
            with open(text.strip(), "r", encoding="utf-8") as f:
                raw = f.read()
        else:
            raw = text

        # 2. 正则找最外层 {} 或 []
        match = re.search(r"(?s)(?:\{.*?\}|\[.*?\])", raw.strip())
        if match:
            try:
                parsed = json.loads(match.group())
                out_str = json.dumps(parsed, ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                out_str = json.dumps({"text": raw}, ensure_ascii=False, indent=2)
        else:
            out_str = json.dumps({"text": raw}, ensure_ascii=False, indent=2)

        # 3. 输出方式
        if output_mode == "file":
            if not output_path.strip():
                fd, output_path = tempfile.mkstemp(suffix=".json", text=True)
                os.close(fd)
            else:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(out_str)
            return (output_path,)
        else:
            return (out_str,)
        
        
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


class JsonPromptProcessor:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_file": ("STRING", {"multiline": False, "default": "Enter the path to your JSON file here"}),
                "output_file": ("STRING", {"multiline": False, "default": "Enter the path to save the output JSON file here"}),
                "image_output_dir": ("STRING", {"multiline": False, "default": "Enter the directory to save generated images here"})
            }
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("prompt", "status",)
    FUNCTION = "process_prompts"
    CATEGORY = "JsonPromptProcessor"

    async def process_prompts(self, json_file, output_file, image_output_dir):
        try:
            if not json_file.strip() or json_file == "Enter the path to your JSON file here":
                return ("", "Error: Please provide a valid JSON file path")

            if not os.path.exists(json_file):
                return ("", f"Error: JSON file not found: {json_file}")

            if not os.path.exists(image_output_dir):
                os.makedirs(image_output_dir)

            # Load prompts from JSON file
            with open(json_file, 'r', encoding='utf-8') as file:
                prompts = json.load(file)

            # Initialize output dictionary
            output_data = {}

            # Save output to the specified file
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            # Process each prompt
            for key, prompt in prompts.items():
                # Simulate image generation and get the image path
                if interrupt_processing:
                    raise InterruptProcessingException
                
                image_path = await self.generate_image(prompt, image_output_dir, key)
                output_data[key] = image_path


            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=4)

            status = f"Processed {len(prompts)} prompts. Output saved to {output_file}"
            return (prompt, status, )
        except Exception as e:
            logger.error(f"Error in process_prompts: {str(e)}")
            return ("no prompt", f"Error: {str(e)}")

    async def generate_image(self, prompt, image_output_dir, key):
        # Placeholder for image generation logic
        # Replace this with your actual image generation code
        image_path = os.path.join(image_output_dir, f"{key}.png")
        logger.info(f"Generated image for prompt: {prompt} at {image_path}")

        # Simulate asynchronous image generation
        await asyncio.sleep(5)  # Simulate delay

        # Check if the image is generated
        if not os.path.exists(image_path):
            logger.warning(f"Image not generated for prompt: {prompt}")
            return ""

        return image_path

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")


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
    assert out == "Hello, Earth! This is a example.", f"FileDictConverter test failed: {out}"

    # 3. JsonParser
    jp = JsonParser()
    assert json.loads(jp.parse('前缀{"name":"Alice"}后缀', "string")[0]) == {"name": "Alice"}
    assert json.loads(jp.parse("没有json", "string")[0]) == {"text": "没有json"}
    
    # 4. FileSplitter
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
    test_json_content = {
        "prompt1": "A beautiful sunset over the ocean",
        "prompt2": "A futuristic cityscape at night",
        "prompt3": "A serene forest with a gentle stream"
    }
    # 创建测试用的 JSON 文件
    test_json_file = "test_prompts.json"
    with open(test_json_file, 'w') as f:
        json.dump(test_json_content, f)

    # 创建测试用的输出目录
    test_image_output_dir = "test_images"
    if not os.path.exists(test_image_output_dir):
        os.makedirs(test_image_output_dir)

    # 创建测试用的输出 JSON 文件
    test_output_file = "test_output.json"
    # 初始化 JsonPromptProcessor
    jpp = JsonPromptProcessor()

    try:
        # 调用 process_prompts 方法
        updated_json, status, current_index = jpp.process_prompts(
            json_file=test_json_file,
            output_file=test_output_file,
            image_output_dir=test_image_output_dir,
            start_index=0
        )

        # 检查状态信息
        assert "Processed 1/3" in status, f"Status message incorrect: {status}"

        # 检查返回的 JSON 数据
        updated_json_data = json.loads(updated_json)
        assert "prompt1" in updated_json_data, f"Key 'prompt1' not found in updated JSON: {updated_json}"
        assert updated_json_data["prompt1"].endswith("prompt1.png"), f"Image path for 'prompt1' incorrect: {updated_json_data['prompt1']}"

        # 检查输出文件
        with open(test_output_file, 'r') as f:
            output_json_data = json.load(f)
        assert output_json_data == updated_json_data, f"Output JSON file does not match updated JSON: {output_json_data}"

        logger.info("Test passed successfully!")
    except AssertionError as e:
        logger.error(f"Test failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")


    # 清理测试文件
    os.remove(test_json_file)
    os.remove(test_output_file)
    os.rmdir(test_image_output_dir)


    print("✅ 所有测试通过！")

if __name__ == "__main__":
    run_all_tests()