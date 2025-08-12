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
                "merge_strategy": (["deep_merge", "a_first", "b_first", "value_ab"], {"default": "deep_merge"}),
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
        elif merge_strategy == "a_first":
            merged = {**b, **a}
        elif merge_strategy == "value_ab":
            merged = {}
            for k, v in a.items():
                if v in b.values():
                    merged[k] = v

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
        
      
class JsonPromptProcessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "metadata": ("STRING", {"multiline": False}),
                "image_output_dir": ("STRING", {"multiline": False})
            }
        }

    RETURN_TYPES = ("STRING", "STRING")   # 第二路输出：json 字符串，包含 key->路径 的映射
    RETURN_NAMES = ("prompts", "meta_json")
    FUNCTION = "process_prompts"
    CATEGORY = "JsonPromptProcessor"

    def process_prompts(self, metadata, image_output_dir):
        if os.path.isfile(metadata):
            with open(metadata, encoding="utf-8") as f:
                prompts = json.load(f)  # {"k1":"prompt1", ...}
        elif isinstance(metadata, str):
            prompts = json.loads(metadata)
        else:
            raise "not str or file like"
        
        Path(image_output_dir).mkdir(parents=True, exist_ok=True)

        pending = {}
        prompt_contents = []
        for key, prompt in prompts.items():
            img_path = Path(image_output_dir) / f"{key}.png"
            pending[key] = str(img_path)
            prompt_contents.append(prompt)

        # 4) 元数据 json
        meta_json = json.dumps(pending, ensure_ascii=False, indent=2)
        return "\n".join(prompt_contents), meta_json
        
def run_all_tests():
    # . JsonParser
    jp = JsonParser()
    assert json.loads(jp.parse('前缀{"name":"Alice"}后缀', "string")[0]) == {"name": "Alice"}
    assert json.loads(jp.parse("没有json", "string")[0]) == {"text": "没有json"}
    
    # 5. JsonPromptProcessor
    print("✅ 所有测试通过！")
    
if __name__ == "__main__":
    run_all_tests()