import os
import re
import json
import tempfile
from pathlib import Path
from .utils import load_json, load_content, logger

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

    def combine(self, json_a, json_b, merge_strategy, indent=None, output_mode="", output_path=""):
        a = load_json(json_a)
        b = load_json(json_b)

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
            import folder_paths
            if not output_path.strip():
                fd, output_path = tempfile.mkstemp(suffix=".json", text=True)
                os.close(fd)
            else:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            output_path = os.path.join(folder_paths.base_path, output_path)
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
    def parse(self, text, output_mode="string", output_path=""):
        # 1. 读取内容
        raw = load_content(text)

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
                "fixed": ("STRING", {"tooltip": "the prompt to be fixed and not generate."}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("prompts.txt", "metadata.txt",)
    FUNCTION = "process_prompts"
    CATEGORY = "JsonPromptProcessor"

    def process_prompts(self, metadata, fixed):
        logger.info(f"json prompt processor: {metadata}")
        prompts = load_json(metadata)
        fixed_list = set(e for e in load_content(fixed).split(",") if fixed)

        prompt_contents = []
        metadata = []
        for key, prompt in prompts.items():
            if key in fixed_list:
                metadata.append(f"{key}:{prompt}:fixed")
            else:
                metadata.append(f"{key}:{prompt}:no-fixed")
                prompt_contents.append(prompt)
        return "\n".join(prompt_contents),"\n".join(metadata)
         
def run_all_tests():
    # . JsonParser
    jp = JsonParser()
    assert json.loads(jp.parse('前缀{"name":"Alice"}后缀', "string")[0]) == {"name": "Alice"}
    assert json.loads(jp.parse("没有json", "string")[0]) == {"text": "没有json"}
    
    # 5. JsonPromptProcessor
    print("✅ 所有测试通过！")
    
if __name__ == "__main__":
    run_all_tests()