import os
import json
import logging

logger = logging.getLogger(__name__)

def load_json(dict_or_file_or_str):
    try:
        if isinstance(dict_or_file_or_str, dict):
          return dict_or_file_or_str
        elif os.path.isfile(dict_or_file_or_str):
            with open(dict_or_file_or_str, "r") as lf:
                return json.load(lf)
        else:
            return json.loads(dict_or_file_or_str)
    except Exception as e:
        logger.warning(f"loaded failed {e.args}, metadata: {dict_or_file_or_str}")
        return {}
      
def load_content(content):
    """加载输入内容"""
    if  os.path.isfile(content.strip()):
        with open(content.strip(), "r", encoding="utf-8") as f:
            return f.read()
    else:
        return content