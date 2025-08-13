import os
import json
import numpy as np
from PIL import Image, PngImagePlugin
from .utils import load_json, load_content, logger
from pathlib import Path
import folder_paths
from comfy.cli_args import args


class SaveImage:
    def __init__(self):
        self.type = "output"
        self.compress_level = 4
        self.output_metadata="output/metadata/metadata.json"
        if os.path.exists(self.output_metadata):
            os.remove(self.output_metadata)
        Path(self.output_metadata).parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "File name prefix, supports %date% etc."}),
            },
            "optional": {
                "labels": ("STRING", {"forceInput": True, "tooltip": "json str mapping for caption"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "SaveImage"
    DESCRIPTION = "Save each image to its own custom path. Empty path falls back to default output folder."
         
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
    
    def get_prompt_key(self, label_metadata, img_idx):
        if 0<= img_idx-1 < len(label_metadata):
            meta = label_metadata[img_idx-1]
            prompt_key, _, fixed = meta.split(":")
            if fixed == "fixed":
                return self.get_prompt_key(label_metadata, img_idx+1)
            else:
                return prompt_key,  img_idx
        
        return "", -1
                 

    def save_images(self, images, filename_prefix="ComfyUI" ,
                    prompt=None, extra_pnginfo=None,  labels=None):
        if labels is None:
            raise "metadata like key:prompt:fixed/no-fixed must given"
        
        out_dir = folder_paths.get_output_directory()
        merged_metadata = load_json(self.output_metadata)
        label_metadata = [x.strip() for x in str(load_content(labels)).splitlines() if len(x.strip().split(":"))>=3]
  
        for _, image in enumerate(images):
            full_out, filename, _, _, _ = folder_paths.get_save_image_path(
                filename_prefix, out_dir, image.shape[1], image.shape[0]
            )
            
            next_img_idx = (merged_metadata.get("idx", 0)) % len(label_metadata) + 1
            prompt_key, next_img_idx = self.get_prompt_key(label_metadata=label_metadata, img_idx=next_img_idx)
            if prompt_key != "" and prompt_key in merged_metadata:
                base_file = merged_metadata[prompt_key] # replace in merged_metadata
            else:
                base_file = f"{filename}_{next_img_idx:05}"

            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            metadata = None
            if not args.disable_metadata:
                metadata = PngImagePlugin.PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for k, v in extra_pnginfo.items():
                        metadata.add_text(k, json.dumps(v))

            png_name = f"{base_file}.png"
            png_path = os.path.join(full_out, png_name)
            merged_metadata[prompt_key] = png_path
            merged_metadata["idx"] = next_img_idx  
            
            logger.info(f"image{next_img_idx} saved to {png_path}, prompt key: {prompt_key}")
            img.save(png_path, pnginfo=metadata, compress_level=self.compress_level)

        with open(self.output_metadata, "w") as fb:
            json.dump(merged_metadata, fb, ensure_ascii=False)
        return json.dumps(merged_metadata)