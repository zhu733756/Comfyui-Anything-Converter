import os
import json
import numpy as np
from PIL import Image, PngImagePlugin
from .utils import load_json, logger
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
                "caption": ("STRING", {"forceInput": True, "tooltip": "One caption per line, same count as images."}),
                "labels": ("STRING", {"forceInput": True, "tooltip": "json str mapping for caption"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING")
    RETURN_NAMES = ("output")
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

    def save_images(self, images, filename_prefix="ComfyUI" ,
                    prompt=None, extra_pnginfo=None, caption=None, labels=None):
        merged_metadata = load_json(self.output_metadata)

        if caption is None:
            caption_list = [None] * len(images)
        else:
            caption_list = [c.strip() for c in str(caption).splitlines()]
            while len(caption_list) < len(images):
                caption_list.append(None)
                
        logger.info(f"get images {len(images)}, shape {image.shape[0]}/{image.shape[1]}, labels: {labels}, captions: {caption}")
                
        label_metadata = {}
        if labels is not None:
            loaded = load_json(labels) 
            label_metadata= {v:k for k,v in loaded.items()}
            
        
        out_dir = folder_paths.get_output_directory()
        
        results = {}
        for idx, image in enumerate(images):
            img_idx = merged_metadata.get("idx", 0) + 1
            
            full_out, filename, _, subfolder, prefix = folder_paths.get_save_image_path(
                filename_prefix, out_dir, image.shape[1], image.shape[0]
            )

            base_file = f"{filename}_{img_idx:05}"

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
            
            
            logger.info(f"image{img_idx} saved to {png_path}, cap: {cap.strip()}")
            img.save(png_path, pnginfo=metadata, compress_level=self.compress_level)

            if img_idx < len(caption_list):
                cap = caption_list[img_idx] is not None
                if cap in label_metadata:
                    results[label_metadata[cap]] = png_path
                else:
                    results[cap] = png_path
            else:
                results[idx] = png_path
            
            results.setdefault("idx", img_idx)

        # 返回 json 字符串，方便下游节点继续用
        results = self._deep_merge(merged_metadata, results)
        output = json.dumps(results)
        with open(self.output_metadata, "w") as fb:
            json.dump(results, fb, ensure_ascii=False)
        return output