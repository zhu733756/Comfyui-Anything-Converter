import os
import json
import numpy as np
import torch
from PIL import Image, ImageOps, PngImagePlugin
from .utils import load_json, load_content, logger
from pathlib import Path
import folder_paths
import node_helpers
from comfy.cli_args import args
import hashlib


class SaveImage:
    def __init__(self):
        self.type = "output"
        self.compress_level = 4
        self.output_metadata="output/metadata/metadata.json"
        if not os.path.exists(self.output_metadata):
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
                png_path = merged_metadata[prompt_key] # replace in merged_metadata
            else:
                png_path = os.path.join(full_out, f"{filename}_{next_img_idx:05}.png")   
                
            merged_metadata[prompt_key] = png_path
            merged_metadata["idx"] = next_img_idx  

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

            logger.info(f"image{next_img_idx} saved to {png_path}, prompt key: {prompt_key}")
            img.save(png_path, pnginfo=metadata, compress_level=self.compress_level)

        with open(self.output_metadata, "w") as fb:
            json.dump(merged_metadata, fb, ensure_ascii=False)
        return json.dumps(merged_metadata)


class LoadImageTextSetFromMetadata:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character2prompt_path": ("STRING", {"default": "character2prompt.json"}),
                "character2img_path": ("STRING", {"default": "character2img.json"}),
            },
            "optional": {
                "resize_method": (["None", "Stretch", "Crop", "Pad"], {"default": "None"}),
                "width":  ("INT", {"default": -1, "min": -1, "max": 10000, "step": 1}),
                "height": ("INT", {"default": -1, "min": -1, "max": 10000, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("IMAGE", "MASK", "CAPTIONS")
    FUNCTION = "load"
    CATEGORY = "image"

    # ---------- 主逻辑 ----------
    def load(self, character2prompt_path, character2img_path, resize_method="None", width=-1, height=-1):
        char2prompt = load_json(character2prompt_path)
        char2imgs   = load_json(character2img_path)

        # 2. 拼路径 & 提示词
        image_paths, captions = [], []
        for char, img_path in char2imgs.items():
            if char == "idx" or not os.path.isfile(img_path):
                continue
            image_paths.append(img_path)
            captions.append(char2prompt.get(char, ""))

        if not image_paths:
            raise RuntimeError("No valid images found in metadata.")

        images, masks = self._load_images(image_paths, resize_method,
                                          width  if width  != -1 else None,
                                          height if height != -1 else None)

        return (images, masks, "\n".join(captions))

    # ---------- 载入图片 ----------
    def _load_images(self, paths, resize_method, w, h):
        out_imgs, out_masks = [], []
        target_w = target_h = None

        for p in paths:
            img = node_helpers.pillow(Image.open, p)
            img = node_helpers.pillow(ImageOps.exif_transpose, img)

            # 16-bit PNG -> 8-bit
            if img.mode == 'I':
                img = img.point(lambda i: i * (1 / 65535))

            # 统一 RGB
            img = img.convert("RGB")

            # 首张尺寸
            if target_w is None:
                target_w, target_h = img.size

            # 缩放
            if (img.width, img.height) != (target_w, target_h):
                img = self._resize(img, target_w, target_h, resize_method)

            # 转 Tensor
            np_img = np.array(img).astype(np.float32) / 255.0
            tensor = torch.from_numpy(np_img)[None,]
            out_imgs.append(tensor)
            out_masks.append(torch.zeros((64, 64), dtype=torch.float32))  # 占位空 mask

        return torch.cat(out_imgs, dim=0), torch.stack(out_masks, dim=0)

    # ---------- 缩放 ----------
    @staticmethod
    def _resize(img, w, h, method):
        if method == "Stretch":
            return img.resize((w, h), Image.LANCZOS)
        elif method == "Crop":
            img = img.copy()
            img.thumbnail((w, h), Image.LANCZOS)
            left = (img.width - w) // 2
            top  = (img.height - h) // 2
            return img.crop((left, top, left + w, top + h))
        elif method == "Pad":
            img.thumbnail((w, h), Image.LANCZOS)
            new_img = Image.new("RGB", (w, h))
            new_img.paste(img, ((w - img.width) // 2, (h - img.height) // 2))
            return new_img
        else:
            raise ValueError("Image sizes do not match and resize_method='None'.")

    # ---------- 变更检测 ----------
    @classmethod
    def IS_CHANGED(cls, character2prompt_path, character2img_path, **kw):
        h = hashlib.sha256()
        for name in (character2prompt_path, character2img_path):
            path = os.path.join(folder_paths.get_input_directory(), name)
            if os.path.isfile(path):
                with open(path, "rb") as f:
                    h.update(f.read())
        return h.hexdigest()

    @classmethod
    def VALIDATE_INPUTS(cls, character2prompt_path, character2img_path, **kw):
        for name in (character2prompt_path, character2img_path):
            path = os.path.join(folder_paths.get_input_directory(), name)
            if not os.path.isfile(path):
                return f"File not found: {name}"
        return True