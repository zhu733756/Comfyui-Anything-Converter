import os
import json
import numpy as np
import torch
from PIL import Image, PngImagePlugin
from .utils import load_json, load_content, logger
from pathlib import Path
import folder_paths
import node_helpers
from comfy.cli_args import args


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
                "clip": ("CLIP",),
            },
            "optional": {
                "resize_method": (["None", "Stretch", "Crop", "Pad"], {"default": "None"}),
                "width": ("INT", {"default": -1, "min": -1, "max": 10000, "step": 1}),
                "height": ("INT", {"default": -1, "min": -1, "max": 10000, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "CONDITIONING", "STRING")
    RETURN_NAMES = ("IMAGE", "CONDITIONING", "CHARS")
    FUNCTION = "load"
    CATEGORY = "LoadImage"
    DESCRIPTION = "Load images & prompts from two JSON files."

    def load(self, character2prompt_path, character2img_path, clip, resize_method="None", width=-1, height=-1):
        char2prompt = load_json(character2prompt_path)
        char2imgs = load_json(character2img_path)
        
        logger.info(f"get metadata: {char2prompt}; {char2imgs}")

        # 2. 拼路径 & 提示词
        image_paths, captions = [], []
        for char, img in char2imgs.items():
            if char == "idx":
                continue
            if os.path.exists(img):
                image_paths.append(img)
                captions.append(char2prompt.get(char, ""))
            else:
                logger.warning(f'image {img} not exists')

        # 3. 载入图片
        output_tensor = self._load(image_paths, resize_method,
                                          width if width != -1 else None,
                                          height if height != -1 else None)

        # 4. 编码提示词
        conds = []
        empty = clip.encode_from_tokens_scheduled(clip.tokenize(""))
        for cap in captions:
            if not cap:
                conds.append(empty)
            else:
                conds.append(clip.encode_from_tokens_scheduled(clip.tokenize(cap)))

        logger.info(f"Loaded {len(output_tensor)} images / {len(conds)} captions.")
        return (output_tensor, conds)
    
    def _load(self, image_files, resize_method="None", w=None, h=None):
        """Utility function to load and process a list of images.

        Args:
            image_files: List of image filenames
            resize_method: How to handle images of different sizes ("None", "Stretch", "Crop", "Pad")

        Returns:
            torch.Tensor: Batch of processed images
        """
        if not image_files:
            raise ValueError("No valid images found in input")

        output_images = []

        for image_path in image_files:
            img = node_helpers.pillow(Image.open, image_path)

            if img.mode == "I":
                img = img.point(lambda i: i * (1 / 255))
            img = img.convert("RGB")

            if w is None and h is None:
                w, h = img.size[0], img.size[1]

            # Resize image to first image
            if img.size[0] != w or img.size[1] != h:
                if resize_method == "Stretch":
                    img = img.resize((w, h), Image.Resampling.LANCZOS)
                elif resize_method == "Crop":
                    img = img.crop((0, 0, w, h))
                elif resize_method == "Pad":
                    img = img.resize((w, h), Image.Resampling.LANCZOS)
                elif resize_method == "None":
                    raise ValueError(
                        "Your input image size does not match the first image in the dataset. Either select a valid resize method or use the same size for all images."
                    )

            img_array = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array)[None,]
            output_images.append(img_tensor)

        return torch.cat(output_images, dim=0)
