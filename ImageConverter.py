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
import json
import os
from PIL import Image, ImageOps
import numpy as np



class SaveImage:
    def __init__(self):
        self.type = "output"
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "labels": ("STRING", {"forceInput": True, "tooltip": "json str mapping for caption"}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "File name prefix, supports %date% etc."}),
            },
            "optional": {
                "metadata_store_path": ("STRING", {"default": "output/metadata/metadata.json"})
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
                 

    def save_images(self, images, labels, filename_prefix="ComfyUI" ,
                    prompt=None, extra_pnginfo=None, metadata_store_path="output/metadata/metadata.json"):
        if not labels or str(labels).count(":") < 3:
            raise "labels like key:prompt:fixed/no-fixed must given"
        
        if not os.path.exists(metadata_store_path):
            Path(metadata_store_path).parent.mkdir(parents=True, exist_ok=True)
        
        out_dir = folder_paths.base_path
        merged_metadata = load_json(metadata_store_path)
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

        with open(metadata_store_path, "w") as fb:
            json.dump(merged_metadata, fb, ensure_ascii=False)
            
        return json.dumps(merged_metadata)


class LoadImage2Kontext:
    """
    1. 根据 character2prompt_path / character2img_path 读取 json
    2. 按顺序加载图片 → VAE 编码 → 串联 latents
    3. 为每张图片追加 ReferenceLatent 到下游条件
    4. 输出：
        LATENT          : 串联后的总 latent
        CONDITIONING    : 已追加所有 reference_latents 的条件
        show_help       : 帮助信息
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character2prompt_path": ("STRING", {
                    "default": "output/novels/character2prompt.json",
                    "multiline": False,
                    "tooltip": "人物到prompt映射的json路径"
                }),
                "character2img_path": ("STRING", {
                    "default": "output/metadata/metadata.json",
                    "multiline": False,
                    "tooltip": "人物图片路径2映射的json路径"
                }),
                "vae": ("VAE",),
            },
            "optional": {
                "conditions": ("CONDITIONING",),  # 来自上游的条件，可为 None
            }
        }

    RETURN_TYPES = ("Image", "LATENT","CONDITIONING", "STRING", "STRING")
    RETURN_NAMES = ("images", "latents", "conditioning", "prompts", "metadata")
    FUNCTION = "load_and_encode"
    CATEGORY = "LoadImage2Kontext"

    # ------------ 工具函数 ------------
    @staticmethod
    def load_image(path):
        """把 ComfyUI 路径 → torch tensor (1,H,W,3)"""
        if not os.path.isabs(path):
            path = os.path.join(folder_paths.get_input_directory(), path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")

        img = Image.open(path).convert("RGB")
        img = ImageOps.exif_transpose(img)  # 处理旋转信息
        img = np.array(img, dtype=np.float32) / 255.0
        img = torch.from_numpy(img)[None,]  # shape (1,H,W,3)
        return img

    # ------------ 主函数 ------------
    def load_and_encode(self,
                        character2prompt_path,
                        character2img_path,
                        vae,
                        conditions):

        # 1. 读取两个 json
        if not os.path.isabs(character2prompt_path):
            character2prompt_path = os.path.join(folder_paths.base_path, character2prompt_path)
        if not os.path.isabs(character2img_path):
            character2img_path = os.path.join(folder_paths.base_path, character2img_path)

        char2prompt = load_json(character2prompt_path)
        char2img =  load_json(character2img_path)
 
        # 2. 逐个图片编码
        if not conditions:
            raise ValueError("conditions must be given.")

        images, captions,latent_list = [], [], []
        labels = {}
        index = 1
        for _, char in enumerate(sorted(char2img.keys()), 1):
            img_path = char2img[char]
            if char == "idx" or not os.path.exists(img_path):
                continue
            
            labels[char] = f'the character in image{index}'
            captions.append(char2prompt.get(char, ""))
            
            # VAEEncode
            pixels = self.load_image(img_path)
            images.append(pixels)
            latent = vae.encode(pixels[:,:,:,:3])
            latent_list.append(latent)
            # ReferenceLatent
            conditions = node_helpers.conditioning_set_values(conditions, {"reference_latents": [latent]}, append=True)
            
            index += 1
          
        stacked = torch.cat(latent_list, dim=0)  # shape (N, C, H/8, W/8)
        return (torch.cat(images, dim=0), {"samples": stacked},  conditions,  "\n".join(captions), json.dumps(labels))
