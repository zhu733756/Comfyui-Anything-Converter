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
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character2prompt_path": ("STRING", {"default": "output/novels/character2prompt.json"}),
                "character2img_path": ("STRING", {"default": "output/metadata/metadata.json"}),
            },
            "optional": {
                "resize_method": (["None", "Stretch", "Crop", "Pad"], {"default": "None"}),
                "width":  ("INT", {"default": -1, "min": -1, "max": 10000, "step": 1}),
                "height": ("INT", {"default": -1, "min": -1, "max": 10000, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("IMAGE", "MASK", "prompts", "labels")
    FUNCTION = "load"
    CATEGORY = "image"

    # ---------- 主逻辑 ----------
    def load(self, character2prompt_path, character2img_path, resize_method="None", width=-1, height=-1):
        out_dir = folder_paths.base_path
        char2prompt = load_json(os.path.join(out_dir, character2prompt_path))
        char2imgs   = load_json(os.path.join(out_dir, character2img_path))

        # 2. 拼路径 & 提示词
        image_paths, captions = [], []
        labels = {}
        index = 1
        for char, img_path in char2imgs.items():
            if char == "idx" or not os.path.exists(img_path):
                continue
            labels[char] = f'image{index}'
            image_paths.append(img_path)
            captions.append(char2prompt.get(char, ""))
            index += 1

        if not image_paths:
            raise RuntimeError("No valid images found in metadata.")

        images, masks = self._load_images(image_paths, resize_method,
                                          width  if width  != -1 else None,
                                          height if height != -1 else None)

        return (images, masks, "\n".join(captions),json.dumps(labels))

    # ---------- 载入图片 ----------
    def _load_images(self,
                     paths,
                     resize_method="None",
                     w=None, h=None,
                     match_image_size=True,
                     spacing_width=0,
                     spacing_color="white"):
        """
        内部参考 ImageStitch 实现：
        1. 统一尺寸（match_image_size=True 时）
        2. 按 resize_method 做 Stretch / Crop / Pad
        3. 支持 spacing
        返回 (batched_images, batched_masks)  NHWC
        """
        import comfy.utils
        color_map = {"white": 1.0, "black": 0.0,
                     "red": (1.0, 0.0, 0.0),
                     "green": (0.0, 1.0, 0.0),
                     "blue": (0.0, 0.0, 1.0)}

        def load_single(path):
            img = node_helpers.pillow(Image.open, path)
            img = node_helpers.pillow(ImageOps.exif_transpose, img)
            if img.mode == 'I':
                img = img.point(lambda i: i * (1 / 65535))

            # 转成 RGBA，方便后面抽 mask
            img_rgba = img.convert("RGBA")
            img_rgb  = img_rgba.convert("RGB")

            # 抽 alpha -> mask
            alpha = np.array(img_rgba.split()[-1]).astype(np.float32) / 255.0
            mask  = 1. - torch.from_numpy(alpha)  # 0 表示不透明

            np_rgb = np.array(img_rgb).astype(np.float32) / 255.0
            tensor = torch.from_numpy(np_rgb)[None,]  # 1HWC
            return tensor, mask.unsqueeze(0)          # 1HW

        # 逐张加载
        images, masks = [], []
        for p in paths:
            im, mk = load_single(p)
            images.append(im)
            masks.append(mk)

        # 以第一张为基准
        target = images[0]
        target_h, target_w = target.shape[1:3]

        # 统一尺寸开关
        if match_image_size:
            for idx, (img, mk) in enumerate(zip(images, masks)):
                h_cur, w_cur = img.shape[1:3]
                if (w_cur, h_cur) != (target_w, target_h):
                    # 计算目标尺寸
                    aspect = w_cur / h_cur
                    if w is not None and h is not None:
                        new_w, new_h = w, h
                    else:
                        if w is not None:
                            new_w, new_h = w, int(w / aspect)
                        elif h is not None:
                            new_w, new_h = int(h * aspect), h
                        else:
                            new_w, new_h = target_w, target_h

                    # 执行缩放
                    img = comfy.utils.common_upscale(
                        img.movedim(-1, 1), new_w, new_h, "lanczos", "disabled"
                    ).movedim(1, -1)
                    mk = comfy.utils.common_upscale(
                        mk.unsqueeze(1), new_w, new_h, "lanczos", "disabled"
                    ).squeeze(1)

                    # 根据 resize_method 对齐到 target
                    pad_w = target_w - new_w
                    pad_h = target_h - new_h
                    if resize_method == "Stretch":
                        img = comfy.utils.common_upscale(
                            img.movedim(-1, 1), target_w, target_h, "lanczos", "disabled"
                        ).movedim(1, -1)
                        mk = comfy.utils.common_upscale(
                            mk.unsqueeze(1), target_w, target_h, "lanczos", "disabled"
                        ).squeeze(1)
                    elif resize_method == "Crop":
                        pad_left = max(0, -pad_w) // 2
                        pad_top  = max(0, -pad_h) // 2
                        img = img[:, pad_top:pad_top + target_h, pad_left:pad_left + target_w, :]
                        mk  = mk[:, pad_top:pad_top + target_h, pad_left:pad_left + target_w]
                    elif resize_method == "Pad":
                        pad_left = max(0, pad_w) // 2
                        pad_top  = max(0, pad_h) // 2
                        img = torch.nn.functional.pad(
                            img, (0, 0, pad_left, pad_w - pad_left,
                                  pad_top, pad_h - pad_top),
                            mode='constant', value=0.0)
                        mk = torch.nn.functional.pad(
                            mk, (pad_left, pad_w - pad_left,
                                 pad_top, pad_h - pad_top),
                            mode='constant', value=0.0)
                    elif resize_method == "None":
                        if (new_w, new_h) != (target_w, target_h):
                            raise ValueError("Image sizes mismatch and resize_method='None'.")

                    images[idx] = img
                    masks[idx]  = mk

        # ---- spacing ----
        if spacing_width > 0:
            spacing_width = spacing_width + (spacing_width % 2)  # 偶数
            color = color_map.get(spacing_color, 1.0)
            spacing_rgb = torch.full((1, target_h, spacing_width, 3), color)
            spacing_msk = torch.zeros((1, target_h, spacing_width))

            out_imgs, out_msks = [], []
            for idx, (img, msk) in enumerate(zip(images, masks)):
                out_imgs.append(img)
                out_msks.append(msk)
                if idx < len(images) - 1:  # 最后一张后面不加
                    out_imgs.append(spacing_rgb)
                    out_msks.append(spacing_msk)
            images = out_imgs
            masks  = out_msks

        # 拼成 batch
        images = torch.cat(images, dim=0)  # NHWC
        masks  = torch.cat(masks,  dim=0)  # NHW
        return images, masks

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
            return img

    @classmethod
    def IS_CHANGED(cls, character2prompt_path, character2img_path, **kw):
        out_dir = folder_paths.base_path
        h = hashlib.sha256()
        for path in (character2prompt_path, character2img_path):
            npw = os.path.join(out_dir, path)
            if os.path.isfile(npw):
                with open(npw, "rb") as f:
                    h.update(f.read())
        return h.hexdigest()

    @classmethod
    def VALIDATE_INPUTS(cls, character2prompt_path, character2img_path, **kw):
        out_dir = folder_paths.base_path
        for path in (character2prompt_path, character2img_path):
            npw = os.path.join(out_dir, path)
            if not os.path.isfile(npw):
                return f"File not found: {npw}"
        return True