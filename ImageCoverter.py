import os
import json
import numpy as np
from PIL import Image, PngImagePlugin
import logging
import folder_paths
from comfy.cli_args import args

logger = logging.getLogger(__name__)

class SaveImage:
    def __init__(self):
        self.type = "output"
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "paths": ("STRING", {"multiline": True, "tooltip": "One absolute path per line, same count as images. Empty → use default output dir."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "File name prefix, supports %date% etc."}),
            },
            "optional": {
                "caption": ("STRING", {"forceInput": True, "tooltip": "One caption per line, same count as images."}),
                "labels": ("STRING", {"forceInput": True, "tooltip": "json str mapping for caption"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("metadata",)
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "SaveImage"
    DESCRIPTION = "Save each image to its own custom path. Empty path falls back to default output folder."

    def save_images(self, images, paths, filename_prefix="ComfyUI", 
                    prompt=None, extra_pnginfo=None, caption=None, labels=None):
        path_list = [p.strip() for p in str(paths).splitlines() if p.strip() != ""]
        while len(path_list) < len(images):
            path_list.append("")  # 用默认

        if caption is None:
            caption_list = [None] * len(images)
        else:
            caption_list = [c.strip() for c in str(caption).splitlines()]
            while len(caption_list) < len(images):
                caption_list.append(None)
                
        label_metadata = {}
        if labels is not None:
            try:
                if os.path.isfile(labels):
                    with open(labels, "r") as lf:
                        loaded = json.load(lf)
                else:
                    loaded = json.loads(labels)
                
                logger.info(f"labels loaded, labels: {labels}")   
                label_metadata = {v:k for k,v in loaded.items()}
            except Exception as e:
                logger.warning(f"labels loaded failed {e.args}, labels: {labels}")
         
        results = {}
        for idx, (image, custom_path, cap) in enumerate(zip(images, path_list, caption_list)):
            # 1. 决定输出目录
            if custom_path and os.path.isabs(custom_path):
                out_dir = custom_path
            else:
                out_dir = folder_paths.get_output_directory()

            # 2. 使用 comfy 自带的计数器机制
            full_out, filename, counter, subfolder, prefix = folder_paths.get_save_image_path(
                filename_prefix, out_dir, image.shape[1], image.shape[0]
            )

            # 3. 组装文件名（带索引）
            base_file = f"{filename}_{counter:05}"

            # 4. 保存 png
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
            
            logger.debug(f"image{idx} saved to {png_path}")
            img.save(png_path, pnginfo=metadata, compress_level=self.compress_level)

            if cap is not None:
                if cap in label_metadata:
                    results[label_metadata[cap]] = png_path
                else:
                    results[cap] = png_path
            else:
                results[idx] = png_path
            counter += 1  # 计数器每图自增

        # 返回 json 字符串，方便下游节点继续用
        return (json.dumps(results, ensure_ascii=False),)


def test_save_image():
    # 1. 准备假数据
    import torch
    import numpy as np

    images = []
    for c in (0, 127, 255):
        arr = np.full((512, 512, 3), c, dtype=np.uint8)
        images.append(torch.from_numpy(arr.astype(np.float32) / 255.0))

    images = torch.stack(images)  # [3, 512, 512, 3]

    # 2. 临时输出目录
    out_dir = os.path.join(os.getcwd(), "test_outputs")
    os.makedirs(out_dir, exist_ok=True)

    # 3. 构造 paths 与 caption
    paths = "\n".join([
        os.path.join(out_dir, "img_black.png"),
        os.path.join(out_dir, "img_gray.png"),
        "",  # 空行 -> 回退到默认
    ])
    captions = "black\ngray\nwhite"

    # 4. 调用节点
    node = SaveImage()
    metadata = node.save_images(
        images=images,
        paths=paths,
        filename_prefix="test",
        caption=captions
    )
    saved_files = json.loads(metadata)

    # 5. 断言
    assert len(saved_files) == 3, f"expect 3 files, got {len(saved_files)}"
    for p in saved_files:
        assert os.path.isfile(p), f"file not found: {p}"
    print("✅ 测试通过，文件如下：")
    for p in saved_files:
        print("   ", p)

if __name__ == "__main__":
    test_save_image()