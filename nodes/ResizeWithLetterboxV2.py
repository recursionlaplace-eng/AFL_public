import torch
import numpy as np
from PIL import Image
import math

class ResizeWithLetterboxV2:
    """
    图像缩放与Letterbox处理节点V2
    功能：将输入图像和掩码按原始比例缩放到不超过指定尺寸，图像用白色像素补齐，掩码用alpha通道（0）补齐
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_width": ("INT", {"default": 512, "min": 1, "step": 1}),
                "target_height": ("INT", {"default": 512, "min": 1, "step": 1}),
            },
            "optional": {  # 将mask改为可选输入
                "mask": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "box_mask", "resize_mask", "width", "height", "original_width", "original_height")
    FUNCTION = "resize_and_letterbox"
    CATEGORY = "AFL/Image Calculator"

    def resize_and_letterbox(self, image, target_width, target_height, mask=None):  # mask默认值为None
        # 输入格式：
        # image: (batch, height, width, channels)，值范围[0,1]
        # mask: (batch, 1, height, width)，值范围[0,1]（符合MASK类型标准），可选
        batch_size = image.shape[0]
        img_results = []
        box_masks = []
        resize_masks = []
        original_widths = []
        original_heights = []
        
        for b in range(batch_size):
            # 处理图像
            img_np = (image[b].cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            original_width, original_height = pil_img.size
            original_widths.append(original_width)
            original_heights.append(original_height)
            
            # 计算缩放比例
            scale = min(target_width / original_width, target_height / original_height)
            scaled_width = int(round(original_width * scale))
            scaled_height = int(round(original_height * scale))
            x_offset = (target_width - scaled_width) // 2
            y_offset = (target_height - scaled_height) // 2
            
            # 缩放图像并粘贴到白色背景
            scaled_img = pil_img.resize((scaled_width, scaled_height), Image.LANCZOS)
            target_img = Image.new('RGB', (target_width, target_height), (255, 255, 255))
            target_img.paste(scaled_img, (x_offset, y_offset))
            
            # 处理box_mask（原mask输出，保持白色补齐）
            box_mask = Image.new('L', (target_width, target_height), 0)
            if x_offset > 0:
                box_mask.paste(255, (0, 0, x_offset, target_height))
                box_mask.paste(255, (x_offset + scaled_width, 0, target_width, target_height))
            if y_offset > 0:
                box_mask.paste(255, (x_offset, 0, x_offset + scaled_width, y_offset))
                box_mask.paste(255, (x_offset, y_offset + scaled_height, x_offset + scaled_width, target_height))
            
            # 处理输入mask（如果存在）
            if mask is not None and batch_size <= len(mask):
                # 将mask从张量转换为PIL图像（单通道）
                mask_np = (mask[b].cpu().numpy().squeeze() * 255).astype(np.uint8)  # 移除通道维度
                pil_mask = Image.fromarray(mask_np, mode='L')
                # 同比例缩放mask
                scaled_mask = pil_mask.resize((scaled_width, scaled_height), Image.LANCZOS)
                # 创建目标尺寸mask（补齐区域为0，对应alpha通道）
                target_mask = Image.new('L', (target_width, target_height), 0)
                target_mask.paste(scaled_mask, (x_offset, y_offset))
            else:
                # 无输入mask时生成空mask（全0）
                target_mask = Image.new('L', (target_width, target_height), 0)
            
            # 转换回张量格式
            img_np = np.array(target_img).astype(np.float32) / 255.0
            box_mask_np = np.array(box_mask).astype(np.float32) / 255.0
            resize_mask_np = np.array(target_mask).astype(np.float32) / 255.0
            
            img_results.append(torch.from_numpy(img_np))
            box_masks.append(torch.from_numpy(box_mask_np))
            resize_masks.append(torch.from_numpy(resize_mask_np))
        
        # 堆叠结果并调整维度
        result_tensor = torch.stack(img_results).permute(0, 3, 1, 2).permute(0, 2, 3, 1)  # (batch, height, width, channels)
        box_mask_tensor = torch.stack(box_masks).unsqueeze(1)  # (batch, 1, height, width)
        resize_mask_tensor = torch.stack(resize_masks).unsqueeze(1)  # (batch, 1, height, width)
        
        original_widths_tensor = torch.tensor(original_widths, dtype=torch.int)
        original_heights_tensor = torch.tensor(original_heights, dtype=torch.int)
        
        return (result_tensor, box_mask_tensor, resize_mask_tensor, target_width, target_height, original_widths_tensor, original_heights_tensor)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "AFL:ResizeWithLetterboxV2": ResizeWithLetterboxV2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AFL:ResizeWithLetterboxV2": "Resize with Letterbox V2"
}
