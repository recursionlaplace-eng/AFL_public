import os
import torch
import numpy as np
from PIL import Image, ImageDraw

# 获取节点目录的相对路径
sCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 构建five star.png的相对路径
FIVE_STAR_PATH = os.path.join(sCRIPT_DIR, '..', 'backup_png', 'five star.png')

class EyeDirectionNode:
    """
    眼睛方向节点：将指定图像缩放到遮罩大小，并与输入图像融合
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output_image",)
    
    FUNCTION = "eye_direction_process"
    
    CATEGORY = "AFL节点"
    
    def _tensor_to_pil(self, tensor):
        """将PyTorch张量转换为PIL图像"""
        # 转换为numpy数组
        img_np = tensor.cpu().numpy()
        # 确保形状正确 (H, W, C) 并处理多批次情况
        if len(img_np.shape) == 4:
            img_np = img_np[0]  # 取第一个批次
        # 缩放至0-255范围
        img_np = (img_np * 255).astype(np.uint8)
        # 创建PIL图像
        return Image.fromarray(img_np, 'RGB')
    
    def _pil_to_tensor(self, img):
        """将PIL图像转换为PyTorch张量"""
        # 转换为numpy数组
        img_np = np.array(img).astype(np.float32) / 255.0
        # 添加批次维度
        return torch.from_numpy(img_np).unsqueeze(0)
    
    def eye_direction_process(self, image, mask):
        """处理图像和遮罩，将五星图像缩放到遮罩大小并融合"""
        # 确保文件存在
        if not os.path.exists(FIVE_STAR_PATH):
            raise FileNotFoundError(f"五星图像文件不存在: {FIVE_STAR_PATH}")
        
        # 读取五星图像
        five_star_img = Image.open(FIVE_STAR_PATH).convert('RGBA')
        
        # 获取输入图像（第一个批次）
        pil_image = self._tensor_to_pil(image[0])
        
        # 获取遮罩（处理多批次情况）
        if len(mask.shape) == 4:
            mask_np = mask[0, 0].cpu().numpy()
        elif len(mask.shape) == 3:
            mask_np = mask[0].cpu().numpy()
        else:
            mask_np = mask.cpu().numpy()
        
        # 创建PIL遮罩图像
        mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8), 'L')
        
        # 计算遮罩的边界框，找到非零像素区域
        bbox = mask_pil.getbbox()
        if bbox is None:
            # 如果遮罩为空，直接返回原始图像
            return (image,)
        
        # 获取遮罩区域的尺寸
        mask_width = bbox[2] - bbox[0]
        mask_height = bbox[3] - bbox[1]
        
        # 缩放五星图像到遮罩大小
        resized_star = five_star_img.resize((mask_width, mask_height), Image.Resampling.LANCZOS)
        
        # 创建一个与输入图像相同大小的图像副本
        result_img = pil_image.copy()
        
        # 创建一个临时图像用于合成
        temp_img = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
        
        # 将缩放后的五星图像粘贴到临时图像的遮罩位置
        temp_img.paste(resized_star, (bbox[0], bbox[1]), resized_star)
        
        # 使用遮罩进行融合
        # 获取临时图像的RGB和Alpha通道
        temp_rgb = temp_img.convert('RGB')
        temp_alpha = temp_img.split()[-1] if temp_img.mode == 'RGBA' else None
        
        # 使用Alpha通道作为遮罩将临时图像融合到结果图像
        if temp_alpha:
            # 转换为numpy数组进行处理
            result_np = np.array(result_img)
            temp_rgb_np = np.array(temp_rgb)
            alpha_np = np.array(temp_alpha).astype(np.float32) / 255.0
            
            # 确保alpha通道形状正确
            if len(alpha_np.shape) == 2:
                alpha_np = alpha_np[:, :, np.newaxis]
            
            # 融合图像
            result_np = (result_np * (1 - alpha_np) + temp_rgb_np * alpha_np).astype(np.uint8)
            result_img = Image.fromarray(result_np, 'RGB')
        
        # 转换回张量格式并添加批次维度
        output_tensor = self._pil_to_tensor(result_img)
        
        return (output_tensor,)

# 节点类映射
NODE_CLASS_MAPPINGS = {
    "AFL_EyeDirection": EyeDirectionNode
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "AFL_EyeDirection": "眼睛方向 (Eye Direction)"
}