import math
import torch

class CalculateOptimalResolution:
    """
    计算图像的最佳对应分辨率节点
    
    以1024×1024作为1 megapixel的基准（与ComfyUI标准一致）
    根据指定的百万像素值和可整除数值，计算保持原始宽高比的最佳分辨率
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "megapixels": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 30.0,
                    "step": 0.1,
                    "tooltip": "目标百万像素值，1.0对应1024×1024像素（ComfyUI标准）"
                }),
                "divisible_by": ("INT", {
                    "default": 64,
                    "min": 1,
                    "max": 1024,
                    "step": 1,
                    "tooltip": "分辨率需要被整除的值，确保与模型兼容"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "calculate_optimal_resolution"
    CATEGORY = "AFL/Image Calculator"
    
    def calculate_optimal_resolution(self, image, megapixels, divisible_by):
        # 获取图像原始尺寸 (ComfyUI中的IMAGE格式为 [batch, height, width, channels])
        _, original_height, original_width, _ = image.shape
        
        # 关键修改：使用1024×1024作为1 megapixel的基数（ComfyUI标准）
        # 1024×1024 = 1,048,576像素，而非传统的1,000,000像素
        base_pixels = 1024 * 1024  
        target_pixel_count = megapixels * base_pixels
        
        # 计算原始宽高比
        aspect_ratio = original_width / original_height
        
        # 根据宽高比和目标总像素计算理想尺寸
        ideal_height = math.sqrt(target_pixel_count / aspect_ratio)
        ideal_width = aspect_ratio * ideal_height
        
        # 调整尺寸使其能被divisible_by整除
        width = round(ideal_width / divisible_by) * divisible_by
        height = round(ideal_height / divisible_by) * divisible_by
        
        # 确保尺寸不会过小
        width = max(width, divisible_by)
        height = max(height, divisible_by)
        
        # 微调以保持正确的宽高比
        adjusted_height = round(width / aspect_ratio)
        adjusted_height = round(adjusted_height / divisible_by) * divisible_by
        adjusted_height = max(adjusted_height, divisible_by)
        
        # 选择更接近目标像素数的尺寸
        if abs(width * adjusted_height - target_pixel_count) < abs(width * height - target_pixel_count):
            height = adjusted_height
        
        return (image, width, height)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "AFL:CalculateOptimalResolution": CalculateOptimalResolution
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AFL:CalculateOptimalResolution": "CalculateOptimalResolution"
}
