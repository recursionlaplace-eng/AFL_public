import math
import torch

class CalculateOptimalResolution:
    """
    计算图像的最佳对应分辨率节点
    
    支持两种模式：
    1. 百万像素模式：以1024×1024作为1 megapixel的基准（与ComfyUI标准一致）
    2. 最长边模式：通过指定最长边来确定图像尺寸
    根据指定参数和可整除数值，计算保持原始宽高比的最佳分辨率
    
    同时支持mask的传递
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "mode": ("COMBO", {
                    "default": "megapixels",
                    "options": ["megapixels", "longest_side"],
                    "tooltip": "选择使用百万像素模式或最长边模式"
                }),
                "megapixels": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 30.0,
                    "step": 0.1,
                    "tooltip": "目标百万像素值，1.0对应1024×1024像素（ComfyUI标准）"
                }),
                "longest_side": ("INT", {
                    "default": 1024,
                    "min": 128,
                    "max": 2048,
                    "step": 64,
                    "tooltip": "图像最长边的目标像素数"
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
    
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "width", "height")
    FUNCTION = "calculate_optimal_resolution"
    CATEGORY = "AFL/Image Calculator"
    
    def calculate_optimal_resolution(self, image, mask, mode, megapixels, longest_side, divisible_by):
        # 获取图像原始尺寸 (ComfyUI中的IMAGE格式为 [batch, height, width, channels])
        _, original_height, original_width, _ = image.shape
        
        # 计算原始宽高比
        aspect_ratio = original_width / original_height
        
        # 根据选择的模式计算目标尺寸
        if mode == "longest_side":
            # 最长边模式：根据指定的最长边计算尺寸
            if original_width > original_height:
                # 宽度是最长边
                width = longest_side
                height = round(longest_side / aspect_ratio)
            else:
                # 高度是最长边
                height = longest_side
                width = round(longest_side * aspect_ratio)
        else:
            # 百万像素模式（默认）：保持原有逻辑
            # 使用1024×1024作为1 megapixel的基数（ComfyUI标准）
            base_pixels = 1024 * 1024  
            target_pixel_count = megapixels * base_pixels
            
            # 根据宽高比和目标总像素计算理想尺寸
            ideal_height = math.sqrt(target_pixel_count / aspect_ratio)
            ideal_width = aspect_ratio * ideal_height
            
            # 调整尺寸使其能被divisible_by整除
            width = round(ideal_width / divisible_by) * divisible_by
            height = round(ideal_height / divisible_by) * divisible_by
        
        # 调整尺寸使其能被divisible_by整除
        width = round(width / divisible_by) * divisible_by
        height = round(height / divisible_by) * divisible_by
        
        # 确保尺寸不会过小
        width = max(width, divisible_by)
        height = max(height, divisible_by)
        
        # 微调以保持正确的宽高比
        adjusted_height = round(width / aspect_ratio)
        adjusted_height = round(adjusted_height / divisible_by) * divisible_by
        adjusted_height = max(adjusted_height, divisible_by)
        
        # 根据模式选择合适的目标值进行比较
        if mode == "longest_side":
            # 在最长边模式下，优先保持最长边为指定值
            current_longest = max(width, height)
            adjusted_longest = max(width, adjusted_height)
            # 如果调整后的最长边更接近目标最长边，则使用调整后的高度
            if abs(adjusted_longest - longest_side) < abs(current_longest - longest_side):
                height = adjusted_height
        else:
            # 百万像素模式下，保持原有逻辑
            target_pixel_count = megapixels * 1024 * 1024
            if abs(width * adjusted_height - target_pixel_count) < abs(width * height - target_pixel_count):
                height = adjusted_height
        
        return (image, mask, width, height)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "AFL:CalculateOptimalResolution": CalculateOptimalResolution
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AFL:CalculateOptimalResolution": "CalculateOptimalResolution"
}
