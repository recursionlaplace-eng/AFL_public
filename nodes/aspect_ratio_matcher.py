import torch

# 定义banana模式的宽高比及其对应的分辨率
BANANA_ASPECT_RATIOS = {
    "1:1": (1024, 1024),
    "4:3": (1184, 864),
    "3:4": (864, 1184),
    "2:3": (832, 1248),
    "3:2": (1248, 832),
    "9:16": (768, 1344),
    "16:9": (1344, 768)
}

# 定义qwen_image模式的宽高比及其对应的分辨率
QWEN_IMAGE_ASPECT_RATIOS = {
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472)
}

BANANA_PRO_ASPECT_RATIOS = {
    "1K": {
        "1:1": (1024, 1024),
        "4:3": (1200, 896),
        "3:4": (896, 1200),
        "2:3": (848, 1264),
        "3:2": (1264, 848),
        "9:16": (768, 1376),
        "16:9": (1376, 768),
        "4:5": (928, 1152),
        "5:4": (1152, 928),
        "21:9": (1584, 672)
    },
    "2K": {
        "1:1": (2048, 2048),
        "4:3": (2400, 1792),
        "3:4": (1792, 2400),
        "2:3": (1696, 2528),
        "3:2": (2528, 1696),
        "9:16": (1536, 2752),
        "16:9": (2752, 1536),
        "4:5": (1856, 2304),
        "5:4": (2304, 1856),
        "21:9": (3168, 1344)
    },
    "4K": {
        "1:1": (4096, 4096),
        "4:3": (4800, 3584),
        "3:4": (3584, 4800),
        "2:3": (3392, 5056),
        "3:2": (5056, 3392),
        "9:16": (3072, 5504),
        "16:9": (5504, 3072),
        "4:5": (3712, 4608),
        "5:4": (4608, 3712),
        "21:9": (6336, 2688)
    }
}


class aspect_ratio_matcher:
    """
    ComfyUI节点：根据输入图像和选择的模式，计算最接近的标准宽高比，并输出对应分辨率
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # 输入图像
                "mode": (["banana", "banana_pro", "qwen_image"],),  # 模式选择
                "banana_pro_resolution":(["1K", "2K", "4K"],),
            }
        }
    
    RETURN_TYPES = ("STRING", "INT", "INT")
    RETURN_NAMES = ("aspect_ratio", "width", "height")
    FUNCTION = "match_aspect_ratio"
    CATEGORY = "AFL/Image Calculator"
    
    def calculate_aspect_ratio(self, width, height):
        if height == 0:
            return 0.0
        return width / height
    
    def find_closest_aspect_ratio(self, input_ratio, aspect_ratios):
        # 根据当前模式的宽高比字典计算最接近的比例
        ratio_values = {k: w/h for k, (w, h) in aspect_ratios.items()}
        
        closest_ratio = None
        min_difference = float('inf')
        
        for ratio_str, ratio_val in ratio_values.items():
            difference = abs(input_ratio - ratio_val)
            if difference < min_difference:
                min_difference = difference
                closest_ratio = ratio_str
        
        return closest_ratio
    
    def match_aspect_ratio(self, image, mode, banana_pro_resolution):
        # 根据选择的模式获取对应的宽高比字典
        if mode == "banana":
            aspect_ratios = BANANA_ASPECT_RATIOS
        elif mode == "banana_pro":
            aspect_ratios = BANANA_PRO_ASPECT_RATIOS
        else:  # qwen_image模式
            aspect_ratios = QWEN_IMAGE_ASPECT_RATIOS
        
        # 获取图像尺寸 (batch, height, width, channels)
        batch_size, height, width, channels = image.shape
        
        input_ratio = self.calculate_aspect_ratio(width, height)
        
        if mode == "banana_pro":
            closest_ratio = self.find_closest_aspect_ratio(input_ratio, aspect_ratios[banana_pro_resolution])
            target_width, target_height = aspect_ratios[banana_pro_resolution][closest_ratio]
        else:
            closest_ratio = self.find_closest_aspect_ratio(input_ratio, aspect_ratios)
            target_width, target_height = aspect_ratios[closest_ratio]
        
        return (closest_ratio, target_width, target_height)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "AFL:aspect_ratio_matcher": aspect_ratio_matcher
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AFL:aspect_ratio_matcher": "aspect_ratio_matcher"
}
