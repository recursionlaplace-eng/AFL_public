import torch

class ImageSwitch:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
            },
            "required": {
                "manual_choice": (["image1", "image2"], {"default": "image1"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image_auto", "image_manual")
    FUNCTION = "switch_images"
    CATEGORY = "AFL/实用工具"
    
    # 调整参数顺序，将无默认值的manual_choice放在前面
    def switch_images(self, manual_choice, image1=None, image2=None):
        # 检查图像是否有效（非空且有内容），增加异常处理
        def is_valid_image(img):
            try:
                return img is not None and isinstance(img, torch.Tensor) and img.numel() > 0 and img.shape[0] > 0
            except:
                return False
        
        # 自动模式处理 - 增加完整的异常处理
        try:
            if is_valid_image(image1):
                auto_result = image1
            else:
                auto_result = image2
        except Exception:
            # 当image1处理出现任何异常时，直接使用image2
            auto_result = image2
        
        # 手动模式处理 - 同样增加异常处理
        try:
            if manual_choice == "image1":
                if is_valid_image(image1):
                    manual_result = image1
                else:
                    manual_result = image2
            else:  # manual_choice == "image2"
                if is_valid_image(image2):
                    manual_result = image2
                else:
                    manual_result = image1
        except Exception:
            # 当手动处理出现异常时，尝试使用有效的图像
            if is_valid_image(image1):
                manual_result = image1
            elif is_valid_image(image2):
                manual_result = image2
            else:
                manual_result = image1 or image2
        
        # 最后确保至少返回一个有效的图像
        if not is_valid_image(auto_result) and is_valid_image(image2):
            auto_result = image2
            
        return (auto_result, manual_result)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "AFL:ImageSwitch": ImageSwitch
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AFL:ImageSwitch": "Image Switch"
}

