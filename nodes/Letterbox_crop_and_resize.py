import torch
import numpy as np
from PIL import Image

class CropAndResize:
    """
    根据遮罩裁剪白色像素区域并缩放回指定尺寸的节点
    功能：使用遮罩识别图像中的有效区域，裁剪后缩放到指定尺寸
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "box_mask": ("MASK",),
                "target_width": ("INT", {"default": 512, "min": 1, "step": 1}),
                "target_height": ("INT", {"default": 512, "min": 1, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "process"
    CATEGORY = "AFL/Image Calculator"

    def process(self, image, box_mask, target_width, target_height):
        # 输入格式:
        # image: (batch, height, width, channels)，值范围[0,1]
        # box_mask: (batch, 1, height, width)，值范围[0,1]，1表示白色补齐区域
        
        batch_size = image.shape[0]
        results = []
        
        for b in range(batch_size):
            # 提取单张图像和对应的遮罩
            img_np = (image[b].cpu().numpy() * 255).astype(np.uint8)
            mask_np = (box_mask[b].cpu().numpy().squeeze() * 255).astype(np.uint8)  # 转为2D数组
            
            # 将numpy数组转换为PIL图像
            pil_img = Image.fromarray(img_np)
            
            # 找到有效区域（非白色补齐区域）的边界
            # 遮罩中0的区域是原始图像内容，1的区域是补齐的白色区域
            non_white_pixels = np.where(mask_np == 0)
            
            if len(non_white_pixels[0]) == 0:
                # 如果没有有效区域，直接返回缩放后的整幅图像
                cropped_img = pil_img
            else:
                # 计算边界框
                min_y = np.min(non_white_pixels[0])
                max_y = np.max(non_white_pixels[0])
                min_x = np.min(non_white_pixels[1])
                max_x = np.max(non_white_pixels[1])
                
                # 裁剪有效区域
                cropped_img = pil_img.crop((min_x, min_y, max_x + 1, max_y + 1))
            
            # 使用Lanczos滤镜缩放到目标尺寸
            resized_img = cropped_img.resize((target_width, target_height), Image.LANCZOS)
            
            # 转换回张量格式
            result_np = np.array(resized_img).astype(np.float32) / 255.0
            results.append(torch.from_numpy(result_np))
        
        # 堆叠处理结果并保持正确的维度格式
        result_tensor = torch.stack(results)
        
        return (result_tensor, target_width, target_height)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "AFL:Letterbox Crop and Resize": CropAndResize
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AFL:Letterbox Crop and Resize": "Letterbox Crop and Resize"
}
    