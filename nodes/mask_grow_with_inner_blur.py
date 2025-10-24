import torch
import numpy as np
import cv2
from scipy.ndimage import binary_dilation, binary_fill_holes

class MaskGrowWithInnerBlur:
    """
    基于原始mask形状的扩展节点，支持边缘向内部模糊
    
    保持原始mask形状特征并按比例扩展，同时可调整遮罩边缘向内部模糊
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "fill_mask_first": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否先填充mask中的内部孔洞，作为所有处理的前置步骤"
                }),
                "grow_factor": ("FLOAT", {
                    "default": 1.2,
                    "min": 1.0,
                    "max": 5.0,
                    "step": 0.05,
                    "tooltip": "扩展倍数，1.0表示不扩展，大于1.0表示按比例扩大"
                }),
                "preserve_details": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "是否保留原始mask的细节特征"
                }),
                "edge_blur_strength": ("INT", {
                    "default": 5,
                    "min": 0,
                    "max": 200,
                    "step": 1,
                    "tooltip": "边缘向内部模糊的强度，0表示不模糊"
                })
            }
        }
    
    RETURN_TYPES = ("MASK",)
    FUNCTION = "process_mask"
    CATEGORY = "AFL/Mask"

    def process_mask(self, mask, fill_mask_first, grow_factor, preserve_details, edge_blur_strength):
        # 确保输入是正确的格式
        if isinstance(mask, torch.Tensor):
            # 转换为numpy数组以便处理，假设mask是单通道
            mask_np = mask.cpu().numpy().squeeze()
        else:
            mask_np = np.array(mask).squeeze()
        
        # 归一化到0-255范围的二值图像
        mask_np = (mask_np * 255).astype(np.uint8)
        _, binary_mask = cv2.threshold(mask_np, 1, 255, cv2.THRESH_BINARY)
        
        # 填充mask中的内部孔洞（前置步骤）
        if fill_mask_first:
            # 将二值图像转换为布尔数组进行孔洞填充
            binary_bool = binary_mask > 0
            # 使用scipy的binary_fill_holes填充内部孔洞
            filled_bool = binary_fill_holes(binary_bool)
            # 转换回0-255范围的uint8
            binary_mask = (filled_bool * 255).astype(np.uint8)
        
        # 如果没有有效区域，直接返回原mask
        if not np.any(binary_mask):
            return (mask,)
        
        # 计算原始mask的边界框和尺寸
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return (mask,)
            
        # 计算所有轮廓的边界框
        x_min, y_min = binary_mask.shape[1], binary_mask.shape[0]
        x_max, y_max = 0, 0
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
        
        # 计算原始mask的宽度和高度
        original_width = x_max - x_min
        original_height = y_max - y_min
        
        # 根据扩展倍数计算需要扩展的像素数
        expand_pixels = int((max(original_width, original_height) * (grow_factor - 1)) / 2)
        expand_pixels = max(0, expand_pixels)  # 确保不小于0
        
        # 选择结构化元素，保留细节时使用较小的圆形核
        if preserve_details:
            kernel_size = min(5, expand_pixels * 2 + 1) if expand_pixels > 0 else 3
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        else:
            kernel_size = expand_pixels * 2 + 1 if expand_pixels > 0 else 3
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # 转换为布尔数组进行形态学操作
        bool_mask = binary_mask.astype(bool)
        
        # 进行多次小步膨胀，更好地保留形状
        processed_mask = bool_mask.copy()
        if expand_pixels > 0:
            steps = max(1, expand_pixels // 2)  # 将膨胀分为多个步骤
            step_size = max(1, expand_pixels // steps)
            
            for _ in range(steps):
                current_kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, 
                    (step_size * 2 + 1, step_size * 2 + 1)
                )
                processed_mask = binary_dilation(processed_mask, current_kernel)
            
            # 处理剩余的膨胀步骤
            remaining = expand_pixels % steps
            if remaining > 0:
                current_kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, 
                    (remaining * 2 + 1, remaining * 2 + 1)
                )
                processed_mask = binary_dilation(processed_mask, current_kernel)
        
        # 转换回0-255范围的uint8
        processed_mask_uint8 = (processed_mask.astype(np.uint8) * 255)
        
        # 边缘向内部模糊处理
        if edge_blur_strength > 0:
            # 计算距离变换，得到每个像素到边缘的距离
            distance = cv2.distanceTransform(processed_mask_uint8, cv2.DIST_L2, 5)
            
            if np.max(distance) > 0:
                # 归一化距离变换结果
                distance = cv2.normalize(distance, None, 0, 1.0, cv2.NORM_MINMAX)
                
                # 创建模糊权重图：距离边缘越近，模糊权重越高
                # 距离超过模糊强度的区域完全不模糊
                max_blur_distance = edge_blur_strength
                weight_map = np.exp(-(distance **2) / (2 * (max_blur_distance / 3)** 2))  # 高斯分布权重
                weight_map = np.clip(weight_map, 0, 1)
                
                # 对整个mask进行模糊处理
                blur_kernel_size = (edge_blur_strength * 2 + 1, edge_blur_strength * 2 + 1)
                blurred_mask = cv2.GaussianBlur(processed_mask_uint8, blur_kernel_size, 0)
                blurred_mask = blurred_mask.astype(np.float32) / 255.0
                
                # 将原始mask转换为0-1范围
                original_float = processed_mask_uint8.astype(np.float32) / 255.0
                
                # 混合原始mask和模糊mask，实现边缘向内模糊效果
                processed_mask = (original_float * (1 - weight_map)) + (blurred_mask * weight_map)
            else:
                # 如果mask太小，直接应用简单模糊
                blur_kernel_size = (edge_blur_strength * 2 + 1, edge_blur_strength * 2 + 1)
                processed_mask = cv2.GaussianBlur(processed_mask_uint8, blur_kernel_size, 0)
                processed_mask = processed_mask.astype(np.float32) / 255.0
        else:
            # 不模糊时直接转换为0-1范围
            processed_mask = processed_mask_uint8.astype(np.float32) / 255.0
        
        # 转换回torch tensor并保持原来的维度
        if len(mask.shape) == 4:  # 处理(batch, 1, h, w)格式
            new_mask = torch.from_numpy(processed_mask).unsqueeze(0).unsqueeze(0).float()
        else:  # 处理(1, h, w)格式
            new_mask = torch.from_numpy(processed_mask).unsqueeze(0).float()
        
        return (new_mask,)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "AFL:MaskGrowWithInnerBlur": MaskGrowWithInnerBlur
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AFL:MaskGrowWithInnerBlur": "Mask Grow With Inner Blur"
}
