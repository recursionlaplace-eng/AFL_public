import math
import torch
import comfy.utils
import node_helpers  # 正确导入node_helpers

class AFL_Qweneditplus_Fastuse:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamic_prompts": True}),
                "choose_latent": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 3,
                    "step": 1,
                    "tooltip": "选择哪个图像作为latent输出: 0=无, 1=image1, 2=image2, 3=image3"
                }),
                "total_int": (["384*384", "nan"], {
                    "default": "384*384",
                    "tooltip": "仅提供384*384和nan两个选项，nan表示不进行缩放"
                }),
            },
            "optional": {
                "vae": ("VAE",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("conditioning", "conditioning_zeroout", "latent")
    FUNCTION = "encode"
    CATEGORY = "AFL"
    
    def encode(self, clip, prompt, choose_latent, total_int, vae=None, image1=None, image2=None, image3=None):
        images = [None, image1, image2, image3]  # 索引0为无，1为image1，2为image2，3为image3
        images_vl = []
        llama_template = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        image_prompt = ""
        
        # 按照ComfyUI ReferenceLatent节点标准，存储latent samples的列表
        reference_latent_samples = []
        
        # 解析total_int参数 - 仅保留384和nan选项
        def parse_total_int(total_str):
            if total_str == "384*384":
                return int(384 * 384)
            elif total_str == "nan":
                # 返回特殊值表示不进行缩放
                return -1
            # 由于INPUT_TYPES已限制选项，此情况理论上不会发生
            # 但保留返回默认值以确保健壮性
            return int(384 * 384)
        
        # 处理所有图像 - 用于reference_latents和clip tokenize
        for i, image in enumerate(images[1:], 1):  # 从1开始，对应image1, image2, image3
            if image is not None:
                # 1. 处理用于clip tokenize的图像，根据total_int参数决定是否缩放
                samples = image.movedim(-1, 1)  # 转换为[B, C, H, W]格式
                total = parse_total_int(total_int)
                
                # 当选择'nan'时不进行缩放，直接使用原始图像
                if total == -1:
                    images_vl.append(samples.movedim(1, -1))  # 转回[B, H, W, C]格式用于tokenize
                else:
                    # 按照指定尺寸缩放图像
                    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                    width = round(samples.shape[3] * scale_by)
                    height = round(samples.shape[2] * scale_by)
                    s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                    images_vl.append(s.movedim(1, -1))  # 转回[B, H, W, C]格式用于tokenize
                
                # 2. 为VAE编码准备图像用于reference_latents，严格按照ComfyUI标准处理
                if vae is not None:
                    try:
                        # 对于VAE编码，使用合理的尺寸，确保是8的倍数
                        total_vae = int(1024 * 1024)
                        scale_by_vae = math.sqrt(total_vae / (samples.shape[3] * samples.shape[2]))
                        
                        # 确保尺寸是8的倍数，这是VAE处理的标准要求
                        width_vae = round(samples.shape[3] * scale_by_vae / 8.0) * 8
                        height_vae = round(samples.shape[2] * scale_by_vae / 8.0) * 8
                        
                        # 确保最小尺寸至少为64x64（避免小图像问题）
                        min_dim = 64
                        width_vae = max(min_dim, width_vae)
                        height_vae = max(min_dim, height_vae)
                        
                        # 进行缩放
                        s_vae = comfy.utils.common_upscale(samples, width_vae, height_vae, "area", "disabled")
                        # 转回[B, H, W, C]格式并只取RGB通道
                        img_for_vae = s_vae.movedim(1, -1)[:, :, :, :3]
                        
                        # 生成latent，获取samples
                        encoded_latent = vae.encode(img_for_vae)
                        
                        # 确保我们有samples张量
                        if isinstance(encoded_latent, dict) and 'samples' in encoded_latent:
                            latent_samples = encoded_latent['samples']
                        elif isinstance(encoded_latent, torch.Tensor):
                            latent_samples = encoded_latent
                        else:
                            raise TypeError(f"Unexpected latent type: {type(encoded_latent)}")
                        
                        # 添加到reference_latents列表，符合ReferenceLatent节点的标准格式
                        reference_latent_samples.append(latent_samples)
                    except Exception as e:
                        print(f"Error processing image {i} for reference_latents: {e}")

                image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i)
        
        # 独立的latent处理逻辑 - 直接对选择的图像进行原始尺寸编码（不缩放）
        selected_latent_samples = None
        if 1 <= choose_latent <= 3 and images[choose_latent] is not None and vae is not None:
            try:
                # 直接获取原始图像（不进行任何缩放）
                selected_image = images[choose_latent]
                
                # 转换格式并只取RGB通道
                samples = selected_image.movedim(-1, 1)  # [B, H, W, C] -> [B, C, H, W]
                img_for_vae = samples.movedim(1, -1)[:, :, :, :3]  # 转回[B, H, W, C]并只取RGB通道
                
                # 直接对原始图像进行VAE编码
                encoded_latent = vae.encode(img_for_vae)
                
                # 确保我们有samples张量
                if isinstance(encoded_latent, dict) and 'samples' in encoded_latent:
                    selected_latent_samples = encoded_latent['samples']
                elif isinstance(encoded_latent, torch.Tensor):
                    selected_latent_samples = encoded_latent
                else:
                    raise TypeError(f"Unexpected latent type: {type(encoded_latent)}")
                    
                print(f"Successfully encoded original size latent from image {choose_latent}")
                
                # 将选择图像的latent替换到对应的reference_latent位置
                # 注意：reference_latent_samples的索引是0-based，而images的索引是1-based
                # 所以我们需要将choose_latent减1来得到正确的索引位置
                if 0 <= (choose_latent - 1) < len(reference_latent_samples):
                    reference_latent_samples[choose_latent - 1] = selected_latent_samples
                    print(f"Replaced reference_latent at position {choose_latent - 1} with selected image's original latent")
                    
            except Exception as e:
                print(f"Error encoding original image {choose_latent} for latent output: {e}")

        # 生成基础conditioning
        if images_vl:
            tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
        else:
            tokens = clip.tokenize(image_prompt + prompt, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        
        # 根据ComfyUI ReferenceLatent节点的标准实现，正确设置reference_latents
        # 注意：reference_latent_samples列表已经被更新，包含了选择图像的原始尺寸latent
        if reference_latent_samples:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": reference_latent_samples}, append=True)
        
        # 创建latent输出 - 确保samples格式正确
        if selected_latent_samples is not None:
            latent_output = {"samples": selected_latent_samples}
        else:
            # 如果没有选择latent或编码失败，输出空latent
            latent_output = {"samples": torch.zeros([1, 4, 64, 64], device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))}
        
        # 创建conditioning_zeroout输出（基于更新后的conditioning）
        conditioning_zeroout = []
        for t in conditioning:
            d = t[1].copy()
            pooled_output = d.get("pooled_output", None)
            if pooled_output is not None:
                d["pooled_output"] = torch.zeros_like(pooled_output)
            
            # 保留reference_latents信息在conditioning_zeroout中
            # 这是重要的，确保reference_latents不会被零化
            
            n = [torch.zeros_like(t[0]), d]
            conditioning_zeroout.append(n)
        
        return (conditioning, conditioning_zeroout, latent_output)
  
  # 确保节点能被ComfyUI加载
NODE_CLASS_MAPPINGS = {
    "AFL:AFL_Qweneditplus_Fastuse": AFL_Qweneditplus_Fastuse
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AFL:AFL_Qweneditplus_Fastuse": "AFL_Qweneditplus_Fastuse",
}