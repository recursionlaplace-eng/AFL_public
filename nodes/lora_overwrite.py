import os
import json
import logging
from typing import Dict, Any, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("lora_prompt_manager.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PromptNodes")


class LoRAOverwriteNode:
    """
    LoRA Overwrite Node
    用于创建可以被LoRAPromptManagerV2读取的over_write数据结构
    包含各种LoRA属性配置选项
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "# recommend_lora_strength (推荐强度)": ("STRING", {"default": "1.0", "placeholder": "输入LoRA的推荐使用强度"}),
                "# trigger_word (触发词)": ("STRING", {"multiline": True, "default": "", "placeholder": "输入LoRA的触发词，支持多行"}),
                "# prompt1 (提示词1)": ("STRING", {"multiline": True, "default": "", "placeholder": "第一个提示词，支持多行"}),
                "# prompt2 (提示词2)": ("STRING", {"multiline": True, "default": "", "placeholder": "第二个提示词，支持多行"}),
                "# prompt3 (提示词3)": ("STRING", {"multiline": True, "default": "", "placeholder": "第三个提示词，支持多行"}),
                "# negative_prompt (负面提示词)": ("STRING", {"multiline": True, "default": "", "placeholder": "负面提示词，支持多行"}),
            }
        }

    RETURN_TYPES = ("STRING",)  # 返回JSON字符串作为over_write数据
    RETURN_NAMES = ("over_write",)
    FUNCTION = "create_overwrite"
    CATEGORY = "AFL/LoRA Tools"

    def create_overwrite(self, **kwargs) -> tuple:
        """
        创建over_write数据结构，以JSON字符串形式返回
        """
        try:
            # 提取实际的输入值，去掉标签前缀
            recommend_lora_strength = kwargs.get('# recommend_lora_strength (推荐强度)', '1.0')
            trigger_word = kwargs.get('# trigger_word (触发词)', '')
            prompt1 = kwargs.get('# prompt1 (提示词1)', '')
            prompt2 = kwargs.get('# prompt2 (提示词2)', '')
            prompt3 = kwargs.get('# prompt3 (提示词3)', '')
            negative_prompt = kwargs.get('# negative_prompt (负面提示词)', '')
            
            # 构建overwrite数据结构
            overwrite_data = {
                "recommend_lora_strength": float(recommend_lora_strength) if recommend_lora_strength else 1.0,
                "trigger_word": trigger_word,
                "prompt1": prompt1,
                "prompt2": prompt2,
                "prompt3": prompt3,
                "negative_prompt": negative_prompt
            }
            
            # 转换为JSON字符串
            json_str = json.dumps(overwrite_data, ensure_ascii=False, indent=2)
            logger.info(f"创建over_write数据: {json_str}")
            
            return (json_str,)
        except Exception as e:
            logger.error(f"创建over_write数据时出错: {str(e)}", exc_info=True)
            # 返回空的JSON结构作为错误处理
            return (json.dumps({}),)


# 注册节点
NODE_CLASS_MAPPINGS = {
    "AFL:LoRAOverwrite": LoRAOverwriteNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AFL:LoRAOverwrite": "Lora Over Write"
}