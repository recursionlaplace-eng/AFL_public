import os
import yaml
import json
import folder_paths
from typing import List, Dict, Optional, Tuple

# 获取当前节点文件所在的文件夹路径
NODE_DIR = os.path.dirname(os.path.abspath(__file__))
# 存储LoRA与prompt的对应关系（位于节点文件夹内）
PROMPT_STORAGE_FILE = os.path.join(NODE_DIR, "lora_prompts.json")
# 支持的LoRA文件扩展名
LORA_EXTENSIONS = [".safetensors", ".ckpt", ".pt"]


class LoraPromptManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        """初始化管理器，加载数据"""
        self.lora_paths: List[str] = []
        self.lora_files: List[str] = []  # 存储格式: "根路径标识::完整相对路径(含后缀)"
        self.lora_categories: Dict[str, List[str]] = {}
        self.lora_prompts: Dict[str, str] = {}
        self.root_path_identifiers: Dict[str, str] = {}  # 根路径 -> 短标识
        self.full_lora_paths: Dict[str, str] = {}  # 存储lora_id到完整路径的映射

        self.load_prompts()
        self.refresh_lora_data()

    def refresh_lora_data(self):
        """刷新LoRA路径和文件列表"""
        self.lora_paths = self.get_lora_paths()
        self.lora_files = self.scan_lora_files()
        self.lora_categories = self.build_lora_categories()

    def get_lora_paths(self) -> List[str]:
        """获取所有LoRA模型的路径，包括额外配置的路径"""
        paths = []
        
        # 添加默认LoRA路径
        default_lora_paths = folder_paths.get_folder_paths("loras")
        if default_lora_paths:
            paths.extend(default_lora_paths)
        
        # 检查额外模型路径配置文件
        extra_paths_file = os.path.join(os.path.dirname(folder_paths.__file__), "extra_model_paths.yaml")
        if os.path.exists(extra_paths_file):
            try:
                with open(extra_paths_file, 'r', encoding='utf-8') as f:
                    extra_config = yaml.safe_load(f)
                    for config in extra_config.values():
                        if isinstance(config, dict) and "loras" in config:
                            lora_path = config["loras"]
                            full_path = os.path.abspath(lora_path) if os.path.isabs(lora_path) else \
                                       os.path.abspath(os.path.join(os.path.dirname(extra_paths_file), lora_path))
                            
                            if full_path not in paths and os.path.exists(full_path) and os.path.isdir(full_path):
                                paths.append(full_path)
            except Exception as e:
                print(f"Error processing extra model paths: {e}")
        
        # 去重并生成根路径标识
        unique_paths = list(set(paths))
        self.root_path_identifiers = {path: f"path_{i}" for i, path in enumerate(unique_paths)}
        return unique_paths

    def scan_lora_files(self) -> List[str]:
        """扫描所有LoRA路径下的模型文件，使用唯一标识避免冲突"""
        lora_files = []
        self.full_lora_paths = {}  # 重置完整路径映射
        
        for base_path in self.lora_paths:
            if not os.path.exists(base_path) or not os.path.isdir(base_path):
                continue
                
            base_id = self.root_path_identifiers[base_path]
            for root, _, files in os.walk(base_path):
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext in LORA_EXTENSIONS:
                        # 保留扩展名的完整相对路径
                        rel_path = os.path.relpath(os.path.join(root, file), base_path)
                        # 使用根路径标识+相对路径作为唯一标识
                        unique_id = f"{base_id}::{rel_path}"
                        lora_files.append(unique_id)
                        # 保存完整路径映射
                        self.full_lora_paths[unique_id] = os.path.join(root, file)
        
        return sorted(lora_files)

    def build_lora_categories(self) -> Dict[str, List[str]]:
        """按文件夹构建LoRA分类"""
        categories = {"All": self.lora_files.copy()}
        
        for lora_id in self.lora_files:
            # 解析唯一标识，提取相对路径部分
            _, lora_name = lora_id.split("::", 1)
            folder = os.path.dirname(lora_name) or "Root"
            
            if folder not in categories:
                categories[folder] = []
            categories[folder].append(lora_id)
            
        return categories

    def load_prompts(self) -> None:
        """加载保存的LoRA对应的prompt"""
        if os.path.exists(PROMPT_STORAGE_FILE):
            try:
                with open(PROMPT_STORAGE_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.lora_prompts = data if isinstance(data, dict) else {}
            except Exception as e:
                print(f"Error loading lora_prompts.json: {e}")
        else:
            self.lora_prompts = {}

    def save_prompts(self) -> None:
        """保存LoRA对应的prompt"""
        try:
            os.makedirs(os.path.dirname(PROMPT_STORAGE_FILE), exist_ok=True)
            with open(PROMPT_STORAGE_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.lora_prompts, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving lora_prompts.json: {e}")

    def get_prompt(self, lora_id: str) -> str:
        """获取指定LoRA的prompt"""
        return self.lora_prompts.get(lora_id, "")

    def update_prompt(self, lora_id: str, prompt: str, overwrite: bool = True) -> str:
        """更新指定LoRA的prompt"""
        if overwrite or lora_id not in self.lora_prompts:
            self.lora_prompts[lora_id] = prompt.strip()
            self.save_prompts()
        return self.lora_prompts[lora_id]

    def get_display_name(self, lora_id: str) -> str:
        """获取用于UI显示的名称（包含文件后缀，兼容COMBO）"""
        _, display_name = lora_id.split("::", 1)
        return display_name


class LoraPromptManagerNode:
    def __init__(self):
        self.manager = LoraPromptManager()

    @classmethod
    def INPUT_TYPES(cls):
        manager = LoraPromptManager()
        # 刷新数据确保获取最新列表
        manager.refresh_lora_data()
        # 转换为显示名称供UI选择（包含文件后缀）
        lora_display_list = [manager.get_display_name(lora_id) for lora_id in manager.lora_files]
        
        return {
            "required": {
                "lora_name": (lora_display_list,),
                "overwrite": (["true", "false"], {"default": "false"}),
            },
            "optional": {
                "prompt_overwrite": ("STRING", {"multiline": True, "placeholder": "Enter prompt for this LoRA..."}),
                "refresh": (["refresh"], {"default": "refresh"}),  # 刷新按钮
            }
        }

    # 只返回显示名称（带后缀）和提示词
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("lora_display_name", "prompt")
    FUNCTION = "process"
    CATEGORY = "AFL/LoRA Tools"

    def process(self, lora_name: str, overwrite: str, 
                prompt_overwrite: Optional[str] = None, refresh: str = "refresh"):
        # 刷新数据（当点击刷新按钮时）
        if refresh == "refresh":
            self.manager.refresh_lora_data()
        
        # 找到选中的LoRA的唯一ID
        selected_lora_id = next(
            (lora_id for lora_id in self.manager.lora_files 
             if self.manager.get_display_name(lora_id) == lora_name),
            None
        )
        
        if not selected_lora_id:
            return ("", "")
        
        # 更新prompt
        if prompt_overwrite is not None and prompt_overwrite.strip():
            self.manager.update_prompt(
                selected_lora_id,
                prompt_overwrite,
                overwrite=(overwrite == "true")
            )
        
        current_prompt = self.manager.get_prompt(selected_lora_id)
        # 返回带后缀的显示名称，兼容COMBO
        return (lora_name, current_prompt)


# 注册节点
NODE_CLASS_MAPPINGS = {
    "AFL:LoraPromptManager": LoraPromptManagerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AFL:LoraPromptManager": "LoRA Prompt Manager"
}