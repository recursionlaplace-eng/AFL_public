import os
import importlib.util
import sys

# 获取当前文件夹路径
current_dir = os.path.dirname(__file__)

# 将nodes目录添加到sys.path
nodes_dir = os.path.join(current_dir, 'nodes')
if nodes_dir not in sys.path:
    sys.path.insert(0, nodes_dir)

# 构建 nodes 子目录路径
nodes_dir = os.path.join(current_dir, 'nodes')

# 确保 nodes 目录存在
if not os.path.exists(nodes_dir):
    raise FileNotFoundError(f"Nodes directory not found: {nodes_dir}")

# AFL_public中包含的节点文件名（不带.py后缀）
node_files = [
    "AFL_Qweneditplus_Fastuse",
    "aspect_ratio_matcher",
    "image_auto_switch",
    "Letterbox_crop_and_resize",
    "lora_overwrite",
    "lora_prompt_manager",
    "lora_prompt_manager_v2",
    "mask_box_crop_node",
    "mask_grow_with_inner_blur",
    "optimal_resolution_node",
    "ResizeWithLetterboxV2",
]

# 初始化节点映射
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# 逐个个加载节点文件
for file in node_files:
    # 构建文件路径 (从 nodes 子目录加载)
    file_path = os.path.join(nodes_dir, f"{file}.py")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"警告：未找到节点文件 {file_path}")
        continue
    
    # 强制导入模块
    spec = importlib.util.spec_from_file_location(file, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[file] = module
    spec.loader.exec_module(module)
    
    # 合并节点映射
    if hasattr(module, "NODE_CLASS_MAPPINGS"):
        NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
    if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
        NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
