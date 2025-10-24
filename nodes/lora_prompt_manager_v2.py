import os
import yaml
import json
import ctypes
import ctypes.wintypes
import logging
from typing import List, Dict, Optional, Tuple

# 配置日志系统 - 便于错误分析
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("lora_prompt_manager.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PromptNodes")


class FilePropertyManager:
    """文件属性管理工具类"""
    GENERIC_READ = 0x80000000
    GENERIC_WRITE = 0x40000000
    OPEN_EXISTING = 3
    CREATE_ALWAYS = 2
    FILE_ATTRIBUTE_NORMAL = 0x80
    INVALID_HANDLE_VALUE = -1

    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

    @classmethod
    def get_custom_property(cls, file_path: str, prop_name: str) -> Optional[str]:
        """读取文件的自定义属性值"""
        try:
            file_path = os.path.abspath(file_path)
            if not os.path.exists(file_path) or not os.path.isfile(file_path):
                logger.warning(f"文件不存在: {file_path}")
                return None
            
            stream_path = f"{file_path}:{prop_name}"
            stream_path_w = ctypes.create_unicode_buffer(stream_path)
            
            h_file = cls.kernel32.CreateFileW(
                stream_path_w,
                cls.GENERIC_READ,
                0,
                None,
                cls.OPEN_EXISTING,
                cls.FILE_ATTRIBUTE_NORMAL,
                None
            )
            
            if h_file == cls.INVALID_HANDLE_VALUE:
                logger.debug(f"属性 '{prop_name}' 不存在于文件 '{file_path}'")
                return None
            
            try:
                buffer = ctypes.create_string_buffer(1024 * 1024)
                bytes_read = ctypes.wintypes.DWORD()
                success = cls.kernel32.ReadFile(
                    h_file, buffer, len(buffer), ctypes.byref(bytes_read), None
                )
                
                if not success:
                    logger.error(f"读取属性 '{prop_name}' 失败")
                    return None
                
                return buffer.raw[:bytes_read.value].decode('utf-8', errors='replace')
            finally:
                cls.kernel32.CloseHandle(h_file)
        except Exception as e:
            logger.error(f"读取属性时发生错误: {str(e)}", exc_info=True)
            return None

    @classmethod
    def set_custom_property(cls, file_path: str, prop_name: str, value: str) -> bool:
        """设置文件的自定义属性值"""
        try:
            file_path = os.path.abspath(file_path)
            if not os.path.exists(file_path) or not os.path.isfile(file_path):
                logger.warning(f"文件不存在: {file_path}")
                return False
            
            stream_path = f"{file_path}:{prop_name}"
            stream_path_w = ctypes.create_unicode_buffer(stream_path)
            
            h_file = cls.kernel32.CreateFileW(
                stream_path_w,
                cls.GENERIC_WRITE,
                0,
                None,
                cls.CREATE_ALWAYS,
                cls.FILE_ATTRIBUTE_NORMAL | 0x02000000,
                None
            )
            
            if h_file == cls.INVALID_HANDLE_VALUE:
                logger.error(f"无法打开属性 '{prop_name}' 的数据流")
                return False
            
            try:
                value_bytes = value.encode('utf-8')
                bytes_written = ctypes.wintypes.DWORD()
                success = cls.kernel32.WriteFile(
                    h_file, value_bytes, len(value_bytes),
                    ctypes.byref(bytes_written), None
                )
                
                result = success and bytes_written.value == len(value_bytes)
                if result:
                    logger.debug(f"成功写入属性 '{prop_name}' 到文件 '{file_path}'")
                else:
                    logger.error(f"写入属性 '{prop_name}' 失败")
                return result
            finally:
                cls.kernel32.CloseHandle(h_file)
        except Exception as e:
            logger.error(f"设置属性时发生错误: {str(e)}", exc_info=True)
            return False


class LoRAPromptManagerV2:
    """改进版LoRA提示词管理节点
    支持多行提示词输入输出和负面提示词管理
    支持通过over_write参数批量设置多个属性
    """
    
    def __init__(self):
        self.lora_paths = self._get_lora_paths()
        self.properties_manager = FilePropertyManager()
        # 自动运行测试
        self._run_self_tests()

    @classmethod
    def INPUT_TYPES(s):
        lora_files = s._get_lora_files()
        
        return {
            "required": {
                "lora_name": (lora_files,),
                "overwrite": ([False, True], {"default": False}),
                "choose_prompt": ([1, 2], {"default": 1, "label": "选择输出的提示词 (1或2)"}),
            },
            "optional": {
                "over_write": ("STRING", {"placeholder": "从LoRA Overwrite节点传入的JSON数据"}),
            }
        }

    # 修改返回类型，使lora_name可以被任何接口接受
    RETURN_TYPES = ("*", "FLOAT", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("lora_name", "lora_strength", "trigger_word", "prompt", "prompt_negative", "all_infor")
    FUNCTION = "process"
    CATEGORY = "AFL/LoRA Tools"

    @staticmethod
    def _get_lora_paths() -> List[str]:
        """获取所有LoRA模型路径（包括配置文件中的路径）"""
        paths = []
        
        # 尝试导入folder_paths
        try:
            import folder_paths
            # 默认LoRA路径
            default_lora_path = os.path.join(folder_paths.models_dir, "loras")
            if os.path.exists(default_lora_path):
                paths.append(default_lora_path)
                logger.info(f"添加默认LoRA路径: {default_lora_path}")
            
            # 从extra_model_paths.yaml添加路径
            extra_config_path = os.path.join(os.path.dirname(folder_paths.__file__), "extra_model_paths.yaml")
            if os.path.exists(extra_config_path):
                try:
                    with open(extra_config_path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                        
                        for key in ['lora_paths', 'loras', 'additional_loras']:
                            if key in config and isinstance(config[key], list):
                                for path in config[key]:
                                    full_path = os.path.abspath(path)
                                    if os.path.exists(full_path) and full_path not in paths:
                                        paths.append(full_path)
                                        logger.info(f"从配置文件添加LoRA路径: {full_path}")
                except Exception as e:
                    logger.error(f"读取extra_model_paths.yaml时出错: {str(e)}")
        except ImportError:
            # 如果无法导入folder_paths，使用默认路径
            logger.warning("无法导入folder_paths模块，使用默认路径")
            default_lora_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "loras")
            if os.path.exists(default_lora_path):
                paths.append(default_lora_path)
        
        return paths

    @staticmethod
    def _get_lora_files() -> List[str]:
        """获取所有LoRA文件（包括子目录）"""
        lora_extensions = ('.safetensors', '.pt', '.bin', '.ckpt')
        lora_files = []
        lora_paths = LoRAPromptManagerV2._get_lora_paths()
        
        for base_path in lora_paths:
            try:
                for root, _, files in os.walk(base_path):
                    for file in files:
                        if file.lower().endswith(lora_extensions):
                            rel_path = os.path.relpath(os.path.join(root, file), base_path)
                            lora_files.append(rel_path)
            except Exception as e:
                logger.error(f"遍历LoRA路径 {base_path} 时出错: {str(e)}")
        
        unique_lora_files = sorted(list(set(lora_files)))
        logger.info(f"找到 {len(unique_lora_files)} 个LoRA文件")
        return unique_lora_files

    def _get_lora_full_path(self, lora_name: str) -> Optional[str]:
        """获取LoRA文件的完整路径"""
        for base_path in self.lora_paths:
            full_path = os.path.join(base_path, lora_name)
            if os.path.exists(full_path) and os.path.isfile(full_path):
                return full_path
        logger.warning(f"未找到LoRA文件: {lora_name}")
        return None

    @staticmethod
    def _split_lines(text: Optional[str], max_lines: int = 3) -> List[str]:
        """将文本按行分割，最多返回指定行数，不足则用空字符串填充"""
        if not text:
            return [""] * max_lines
            
        lines = [line.rstrip('\r\n') for line in text.splitlines()]
        while len(lines) < max_lines:
            lines.append("")
        return lines[:max_lines]

    @staticmethod
    def _join_lines(lines: List[str]) -> str:
        """将多行文本连接成一个字符串"""
        return '\n'.join(lines)

    def _load_prompt_properties(self, lora_path: str) -> Tuple[Dict[str, str], Dict[str, str]]:
        """加载LoRA文件中的prompt属性"""
        # 正面提示词
        prompts = {
            "prompt1": "",
            "prompt2": "",
            "prompt3": ""
        }
        
        # 负面提示词
        negative_prompts = {
            "prompt_negative1": ""
        }
        
        # 读取现有属性
        for prompt_key in prompts.keys():
            value = self.properties_manager.get_custom_property(lora_path, prompt_key)
            if value is not None:
                prompts[prompt_key] = value
                
        for prompt_key in negative_prompts.keys():
            value = self.properties_manager.get_custom_property(lora_path, prompt_key)
            if value is not None:
                negative_prompts[prompt_key] = value
                
        return prompts, negative_prompts

    def _save_prompt_properties(self, lora_path: str, prompts: Dict[str, str], negative_prompts: Dict[str, str]) -> None:
        """保存LoRA文件的prompt属性"""
        for prompt_key, value in prompts.items():
            self.properties_manager.set_custom_property(lora_path, prompt_key, value)
            
        for prompt_key, value in negative_prompts.items():
            self.properties_manager.set_custom_property(lora_path, prompt_key, value)

    def _load_all_properties(self, lora_path: str) -> Dict[str, str]:
        """加载LoRA文件的所有自定义属性"""
        # 定义需要加载的所有属性
        properties = {
            "prompt1": "",
            "prompt2": "",
            "prompt3": "",
            "prompt_negative1": "",
            "trigger_word": "",
            "recommend_lora_strength": "1.0"
        }
        
        # 读取现有属性
        for key in properties.keys():
            value = self.properties_manager.get_custom_property(lora_path, key)
            if value is not None:
                properties[key] = value
        
        return properties
    
    def _save_all_properties(self, lora_path: str, properties: Dict[str, str]) -> None:
        """保存LoRA文件的所有自定义属性"""
        for key, value in properties.items():
            self.properties_manager.set_custom_property(lora_path, key, value)
    
    def _format_all_infor(self, properties: Dict[str, str]) -> str:
        """格式化所有信息，使其易于识别"""
        lines = []
        lines.append("=== LoRA 详细信息 ===")
        lines.append(f"[触发词trigger_word]: {properties.get('trigger_word', '')}")
        lines.append(f"[推荐强度recommend_lora_strength]: {properties.get('recommend_lora_strength', '1.0')}")
        lines.append("[提示词1prompt1]:")
        lines.append(properties.get('prompt1', ''))
        lines.append("[提示词2prompt2]:")
        lines.append(properties.get('prompt2', ''))
        lines.append("[提示词3prompt3]:")
        lines.append(properties.get('prompt3', ''))
        lines.append("[负面提示词]:")
        lines.append(properties.get('prompt_negative1', ''))
        lines.append("===================")
        return '\n'.join(lines)
    
    def process(self, lora_name: str, overwrite: bool, choose_prompt: int, over_write: Optional[str] = None):
        """处理LoRA提示词属性"""
        try:
            logger.info(f"处理LoRA: {lora_name}, overwrite: {overwrite}, choose_prompt: {choose_prompt}")
            
            # 获取LoRA文件完整路径
            lora_path = self._get_lora_full_path(lora_name)
            if not lora_path:
                return (lora_name, 1.0, "", "", "", "未找到LoRA文件路径")
            
            # 加载所有属性
            properties = self._load_all_properties(lora_path)
            
            # 处理over_write输入
            if over_write and overwrite:
                try:
                    # 解析JSON数据
                    overwrite_data = json.loads(over_write)
                    
                    # 只更新非空属性
                    if overwrite_data.get('recommend_lora_strength') is not None:
                        properties['recommend_lora_strength'] = str(overwrite_data['recommend_lora_strength'])
                    
                    if 'trigger_word' in overwrite_data and overwrite_data['trigger_word'].strip() != "":
                        properties['trigger_word'] = overwrite_data['trigger_word']
                    
                    if 'prompt1' in overwrite_data and overwrite_data['prompt1'].strip() != "":
                        properties['prompt1'] = overwrite_data['prompt1']
                    
                    if 'prompt2' in overwrite_data and overwrite_data['prompt2'].strip() != "":
                        properties['prompt2'] = overwrite_data['prompt2']
                    
                    if 'negative_prompt' in overwrite_data and overwrite_data['negative_prompt'].strip() != "":
                        properties['prompt_negative1'] = overwrite_data['negative_prompt']
                    
                    logger.info(f"应用over_write数据到 {lora_name}")
                except json.JSONDecodeError:
                    logger.error(f"over_write数据格式错误: {over_write}")
            
            # 保存更新后的属性
            self._save_all_properties(lora_path, properties)
            
            # 根据用户选择的choose_prompt选择输出的提示词
            if choose_prompt == 2 and properties.get('prompt2', '').strip():
                output_prompt = properties['prompt2']
            else:
                output_prompt = properties['prompt1']
            
            # 准备输出
            output_trigger_word = properties.get('trigger_word', '')
            output_negative_prompt = properties.get('prompt_negative1', '')
            output_all_infor = self._format_all_infor(properties)
            
            # 处理lora_strength输出为float类型，默认值为1.0
            try:
                output_lora_strength = float(properties.get('recommend_lora_strength', '1.0'))
            except (ValueError, TypeError):
                output_lora_strength = 1.0
            
            return (lora_name, output_lora_strength, output_trigger_word, output_prompt, output_negative_prompt, output_all_infor)
        except Exception as e:
            logger.error(f"处理过程中发生错误: {str(e)}", exc_info=True)
            return (lora_name, 1.0, "", "", "", f"处理错误: {str(e)}")

    def _run_self_tests(self):
        """自动运行节点功能测试"""
        logger.info("开始执行节点自动测试...")
        
        try:
            # 测试1: 行分割功能测试
            test_text = "第一行\n第二行\n第三行\n第四行"
            lines = self._split_lines(test_text)
            assert len(lines) == 3, f"行分割测试失败: 预期3行，实际{len(lines)}行"
            assert lines == ["第一行", "第二行", "第三行"], f"行分割内容错误: {lines}"
            
            # 测试2: 行连接功能测试
            test_lines = ["a", "b", "c"]
            text = self._join_lines(test_lines)
            assert text == "a\nb\nc", f"行连接测试失败: 预期'a\nb\nc'，实际'{text}'"
            
            # 测试3: 空输入处理测试
            empty_lines = self._split_lines(None)
            assert empty_lines == ["", "", ""], f"空输入处理失败: {empty_lines}"
            
            logger.info("所有自动测试通过!")
        except AssertionError as e:
            logger.error(f"测试失败: {str(e)}")
        except Exception as e:
            logger.error(f"测试过程中发生错误: {str(e)}", exc_info=True)


class LineSplitterNode:
    """简化版多行文本分割节点
    仅返回选中行的内容，支持更宽的行号范围
    """
    
    def __init__(self):
        # 自动运行测试
        self._run_self_tests()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "multiline_text": ("STRING", {"multiline": True, "placeholder": "输入多行文本，将提取选中行"}),
                "selected_line": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1,  # 行号范围扩大到1-100
                                         "label": "选择输出的行号"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("selected_line",)
    FUNCTION = "split_lines"
    CATEGORY = "AFL/Text"

    def split_lines(self, multiline_text: str, selected_line: int) -> Tuple[str]:
        """分割多行文本并返回选中行内容"""
        try:
            # 将文本按行分割
            lines = [line.rstrip('\r\n') for line in multiline_text.splitlines()]
            
            # 获取选中的行（索引从0开始），超出范围返回空字符串
            if 0 <= selected_line - 1 < len(lines):
                selected = lines[selected_line - 1]
            else:
                selected = ""
                
            logger.debug(f"行分割: 选中第{selected_line}行，内容: {selected}")
            return (selected,)
        except Exception as e:
            logger.error(f"行分割错误: {str(e)}", exc_info=True)
            return ("",)

    def _run_self_tests(self):
        """自动运行节点测试"""
        logger.info("开始执行LineSplitterNode自动测试...")
        
        try:
            # 测试1: 正常输入测试
            test_text = "line1\nline2\nline3"
            result = self.split_lines(test_text, 2)
            assert result[0] == "line2", f"行分割测试失败: 预期'line2'，实际'{result[0]}'"
            
            # 测试2: 行数不足测试
            test_text = "only one line"
            result = self.split_lines(test_text, 3)
            assert result[0] == "", f"不足行数处理失败: 预期空字符串，实际'{result[0]}'"
            
            # 测试3: 较宽范围行号测试
            test_lines = [f"line{i}" for i in range(1, 50)]  # 创建49行文本
            test_text = "\n".join(test_lines)
            result = self.split_lines(test_text, 49)
            assert result[0] == "line49", f"宽范围行号测试失败: 预期'line49'，实际'{result[0]}'"
            
            # 测试4: 超出文本行数测试
            result = self.split_lines(test_text, 50)
            assert result[0] == "", f"超出范围处理失败: 预期空字符串，实际'{result[0]}'"
            
            # 测试5: 空输入测试
            result = self.split_lines("", 1)
            assert result[0] == "", f"空输入处理失败: 预期空字符串，实际'{result[0]}'"
            
            logger.info("LoRAPromptManagerV2所有测试通过!")
        except AssertionError as e:
            logger.error(f"LoRAPromptManagerV2测试失败: {str(e)}")
        except Exception as e:
            logger.error(f"LoRAPromptManagerV2测试错误: {str(e)}", exc_info=True)


# 注册节点
NODE_CLASS_MAPPINGS = {
    "AFL:LoRAPromptManagerV2": LoRAPromptManagerV2
}

# 导入lora_overwrite模块的节点注册
import importlib.util
import sys

try:
    # 尝试导入lora_overwrite模块
    lora_overwrite_path = os.path.join(os.path.dirname(__file__), "lora_overwrite.py")
    if os.path.exists(lora_overwrite_path):
        spec = importlib.util.spec_from_file_location("lora_overwrite", lora_overwrite_path)
        if spec and spec.loader:
            lora_overwrite_module = importlib.util.module_from_spec(spec)
            sys.modules["lora_overwrite"] = lora_overwrite_module
            spec.loader.exec_module(lora_overwrite_module)
            
            # 合并节点注册
            if hasattr(lora_overwrite_module, "NODE_CLASS_MAPPINGS"):
                NODE_CLASS_MAPPINGS.update(lora_overwrite_module.NODE_CLASS_MAPPINGS)
            if hasattr(lora_overwrite_module, "NODE_DISPLAY_NAME_MAPPINGS"):
                if 'NODE_DISPLAY_NAME_MAPPINGS' not in globals():
                    global NODE_DISPLAY_NAME_MAPPINGS
                    NODE_DISPLAY_NAME_MAPPINGS = {}
                NODE_DISPLAY_NAME_MAPPINGS.update(lora_overwrite_module.NODE_DISPLAY_NAME_MAPPINGS)
            
            logger.info("成功导入LoRA Overwrite节点")
except Exception as e:
    logger.error(f"导入LoRA Overwrite节点时出错: {str(e)}")

NODE_DISPLAY_NAME_MAPPINGS = {
    "AFL:LoRAPromptManagerV2": "LoRA Prompt Manager V2"
}
