import gc

import torch

try:
    import comfy.model_management as model_management
except Exception:
    model_management = None


GIB = 1024 * 1024 * 1024


def resolve_device(device):
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was selected, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if device == "cpu":
        return torch.device("cpu")
    if model_management is not None:
        return model_management.get_torch_device()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def module_size(module):
    if module is None:
        return 0
    if model_management is not None:
        try:
            return int(model_management.module_size(module))
        except Exception:
            pass
    total = 0
    for tensor in list(module.parameters(recurse=True)) + list(module.buffers(recurse=True)):
        total += tensor.numel() * tensor.element_size()
    return int(total)


def request_vram(device, model_bytes=0, extra_bytes=0):
    device = torch.device(device)
    if device.type == "cpu" or model_management is None:
        return
    required = int(max(0, model_bytes) + max(0, extra_bytes))
    if required <= 0:
        required = GIB
    model_management.free_memory(required, device)


def check_interrupted():
    if model_management is not None:
        model_management.throw_exception_if_processing_interrupted()


def clear_torch_memory(force=True):
    gc.collect()
    if model_management is not None:
        try:
            model_management.soft_empty_cache(force=force)
            return
        except Exception:
            pass
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def release_module(module):
    if module is None:
        return
    try:
        module.to("cpu")
    except Exception:
        pass


def move_module(module, device):
    if module is None:
        return
    try:
        module.to(device)
    except Exception:
        pass
