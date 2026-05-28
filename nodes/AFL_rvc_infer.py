import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import folder_paths


PLUGIN_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RVC_REPO_DIR = PLUGIN_DIR / "vendor" / "rvc_repo"
RVC_VENDOR_DIR = PLUGIN_DIR / "vendor" / "rvc_wrapper"
RVC_MODELS_DIR = Path(folder_paths.models_dir) / "tts" / "RVC"

if str(RVC_VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(RVC_VENDOR_DIR))

from afl_rvc_wrapper import RVCInferenceError, infer_rvc_audio


def _ensure_rvc_model_path_registered():
    RVC_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    folder_paths.add_model_folder_path("afl_rvc", str(RVC_MODELS_DIR), is_default=True)


def _list_rvc_models():
    _ensure_rvc_model_path_registered()
    models = []
    for rel_path in folder_paths.get_filename_list("afl_rvc"):
        if rel_path.lower().endswith(".pth"):
            models.append(rel_path.replace("\\", "/"))
    return sorted(models)


def _resolve_model_bundle(model_name):
    if not model_name or model_name == "__manual__":
        return "", ""

    model_path = folder_paths.get_full_path("afl_rvc", model_name)
    if not model_path:
        return "", ""

    model_file = Path(model_path)
    candidates = sorted(model_file.parent.glob("*.index"))
    index_path = str(candidates[0]) if candidates else ""
    return str(model_file), index_path


def _audio_to_temp_wav(audio, prefix):
    waveform = audio["waveform"].detach().cpu().float()
    sample_rate = int(audio["sample_rate"])
    if waveform.ndim == 3:
        waveform = waveform[0]
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    audio_np = waveform.numpy().T
    handle = tempfile.NamedTemporaryFile(prefix=prefix, suffix=".wav", delete=False)
    handle.close()
    sf.write(handle.name, audio_np, sample_rate)
    return handle.name


def _as_audio(audio_np, sample_rate):
    audio_np = np.asarray(audio_np, dtype=np.float32)
    if audio_np.ndim == 1:
        waveform = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0)
    elif audio_np.ndim == 2:
        waveform = torch.from_numpy(audio_np.T).unsqueeze(0)
    else:
        raise ValueError(f"Unsupported output audio shape: {audio_np.shape}")
    return {"waveform": waveform.contiguous(), "sample_rate": int(sample_rate)}


class AFLRVCConvert:
    @classmethod
    def INPUT_TYPES(cls):
        model_options = ["__manual__"] + _list_rvc_models()
        return {
            "required": {
                "audio": ("AUDIO",),
                "model_name": (model_options, {
                    "default": model_options[0],
                    "tooltip": "Select a bundled RVC .pth under ComfyUI/models/tts/RVC. Use __manual__ to type paths below.",
                }),
                "pitch_shift": ("INT", {
                    "default": 0,
                    "min": -24,
                    "max": 24,
                    "step": 1,
                    "tooltip": "Transpose in semitones.",
                }),
                "f0_method": (["rmvpe", "fcpe", "harvest", "pm", "crepe"], {
                    "default": "rmvpe",
                    "tooltip": "Pitch extraction method used by RVC.",
                }),
                "index_rate": ("FLOAT", {
                    "default": 0.66,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "How strongly to use the retrieval index. 0 disables it.",
                }),
                "protect": ("FLOAT", {
                    "default": 0.33,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.01,
                    "tooltip": "Protect consonants / breathiness. Higher can reduce artifacts.",
                }),
            },
            "optional": {
                "rvc_repo_path": ("STRING", {
                    "default": str(DEFAULT_RVC_REPO_DIR),
                    "multiline": False,
                    "tooltip": "Path to the bundled RVC repository clone.",
                }),
                "model_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Manual absolute path to the RVC .pth model. Only used when model_name is __manual__.",
                }),
                "index_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Manual .index path. Leave empty to auto-pair from the model folder when possible.",
                }),
                "filter_radius": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 7,
                    "step": 1,
                    "tooltip": "Median filter radius for harvest-style F0 smoothing inside RVC.",
                }),
                "resample_sr": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 96000,
                    "step": 1000,
                    "tooltip": "0 keeps the model sample rate. Otherwise resamples output.",
                }),
                "rms_mix_rate": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "RVC loudness envelope mix rate.",
                }),
                "device": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Optional torch device override, e.g. cuda:0 or cpu.",
                }),
                "is_half": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use fp16 when supported by the installed RVC environment.",
                }),
            },
        }

    CATEGORY = "AFL/Audio"
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "info")
    FUNCTION = "convert"

    def convert(
        self,
        audio,
        model_name,
        pitch_shift,
        f0_method,
        index_rate,
        protect,
        rvc_repo_path=str(DEFAULT_RVC_REPO_DIR),
        model_path="",
        index_path="",
        filter_radius=3,
        resample_sr=0,
        rms_mix_rate=1.0,
        device="",
        is_half=True,
    ):
        resolved_model_path = model_path.strip()
        resolved_index_path = index_path.strip()

        auto_model_path, auto_index_path = _resolve_model_bundle(model_name)
        if auto_model_path:
            resolved_model_path = auto_model_path
            if not resolved_index_path:
                resolved_index_path = auto_index_path

        if not resolved_model_path:
            raise RuntimeError(
                "No RVC model selected. Put .pth files under ComfyUI/models/tts/RVC or use manual model_path."
            )

        source_path = None
        output_path = None
        try:
            source_path = _audio_to_temp_wav(audio, prefix="afl_rvc_input_")
            output_handle = tempfile.NamedTemporaryFile(prefix="afl_rvc_output_", suffix=".wav", delete=False)
            output_handle.close()
            output_path = output_handle.name

            info = infer_rvc_audio(
                repo_path=rvc_repo_path,
                input_wav_path=source_path,
                output_wav_path=output_path,
                model_path=resolved_model_path,
                index_path=resolved_index_path,
                pitch_shift=pitch_shift,
                f0_method=f0_method,
                index_rate=index_rate,
                filter_radius=filter_radius,
                resample_sr=resample_sr,
                rms_mix_rate=rms_mix_rate,
                protect=protect,
                device=device,
                is_half=is_half,
            )

            audio_np, sample_rate = sf.read(output_path, always_2d=False, dtype="float32")
            return (_as_audio(audio_np, sample_rate), info)
        except RVCInferenceError as exc:
            raise RuntimeError(str(exc)) from exc
        finally:
            for path in (source_path, output_path):
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass


NODE_CLASS_MAPPINGS = {
    "AFL:RVCConvert": AFLRVCConvert,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AFL:RVCConvert": "AFL RVC Convert",
}
