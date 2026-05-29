import os
import sys
import tempfile
import threading

import numpy as np
import soundfile as sf
import torch
import torchaudio

from AFL_audio_memory import GIB, clear_torch_memory, request_vram, resolve_device


PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR_DIR = os.path.join(PLUGIN_DIR, "vendor")
if VENDOR_DIR not in sys.path:
    sys.path.insert(0, VENDOR_DIR)

from afl_seedvc_core import SeedVCWrapper


SEEDVC = None
SEEDVC_UNLOAD_TIMER = None
SEEDVC_IDLE_UNLOAD_SECONDS = 300


def _clear_torch_memory():
    clear_torch_memory()


def _cancel_seedvc_idle_timer():
    global SEEDVC_UNLOAD_TIMER
    if SEEDVC_UNLOAD_TIMER is not None:
        try:
            SEEDVC_UNLOAD_TIMER.cancel()
        except Exception:
            pass
        SEEDVC_UNLOAD_TIMER = None


def unload_seedvc():
    global SEEDVC
    _cancel_seedvc_idle_timer()
    if SEEDVC is not None:
        try:
            SEEDVC.unload()
        finally:
            SEEDVC = None
    _clear_torch_memory()


def offload_seedvc_to_cpu(schedule_full_unload=True):
    global SEEDVC, SEEDVC_UNLOAD_TIMER
    if SEEDVC is not None:
        SEEDVC.offload_to_cpu()
    _clear_torch_memory()
    if schedule_full_unload:
        _cancel_seedvc_idle_timer()
        SEEDVC_UNLOAD_TIMER = threading.Timer(SEEDVC_IDLE_UNLOAD_SECONDS, unload_seedvc)
        SEEDVC_UNLOAD_TIMER.daemon = True
        SEEDVC_UNLOAD_TIMER.start()


def _audio_to_temp_wav(audio):
    waveform = audio["waveform"].detach().cpu().float()
    sample_rate = int(audio["sample_rate"])
    if waveform.ndim == 3:
        waveform = waveform[0]
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    audio_np = waveform.numpy().T
    handle = tempfile.NamedTemporaryFile(prefix="afl_seedvc_", suffix=".wav", delete=False)
    handle.close()
    sf.write(handle.name, audio_np, sample_rate)
    return handle.name


def _as_audio(audio_np, sample_rate):
    waveform = torch.from_numpy(np.asarray(audio_np, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    return {"waveform": waveform, "sample_rate": int(sample_rate)}


def _mono(waveform):
    if waveform.ndim == 3:
        waveform = waveform[0]
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    return waveform.float()


def _rms_db(waveform):
    rms = torch.sqrt(torch.mean(waveform.square()).clamp_min(1e-12))
    return 20.0 * torch.log10(rms).item()


def _lufs_or_rms(waveform, sample_rate, mode):
    mono = _mono(waveform).unsqueeze(0)
    if mode == "lufs":
        try:
            return float(torchaudio.functional.loudness(mono, int(sample_rate)).item())
        except Exception:
            return _rms_db(mono)
    return _rms_db(mono)


def _peak_dbfs(waveform):
    peak = waveform.abs().max().clamp_min(1e-12)
    return 20.0 * torch.log10(peak).item()


class AFLSeedVC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_audio": ("AUDIO",),
                "ref_audio": ("AUDIO",),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                    "tooltip": "Controls SeedVC diffusion noise. Same seed is repeatable; different seeds change details.",
                }),
                "steps": ("INT", {"default": 30, "min": 1, "max": 200, "step": 1}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "inference_cfg_rate": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
                "f0_condition": ("BOOLEAN", {"default": False, "tooltip": "Enable for singing voice conversion."}),
                "auto_f0_adjust": ("BOOLEAN", {"default": True}),
                "pitch_shift": ("INT", {"default": 0, "min": -24, "max": 24, "step": 1}),
                "max_ref_seconds": ("FLOAT", {"default": 10.0, "min": 3.0, "max": 25.0, "step": 0.5}),
                "source_chunk_seconds": ("FLOAT", {"default": 10.0, "min": 5.0, "max": 20.0, "step": 0.5}),
                "unload_model": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Kept for workflow compatibility. AFL Seed VC releases VRAM after each run and keeps a short CPU cache for repeated SeedVC calls.",
                }),
            },
        }

    CATEGORY = "AFL/Audio"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "run"

    def run(
        self, source_audio, ref_audio, seed, steps, speed, inference_cfg_rate, f0_condition,
        auto_f0_adjust, pitch_shift, max_ref_seconds, source_chunk_seconds, unload_model
    ):
        global SEEDVC
        source_path = None
        ref_path = None
        try:
            _cancel_seedvc_idle_timer()
            if SEEDVC is None:
                request_vram(resolve_device("auto"), extra_bytes=8 * GIB)
                SEEDVC = SeedVCWrapper()
            else:
                request_vram(SEEDVC.device, extra_bytes=8 * GIB)
                SEEDVC.move_to_device(SEEDVC.device)

            source_path = _audio_to_temp_wav(source_audio)
            ref_path = _audio_to_temp_wav(ref_audio)
            audio_np, sample_rate = SEEDVC.convert_voice(
                source=source_path,
                target=ref_path,
                diffusion_steps=steps,
                length_adjust=speed,
                inference_cfg_rate=inference_cfg_rate,
                f0_condition=f0_condition,
                auto_f0_adjust=auto_f0_adjust,
                pitch_shift=pitch_shift,
                max_ref_seconds=max_ref_seconds,
                source_chunk_seconds=source_chunk_seconds,
                seed=seed,
            )
        finally:
            for path in (source_path, ref_path):
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            offload_seedvc_to_cpu(schedule_full_unload=True)

        return (_as_audio(audio_np, sample_rate),)


class AFLUnloadSeedVC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "force_gc": ("BOOLEAN", {"default": True}),
            },
        }

    CATEGORY = "AFL/Audio"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "unload"

    def unload(self, force_gc=True):
        unload_seedvc()
        if force_gc:
            _clear_torch_memory()
        return ("AFL Seed VC unloaded",)


class AFLMatchAudioLoudness:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "reference_audio": ("AUDIO",),
                "mode": (["lufs", "rms"], {"default": "lufs"}),
                "mix": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "max_gain_db": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 36.0, "step": 0.5}),
                "peak_limit_dbfs": ("FLOAT", {"default": -1.0, "min": -12.0, "max": 0.0, "step": 0.1}),
            },
        }

    CATEGORY = "AFL/Audio"
    RETURN_TYPES = ("AUDIO", "FLOAT")
    RETURN_NAMES = ("audio", "applied_gain_db")
    FUNCTION = "match"

    def match(self, audio, reference_audio, mode, mix, max_gain_db, peak_limit_dbfs):
        waveform = audio["waveform"].detach().cpu().float()
        sample_rate = int(audio["sample_rate"])
        ref_waveform = reference_audio["waveform"].detach().cpu().float()
        ref_sample_rate = int(reference_audio["sample_rate"])

        target_level = _lufs_or_rms(waveform, sample_rate, mode)
        reference_level = _lufs_or_rms(ref_waveform, ref_sample_rate, mode)
        desired_gain_db = (reference_level - target_level) * float(mix)
        desired_gain_db = max(-float(max_gain_db), min(float(max_gain_db), desired_gain_db))

        current_peak_db = _peak_dbfs(waveform)
        allowed_gain_db = float(peak_limit_dbfs) - current_peak_db
        applied_gain_db = min(desired_gain_db, allowed_gain_db)
        gain = 10.0 ** (applied_gain_db / 20.0)
        matched = (waveform * gain).clamp(-1.0, 1.0)

        return ({"waveform": matched, "sample_rate": sample_rate}, float(applied_gain_db))


NODE_CLASS_MAPPINGS = {
    "AFL:SeedVC": AFLSeedVC,
    "AFL:UnloadSeedVC": AFLUnloadSeedVC,
    "AFL:MatchAudioLoudness": AFLMatchAudioLoudness,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AFL:SeedVC": "AFL Seed VC",
    "AFL:UnloadSeedVC": "AFL Unload Seed VC",
    "AFL:MatchAudioLoudness": "AFL Match Audio Loudness",
}
