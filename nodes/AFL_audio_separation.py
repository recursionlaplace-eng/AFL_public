import torch
from torchaudio.transforms import Fade, Resample
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS

from AFL_local_ai import ensure_local_ai_paths
from AFL_audio_memory import (
    GIB,
    check_interrupted,
    clear_torch_memory,
    module_size,
    release_module,
    request_vram,
    resolve_device,
)


ensure_local_ai_paths(required=False)


FADE_SHAPES = ["linear", "half_sine", "logarithmic", "exponential"]
DEVICES = ["auto", "cuda", "cpu"]


def _resolve_device(device):
    return resolve_device(device)


def _ensure_batched_stereo(waveform):
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 3:
        raise ValueError(f"Expected AUDIO waveform shaped [batch, channels, frames], got {tuple(waveform.shape)}")

    channels = waveform.shape[1]
    if channels == 2:
        return waveform
    if channels == 1:
        return waveform.repeat(1, 2, 1)

    mono = waveform[:, :2, :].mean(dim=1, keepdim=True)
    return mono.repeat(1, 2, 1)


def _as_audio(waveform, sample_rate):
    return {
        "waveform": waveform.detach().cpu(),
        "sample_rate": int(sample_rate),
    }


def _normalize_waveform(waveform):
    ref = waveform.mean(dim=1, keepdim=True)
    mean = ref.mean(dim=-1, keepdim=True)
    std = ref.std(dim=-1, keepdim=True).clamp_min(1e-8)
    return (waveform - mean) / std, mean, std


def _separate_sources(model, mix, sample_rate, chunk_length, chunk_overlap, device, fade_shape):
    batch, channels, length = mix.shape
    chunk_length = max(float(chunk_length), 1.0)
    chunk_overlap = max(float(chunk_overlap), 0.0)

    stride = max(int(sample_rate * chunk_length), 1)
    overlap_frames = min(int(sample_rate * chunk_overlap), max(stride - 1, 0))
    chunk_frames = stride + overlap_frames

    final = torch.zeros(batch, len(model.sources), channels, length, device=device)

    start = 0
    is_first = True
    while start < length:
        check_interrupted()
        end = min(start + chunk_frames, length)
        chunk = mix[:, :, start:end]

        with torch.no_grad():
            out = model(chunk)
        check_interrupted()

        current_len = out.shape[-1]
        fade_in = 0 if is_first else min(overlap_frames, current_len)
        fade_out = 0 if end >= length else min(overlap_frames, current_len)
        if fade_in or fade_out:
            fade = Fade(fade_in_len=fade_in, fade_out_len=fade_out, fade_shape=fade_shape)
            out = fade(out)

        final[:, :, :, start : start + current_len] += out[:, :, :, : min(current_len, length - start)]

        if end >= length:
            break
        start += stride
        is_first = False

    return final


def _run_hdemucs(audio, chunk_length, chunk_overlap, fade_shape, device):
    target_device = _resolve_device(device)
    input_sample_rate = int(audio["sample_rate"])
    bundle = HDEMUCS_HIGH_MUSDB_PLUS
    model = None
    try:
        request_vram(target_device, extra_bytes=2 * GIB)
        model = bundle.get_model()
        model.eval()
        model_sources = list(model.sources)
        model_sample_rate = int(bundle.sample_rate)
        request_vram(target_device, model_bytes=module_size(model), extra_bytes=2 * GIB)
        model.to(target_device)

        waveform = audio["waveform"].to(target_device)
        waveform = _ensure_batched_stereo(waveform)
        if input_sample_rate != model_sample_rate:
            resample = Resample(input_sample_rate, model_sample_rate).to(target_device)
            waveform = resample(waveform)

        waveform, mean, std = _normalize_waveform(waveform)
        sources = _separate_sources(
            model=model,
            mix=waveform,
            sample_rate=model_sample_rate,
            chunk_length=chunk_length,
            chunk_overlap=chunk_overlap,
            device=target_device,
            fade_shape=fade_shape,
        )
        sources = (sources * std[:, None, :, :] + mean[:, None, :, :]).detach().cpu()
        return dict(zip(model_sources, sources.unbind(dim=1))), model_sample_rate
    finally:
        release_module(model)
        del model
        clear_torch_memory()


def _run_demucs_model(audio, model_name, chunk_length, chunk_overlap, device):
    ensure_local_ai_paths(required=False)
    import demucs.pretrained
    from demucs.apply import apply_model

    target_device = _resolve_device(device)
    model = None
    try:
        request_vram(target_device, extra_bytes=2 * GIB)
        model = demucs.pretrained.get_model(model_name)
        model.eval()
        model_sources = list(model.sources)
        model_sample_rate = int(model.samplerate)
        request_vram(target_device, model_bytes=module_size(model), extra_bytes=2 * GIB)
        model.to(target_device)

        waveform = audio["waveform"].to(target_device)
        waveform = _ensure_batched_stereo(waveform)
        if int(audio["sample_rate"]) != model_sample_rate:
            waveform = Resample(int(audio["sample_rate"]), model_sample_rate).to(target_device)(waveform)

        waveform, mean, std = _normalize_waveform(waveform)
        segment = _resolve_demucs_segment(model, chunk_length)
        overlap_ratio = max(0.0, min(0.95, float(chunk_overlap) / max(segment, 0.1)))
        sources = apply_model(
            model,
            waveform,
            device=target_device,
            split=True,
            overlap=overlap_ratio,
            progress=False,
            shifts=1,
            segment=segment,
        )
        check_interrupted()
        sources = (sources * std[:, None, :, :] + mean[:, None, :, :]).detach().cpu()
        return dict(zip(model_sources, sources.unbind(dim=1))), model_sample_rate
    finally:
        release_module(model)
        del model
        clear_torch_memory()


def _resolve_demucs_segment(model, chunk_length):
    segment = max(float(chunk_length), 0.1)
    max_segment = None

    model_segment = getattr(model, "segment", None)
    if getattr(model, "use_train_segment", False) and model_segment is not None:
        max_segment = float(model_segment)

    bag_max_segment = getattr(model, "max_allowed_segment", None)
    if bag_max_segment is not None:
        bag_max_segment = float(bag_max_segment)
        if bag_max_segment > 0:
            max_segment = bag_max_segment if max_segment is None else min(max_segment, bag_max_segment)

    if max_segment is not None and max_segment > 0:
        segment = min(segment, max_segment)
    return max(segment, 0.1)


class AFL_AudioSeparateDemucs4Stem:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
            "optional": {
                "chunk_length": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 120.0, "step": 0.5}),
                "chunk_overlap": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 10.0, "step": 0.05}),
                "chunk_fade_shape": (FADE_SHAPES, {"default": "linear"}),
                "device": (DEVICES, {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO")
    RETURN_NAMES = ("bass", "drums", "other", "vocals", "instrumental")
    FUNCTION = "separate"
    CATEGORY = "AFL/Audio"

    def separate(self, audio, chunk_length=10.0, chunk_overlap=0.1, chunk_fade_shape="linear", device="auto"):
        sources, sample_rate = _run_hdemucs(audio, chunk_length, chunk_overlap, chunk_fade_shape, device)
        bass = sources["bass"]
        drums = sources["drums"]
        other = sources["other"]
        vocals = sources["vocals"]
        instrumental = bass + drums + other
        return (
            _as_audio(bass, sample_rate),
            _as_audio(drums, sample_rate),
            _as_audio(other, sample_rate),
            _as_audio(vocals, sample_rate),
            _as_audio(instrumental, sample_rate),
        )


class AFL_AudioSeparateVoiceBackground:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
            "optional": {
                "chunk_length": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 120.0, "step": 0.5}),
                "chunk_overlap": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 10.0, "step": 0.05}),
                "chunk_fade_shape": (FADE_SHAPES, {"default": "linear"}),
                "device": (DEVICES, {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("AUDIO", "AUDIO")
    RETURN_NAMES = ("voice", "background")
    FUNCTION = "separate"
    CATEGORY = "AFL/Audio"

    def separate(self, audio, chunk_length=10.0, chunk_overlap=0.1, chunk_fade_shape="linear", device="auto"):
        sources, sample_rate = _run_hdemucs(audio, chunk_length, chunk_overlap, chunk_fade_shape, device)
        voice = sources["vocals"]
        background = sources["bass"] + sources["drums"] + sources["other"]
        return (_as_audio(voice, sample_rate), _as_audio(background, sample_rate))


class AFL_AudioSeparateDemucs6StemBeta:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
            "optional": {
                "chunk_length": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 120.0, "step": 0.5}),
                "chunk_overlap": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 10.0, "step": 0.05}),
                "device": (DEVICES, {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO")
    RETURN_NAMES = ("drums", "bass", "other", "vocals", "guitar", "piano", "instrumental")
    FUNCTION = "separate"
    CATEGORY = "AFL/Audio"

    def separate(self, audio, chunk_length=10.0, chunk_overlap=0.25, device="auto"):
        sources, sample_rate = _run_demucs_model(audio, "htdemucs_6s", chunk_length, chunk_overlap, device)
        drums = sources["drums"]
        bass = sources["bass"]
        other = sources["other"]
        vocals = sources["vocals"]
        guitar = sources["guitar"]
        piano = sources["piano"]
        instrumental = drums + bass + other + guitar + piano
        return (
            _as_audio(drums, sample_rate),
            _as_audio(bass, sample_rate),
            _as_audio(other, sample_rate),
            _as_audio(vocals, sample_rate),
            _as_audio(guitar, sample_rate),
            _as_audio(piano, sample_rate),
            _as_audio(instrumental, sample_rate),
        )


class AFL_AudioMixStems:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stem_a": ("AUDIO",),
                "stem_b": ("AUDIO",),
            },
            "optional": {
                "stem_c": ("AUDIO",),
                "stem_d": ("AUDIO",),
                "gain_a": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
                "gain_b": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
                "gain_c": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
                "gain_d": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "mix"
    CATEGORY = "AFL/Audio"

    def mix(self, stem_a, stem_b, stem_c=None, stem_d=None, gain_a=1.0, gain_b=1.0, gain_c=1.0, gain_d=1.0):
        stems = [(stem_a, gain_a), (stem_b, gain_b), (stem_c, gain_c), (stem_d, gain_d)]
        base_rate = int(stem_a["sample_rate"])
        waveforms = []
        for stem, gain in stems:
            if stem is None:
                continue
            waveform = _ensure_batched_stereo(stem["waveform"].float())
            sample_rate = int(stem["sample_rate"])
            if sample_rate != base_rate:
                waveform = Resample(sample_rate, base_rate)(waveform)
            waveforms.append(waveform * float(gain))

        max_len = max(item.shape[-1] for item in waveforms)
        padded = []
        for waveform in waveforms:
            if waveform.shape[-1] < max_len:
                waveform = torch.nn.functional.pad(waveform, (0, max_len - waveform.shape[-1]))
            padded.append(waveform)

        mixed = torch.stack(padded, dim=0).sum(dim=0).clamp(-1.0, 1.0)
        return (_as_audio(mixed, base_rate),)


NODE_CLASS_MAPPINGS = {
    "AFL:AudioSeparateDemucs4Stem": AFL_AudioSeparateDemucs4Stem,
    "AFL:AudioSeparateVoiceBackground": AFL_AudioSeparateVoiceBackground,
    "AFL:AudioSeparateDemucs6StemBeta": AFL_AudioSeparateDemucs6StemBeta,
    "AFL:AudioMixStems": AFL_AudioMixStems,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AFL:AudioSeparateDemucs4Stem": "AFL Audio Separate Demucs 4 Stem",
    "AFL:AudioSeparateVoiceBackground": "AFL Audio Separate Voice Background",
    "AFL:AudioSeparateDemucs6StemBeta": "AFL Audio Separate Demucs 6 Stem Beta",
    "AFL:AudioMixStems": "AFL Audio Mix Stems",
}
