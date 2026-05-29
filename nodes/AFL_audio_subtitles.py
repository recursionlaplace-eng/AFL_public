import json
import re

import numpy as np
import torch

from AFL_local_ai import ensure_local_ai_paths, resolve_manifest_model
from AFL_audio_memory import GIB, clear_torch_memory, release_module, request_vram


DEVICES = ["auto", "cuda", "cpu"]
MODELS = ["Qwen/Qwen3-ASR-0.6B", "Qwen/Qwen3-ASR-1.7B"]
LANGUAGES = [
    "auto",
    "Chinese",
    "English",
    "Cantonese",
    "Arabic",
    "German",
    "French",
    "Spanish",
    "Portuguese",
    "Indonesian",
    "Italian",
    "Korean",
    "Russian",
    "Thai",
    "Vietnamese",
    "Japanese",
    "Turkish",
    "Hindi",
    "Malay",
    "Dutch",
    "Swedish",
    "Danish",
    "Finnish",
    "Polish",
    "Czech",
    "Filipino",
    "Persian",
    "Greek",
    "Romanian",
    "Hungarian",
    "Macedonian",
]

def _srt_time(seconds):
    seconds = max(0.0, float(seconds or 0))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        secs += 1
        millis -= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _is_punctuation(char):
    return bool(char and re.match(r"[\u3002\uff01\uff1f\uff0c\uff1b\uff1a\uff08\uff09\u3001\uff0e.!?,;:()\[\]{}\"']", char))


def _restore_punctuation_to_words(transcript, words):
    if not transcript or not words:
        return words
    normalized_words = []
    cursor = 0
    text = str(transcript)
    for item in words:
        token = str(item.get("text") or item.get("word") or "").strip()
        if not token:
            continue
        next_item = dict(item)
        found_at = text.find(token, cursor)
        if found_at < 0:
            normalized_words.append(next_item)
            continue
        cursor = max(cursor, found_at + len(token))
        punctuation = ""
        while cursor < len(text):
            char = text[cursor]
            if char.isspace():
                cursor += 1
                continue
            if _is_punctuation(char):
                punctuation += char
                cursor += 1
                continue
            break
        if punctuation and not str(next_item.get("text", "")).endswith(punctuation):
            next_item["text"] = f"{token}{punctuation}"
        normalized_words.append(next_item)
    return normalized_words


def _split_to_subtitles(words, max_chars=30, max_seconds=5.2, gap_seconds=0.65):
    subtitles = []
    current = []
    hard_sentence_marks = set("。！？.!?")
    soft_sentence_marks = set("，,；;：:")
    sentence_marks = hard_sentence_marks | soft_sentence_marks
    cjk_pattern = re.compile(r"[\u3400-\u9fff]")

    def join_parts(parts):
        text = ""
        for part in parts:
            token = str(part["text"])
            if not text:
                text = token
            elif cjk_pattern.search(text[-1:]) or cjk_pattern.search(token[:1]) or token[:1] in sentence_marks or token[:1] in ",.:;)]}":
                text += token
            else:
                text += " " + token
        return text

    for item in words:
        text = str(item.get("text") or item.get("word") or "").strip()
        if not text:
            continue
        current.append({"text": text, "start": float(item.get("start", 0)), "end": float(item.get("end", item.get("start", 0)))})
        joined = join_parts(current)
        duration = current[-1]["end"] - current[0]["start"]
        gap = 0.0
        if len(current) > 1:
            gap = max(0.0, current[-1]["start"] - current[-2]["end"])
        last_char = text[-1:]
        should_cut = (
            last_char in hard_sentence_marks
            or (last_char in soft_sentence_marks and (len(joined) >= 12 or duration >= 2.2))
            or (gap >= float(gap_seconds) and len(joined) >= 8)
            or len(joined) >= int(max_chars)
            or duration >= float(max_seconds)
        )
        if should_cut:
            subtitles.append({"start": current[0]["start"], "end": current[-1]["end"], "text": joined})
            current = []
    if current:
        subtitles.append({"start": current[0]["start"], "end": current[-1]["end"], "text": join_parts(current)})
    return subtitles


def _to_srt(subtitles):
    blocks = []
    for index, item in enumerate(subtitles, 1):
        blocks.append(f"{index}\n{_srt_time(item['start'])} --> {_srt_time(item['end'])}\n{item['text']}")
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def _normalize_result(result):
    if isinstance(result, list) and result:
        result = result[0]
    if hasattr(result, "__dict__"):
        result = result.__dict__
    if not isinstance(result, dict):
        result = {"text": str(result)}
    text = result.get("text") or result.get("transcript") or result.get("sentence") or ""
    words = result.get("words") or result.get("timestamps") or result.get("word_timestamps") or []
    if not words:
        time_stamps = result.get("time_stamps")
        items = getattr(time_stamps, "items", None)
        if items:
            words = [
                {
                    "text": getattr(item, "text", ""),
                    "start": getattr(item, "start_time", 0),
                    "end": getattr(item, "end_time", getattr(item, "start_time", 0)),
                }
                for item in items
            ]
    normalized_words = []
    for item in words:
        if hasattr(item, "__dict__"):
            item = item.__dict__
        if not isinstance(item, dict):
            continue
        normalized_words.append(
            {
                "text": item.get("text") or item.get("word") or item.get("token") or "",
                "start": item.get("start") or item.get("start_time") or 0,
                "end": item.get("end") or item.get("end_time") or item.get("start") or 0,
            }
        )
    return str(text), normalized_words


def _audio_to_numpy(audio):
    waveform = audio["waveform"]
    if hasattr(waveform, "detach"):
        waveform = waveform.detach().cpu()
    waveform = torch.as_tensor(waveform, dtype=torch.float32)
    if waveform.ndim == 3:
        waveform = waveform[0]
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    if waveform.ndim != 1:
        raise ValueError(f"Expected AUDIO waveform shaped [batch, channels, frames], got {tuple(waveform.shape)}")
    return waveform.numpy().astype(np.float32), int(audio["sample_rate"])


def _load_model(model_name, device):
    ensure_local_ai_paths(required=True)
    from qwen_asr import Qwen3ASRModel

    use_cuda = device == "cuda" or (device == "auto" and torch.cuda.is_available())
    dtype = torch.bfloat16 if use_cuda else torch.float32
    device_map = "cuda:0" if use_cuda else "cpu"
    model_key = "qwen3_asr_1_7b" if model_name == "Qwen/Qwen3-ASR-1.7B" else "qwen3_asr_0_6b"
    model_source = resolve_manifest_model(model_key, model_name)
    aligner_source = resolve_manifest_model("qwen3_forced_aligner_0_6b", "Qwen/Qwen3-ForcedAligner-0.6B")
    if use_cuda:
        estimated_model_memory = 6 * GIB if model_name == "Qwen/Qwen3-ASR-1.7B" else 4 * GIB
        request_vram(torch.device("cuda"), extra_bytes=estimated_model_memory)
    return Qwen3ASRModel.from_pretrained(
        model_source,
        dtype=dtype,
        device_map=device_map,
        forced_aligner=aligner_source,
        forced_aligner_kwargs={"dtype": dtype, "device_map": device_map},
        max_inference_batch_size=8 if use_cuda else 1,
        max_new_tokens=256,
    )


def _release_model(asr):
    if asr is None:
        return
    release_module(getattr(asr, "model", None))
    forced_aligner = getattr(asr, "forced_aligner", None)
    if forced_aligner is not None:
        release_module(getattr(forced_aligner, "model", None))
        try:
            forced_aligner.model = None
        except Exception:
            pass
    for attr in ("model", "processor", "sampling_params", "forced_aligner"):
        try:
            setattr(asr, attr, None)
        except Exception:
            pass


class AFL_AudioQwenSubtitles:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
            "optional": {
                "model": (MODELS, {"default": "Qwen/Qwen3-ASR-0.6B"}),
                "device": (DEVICES, {"default": "auto"}),
                "language": (LANGUAGES, {"default": "auto"}),
                "context": ("STRING", {"default": "", "multiline": True}),
                "max_chars": ("INT", {"default": 30, "min": 8, "max": 120, "step": 1}),
                "max_seconds": ("FLOAT", {"default": 5.2, "min": 1.0, "max": 20.0, "step": 0.1}),
                "gap_seconds": ("FLOAT", {"default": 0.65, "min": 0.1, "max": 3.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("srt", "text", "json")
    FUNCTION = "transcribe"
    CATEGORY = "AFL/Audio"

    def transcribe(self, audio, model="Qwen/Qwen3-ASR-0.6B", device="auto", language="auto", context="", max_chars=30, max_seconds=5.2, gap_seconds=0.65):
        asr = None
        try:
            asr = _load_model(model, device)
            waveform, sample_rate = _audio_to_numpy(audio)
            language_arg = None if language == "auto" else language
            result = asr.transcribe([(waveform, sample_rate)], context=context or "", language=language_arg, return_time_stamps=True)
            text, words = _normalize_result(result)
            if not words and text:
                tokens = [token for token in re.split(r"(\s+)", text) if token.strip()]
                words = [{"text": token, "start": index * 1.2, "end": (index + 1) * 1.2} for index, token in enumerate(tokens)]
            words = _restore_punctuation_to_words(text, words)
            subtitles = _split_to_subtitles(words, max_chars=max_chars, max_seconds=max_seconds, gap_seconds=gap_seconds)
            srt = _to_srt(subtitles)
            payload = {
                "text": text,
                "srt": srt,
                "subtitles": subtitles,
                "model": model,
                "language": language,
                "diarization": False,
            }
            return (srt, text, json.dumps(payload, ensure_ascii=False, indent=2))
        finally:
            _release_model(asr)
            del asr
            clear_torch_memory()


class AFL_ShowText:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING", {"forceInput": True, "multiline": True})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "show"
    OUTPUT_NODE = True
    CATEGORY = "AFL/Audio"

    def show(self, text):
        return {"ui": {"text": [text]}, "result": (text,)}


NODE_CLASS_MAPPINGS = {
    "AFL:AudioQwenSubtitles": AFL_AudioQwenSubtitles,
    "AFL:ShowText": AFL_ShowText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AFL:AudioQwenSubtitles": "AFL Audio Qwen Subtitles",
    "AFL:ShowText": "AFL Show Text",
}
