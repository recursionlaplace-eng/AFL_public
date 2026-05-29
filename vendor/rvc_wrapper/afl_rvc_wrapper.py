import importlib.util
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path


class RVCInferenceError(RuntimeError):
    pass


def _load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RVCInferenceError(f"Unable to load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def _temp_environ(values):
    old_values = {}
    try:
        for key, value in values.items():
            old_values[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


@contextmanager
def _temp_sys_path(path):
    path = str(path)
    if path in sys.path:
        yield
        return
    sys.path.insert(0, path)
    try:
        yield
    finally:
        try:
            sys.path.remove(path)
        except ValueError:
            pass


@contextmanager
def _temp_cwd(path):
    old_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


@contextmanager
def _temp_argv(argv=None):
    old_argv = sys.argv[:]
    sys.argv = list(argv or [old_argv[0] if old_argv else "afl_rvc"])
    try:
        yield
    finally:
        sys.argv = old_argv


@contextmanager
def _patched_torch_load():
    try:
        import torch
    except Exception:
        yield
        return

    original_torch_load = torch.load

    def _torch_load_with_legacy_default(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = _torch_load_with_legacy_default
    try:
        yield
    finally:
        torch.load = original_torch_load


def _resolve_ffmpeg_dir():
    try:
        import imageio_ffmpeg

        source_exe = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
        target_dir = Path(__file__).resolve().parent / "_ffmpeg_bin"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_exe = target_dir / "ffmpeg.exe"
        if (not target_exe.exists()) or target_exe.stat().st_size != source_exe.stat().st_size:
            shutil.copy2(source_exe, target_exe)
        return target_dir
    except Exception:
        explicit_dir = Path(r"C:\ffmpeg6.0\bin")
        if explicit_dir.exists():
            return explicit_dir
        return None


def infer_rvc_audio(
    repo_path,
    input_wav_path,
    output_wav_path,
    model_path,
    index_path="",
    pitch_shift=0,
    f0_method="rmvpe",
    index_rate=0.66,
    filter_radius=3,
    resample_sr=0,
    rms_mix_rate=1.0,
    protect=0.33,
    device="",
    is_half=True,
):
    repo = Path(repo_path).expanduser().resolve()
    if not repo.exists():
        raise RVCInferenceError(f"RVC repo path does not exist: {repo}")
    if not (repo / "configs" / "config.py").exists():
        raise RVCInferenceError(
            f"RVC repo path does not look valid: missing configs/config.py under {repo}"
        )

    model = Path(model_path).expanduser().resolve()
    if not model.exists():
        raise RVCInferenceError(f"RVC model not found: {model}")
    if model.suffix.lower() != ".pth":
        raise RVCInferenceError(f"RVC model must be a .pth file: {model}")

    index = ""
    if index_path:
        index_file = Path(index_path).expanduser().resolve()
        if not index_file.exists():
            raise RVCInferenceError(f"RVC index not found: {index_file}")
        index = str(index_file)

    env_updates = {
        "weight_root": str(model.parent),
        "weight_uvr5_root": str(repo / "assets" / "uvr5_weights"),
        "index_root": str(Path(index).parent if index else repo / "logs"),
        "outside_index_root": str(Path(index).parent if index else repo / "logs"),
        "rmvpe_root": str(repo / "assets" / "rmvpe"),
    }

    ffmpeg_dir = _resolve_ffmpeg_dir()
    if ffmpeg_dir is not None:
        current_path = os.environ.get("PATH", "")
        env_updates["PATH"] = f"{ffmpeg_dir};{current_path}" if current_path else str(ffmpeg_dir)

    config_module_path = repo / "configs" / "config.py"
    vc_module_path = repo / "infer" / "modules" / "vc" / "modules.py"

    with (
        _temp_cwd(repo),
        _temp_sys_path(repo),
        _temp_environ(env_updates),
        _temp_argv(["afl_rvc"]),
        _patched_torch_load(),
    ):
        try:
            config_module = _load_module("afl_rvc_config", config_module_path)
            vc_module = _load_module("afl_rvc_modules", vc_module_path)
        except Exception as exc:
            raise RVCInferenceError(
                "Failed to import RVC modules. Check that the RVC repo dependencies are installed."
            ) from exc

        try:
            config = config_module.Config()
            if device:
                config.device = device
            if is_half is not None:
                config.is_half = bool(is_half)

            vc = vc_module.VC(config)
            vc.get_vc(model.name)
            info, wav_opt = vc.vc_single(
                0,
                str(Path(input_wav_path).resolve()),
                int(pitch_shift),
                None,
                f0_method,
                index,
                None,
                float(index_rate),
                int(filter_radius),
                int(resample_sr),
                float(rms_mix_rate),
                float(protect),
            )
        except BaseException as exc:
            raise RVCInferenceError(f"RVC inference failed: {exc}") from exc

    output_sr, output_audio = wav_opt
    if output_sr is None or output_audio is None:
        raise RVCInferenceError(f"RVC did not return audio.\n{info}")

    try:
        import soundfile as sf
        sf.write(str(Path(output_wav_path).resolve()), output_audio, output_sr)
    except Exception as exc:
        raise RVCInferenceError(f"Failed to save RVC output WAV: {exc}") from exc

    return str(info)
