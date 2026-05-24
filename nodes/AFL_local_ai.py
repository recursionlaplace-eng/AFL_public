import json
import os
import sys
from pathlib import Path


def plugin_root():
    return Path(__file__).resolve().parents[1]


def comfy_root():
    root = plugin_root()
    if root.parent.name == "custom_nodes":
        return root.parent.parent
    return root.parent


def local_ai_candidates():
    env_root = os.environ.get("AFL_LOCAL_AI_RESOURCE_ROOT", "").strip()
    root = plugin_root()
    comfy = comfy_root()
    roots = [
        root / "afl_local_ai",
        comfy / "afl_local_ai",
    ]
    if env_root:
        roots.append(Path(env_root))
    roots.append(comfy.parent / "afl_local_ai")

    unique_roots = []
    seen = set()
    for candidate in roots:
        try:
            key = str(candidate.resolve()).lower()
        except Exception:
            key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        unique_roots.append(candidate)
    return unique_roots


def find_local_ai_root(required=False):
    for candidate in local_ai_candidates():
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        if (resolved / "manifest.json").exists() or (resolved / "site-packages").exists():
            return resolved

    if required:
        checked = "\n".join(str(path) for path in local_ai_candidates())
        raise FileNotFoundError(
            "AFL local AI resources were not found. Put afl_local_ai in one of these locations:\n"
            f"{checked}\n"
            "Recommended portable layouts: ComfyUI/custom_nodes/AFL_public/afl_local_ai or ComfyUI/afl_local_ai. "
            "Do not depend on the parent folder name or drive letter."
        )
    return None


def ensure_local_ai_paths(required=False):
    root = find_local_ai_root(required=required)
    if root is None:
        return None

    site_packages = root / "site-packages"
    if site_packages.exists():
        site_path = str(site_packages)
        if site_path not in sys.path:
            sys.path.insert(0, site_path)

    models = root / "models"
    os.environ.setdefault("AFL_LOCAL_AI_RESOURCE_ROOT", str(root))
    os.environ.setdefault("AFL_LOCAL_AI_MODEL_DIR", str(models))
    os.environ.setdefault("AFL_LOCAL_AI_BUNDLE_MANIFEST", str(root / "manifest.json"))
    os.environ.setdefault("TORCH_HOME", str(models / "torch"))
    os.environ.setdefault("DEMUCS_CACHE", str(models / "demucs"))
    return root


def load_manifest():
    root = ensure_local_ai_paths(required=False)
    if root is None:
        return {}
    manifest_path = root / "manifest.json"
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def resolve_manifest_model(model_key, fallback):
    root = ensure_local_ai_paths(required=False)
    manifest = load_manifest()
    models = manifest.get("models") if isinstance(manifest, dict) else {}
    entry = models.get(model_key) if isinstance(models, dict) else None
    if not isinstance(entry, dict):
        return fallback

    path_value = str(entry.get("path") or "").strip()
    if not path_value:
        return fallback

    candidate = Path(path_value)
    if not candidate.is_absolute() and root is not None:
        candidate = root / candidate
    return str(candidate) if candidate.exists() else fallback
