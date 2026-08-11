"""
BSAI_ComfyUI_IndexTTS-2.5 - Model Manager
Handles automatic model download with priority: local -> ModelScope -> HuggingFace
"""

import os
import sys
import shutil
import subprocess
import folder_paths


# Model repo identifiers for IndexTTS-2.5
MODELSCOPE_MODEL_ID = "IndexTeam/IndexTTS-2.5"
HUGGINGFACE_MODEL_ID = "IndexTeam/IndexTTS-2.5"
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"

# Use ComfyUI's standard model directory
MODEL_FOLDER_NAME = "IndexTTS2.5"

def get_model_dir():
    """Get or create the IndexTTS-2.5 model directory under ComfyUI's models folder."""
    try:
        model_dir = folder_paths.get_folder_paths(MODEL_FOLDER_NAME)
        if model_dir:
            return model_dir[0]
    except Exception:
        pass
    # Fallback: create under ComfyUI/models/
    base = os.path.join(folder_paths.base_path, "models", MODEL_FOLDER_NAME)
    os.makedirs(base, exist_ok=True)
    return base


def get_config_path(model_dir=None):
    """Get the config.yaml path."""
    if model_dir is None:
        model_dir = get_model_dir()
    return os.path.join(model_dir, "config.yaml")


def check_model_exists(model_dir=None):
    """Check if the IndexTTS-2.5 model files exist locally."""
    if model_dir is None:
        model_dir = get_model_dir()

    # Key files that must exist
    required_files = [
        "config.yaml",
        "gpt.pth",
        "s2mel.pth",
        "bpe.model",
    ]

    for f in required_files:
        if not os.path.exists(os.path.join(model_dir, f)):
            return False

    # Check for bigvgan directory or bigvgan checkpoint
    bigvgan_dir = os.path.join(model_dir, "bigvgan")
    if not os.path.exists(bigvgan_dir):
        # Maybe bigvgan_checkpoint is a file in model_dir
        has_bigvgan = any(
            f.startswith("bigvgan") and f.endswith((".pth", ".pt", ".safetensors"))
            for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f))
        ) if os.path.exists(model_dir) else False
        if not has_bigvgan:
            return False

    return True


def _run_subprocess(cmd, env=None):
    """Run a subprocess and return success status."""
    try:
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)

        result = subprocess.run(
            cmd,
            env=merged_env,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout for large model downloads
        )
        if result.returncode != 0:
            print(f"[BSAI_IndexTTS2.5] Command failed: {' '.join(cmd)}")
            if result.stderr:
                print(f"[BSAI_IndexTTS2.5] stderr: {result.stderr[:500]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print("[BSAI_IndexTTS2.5] Model download timed out (1 hour limit).")
        return False
    except Exception as e:
        print(f"[BSAI_IndexTTS2.5] Exception during download: {e}")
        return False


def download_from_modelscope(target_dir):
    """Download model from ModelScope (preferred for China users)."""
    print(f"[BSAI_IndexTTS2.5] Downloading from ModelScope: {MODELSCOPE_MODEL_ID}")

    # Try using modelscope CLI
    try:
        import modelscope
        print("[BSAI_IndexTTS2.5] modelscope package found, using Python API.")
        from modelscope import snapshot_download
        snapshot_download(
            model_id=MODELSCOPE_MODEL_ID,
            local_dir=target_dir,
        )
        print(f"[BSAI_IndexTTS2.5] ModelScope download complete -> {target_dir}")
        return True
    except ImportError:
        print("[BSAI_IndexTTS2.5] modelscope not installed, trying CLI...")
    except Exception as e:
        print(f"[BSAI_IndexTTS2.5] ModelScope Python API failed: {e}, trying CLI...")

    # Try CLI fallback
    cmd = [
        sys.executable, "-m", "pip", "install", "-q", "modelscope",
    ]
    _run_subprocess(cmd)

    try:
        from modelscope import snapshot_download
        snapshot_download(
            model_id=MODELSCOPE_MODEL_ID,
            local_dir=target_dir,
        )
        print(f"[BSAI_IndexTTS2.5] ModelScope download complete -> {target_dir}")
        return True
    except Exception as e:
        print(f"[BSAI_IndexTTS2.5] ModelScope download failed: {e}")
        return False


def download_from_huggingface(target_dir, use_mirror=True):
    """Download model from HuggingFace (with optional mirror for China)."""
    env = {}
    if use_mirror:
        env["HF_ENDPOINT"] = HF_MIRROR_ENDPOINT
        print(f"[BSAI_IndexTTS2.5] Downloading from HuggingFace mirror: {HUGGINGFACE_MODEL_ID}")
    else:
        print(f"[BSAI_IndexTTS2.5] Downloading from HuggingFace: {HUGGINGFACE_MODEL_ID}")

    # Try using huggingface_hub Python API
    try:
        from huggingface_hub import snapshot_download as hf_snapshot_download
        hf_snapshot_download(
            repo_id=HUGGINGFACE_MODEL_ID,
            local_dir=target_dir,
            env=env if env else None,
        )
        print(f"[BSAI_IndexTTS2.5] HuggingFace download complete -> {target_dir}")
        return True
    except ImportError:
        print("[BSAI_IndexTTS2.5] huggingface_hub not installed, installing...")
    except Exception as e:
        print(f"[BSAI_IndexTTS2.5] HuggingFace Python API failed: {e}, trying CLI...")

    # Install huggingface_hub
    cmd = [sys.executable, "-m", "pip", "install", "-q", "huggingface_hub[cli,hf_xet]"]
    _run_subprocess(cmd)

    try:
        from huggingface_hub import snapshot_download as hf_snapshot_download
        # Set environment variable for mirror
        if use_mirror:
            os.environ["HF_ENDPOINT"] = HF_MIRROR_ENDPOINT
        hf_snapshot_download(
            repo_id=HUGGINGFACE_MODEL_ID,
            local_dir=target_dir,
        )
        print(f"[BSAI_IndexTTS2.5] HuggingFace download complete -> {target_dir}")
        return True
    except Exception as e:
        print(f"[BSAI_IndexTTS2.5] HuggingFace download failed: {e}")
        if use_mirror:
            print("[BSAI_IndexTTS2.5] Retrying without mirror...")
            return download_from_huggingface(target_dir, use_mirror=False)
        return False


def ensure_model_available(model_dir=None):
    """
    Ensure IndexTTS-2.5 model is available locally.
    Download priority: local -> ModelScope -> HuggingFace (with mirror) -> HuggingFace (direct)

    Returns:
        str: Path to the model directory, or None if download failed.
    """
    if model_dir is None:
        model_dir = get_model_dir()

    print(f"[BSAI_IndexTTS2.5] Model directory: {model_dir}")

    # Step 1: Check if model already exists locally
    if check_model_exists(model_dir):
        print("[BSAI_IndexTTS2.5] Model found locally, skipping download.")
        return model_dir

    print("[BSAI_IndexTTS2.5] Model not found locally, starting download...")

    # Step 2: Try ModelScope first (better for China users)
    if download_from_modelscope(model_dir):
        if check_model_exists(model_dir):
            print("[BSAI_IndexTTS2.5] Model successfully downloaded from ModelScope.")
            return model_dir

    # Step 3: Try HuggingFace with mirror
    if download_from_huggingface(model_dir, use_mirror=True):
        if check_model_exists(model_dir):
            print("[BSAI_IndexTTS2.5] Model successfully downloaded from HuggingFace (mirror).")
            return model_dir

    # Step 4: Last resort - direct HuggingFace
    if download_from_huggingface(model_dir, use_mirror=False):
        if check_model_exists(model_dir):
            print("[BSAI_IndexTTS2.5] Model successfully downloaded from HuggingFace (direct).")
            return model_dir

    print("[BSAI_IndexTTS2.5] ERROR: All download methods failed!")
    print(f"[BSAI_IndexTTS2.5] Please manually download the model from:")
    print(f"  ModelScope: https://modelscope.cn/models/{MODELSCOPE_MODEL_ID}")
    print(f"  HuggingFace: https://huggingface.co/{HUGGINGFACE_MODEL_ID}")
    print(f"  And extract to: {model_dir}")
    return None


def get_python_executable():
    """Get the Python executable path for subprocess calls."""
    return sys.executable
