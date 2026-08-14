"""
BSAI_ComfyUI_IndexTTS-2.5
ComfyUI custom nodes for IndexTTS-2.5 voice cloning and emotion-controllable TTS.

Nodes:
  - BSAI IndexTTS2.5 Load Audio:     Load audio files (self-contained, no external deps)
  - BSAI IndexTTS2.5 Loader:         Load model with auto-download (supports use_qwen_emo)
  - BSAI IndexTTS2.5 Synthesis:      Text-to-speech with emotion control, speed control, cross-language
  - BSAI IndexTTS2.5 Emotion Vector: Construct 8D emotion vector [happy, angry, sad, fear, disgust, melancholy, surprise, calm]
  - BSAI IndexTTS2.5 Save Audio:     Save generated audio (outputs file_path + audio passthrough)
  - BSAI IndexTTS2.5 Unload Model:   Free VRAM

Author: BSAI Team
License: MIT
"""

import logging
import os
import sys
import shutil

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compatibility shim management: ensure indextts works with transformers 5.x
# This runs every time ComfyUI loads the node, so it survives indextts updates.
# ---------------------------------------------------------------------------
def _ensure_compat_shim():
    """Ensure the transformers 5.x compatibility shim is installed in indextts.

    For transformers 5.x: install/update the shim.
    For transformers 4.x: remove any stale shim that may have been left from
    a previous transformers 5.x installation.
    """
    try:
        import transformers
        tf_major = int(transformers.__version__.split('.')[0])
    except Exception:
        return  # Can't check, let it fail naturally

    try:
        import indextts
        indextts_dir = os.path.dirname(indextts.__file__)
    except ImportError:
        return  # indextts not installed yet
    except Exception:
        return

    compat_dst = os.path.join(indextts_dir, "_compat.py")
    init_path = os.path.join(indextts_dir, "__init__.py")
    node_dir = os.path.dirname(os.path.abspath(__file__))
    compat_src = os.path.join(node_dir, "indextts_compat.py")

    # ------------------------------------------------------------------
    # For transformers 4.x: clean up any stale shim
    # ------------------------------------------------------------------
    if tf_major < 5:
        # Remove the _compat.py file
        if os.path.exists(compat_dst):
            try:
                os.remove(compat_dst)
                logger.info("[BSAI_IndexTTS2.5] Removed stale transformers 5.x compat shim (transformers 4.x detected)")
            except Exception:
                pass

        # Remove cached .pyc
        pyc_path = os.path.join(indextts_dir, "__pycache__", "_compat.cpython-313.pyc")
        if os.path.exists(pyc_path):
            try:
                os.remove(pyc_path)
            except Exception:
                pass

        # Unpatch __init__.py: remove the "from . import _compat" line
        if os.path.exists(init_path):
            try:
                with open(init_path, 'r', encoding='utf-8') as f:
                    init_content = f.read()
                if '_compat' in init_content:
                    lines = init_content.split('\n')
                    cleaned_lines = [l for l in lines if '_compat' not in l]
                    # Remove leading empty/comment-only lines that were part of the patch
                    while cleaned_lines and (cleaned_lines[0].strip().startswith('#') or not cleaned_lines[0].strip()):
                        if 'compat' in cleaned_lines[0].lower():
                            cleaned_lines.pop(0)
                        else:
                            break
                    cleaned = '\n'.join(cleaned_lines).strip()
                    if not cleaned:
                        cleaned = "# indextts package\n"
                    with open(init_path, 'w', encoding='utf-8') as f:
                        f.write(cleaned + '\n')
                    logger.info("[BSAI_IndexTTS2.5] Unpatched indextts __init__.py (removed stale compat import)")
            except Exception:
                pass
        return  # transformers 4.x, no shim needed

    # ------------------------------------------------------------------
    # For transformers 5.x: install/update the shim
    # ------------------------------------------------------------------
    try:
        # Copy shim if it doesn't exist or is outdated
        need_copy = True
        if os.path.exists(compat_dst) and os.path.exists(compat_src):
            src_mtime = os.path.getmtime(compat_src)
            dst_mtime = os.path.getmtime(compat_dst)
            need_copy = src_mtime > dst_mtime

        if need_copy and os.path.exists(compat_src):
            shutil.copy2(compat_src, compat_dst)
            logger.info("[BSAI_IndexTTS2.5] Updated transformers 5.x compat shim")

        # Patch indextts __init__.py to load compat first
        if os.path.exists(init_path):
            with open(init_path, 'r', encoding='utf-8') as f:
                init_content = f.read()
            if '_compat' not in init_content:
                with open(init_path, 'w', encoding='utf-8') as f:
                    f.write(
                        "# Load compatibility shim first (for transformers 5.x)\n"
                        "from . import _compat  # noqa: F401\n"
                        + init_content
                    )
                    logger.info("[BSAI_IndexTTS2.5] Patched indextts __init__.py for transformers 5.x")

    except Exception as e:
        logger.warning(f"[BSAI_IndexTTS2.5] Could not apply compat shim: {e}")


# Apply compatibility shim before importing node code
_ensure_compat_shim()

try:
    from .BSAI_IndexTTS import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except Exception as e:
    logger.error(f"Failed to load BSAI_IndexTTS nodes: {e}")
    import traceback
    traceback.print_exc()
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
