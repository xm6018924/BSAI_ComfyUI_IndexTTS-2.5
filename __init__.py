"""
BSAI_ComfyUI_IndexTTS-2.5
ComfyUI custom nodes for IndexTTS-2 voice cloning and emotion-controllable TTS.

Nodes:
  - BSAI IndexTTS2 Loader:      Load model with auto-download
  - BSAI IndexTTS2 Synthesis:   Text-to-speech with reference audio
  - BSAI IndexTTS2 Save Audio:  Save generated audio
  - BSAI IndexTTS2 Unload Model: Free VRAM

Author: BSAI Team
License: MIT
"""

import logging
import os
import sys
import shutil

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compatibility shim: ensure indextts works with transformers 5.x
# This runs every time ComfyUI loads the node, so it survives indextts updates.
# ---------------------------------------------------------------------------
def _ensure_compat_shim():
    """Ensure the transformers 5.x compatibility shim is installed in indextts."""
    try:
        import transformers
        tf_major = int(transformers.__version__.split('.')[0])
    except Exception:
        return  # Can't check, let it fail naturally

    if tf_major < 5:
        return  # transformers 4.x, no shim needed

    try:
        import indextts
        indextts_dir = os.path.dirname(indextts.__file__)

        # Path to the compat shim bundled with this node
        node_dir = os.path.dirname(os.path.abspath(__file__))
        compat_src = os.path.join(node_dir, "indextts_compat.py")
        compat_dst = os.path.join(indextts_dir, "_compat.py")

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
        init_path = os.path.join(indextts_dir, "__init__.py")
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

    except ImportError:
        logger.warning(
            "[BSAI_IndexTTS2.5] indextts package not found. "
            "Please run install.py or install_bsai_indextts.bat"
        )
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
