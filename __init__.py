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

logger = logging.getLogger(__name__)

try:
    from .BSAI_IndexTTS import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except Exception as e:
    logger.error(f"Failed to load BSAI_IndexTTS nodes: {e}")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
