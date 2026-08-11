"""
BSAI_ComfyUI_IndexTTS-2.5 - Core TTS Nodes
ComfyUI custom nodes for IndexTTS-2.5 voice cloning and emotion-controllable TTS.

Nodes:
  - BSAI_IndexTTS2.5Loader:     Load IndexTTS-2.5 model (auto-download)
  - BSAI_IndexTTS2.5Synthesis:  Text-to-speech synthesis with reference audio
  - BSAI_IndexTTS2.5SaveAudio:  Save generated audio to disk
  - BSAI_IndexTTS2.5UnloadModel: Unload model to free VRAM
"""

import os
import sys
import json
import tempfile
import traceback
import numpy as np
import torch
import torchaudio
import folder_paths

from .model_manager import ensure_model_available, get_model_dir, get_config_path


# ---------------------------------------------------------------------------
#  Lazy import of indextts package (it's heavy, so only import when needed)
# ---------------------------------------------------------------------------
_INDEXTTS_INSTANCE = None
_INDEXTTS_MODEL_DIR = None


def _get_indextts(use_fp16=True, device=None):
    """Get or create the IndexTTS singleton instance."""
    global _INDEXTTS_INSTANCE, _INDEXTTS_MODEL_DIR

    if _INDEXTTS_INSTANCE is not None:
        return _INDEXTTS_INSTANCE

    # Ensure model is available
    model_dir = ensure_model_available()
    if model_dir is None:
        raise RuntimeError(
            "IndexTTS-2.5 model not available. "
            "Please check your network connection or manually download the model."
        )

    cfg_path = get_config_path(model_dir)
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    print(f"[BSAI_IndexTTS2.5] Loading IndexTTS-2.5 model from: {model_dir}")
    print(f"[BSAI_IndexTTS2.5] Config: {cfg_path}")
    print(f"[BSAI_IndexTTS2.5] FP16: {use_fp16}, Device: {device or 'auto'}")

    # Import indextts (lazy import)
    try:
        from indextts.infer import IndexTTS
    except ImportError:
        print("[BSAI_IndexTTS2.5] indextts package not found, installing...")
        import subprocess
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "indextts"],
            check=False,
        )
        from indextts.infer import IndexTTS

    tts = IndexTTS(
        cfg_path=cfg_path,
        model_dir=model_dir,
        use_fp16=use_fp16,
        device=device,
    )

    _INDEXTTS_INSTANCE = tts
    _INDEXTTS_MODEL_DIR = model_dir
    print("[BSAI_IndexTTS2.5] Model loaded successfully!")
    return tts


def _unload_indextts():
    """Unload the IndexTTS model to free VRAM."""
    global _INDEXTTS_INSTANCE, _INDEXTTS_MODEL_DIR
    if _INDEXTTS_INSTANCE is not None:
        del _INDEXTTS_INSTANCE
        _INDEXTTS_INSTANCE = None
        _INDEXTTS_MODEL_DIR = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[BSAI_IndexTTS2.5] Model unloaded, VRAM cleared.")


# ---------------------------------------------------------------------------
#  Helper: audio tensor <-> file conversion
# ---------------------------------------------------------------------------
def _audio_tensor_to_dict(waveform, sample_rate):
    """Convert a raw waveform tensor to ComfyUI AUDIO dict format."""
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # [1, T]
    return {"waveform": waveform.cpu(), "sample_rate": sample_rate}


def _load_audio_file(audio_path, target_sr=24000):
    """Load an audio file and return waveform tensor and sample rate."""
    waveform, sr = torchaudio.load(audio_path)
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    return waveform, target_sr


# ===========================================================================
#  Node 1: BSAI_IndexTTS2.5Loader
# ===========================================================================
class BSAI_IndexTTS2_5Loader:
    """
    Load IndexTTS-2.5 model with automatic model download support.
    Downloads from ModelScope (preferred) or HuggingFace (fallback).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_fp16": ("BOOLEAN", {"default": True}),
                "device": (["auto", "cuda:0", "cpu"], {"default": "auto"}),
            },
            "optional": {
                "force_reload": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("INDEX_TTS_MODEL",)
    RETURN_NAMES = ("tts_model",)
    FUNCTION = "load_model"
    CATEGORY = "BSAI"

    def load_model(self, use_fp16=True, device="auto", force_reload=False):
        if force_reload:
            _unload_indextts()

        device = None if device == "auto" else device
        tts = _get_indextts(use_fp16=use_fp16, device=device)
        return (tts,)


# ===========================================================================
#  Node 2: BSAI_IndexTTS2.5Synthesis
# ===========================================================================
class BSAI_IndexTTS2_5Synthesis:
    """
    IndexTTS-2.5 voice cloning synthesis node.
    Generate speech from text using a reference audio for voice cloning.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tts_model": ("INDEX_TTS_MODEL",),
                "text": ("STRING", {
                    "default": "Hello, this is a test of IndexTTS-2.5 voice cloning.",
                    "multiline": True,
                }),
                "reference_audio": ("AUDIO",),
            },
            "optional": {
                "use_fast_inference": ("BOOLEAN", {"default": True}),
                "max_text_tokens_per_segment": ("INT", {
                    "default": 100, "min": 20, "max": 600, "step": 10,
                }),
                "max_mel_tokens": ("INT", {
                    "default": 600, "min": 100, "max": 1815, "step": 50,
                }),
                "temperature": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 2.0, "step": 0.05,
                }),
                "top_p": ("FLOAT", {
                    "default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05,
                }),
                "top_k": ("INT", {
                    "default": 30, "min": 1, "max": 100, "step": 1,
                }),
                "length_penalty": ("FLOAT", {
                    "default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1,
                }),
                "num_beams": ("INT", {
                    "default": 3, "min": 1, "max": 10, "step": 1,
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": 10.0, "min": 1.0, "max": 20.0, "step": 0.5,
                }),
                "remove_silence": ("BOOLEAN", {"default": True}),
                "verbose": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "status")
    FUNCTION = "synthesize"
    CATEGORY = "BSAI"

    def synthesize(
        self,
        tts_model,
        text,
        reference_audio,
        use_fast_inference=True,
        max_text_tokens_per_segment=100,
        max_mel_tokens=600,
        temperature=1.0,
        top_p=0.8,
        top_k=30,
        length_penalty=0.0,
        num_beams=3,
        repetition_penalty=10.0,
        remove_silence=True,
        verbose=False,
    ):
        if not text or not text.strip():
            raise ValueError("Text input cannot be empty.")

        # Save reference audio to a temp file (IndexTTS expects a file path)
        ref_waveform = reference_audio["waveform"]
        ref_sr = reference_audio["sample_rate"]

        # Ensure 2D shape [1, T]
        if ref_waveform.dim() == 1:
            ref_waveform = ref_waveform.unsqueeze(0)

        temp_dir = tempfile.mkdtemp(prefix="bsai_indextts2_5_")
        ref_audio_path = os.path.join(temp_dir, "reference.wav")
        out_audio_path = os.path.join(temp_dir, "output.wav")

        # Save reference audio
        torchaudio.save(ref_audio_path, ref_waveform.cpu(), ref_sr)
        print(f"[BSAI_IndexTTS2.5] Reference audio saved: {ref_audio_path} (sr={ref_sr})")

        # Build generation kwargs
        generation_kwargs = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "length_penalty": length_penalty,
            "num_beams": num_beams,
            "repetition_penalty": repetition_penalty,
            "max_mel_tokens": max_mel_tokens,
            "do_sample": True,
        }

        try:
            if use_fast_inference:
                print("[BSAI_IndexTTS2.5] Using fast inference mode...")
                tts_model.infer_fast(
                    audio_prompt=ref_audio_path,
                    text=text,
                    output_path=out_audio_path,
                    verbose=verbose,
                    max_text_tokens_per_segment=max_text_tokens_per_segment,
                    **generation_kwargs,
                )
            else:
                print("[BSAI_IndexTTS2.5] Using standard inference mode...")
                # Build parameters for standard infer
                params = {
                    "verbose": verbose,
                    **generation_kwargs,
                }
                # Check if infer method supports remove_silence
                if remove_silence:
                    params["remove_long_silence"] = True
                tts_model.infer(
                    audio_prompt=ref_audio_path,
                    text=text,
                    output_path=out_audio_path,
                    **params,
                )

            # Load generated audio
            if not os.path.exists(out_audio_path):
                raise RuntimeError("TTS inference completed but output file not found.")

            audio_waveform, audio_sr = torchaudio.load(out_audio_path)
            # Convert to mono if needed
            if audio_waveform.shape[0] > 1:
                audio_waveform = audio_waveform.mean(dim=0, keepdim=True)

            # Return as ComfyUI AUDIO dict
            audio_dict = _audio_tensor_to_dict(audio_waveform, audio_sr)

            duration = audio_waveform.shape[-1] / audio_sr
            status = f"Success | Duration: {duration:.2f}s | Sample Rate: {audio_sr}Hz | Mode: {'fast' if use_fast_inference else 'standard'}"
            print(f"[BSAI_IndexTTS2.5] {status}")

            return (audio_dict, status)

        except Exception as e:
            error_msg = f"TTS synthesis failed: {e}"
            print(f"[BSAI_IndexTTS2.5] ERROR: {error_msg}")
            traceback.print_exc()
            raise RuntimeError(error_msg)
        finally:
            # Cleanup temp files
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass


# ===========================================================================
#  Node 3: BSAI_IndexTTS2.5SaveAudio
# ===========================================================================
class BSAI_IndexTTS2_5SaveAudio:
    """
    Save generated audio to disk with format options.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "BSAI_IndexTTS2_5"}),
                "format": (["wav", "mp3", "flac"], {"default": "wav"}),
            },
            "optional": {
                "mp3_bitrate": ("INT", {"default": 192, "min": 64, "max": 320, "step": 32}),
                "output_gain": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path")
    FUNCTION = "save_audio"
    CATEGORY = "BSAI"
    OUTPUT_NODE = True

    def save_audio(self, audio, filename_prefix="BSAI_IndexTTS2_5", format="wav", mp3_bitrate=192, output_gain=1.0):
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        # Apply gain
        if output_gain != 1.0:
            waveform = waveform * output_gain
            # Clamp to valid range
            waveform = torch.clamp(waveform, -1.0, 1.0)

        # Generate output path
        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)

        # Find unique filename
        base_name = filename_prefix
        counter = 1
        while True:
            ext = format if format != "wav" else "wav"
            filename = f"{base_name}_{counter:05d}.{ext}"
            filepath = os.path.join(output_dir, filename)
            if not os.path.exists(filepath):
                break
            counter += 1

        # Save audio
        if format == "wav":
            torchaudio.save(filepath, waveform.cpu(), sample_rate)
        elif format == "flac":
            torchaudio.save(filepath, waveform.cpu(), sample_rate, format="flac")
        elif format == "mp3":
            # torchaudio supports mp3 with ffmpeg backend
            try:
                torchaudio.save(filepath, waveform.cpu(), sample_rate, format="mp3",
                                bits_per_sample=mp3_bitrate * 1000)
            except Exception:
                # Fallback: save as wav and convert
                wav_path = filepath.replace(".mp3", ".wav")
                torchaudio.save(wav_path, waveform.cpu(), sample_rate)
                filepath = wav_path
                print("[BSAI_IndexTTS2.5] MP3 encoding failed, saved as WAV instead.")

        print(f"[BSAI_IndexTTS2.5] Audio saved: {filepath}")
        return (filepath,)


# ===========================================================================
#  Node 4: BSAI_IndexTTS2.5UnloadModel
# ===========================================================================
class BSAI_IndexTTS2_5UnloadModel:
    """Unload IndexTTS-2.5 model to free VRAM."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "any_input": ("*",),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "unload_model"
    CATEGORY = "BSAI"
    OUTPUT_NODE = True

    def unload_model(self, any_input=None):
        _unload_indextts()
        return {}


# ===========================================================================
#  Node Mappings
# ===========================================================================
NODE_CLASS_MAPPINGS = {
    "BSAI_IndexTTS2.5Loader": BSAI_IndexTTS2_5Loader,
    "BSAI_IndexTTS2.5Synthesis": BSAI_IndexTTS2_5Synthesis,
    "BSAI_IndexTTS2.5SaveAudio": BSAI_IndexTTS2_5SaveAudio,
    "BSAI_IndexTTS2.5UnloadModel": BSAI_IndexTTS2_5UnloadModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSAI_IndexTTS2.5Loader": "BSAI IndexTTS2.5 Loader",
    "BSAI_IndexTTS2.5Synthesis": "BSAI IndexTTS2.5 Synthesis",
    "BSAI_IndexTTS2.5SaveAudio": "BSAI IndexTTS2.5 Save Audio",
    "BSAI_IndexTTS2.5UnloadModel": "BSAI IndexTTS2.5 Unload Model",
}
