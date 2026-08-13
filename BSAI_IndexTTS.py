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
# Monkey-patch torchaudio.save/load to use soundfile fallback
# torchaudio 2.11+ requires torchcodec for save/load, which may not be
# installed. This patch makes ALL torchaudio.save/load calls (including
# from indextts library internals) fall back to soundfile automatically.
# ---------------------------------------------------------------------------
_torchaudio_save_orig = torchaudio.save
_torchaudio_load_orig = torchaudio.load


def _torchaudio_save_patched(filepath, src, sample_rate, **kwargs):
    """Wrap torchaudio.save with soundfile fallback."""
    try:
        return _torchaudio_save_orig(filepath, src, sample_rate, **kwargs)
    except (ImportError, RuntimeError):
        import soundfile as sf
        wav_np = src.cpu().numpy() if hasattr(src, 'cpu') else np.array(src)
        if wav_np.ndim == 3:
            wav_np = wav_np[0]  # squeeze batch dim
        if wav_np.ndim == 2:
            wav_np = wav_np.T  # (channels, samples) -> (samples, channels)
        elif wav_np.ndim == 1:
            wav_np = wav_np.reshape(-1, 1)
        sf.write(filepath, wav_np, sample_rate)


def _torchaudio_load_patched(filepath, **kwargs):
    """Wrap torchaudio.load with soundfile fallback."""
    try:
        return _torchaudio_load_orig(filepath, **kwargs)
    except (ImportError, RuntimeError):
        import soundfile as sf
        data, sr = sf.read(filepath)
        waveform = torch.from_numpy(data).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.T  # (samples, channels) -> (channels, samples)
        return waveform, sr


torchaudio.save = _torchaudio_save_patched
torchaudio.load = _torchaudio_load_patched


# ---------------------------------------------------------------------------
#  Lazy import of indextts package (it's heavy, so only import when needed)
# ---------------------------------------------------------------------------
_INDEXTTS_INSTANCE = None
_INDEXTTS_MODEL_DIR = None


def _reapply_compat_shim():
    """Re-apply the transformers 5.x compat shim after indextts is installed."""
    import shutil
    try:
        import transformers
        tf_major = int(transformers.__version__.split('.')[0])
    except Exception:
        return
    if tf_major < 5:
        return
    try:
        import indextts
        indextts_dir = os.path.dirname(indextts.__file__)
    except ImportError:
        return
    node_dir = os.path.dirname(os.path.abspath(__file__))
    compat_src = os.path.join(node_dir, "indextts_compat.py")
    compat_dst = os.path.join(indextts_dir, "_compat.py")
    init_path = os.path.join(indextts_dir, "__init__.py")
    if os.path.exists(compat_src):
        shutil.copy2(compat_src, compat_dst)
        print(f"[BSAI_IndexTTS2.5] Copied compat shim to: {compat_dst}")
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
            print("[BSAI_IndexTTS2.5] Patched indextts __init__.py for compat")


def _get_indextts(use_bf16=True, device=None):
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
    print(f"[BSAI_IndexTTS2.5] BF16: {use_bf16}, Device: {device or 'auto'}")

    # Import indextts v2.5 (lazy import)
    try:
        from indextts.infer_v2_5 import IndexTTS2
    except ImportError as _import_err:
        _import_err_msg = str(_import_err)
        print(f"[BSAI_IndexTTS2.5] ImportError: {_import_err_msg}")
        print("[BSAI_IndexTTS2.5] indextts not available, attempting auto-install...")

        # Diagnostic: check if indextts package exists at all
        try:
            import indextts
            print(f"[BSAI_IndexTTS2.5] indextts package found at: {indextts.__file__}")
            # Package exists but infer_v2_5 missing — likely wrong version
            import os as _os
            _pkg_dir = _os.path.dirname(indextts.__file__)
            _files = _os.listdir(_pkg_dir) if _os.path.isdir(_pkg_dir) else []
            print(f"[BSAI_IndexTTS2.5] indextts package files: {_files}")
            if 'infer_v2_5.py' not in _files:
                print("[BSAI_IndexTTS2.5] WARNING: infer_v2_5.py not found in indextts package!")
                print("[BSAI_IndexTTS2.5] The installed indextts version may be too old. Reinstalling...")
        except ImportError:
            print("[BSAI_IndexTTS2.5] indextts package NOT installed")

        import subprocess
        node_dir = os.path.dirname(os.path.abspath(__file__))
        install_script = os.path.join(node_dir, "install.py")

        if os.path.exists(install_script):
            print(f"[BSAI_IndexTTS2.5] Running install.py with {sys.executable}...")
            result = subprocess.run(
                [sys.executable, install_script],
                cwd=node_dir,
                capture_output=True,
                text=True,
                check=False,
            )
            # Print install output so user can see what happened
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
            print(f"[BSAI_IndexTTS2.5] install.py exited with code: {result.returncode}")
        else:
            print("[BSAI_IndexTTS2.5] install.py not found, trying direct pip install...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install",
                 "--no-deps", "--ignore-requires-python",
                 "https://github.com/index-tts/index-tts/archive/refs/heads/main.zip"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)

        # Re-apply compat shim after install
        try:
            _reapply_compat_shim()
        except Exception as _e:
            print(f"[BSAI_IndexTTS2.5] Warning: could not apply compat shim: {_e}")

        # Try importing again
        try:
            from indextts.infer_v2_5 import IndexTTS2
        except ImportError as _retry_err:
            _retry_msg = str(_retry_err)
            print(f"[BSAI_IndexTTS2.5] Import still failing after install: {_retry_msg}")
            raise ImportError(
                f"Failed to import indextts after auto-install.\n"
                f"  Original error: {_import_err_msg}\n"
                f"  After install:  {_retry_msg}\n"
                f"Please run install_bsai_indextts.bat manually, or execute:\n"
                f"  pip install --no-deps --ignore-requires-python "
                f"https://github.com/index-tts/index-tts/archive/refs/heads/main.zip"
            )

    tts = IndexTTS2(
        cfg_path=cfg_path,
        model_dir=model_dir,
        use_bf16=use_bf16,
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
    """Convert a raw waveform tensor to ComfyUI AUDIO dict format.

    ComfyUI AUDIO format: (batch, channels, samples) 3D tensor.
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
    elif waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)  # [1, C, T]
    return {"waveform": waveform.cpu(), "sample_rate": sample_rate}


def _save_audio_file(path, waveform, sample_rate, fmt=None):
    """Save audio file, using soundfile as fallback when torchcodec is unavailable.

    torchaudio 2.11+ requires torchcodec for save/load. This helper tries
    torchaudio first, then falls back to soundfile (which is pre-installed).
    """
    # Try torchaudio.save first (uses torchcodec if available)
    try:
        if fmt:
            torchaudio.save(path, waveform.cpu(), sample_rate, format=fmt)
        else:
            torchaudio.save(path, waveform.cpu(), sample_rate)
        return
    except (ImportError, RuntimeError):
        pass

    # Fallback to soundfile
    import soundfile as sf
    wav_np = waveform.cpu().numpy()
    # Handle 3D (batch, channels, samples) -> squeeze batch dimension
    if wav_np.ndim == 3:
        wav_np = wav_np[0]  # (channels, samples)
    # torchaudio: (channels, samples), soundfile: (samples, channels)
    if wav_np.ndim == 2:
        wav_np = wav_np.T
    elif wav_np.ndim == 1:
        wav_np = wav_np.reshape(-1, 1)

    # Determine soundfile format from fmt or file extension
    sf_fmt = None
    ext_lower = path.lower().rsplit('.', 1)[-1] if '.' in path else ''
    target = (fmt or ext_lower).lower() if (fmt or ext_lower) else 'wav'
    if target == 'flac':
        sf_fmt = 'FLAC'
    elif target == 'mp3':
        sf_fmt = 'MP3'

    try:
        if sf_fmt:
            sf.write(path, wav_np, sample_rate, format=sf_fmt)
        else:
            sf.write(path, wav_np, sample_rate)
    except Exception as e:
        # If MP3 fails (libsndfile too old), save as WAV
        if target == 'mp3':
            wav_path = path.rsplit('.', 1)[0] + '.wav'
            sf.write(wav_path, wav_np, sample_rate)
            print(f"[BSAI_IndexTTS2.5] MP3 save failed ({e}), saved as WAV: {wav_path}")
            return wav_path
        raise


def _load_audio_file(audio_path, target_sr=24000):
    """Load an audio file and return waveform tensor and sample rate.

    Uses soundfile as fallback when torchaudio.load requires torchcodec.
    """
    try:
        waveform, sr = torchaudio.load(audio_path)
    except (ImportError, RuntimeError):
        # Fallback to soundfile
        import soundfile as sf
        data, sr = sf.read(audio_path)  # (samples, channels) numpy
        waveform = torch.from_numpy(data).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # (1, samples)
        else:
            waveform = waveform.T  # (samples, channels) -> (channels, samples)

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
                "use_bf16": ("BOOLEAN", {"default": True}),
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

    def load_model(self, use_bf16=True, device="auto", force_reload=False):
        if force_reload:
            _unload_indextts()

        device = None if device == "auto" else device
        tts = _get_indextts(use_bf16=use_bf16, device=device)
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
                "lang": (["ZH", "EN", "JA", "ES", "zhen"], {"default": "ZH"}),
            },
            "optional": {
                "max_text_tokens_per_segment": ("INT", {
                    "default": 100, "min": 20, "max": 600, "step": 10,
                }),
                "max_mel_tokens": ("INT", {
                    "default": 1500, "min": 100, "max": 1815, "step": 50,
                }),
                "temperature": ("FLOAT", {
                    "default": 0.8, "min": 0.1, "max": 2.0, "step": 0.05,
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
                "do_sample": ("BOOLEAN", {"default": True}),
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
        lang="ZH",
        max_text_tokens_per_segment=100,
        max_mel_tokens=1500,
        temperature=0.8,
        top_p=0.8,
        top_k=30,
        length_penalty=0.0,
        num_beams=3,
        repetition_penalty=10.0,
        do_sample=True,
        verbose=False,
    ):
        if not text or not text.strip():
            raise ValueError("Text input cannot be empty.")

        # Save reference audio to a temp file (IndexTTS expects a file path)
        ref_waveform = reference_audio["waveform"]
        ref_sr = reference_audio["sample_rate"]

        # Ensure 2D shape [channels, samples] (ComfyUI AUDIO is 3D: batch/channels/samples)
        if ref_waveform.dim() == 3:
            ref_waveform = ref_waveform[0]  # Remove batch dim
        elif ref_waveform.dim() == 1:
            ref_waveform = ref_waveform.unsqueeze(0)

        temp_dir = tempfile.mkdtemp(prefix="bsai_indextts2_5_")
        ref_audio_path = os.path.join(temp_dir, "reference.wav")
        out_audio_path = os.path.join(temp_dir, "output.wav")

        # Save reference audio
        _save_audio_file(ref_audio_path, ref_waveform, ref_sr)
        print(f"[BSAI_IndexTTS2.5] Reference audio saved: {ref_audio_path} (sr={ref_sr})")

        # Build generation kwargs for v2.5 infer()
        generation_kwargs = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "length_penalty": length_penalty,
            "num_beams": num_beams,
            "repetition_penalty": repetition_penalty,
            "max_mel_tokens": max_mel_tokens,
            "do_sample": do_sample,
        }

        try:
            print(f"[BSAI_IndexTTS2.5] Synthesizing (lang={lang})...")
            tts_model.infer(
                spk_audio_prompt=ref_audio_path,
                text=text,
                output_path=out_audio_path,
                lang=lang,
                verbose=verbose,
                max_text_tokens_per_segment=max_text_tokens_per_segment,
                **generation_kwargs,
            )

            # Load generated audio (with soundfile fallback for torchaudio 2.11+)
            if not os.path.exists(out_audio_path):
                raise RuntimeError("TTS inference completed but output file not found.")

            try:
                audio_waveform, audio_sr = torchaudio.load(out_audio_path)
            except (ImportError, RuntimeError):
                import soundfile as sf
                data, audio_sr = sf.read(out_audio_path)
                audio_waveform = torch.from_numpy(data).float()
                if audio_waveform.dim() == 1:
                    audio_waveform = audio_waveform.unsqueeze(0)
                else:
                    audio_waveform = audio_waveform.T
            # Convert to mono if needed
            if audio_waveform.shape[0] > 1:
                audio_waveform = audio_waveform.mean(dim=0, keepdim=True)

            # Return as ComfyUI AUDIO dict
            audio_dict = _audio_tensor_to_dict(audio_waveform, audio_sr)

            duration = audio_waveform.shape[-1] / audio_sr
            status = f"Success | Duration: {duration:.2f}s | Sample Rate: {audio_sr}Hz | Lang: {lang}"
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
            _save_audio_file(filepath, waveform, sample_rate)
        elif format == "flac":
            _save_audio_file(filepath, waveform, sample_rate, fmt="flac")
        elif format == "mp3":
            # Try MP3, fallback to WAV
            result = _save_audio_file(filepath, waveform, sample_rate, fmt="mp3")
            if result and result != filepath:
                filepath = result

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
