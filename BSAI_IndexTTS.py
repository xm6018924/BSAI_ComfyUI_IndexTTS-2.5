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

# Python 3.14 下 ComfyUI 以 spec_from_file_location 加载插件（无包上下文），
# 相对导入会因加载顺序/独立导入而失败；改为自包含绝对导入。
_NODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _NODE_DIR not in sys.path:
    sys.path.insert(0, _NODE_DIR)

from model_manager import ensure_model_available, get_model_dir, get_config_path

# ==== BSAI 插件协同 SDK：加载即自动注册（失败不拖垮插件） ====
try:
    import sys as _bsai_sys, os as _bsai_os
    _BSAI_ORCH_DIR = _bsai_os.path.join(
        _bsai_os.path.dirname(_bsai_os.path.abspath(__file__)),
        "..", "BSAI-ComfyUI-Orchestrator")
    if _bsai_os.path.isdir(_BSAI_ORCH_DIR) and _BSAI_ORCH_DIR not in _bsai_sys.path:
        _bsai_sys.path.insert(0, _BSAI_ORCH_DIR)
    from bsai_orch_client import BSAIOrch  # noqa: E402
    BSAIOrch.register(
        name="BSAI-IndexTTS-2.5",
        kind="tts_infer",
        hardware=["cuda", "cpu"],
    )
except Exception as _bsai_e:  # 注册失败不得拖垮插件
    print(f"[BSAI SDK] BSAI-IndexTTS-2.5 注册失败(忽略): {_bsai_e}")
# ==== BSAI SDK 块结束 ====


# ---------------------------------------------------------------------------
# Monkey-patch torchaudio.save/load to use soundfile fallback
# torchaudio 2.11+ requires torchcodec for save/load, which may not be
# installed. This patch makes ALL torchaudio.save/load calls (including
# from indextts library internals) fall back to soundfile automatically.
# ---------------------------------------------------------------------------
_torchaudio_save_orig = torchaudio.save
_torchaudio_load_orig = torchaudio.load


def _torchaudio_save_patched(filepath, src, sample_rate, **kwargs):
    """Wrap torchaudio.save with batch-dim normalization and soundfile fallback."""
    try:
        save_src = src
        if hasattr(src, 'cpu'):
            save_src = src.cpu()
        if getattr(save_src, 'ndim', 0) == 3:
            save_src = save_src[0]  # [batch, channels, samples] -> [channels, samples]
        return _torchaudio_save_orig(filepath, save_src, sample_rate, **kwargs)
    except (ImportError, RuntimeError, ValueError):
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


def _clear_indextts_modules():
    """Remove indextts and its submodules from sys.modules cache.

    When indextts is partially imported (and fails) before the compat shim is
    applied, Python caches the broken module in sys.modules. Subsequent import
    attempts reuse the cached broken module instead of re-running __init__.py.
    Clearing the cache forces a fresh import that picks up the compat shim.
    """
    keys_to_remove = [k for k in sys.modules if k == 'indextts' or k.startswith('indextts.')]
    for key in keys_to_remove:
        del sys.modules[key]
    if keys_to_remove:
        print(f"[BSAI_IndexTTS2.5] Cleared {len(keys_to_remove)} cached indextts modules from sys.modules")


def _ensure_transformers_compat():
    """Proactively inject critical missing symbols into transformers before importing indextts.

    In transformers 5.x, several symbols were removed or relocated. The compat
    shim (_compat.py) handles this, but only if it's loaded via indextts/__init__.py.
    This function provides a belt-and-suspenders fallback: it directly injects
    the most critical missing symbol (ExtensionsTrie) so that the import succeeds
    even if the compat shim hasn't been loaded yet.
    """
    try:
        import transformers
        tf_major = int(transformers.__version__.split('.')[0])
    except Exception:
        return

    if tf_major < 5:
        return  # transformers 4.x — symbols should be available natively

    # --- Inject ExtensionsTrie into transformers.tokenization_utils ---
    # In transformers 5.x, ExtensionsTrie was moved to tokenization_python.
    # indextts imports it from tokenization_utils, so we inject it there.
    try:
        import transformers.tokenization_utils as _tu
        if not hasattr(_tu, 'ExtensionsTrie'):
            try:
                from transformers.tokenization_python import ExtensionsTrie
                _tu.ExtensionsTrie = ExtensionsTrie
                print("[BSAI_IndexTTS2.5] Injected ExtensionsTrie (from tokenization_python) into transformers.tokenization_utils")
            except ImportError:
                # tokenization_python not available — define a minimal ExtensionsTrie
                class ExtensionsTrie:
                    """Minimal ExtensionsTrie fallback (matches transformers 4.x API)."""
                    def __init__(self, vocab=None):
                        self.data = {}
                        if vocab:
                            for token in vocab:
                                self.add(token)

                    def add(self, word):
                        node = self.data
                        for ch in word:
                            if ch not in node:
                                node[ch] = {}
                            node = node[ch]
                        node[None] = word

                    def split(self, text):
                        states = [(self.data, 0)]
                        results = []
                        for i, ch in enumerate(text):
                            new_states = []
                            for state, start in states:
                                if ch in state:
                                    new_states.append((state[ch], start))
                                if None in state:
                                    results.append((state[None], start, i))
                            if ch in self.data:
                                new_states.append((self.data[ch], i))
                            states = new_states
                        for state, start in states:
                            if None in state:
                                results.append((state[None], start, len(text)))
                        return results

                _tu.ExtensionsTrie = ExtensionsTrie
                print("[BSAI_IndexTTS2.5] Injected fallback ExtensionsTrie into transformers.tokenization_utils")
    except Exception as e:
        print(f"[BSAI_IndexTTS2.5] Warning: could not inject ExtensionsTrie: {e}")


def _patch_transformers_config_compat():
    """Patch PretrainedConfig._get_non_default_generation_parameters for transformers 5.x.

    transformers 5.x removed the _get_non_default_generation_parameters method
    from PretrainedConfig, but indextts still calls it during generation
    (transformers_generation_utils.py) and model saving
    (transformers_modeling_utils.py), causing:
      AttributeError: 'GPT2Config' object has no attribute
      '_get_non_default_generation_parameters'

    This method returns a dict of config attributes that differ from the
    default GenerationConfig values. We re-implement it using hasattr()
    instead of `in` since GenerationConfig 5.x no longer supports __contains__.
    """
    try:
        import transformers
        tf_major = int(transformers.__version__.split('.')[0])
    except Exception:
        return

    if tf_major < 5:
        return  # transformers 4.x — method exists natively

    try:
        from transformers.configuration_utils import PretrainedConfig
        from transformers import GenerationConfig

        if hasattr(PretrainedConfig, '_get_non_default_generation_parameters'):
            return  # already patched or available

        def _get_non_default_generation_parameters(self):
            """Compare config attributes with default GenerationConfig values.

            Returns a dict of parameters with non-default values that exist
            in the default GenerationConfig. Replicates transformers 4.x.
            """
            config_dict = self.to_dict()
            try:
                generation_config = GenerationConfig()
            except Exception:
                return {}

            non_default = {}
            for config_key in config_dict:
                if config_key == "is_decoder":
                    continue
                if hasattr(generation_config, config_key):
                    config_val = config_dict[config_key]
                    gen_val = getattr(generation_config, config_key)
                    if config_val != gen_val:
                        non_default[config_key] = config_val
            return non_default

        PretrainedConfig._get_non_default_generation_parameters = _get_non_default_generation_parameters
        print(f"[BSAI_IndexTTS2.5] Patched PretrainedConfig._get_non_default_generation_parameters for transformers {transformers.__version__}")
    except Exception as e:
        print(f"[BSAI_IndexTTS2.5] Warning: could not patch PretrainedConfig._get_non_default_generation_parameters: {e}")


def _patch_generation_config_compat():
    """Patch GenerationConfig and PretrainedConfig with missing attributes for transformers 5.x.

    transformers 5.x removed several attributes from GenerationConfig that indextts
    accesses during generation:
      - return_legacy_cache: controlled legacy KV cache format (removed in 5.x)
      - _original_object_hash: config migration check (removed in 5.x)
      - _bos_token_tensor, _eos_token_tensor, _pad_token_tensor,
        _decoder_start_token_tensor: cached tensor versions of token IDs

    Also patches PretrainedConfig:
      - _pre_quantization_dtype: quantization dtype tracking
      - sliding_window: sliding window attention config

    Each attribute is guarded with hasattr so this is a no-op on transformers 4.x
    (where all attributes exist natively). This provides a belt-and-suspenders
    fallback in case the _compat.py shim hasn't been loaded yet.
    """
    try:
        import transformers
        tf_major = int(transformers.__version__.split('.')[0])
    except Exception:
        return

    if tf_major < 5:
        return  # transformers 4.x — attributes exist natively

    try:
        from transformers import GenerationConfig as _GC
        from transformers.configuration_utils import PretrainedConfig as _PC

        _patched = []

        # --- GenerationConfig class-level defaults ---
        # return_legacy_cache: In 4.x, controlled whether to return tuple-based
        # legacy KV cache. Removed in 5.x — DynamicCache is the only format.
        # Default False = never convert to legacy format (safe for 5.x).
        if not hasattr(_GC, 'return_legacy_cache'):
            _GC.return_legacy_cache = False
            _patched.append('return_legacy_cache')

        # _original_object_hash: In 4.x, set in __init__ to detect config
        # modifications. Used in generation config migration check (line ~1523).
        # Setting to 0 makes the equality check fail safely (skips migration).
        if not hasattr(_GC, '_original_object_hash'):
            _GC._original_object_hash = 0
            _patched.append('_original_object_hash')

        # _bos_token_tensor, _eos_token_tensor, _pad_token_tensor,
        # _decoder_start_token_tensor: In 4.x, cached tensor versions of token
        # IDs. indextts sets these on instances before generation (lines ~1898-1901).
        # Default None ensures safe access before they're explicitly set.
        for _attr in ('_bos_token_tensor', '_eos_token_tensor',
                      '_pad_token_tensor', '_decoder_start_token_tensor'):
            if not hasattr(_GC, _attr):
                setattr(_GC, _attr, None)
                _patched.append(_attr)

        # --- PretrainedConfig class-level defaults ---
        # _pre_quantization_dtype: accessed during model saving/loading.
        # Some indextts code paths already use hasattr guard, but not all.
        if not hasattr(_PC, '_pre_quantization_dtype'):
            _PC._pre_quantization_dtype = None
            _patched.append('_pre_quantization_dtype')

        # sliding_window: accessed during model config initialization.
        if not hasattr(_PC, 'sliding_window'):
            _PC.sliding_window = None
            _patched.append('sliding_window')

        if _patched:
            print(f"[BSAI_IndexTTS2.5] Patched missing attributes for transformers {transformers.__version__}: {_patched}")
    except Exception as e:
        print(f"[BSAI_IndexTTS2.5] Warning: could not patch GenerationConfig/PretrainedConfig attributes: {e}")


def _patch_beam_search_compat():
    """Patch BeamSearchScorer to ensure it has 'is_done' for transformers 5.x.

    transformers 5.x removed the entire beam_search module. The compat shim
    creates a dummy BeamSearchScorer, but earlier versions lacked the 'is_done'
    property that indextts calls during beam search:
      AttributeError: 'BeamSearchScorer' object has no attribute 'is_done'

    indextts ships its own complete transformers_beam_search.py. If the
    BeamSearchScorer from the compat shim is missing 'is_done', we replace it
    in the transformers.generation.beam_search module with the local version.
    """
    try:
        import transformers
        tf_major = int(transformers.__version__.split('.')[0])
    except Exception:
        return

    if tf_major < 5:
        return  # transformers 4.x — beam_search module exists natively

    try:
        # Check if beam_search module exists (created by compat shim)
        beam_search_mod = sys.modules.get('transformers.generation.beam_search')
        if beam_search_mod is None:
            return

        BSS = getattr(beam_search_mod, 'BeamSearchScorer', None)
        if BSS is None:
            return

        if hasattr(BSS, 'is_done'):
            return  # already has is_done (compat shim updated or real class)

        # Try to use indextts's local complete implementation
        try:
            from indextts.gpt.transformers_beam_search import (
                BeamScorer as _LocalBeamScorer,
                BeamSearchScorer as _LocalBSS,
                ConstrainedBeamSearchScorer as _LocalCBSS,
            )
            beam_search_mod.BeamScorer = _LocalBeamScorer
            beam_search_mod.BeamSearchScorer = _LocalBSS
            beam_search_mod.ConstrainedBeamSearchScorer = _LocalCBSS
            print("[BSAI_IndexTTS2.5] Replaced compat BeamSearchScorer with indextts local implementation (is_done)")
        except ImportError:
            # Local module not available yet — add is_done property as fallback
            _orig_init = BSS.__init__

            def _patched_init(self, *args, **kwargs):
                _orig_init(self, *args, **kwargs)
                if not hasattr(self, '_done'):
                    import torch as _torch
                    batch_size = getattr(self, 'batch_size', 1)
                    num_beam_groups = getattr(self, 'num_beam_groups', 1)
                    device = getattr(self, 'device', 'cpu')
                    self._done = _torch.tensor(
                        [False for _ in range(batch_size * num_beam_groups)],
                        dtype=_torch.bool, device=device,
                    )

            @property
            def _is_done(self):
                return self._done.all()

            BSS.__init__ = _patched_init
            BSS.is_done = _is_done
            print("[BSAI_IndexTTS2.5] Added is_done property to compat BeamSearchScorer (fallback)")
    except Exception as e:
        print(f"[BSAI_IndexTTS2.5] Warning: could not patch BeamSearchScorer.is_done: {e}")


def _patch_bigvgan_compat():
    """Patch BigVGAN._from_pretrained for huggingface_hub 1.0+ compatibility.

    huggingface_hub 1.0+ removed 'proxies' and 'resume_download' from the
    from_pretrained -> _from_pretrained call chain, but BigVGAN defines them
    as required keyword-only arguments (no default), causing:
      TypeError: BigVGAN._from_pretrained() missing 2 required keyword-only
      arguments: 'proxies' and 'resume_download'

    This wrapper injects defaults (proxies=None, resume_download=False) so the
    method works on both huggingface_hub 0.x and 1.x. The hf_hub_download call
    inside _from_pretrained is unaffected — huggingface_hub 1.x silently
    ignores these deprecated parameters via smoothly_deprecate_legacy_arguments.
    """
    try:
        import huggingface_hub
        hf_major = int(huggingface_hub.__version__.split('.')[0])
    except Exception:
        return

    if hf_major < 1:
        return  # huggingface_hub < 1.0 — no patch needed

    patched_modules = []

    # Patch the BigVGAN used by infer_v2_5.py
    for _mod_path in [
        "indextts.s2mel.modules.bigvgan.bigvgan",
        "indextts.BigVGAN.bigvgan",
    ]:
        try:
            _mod = __import__(_mod_path, fromlist=["BigVGAN"])
            _BigVGAN = getattr(_mod, "BigVGAN", None)
            if _BigVGAN is None:
                continue
            _orig_fp = _BigVGAN._from_pretrained

            # Skip if already patched
            if getattr(_orig_fp, "_bsai_patched", False):
                patched_modules.append(f"{_mod_path} (already patched)")
                continue

            @classmethod
            def _patched_from_pretrained(cls, **kwargs):
                kwargs.setdefault("proxies", None)
                kwargs.setdefault("resume_download", False)
                return _orig_fp.__func__(cls, **kwargs)

            _patched_from_pretrained._bsai_patched = True
            _BigVGAN._from_pretrained = _patched_from_pretrained
            patched_modules.append(_mod_path)
        except Exception:
            pass

    if patched_modules:
        print(f"[BSAI_IndexTTS2.5] Patched BigVGAN._from_pretrained for huggingface_hub {huggingface_hub.__version__}: {patched_modules}")


def _patch_librosa_numba_compat():
    """Patch librosa.zero_crossings to avoid numba @stencil bug with numpy 2.x.

    numba 0.66.0 + numpy 2.x has a bug where the @stencil decorator internally
    calls np.empty(shape, dtype=bool) which numba cannot compile, causing:
      TypeError: No implementation of function Function(<built-in function empty>)
      found for signature: empty(UniTuple(int64 x 1), dtype=Function(<class 'bool'>))

    This replaces librosa's numba-accelerated zero_crossings with a pure numpy
    implementation that produces identical results.
    """
    try:
        import numba
        import numpy as _np
        numba_major = int(numba.__version__.split('.')[0])
        numpy_major = int(_np.__version__.split('.')[0])
    except Exception:
        return

    # Only patch if numba + numpy 2.x (the buggy combination)
    if numpy_major < 2:
        return

    try:
        import librosa.core.audio as _lca

        # Check if already patched
        if getattr(_lca.zero_crossings, '_bsai_patched', False):
            return

        def _zero_crossings_numpy(
            y,
            *,
            threshold=1e-10,
            ref_magnitude=None,
            pad=True,
            zero_pos=True,
            axis=-1,
        ):
            """Pure numpy replacement for librosa.zero_crossings.

            Reproduces the exact semantics of the numba @stencil version:
            - Threshold: values with |x| <= threshold are clipped to 0
            - zero_pos=True: uses signbit (0 treated as positive)
            - zero_pos=False: uses sign (0 distinct from +/-1)
            - Output[i] = sign(x[i]) != sign(x[i-1]) for i > 0
            - Output[0] = pad
            """
            if callable(ref_magnitude):
                threshold = threshold * ref_magnitude(_np.abs(y))
            elif ref_magnitude is not None:
                threshold = threshold * ref_magnitude

            yi = y.swapaxes(-1, axis)

            # Clip values within threshold to 0 (same as stencil)
            x = yi.copy()
            mask = _np.abs(x) <= threshold
            x[mask] = 0

            # Compute sign for each element
            if zero_pos:
                signs = _np.signbit(x)
            else:
                signs = _np.sign(x)

            # Zero crossing: sign change between consecutive samples
            z = _np.empty(yi.shape, dtype=bool)
            z[..., 1:] = signs[..., 1:] != signs[..., :-1]
            z[..., 0] = pad

            # Swap back to original axis
            z = z.swapaxes(-1, axis)
            return z

        _zero_crossings_numpy._bsai_patched = True

        # Also patch the librosa top-level import
        import librosa
        _lca.zero_crossings = _zero_crossings_numpy
        librosa.zero_crossings = _zero_crossings_numpy

        print(f"[BSAI_IndexTTS2.5] Patched librosa.zero_crossings with pure numpy implementation (numba {numba.__version__} + numpy {_np.__version__})")
    except Exception as e:
        print(f"[BSAI_IndexTTS2.5] Warning: could not patch librosa.zero_crossings: {e}")


# ---------------------------------------------------------------------------
#  Isolated venv support: run indextts in a separate Python process with
#  old transformers/accelerate/tokenizers to avoid conflicts with nunchaku.
# ---------------------------------------------------------------------------
_VENV_PYTHON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "python_envs", "indextts_env", "Scripts", "python.exe"
)

_BRIDGE_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "indextts_bridge.py")


class IndexTTSBridge:
    """Wrapper that communicates with the indextts bridge subprocess."""

    def __init__(self, proc):
        self._proc = proc

    def _send(self, msg):
        self._proc.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()

    def _recv(self):
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("Bridge process closed unexpectedly")
        return json.loads(line)

    def infer(self, **kwargs):
        msg = {"cmd": "synthesize"}
        msg.update(kwargs)
        self._send(msg)
        resp = self._recv()
        if resp.get("status") != "ok":
            raise RuntimeError(resp.get("msg", "Unknown bridge error"))
        return resp.get("output")

    def close(self):
        try:
            self._send({"cmd": "unload"})
            self._recv()
        except Exception:
            pass
        try:
            self._proc.terminate()
            self._proc.wait(timeout=10)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass

    def is_alive(self):
        return self._proc.poll() is None


def _venv_has_indextts(venv_python):
    """Check that the dedicated venv is actually usable (indextts + torch present).

    A venv directory may exist but be an empty/broken shell (missing torch or
    indextts), e.g. when the venv was created but its install never completed.
    Using it would make the bridge subprocess die with
    `ModuleNotFoundError: No module named 'indextts'`. Probe it cheaply and
    return False so the caller falls back to the main embedded python.
    """
    try:
        import subprocess
        r = subprocess.run(
            [venv_python, "-c", "import torch, indextts"],
            capture_output=True, timeout=60,
        )
        return r.returncode == 0
    except Exception:
        return False


def _get_venv_python():
    """Return the python to run the bridge subprocess in.

    Priority: dedicated venv (old transformers) -> current embedded python.
    The bridge runs synthesis in a *clean subprocess*, isolating it from
    ComfyUI-process-wide environment factors (custom-node patches, benchmark
    caches, offload streams...) that can corrupt BigVGAN output into a 1-bit
    square wave on this user's setup. Set BSAI_INDEXTTS_DIRECT=1 to force
    in-process synthesis (debug only).
    """
    if os.environ.get("BSAI_INDEXTTS_DIRECT", "0") == "1":
        return None  # debug: force direct in-process synthesis
    if (os.path.isfile(_VENV_PYTHON) and os.path.isfile(_BRIDGE_SCRIPT)
            and _venv_has_indextts(_VENV_PYTHON)):
        return _VENV_PYTHON
    if sys.executable:
        return sys.executable  # embedded python already ships indextts
    return None


def _start_bridge(use_bf16, device, use_qwen_emo):
    """Start the bridge subprocess in the isolated venv."""
    model_dir = ensure_model_available()
    if model_dir is None:
        raise RuntimeError("IndexTTS-2.5 model not available.")
    cfg_path = get_config_path(model_dir)
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    venv_python = _get_venv_python()
    print(f"[BSAI_IndexTTS2.5] Bridge python: {venv_python}")
    print(f"[BSAI_IndexTTS2.5] Bridge script: {_BRIDGE_SCRIPT}")

    import subprocess
    proc = subprocess.Popen(
        [venv_python, _BRIDGE_SCRIPT],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        cwd=os.path.dirname(_BRIDGE_SCRIPT),
    )

    # Send load command
    load_msg = {
        "cmd": "load",
        "cfg_path": cfg_path,
        "model_dir": model_dir,
        "use_bf16": use_bf16,
        "device": device,
        "use_qwen_emo": use_qwen_emo,
        # 与直连路径一致：禁用 BigVGAN 自定义 CUDA kernel（RTX 50 上数值错误→方波）
        "use_cuda_kernel": False,
    }
    proc.stdin.write(json.dumps(load_msg, ensure_ascii=False) + "\n")
    proc.stdin.flush()

    # Read stderr in a thread to avoid blocking
    import threading
    stderr_lines = []
    def _read_stderr():
        for line in proc.stderr:
            stderr_lines.append(line)
            print(f"[IndexTTS-Bridge] {line.rstrip()}")
    t = threading.Thread(target=_read_stderr, daemon=True)
    t.start()

    # Wait for response
    resp_line = proc.stdout.readline()
    if not resp_line:
        raise RuntimeError("Bridge process closed during model load")
    resp = json.loads(resp_line)
    if resp.get("status") != "ok":
        raise RuntimeError(f"Bridge load failed: {resp.get('msg', 'unknown')}\n{resp.get('tb', '')}")

    print("[BSAI_IndexTTS2.5] Model loaded successfully (via bridge)!")
    return IndexTTSBridge(proc)


def _get_indextts(use_bf16=True, device=None, use_qwen_emo=False):
    """Get or create the IndexTTS singleton instance."""
    global _INDEXTTS_INSTANCE, _INDEXTTS_MODEL_DIR

    if _INDEXTTS_INSTANCE is not None:
        if isinstance(_INDEXTTS_INSTANCE, IndexTTSBridge):
            if _INDEXTTS_INSTANCE.is_alive():
                return _INDEXTTS_INSTANCE
            else:
                _INDEXTTS_INSTANCE = None
                _INDEXTTS_MODEL_DIR = None
        else:
            return _INDEXTTS_INSTANCE

    # --- Bridge subprocess path (clean process: immune to ComfyUI-process-wide
    #     environment factors that corrupt BigVGAN output into square waves) ---
    if _get_venv_python() is not None:
        print("[BSAI_IndexTTS2.5] Starting bridge subprocess for isolated synthesis...")
        tts = _start_bridge(use_bf16=use_bf16, device=device, use_qwen_emo=use_qwen_emo)
        _INDEXTTS_INSTANCE = tts
        _INDEXTTS_MODEL_DIR = ensure_model_available()
        return tts

    # --- Direct import path (fallback: use main Python environment) ---
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
    print(f"[BSAI_IndexTTS2.5] BF16: {use_bf16}, Device: {device or 'auto'}, QwenEmo: {use_qwen_emo}")

    # Proactively inject missing transformers symbols before importing indextts.
    # This ensures ExtensionsTrie (moved in transformers 5.x) is available even
    # if the _compat.py shim hasn't been loaded yet.
    _ensure_transformers_compat()

    # Patch PretrainedConfig._get_non_default_generation_parameters for transformers 5.x
    _patch_transformers_config_compat()

    # Patch GenerationConfig/PretrainedConfig missing attributes for transformers 5.x
    _patch_generation_config_compat()

    # Patch BeamSearchScorer.is_done for transformers 5.x
    _patch_beam_search_compat()

    # Patch BigVGAN for huggingface_hub 1.0+ compatibility (proxies/resume_download)
    _patch_bigvgan_compat()

    # Patch librosa.zero_crossings for numba/numpy 2.x compatibility
    _patch_librosa_numba_compat()

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

        # Clear cached indextts modules so the retry gets a fresh import.
        # The first import may have cached a broken indextts module (without the
        # compat shim), which would cause the retry to fail even after the shim
        # is applied on disk.
        _clear_indextts_modules()

        # Re-inject transformers compat symbols (in case install changed anything)
        _ensure_transformers_compat()

        # Re-apply PretrainedConfig compat patch after install
        _patch_transformers_config_compat()

        # Re-apply GenerationConfig/PretrainedConfig attribute compat patch after install
        _patch_generation_config_compat()

        # Re-apply BeamSearchScorer compat patch after install
        _patch_beam_search_compat()

        # Re-apply BigVGAN compat patch after install
        _patch_bigvgan_compat()

        # Re-apply librosa numba compat patch after install
        _patch_librosa_numba_compat()

        # Belt-and-suspenders: ensure critical dependencies are installed
        # even if install.py missed them (e.g., omegaconf, einops, etc.)
        _critical_deps = {
            "omegaconf": "omegaconf",
            "einops": "einops",
            "librosa": "librosa",
            "jieba": "jieba",
            "modelscope": "modelscope",
        }
        _missing_critical = []
        for _imp_name, _pip_name in _critical_deps.items():
            try:
                __import__(_imp_name)
            except ImportError:
                _missing_critical.append(_pip_name)
        if _missing_critical:
            print(f"[BSAI_IndexTTS2.5] Installing critical missing deps: {_missing_critical}")
            subprocess.run(
                [sys.executable, "-m", "pip", "install"] + _missing_critical,
                capture_output=True, text=True, check=False,
            )

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
        use_qwen_emo=use_qwen_emo,
        # 关键修复：强制禁用 BigVGAN 自定义 CUDA kernel。
        # 库在 device=None(Loader 'auto') 时会默认 use_cuda_kernel=True 并加载
        # anti_alias_activation_cuda，该 kernel 在 RTX 50/torch2.13 上数值错误，
        # 导致 vocoder 输出爆炸->满幅方波（且极慢）。禁用后走官方 torch 实现，输出正常。
        use_cuda_kernel=False,
    )

    _INDEXTTS_INSTANCE = tts
    _INDEXTTS_MODEL_DIR = model_dir
    print("[BSAI_IndexTTS2.5] Model loaded successfully!")
    return tts


def _unload_indextts():
    """Unload the IndexTTS model to free VRAM."""
    global _INDEXTTS_INSTANCE, _INDEXTTS_MODEL_DIR
    if _INDEXTTS_INSTANCE is not None:
        if isinstance(_INDEXTTS_INSTANCE, IndexTTSBridge):
            _INDEXTTS_INSTANCE.close()
        else:
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


def _is_degenerate_audio(waveform):
    """Detect numerically-broken TTS output (the '1-bit square wave' failure).

    When the GPT/CFM pipeline hits a numerical blowup (Inf/NaN), the vocoder
    output amplitude explodes and the library's int16 conversion clamps it to a
    rail-to-rail square wave: values only {-1, 0, +1} with a very high fraction
    of samples at full scale, or implausibly loud RMS. Healthy speech never
    exceeds a few percent of clipped samples and sits around -24..-15 dBFS.

    Returns (degenerate: bool, reason: str).
    """
    if waveform is None or waveform.numel() == 0:
        return True, "empty output"
    w = waveform.float()
    if not bool(torch.isfinite(w).all()):
        return True, "waveform contains NaN/Inf"
    x = w.abs()
    sat_frac = float((x > 0.999).float().mean().item())
    rms = float(torch.sqrt((w ** 2).mean()).item())
    if sat_frac > 0.3:
        return True, f"clipped-to-rail ({sat_frac * 100:.1f}% samples at full scale)"
    if rms > 0.25:
        return True, f"implausible level (rms={rms:.3f}, > -12dBFS)"
    return False, "ok"


def _save_audio_file(path, waveform, sample_rate, fmt=None):
    """Save audio file, using soundfile as fallback when torchcodec is unavailable.

    torchaudio 2.11+ requires torchcodec for save/load. This helper tries
    torchaudio first, then falls back to soundfile (which is pre-installed).
    """
    # ComfyUI AUDIO waveform is [batch, channels, samples] (3D); torchaudio.save
    # accepts only 1D (mono) or 2D (channels, samples), so normalize here first.
    wav_t = waveform.cpu()
    if wav_t.dim() == 3:
        wav_t = wav_t[0]  # [batch, channels, samples] -> [channels, samples]
    elif wav_t.dim() == 1:
        wav_t = wav_t.unsqueeze(0)  # [samples] -> [1, samples]

    # Try torchaudio.save first (uses torchcodec if available)
    try:
        if fmt:
            torchaudio.save(path, wav_t, sample_rate, format=fmt)
        else:
            torchaudio.save(path, wav_t, sample_rate)
        return
    except (ImportError, RuntimeError, ValueError):
        pass

    # Fallback to soundfile
    import soundfile as sf
    wav_np = wav_t.numpy()
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
#  Node 0: BSAI_IndexTTS2.5LoadAudio (self-contained audio loader)
# ===========================================================================
class BSAI_IndexTTS2_5LoadAudio:
    """
    Load an audio file from ComfyUI's input directory.
    Self-contained — does not depend on ComfyUI built-in audio nodes.
    Supports file upload via ComfyUI's standard upload widget.
    """

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if f.lower().endswith(
            ('.wav', '.mp3', '.flac', '.ogg', '.m4a'))] if os.path.isdir(input_dir) else []
        return {
            "required": {
                "audio": (sorted(files) if files else ["upload_your_audio_file.wav"], ),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "load_audio"
    CATEGORY = "BSAI"
    OUTPUT_NODE = True

    @classmethod
    def VALIDATE_INPUTS(cls, audio):
        # Don't reject placeholder filenames during validation — users will
        # upload their own files before executing. File existence is checked
        # at execution time in load_audio() with a clear error message.
        return True

    @classmethod
    def IS_CHANGED(cls, audio):
        try:
            import hashlib
            audio_path = folder_paths.get_annotated_filepath(audio)
            m = hashlib.sha256()
            with open(audio_path, 'rb') as f:
                m.update(f.read())
            return m.digest().hex()
        except Exception:
            return ""

    def load_audio(self, audio):
        if not audio or audio.startswith("upload_"):
            raise ValueError("Please upload an audio file using the upload button, or select one from the dropdown.")

        audio_path = folder_paths.get_annotated_filepath(audio)
        if not os.path.isfile(audio_path):
            # Try input directory directly
            input_dir = folder_paths.get_input_directory()
            audio_path = os.path.join(input_dir, audio)
            if not os.path.isfile(audio_path):
                # Generate a silent placeholder so the workflow can execute
                # without crashing. The user will see a warning and should
                # upload their own reference audio for proper voice cloning.
                print(f"[BSAI_IndexTTS2.5] WARNING: Audio file '{audio}' not found. "
                      f"Using a 3-second silent placeholder. "
                      f"Please upload your own audio file for voice cloning.")
                sample_rate = 22050
                duration = 3
                waveform = torch.zeros(1, duration * sample_rate)
                audio_dict = {
                    "waveform": waveform.unsqueeze(0),
                    "sample_rate": sample_rate,
                }
                return (audio_dict,)

        print(f"[BSAI_IndexTTS2.5] Loading audio: {audio_path}")

        # Load audio (torchaudio first, soundfile fallback)
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
        except (ImportError, RuntimeError):
            import soundfile as sf
            data, sr = sf.read(audio_path)
            waveform = torch.from_numpy(data).float()
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.T
            sample_rate = sr

        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # ComfyUI AUDIO format: {"waveform": tensor[batch, channels, samples], "sample_rate": int}
        audio_dict = {
            "waveform": waveform.unsqueeze(0),  # Add batch dim: [1, channels, samples]
            "sample_rate": sample_rate,
        }
        print(f"[BSAI_IndexTTS2.5] Audio loaded: sr={sample_rate}, duration={waveform.shape[-1]/sample_rate:.2f}s")
        return (audio_dict,)


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
                # 默认 fp32：官方 CLI 默认精度，数值最稳定。BF16 在部分环境(如 RTX 50 + ComfyUI --fast)
                # 下可能偶发数值爆炸导致输出退化为满幅方波，需用时手动开启，且合成节点带自动降级重试保护。
                "use_bf16": ("BOOLEAN", {"default": False}),
                "device": (["auto", "cuda:0", "cpu"], {"default": "auto"}),
            },
            "optional": {
                "use_qwen_emo": ("BOOLEAN", {"default": False}),
                "force_reload": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("INDEX_TTS_MODEL",)
    RETURN_NAMES = ("tts_model",)
    FUNCTION = "load_model"
    CATEGORY = "BSAI"

    def load_model(self, use_bf16=True, device="auto", use_qwen_emo=False, force_reload=False):
        if force_reload:
            _unload_indextts()

        device = None if device == "auto" else device
        tts = _get_indextts(use_bf16=use_bf16, device=device, use_qwen_emo=use_qwen_emo)
        return (tts,)


# ===========================================================================
#  Node 2: BSAI_IndexTTS2.5Synthesis
# ===========================================================================
class BSAI_IndexTTS2_5Synthesis:
    """
    IndexTTS-2.5 voice cloning synthesis node.
    Generate speech from text using a reference audio for voice cloning.
    Supports cross-language synthesis, emotion control, and speed control.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tts_model_TTS模型": ("INDEX_TTS_MODEL",),
                # 默认文本与默认语言(ZH)对齐，避免 英文文本+ZH 语言 的失配组合诱发退化生成
                "text": ("STRING", {
                    "default": "同学们，今天我们来学习百声电影工厂的声音克隆功能。",
                    "multiline": True,
                }),
                "reference_audio": ("AUDIO",),
                "lang": (["ZH", "EN", "JA", "ES", "AR", "zhen"], {"default": "ZH"}),
            },
            "optional": {
                # --- Emotion control ---
                "emo_audio_prompt": ("AUDIO", ),
                "emo_alpha": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                }),
                "emo_vector": ("EMO_VECTOR", ),
                "use_emo_text": ("BOOLEAN", {"default": False}),
                "emo_text": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
                "use_random": ("BOOLEAN", {"default": False}),
                # --- Speed control ---
                "duration_factor": ("FLOAT", {
                    "default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05,
                }),
                # --- Generation parameters ---
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
                "top_k_TopK采样": ("INT", {
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
        tts_model_TTS模型,
        text,
        reference_audio,
        lang="ZH",
        emo_audio_prompt=None,
        emo_alpha=1.0,
        emo_vector=None,
        use_emo_text=False,
        emo_text="",
        use_random=False,
        duration_factor=1.0,
        max_text_tokens_per_segment=100,
        max_mel_tokens=1500,
        temperature=0.8,
        top_p=0.8,
        top_k_TopK采样=30,
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

        # --- Handle emotion reference audio ---
        emo_audio_path = None
        if emo_audio_prompt is not None:
            emo_wf = emo_audio_prompt["waveform"]
            emo_sr = emo_audio_prompt["sample_rate"]
            if emo_wf.dim() == 3:
                emo_wf = emo_wf[0]
            elif emo_wf.dim() == 1:
                emo_wf = emo_wf.unsqueeze(0)
            emo_audio_path = os.path.join(temp_dir, "emotion_ref.wav")
            _save_audio_file(emo_audio_path, emo_wf, emo_sr)
            print(f"[BSAI_IndexTTS2.5] Emotion reference audio saved: {emo_audio_path} (sr={emo_sr})")

        # --- Handle emotion text ---
        # Empty string means no custom emotion text
        emo_text_param = emo_text.strip() if emo_text and emo_text.strip() else None

        # --- Handle emotion vector ---
        # emo_vector comes from BSAI_IndexTTS2.5EmotionVector node (a list of 8 floats)
        # or None if not connected
        emo_vector_param = emo_vector if emo_vector is not None else None

        # Build generation kwargs for v2.5 infer()
        generation_kwargs = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k_TopK采样,
            "length_penalty": length_penalty,
            "num_beams": num_beams,
            "repetition_penalty": repetition_penalty,
            "max_mel_tokens": max_mel_tokens,
            "do_sample": do_sample,
        }

        # Log emotion settings
        emotion_mode = "default (from speaker voice)"
        if emo_vector_param is not None:
            emotion_mode = f"emotion vector: {emo_vector_param}"
        elif use_emo_text:
            emotion_mode = f"text-based emotion (emo_text={'auto' if emo_text_param is None else repr(emo_text_param[:50])})"
        elif emo_audio_path is not None:
            emotion_mode = f"emotion reference audio (alpha={emo_alpha})"
        print(f"[BSAI_IndexTTS2.5] Emotion mode: {emotion_mode}")
        print(f"[BSAI_IndexTTS2.5] Duration factor: {duration_factor}, Use random: {use_random}")

        # 数值爆炸防护：退化输出(1bit 方波/NaN)检测 + 自动降级重试。
        # 首次退化若当前为 BF16 实例 → 卸载并按 fp32 重载重试；仍失败则明确报错，绝不静默保存废音频。
        try:
            attempt = 0
            max_attempts = 2
            while True:
                print(f"[BSAI_IndexTTS2.5] Synthesizing (lang={lang})...")
                # ---- BSAI 协同：GPU TTS 推理租约（失败不阻断合成，仅叠加租约管理） ----
                _bsai_alloc = None
                try:
                    _bsai_alloc = BSAIOrch.allocate("tts_infer", requester="8191")
                    if not _bsai_alloc.ok:
                        print(f"[BSAI_IndexTTS2.5] SDK 未取得 tts_infer 租约({_bsai_alloc.reason})，仍按原逻辑合成")
                except Exception as _bsai_ae:
                    print(f"[BSAI_IndexTTS2.5] SDK allocate 异常(忽略): {_bsai_ae}")
                    _bsai_alloc = None
                try:
                    tts_model_TTS模型.infer(
                        spk_audio_prompt=ref_audio_path,
                        text=text,
                        output_path=out_audio_path,
                        lang=lang,
                        emo_audio_prompt=emo_audio_path,
                        emo_alpha=emo_alpha,
                        emo_vector=emo_vector_param,
                        use_emo_text=use_emo_text,
                        emo_text=emo_text_param,
                        use_random=use_random,
                        duration_factor=duration_factor,
                        verbose=verbose,
                        max_text_tokens_per_segment=max_text_tokens_per_segment,
                        **generation_kwargs,
                    )
                finally:
                    if _bsai_alloc is not None:
                        try:
                            _bsai_alloc.release()
                        except Exception:
                            pass

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

                # 退化输出检测（满幅方波 / NaN / 爆响），健康语音削波占比接近 0
                is_degenerate, reason = _is_degenerate_audio(audio_waveform)
                if not is_degenerate:
                    # Return as ComfyUI AUDIO dict
                    audio_dict = _audio_tensor_to_dict(audio_waveform, audio_sr)

                    duration = audio_waveform.shape[-1] / audio_sr
                    status_parts = [
                        f"Success | Duration: {duration:.2f}s",
                        f"Sample Rate: {audio_sr}Hz",
                        f"Lang: {lang}",
                        f"Emotion: {emotion_mode}",
                        f"Speed: {duration_factor:.2f}x",
                    ]
                    status = " | ".join(status_parts)
                    print(f"[BSAI_IndexTTS2.5] {status}")

                    return (audio_dict, status)

                # ---- degenerate output: save a diagnostic copy first ----
                try:
                    _diag_dir = folder_paths.get_output_directory()
                    _diag_path = os.path.join(
                        _diag_dir,
                        f"BSAI_IndexTTS25_degenerate_{time.strftime('%Y%m%d_%H%M%S')}.wav",
                    )
                    _save_audio_file(_diag_path, audio_waveform, audio_sr)
                    print(f"[BSAI_IndexTTS2.5] 退化样本已留存诊断: {_diag_path}")
                except Exception:
                    pass

                # ---- rescue: implausible level (too loud but NOT rail-clipped) ----
                # Peak-normalize and verify speech-like morphology (low zero-crossing
                # rate + rich unique values). Loud-but-healthy speech passes; real
                # noise/square waves still fail and are rejected.
                if "implausible" in reason:
                    _pk = audio_waveform.abs().max().clamp_min(1e-6)
                    _norm = audio_waveform / _pk * 0.9
                    _nf = _norm.float()
                    _finite_ok = bool(torch.isfinite(_nf).all())
                    _sat_n = float((_nf.abs() > 0.999).float().mean().item())
                    _z = _nf.flatten()
                    _zcr = float(((_z[1:] * _z[:-1] < 0).float().mean()).item())
                    _uniq = len(torch.unique(torch.round(_z, decimals=3)))
                    if _finite_ok and _sat_n < 0.3 and _zcr < 0.25 and _uniq > 80:
                        _norm_rms = float(torch.sqrt((_nf ** 2).mean()).item())
                        print(
                            f"[BSAI_IndexTTS2.5] 高电平输出({reason})，峰值正常化后通过"
                            f"(rms={_norm_rms:.3f}, zcr={_zcr:.3f}, uniq={_uniq})，已放行"
                        )
                        audio_waveform = _norm
                        audio_dict = _audio_tensor_to_dict(audio_waveform, audio_sr)
                        duration = audio_waveform.shape[-1] / audio_sr
                        status = (
                            f"Success(normalized) | Duration: {duration:.2f}s | "
                            f"Sample Rate: {audio_sr}Hz | Lang: {lang} | "
                            f"Raw rms: {float(torch.sqrt((audio_waveform.float() ** 2).mean())):.3f}"
                        )
                        print(f"[BSAI_IndexTTS2.5] {status}")
                        return (audio_dict, status)

                # 退化 → 重试
                attempt += 1
                print(f"[BSAI_IndexTTS2.5] WARNING: 检测到退化音频 ({reason})，触发保护重试 {attempt}/{max_attempts}...")
                if attempt >= max_attempts:
                    raise RuntimeError(
                        f"IndexTTS-2.5 连续 {max_attempts} 次生成退化音频（{reason}），已拒绝输出。"
                        f"建议：1) 保持 Loader 节点 use_bf16 关闭（fp32 数值最稳定）；"
                        f"2) 更换参考音频或缩短文本后重试。"
                    )
                if getattr(tts_model_TTS模型, "dtype", None) is not None:
                    _old_dev = getattr(tts_model_TTS模型, "device", "auto")
                    _old_emo = getattr(tts_model_TTS模型, "qwen_emo", None)
                    _unload_indextts()
                    tts_model_TTS模型 = _get_indextts(
                        use_bf16=False,
                        device=_old_dev,
                        use_qwen_emo=_old_emo is not None,
                    )
                    print("[BSAI_IndexTTS2.5] 已自动卸载并降级为 fp32 实例，重试合成...")
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
#  Node 2b: BSAI_IndexTTS2.5EmotionVector (helper)
# ===========================================================================
class BSAI_IndexTTS2_5EmotionVector:
    """
    Construct an 8-dimensional emotion vector for IndexTTS-2.5.
    Emotion order: [Happy, Angry, Sad, Fear, Disgust, Melancholy, Surprise, Calm]
    Each value range: 0.0 - 1.0
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "happy": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "angry": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "sad": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "fear": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "disgust": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "melancholy": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "surprise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "calm": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "preset": (["none", "happy", "angry", "sad", "fear", "disgust",
                            "melancholy", "surprise", "calm"], {"default": "none"}),
            },
        }

    RETURN_TYPES = ("EMO_VECTOR",)
    RETURN_NAMES = ("emo_vector",)
    FUNCTION = "build_vector"
    CATEGORY = "BSAI"

    def build_vector(self, happy, angry, sad, fear, disgust, melancholy, surprise, calm, preset="none"):
        # Apply preset if selected (overrides individual values)
        presets = {
            "happy":      [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "angry":      [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "sad":        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "fear":       [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "disgust":    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "melancholy": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            "surprise":   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "calm":       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        }

        if preset != "none" and preset in presets:
            vec = presets[preset]
            print(f"[BSAI_IndexTTS2.5] Emotion vector preset: {preset} -> {vec}")
        else:
            vec = [happy, angry, sad, fear, disgust, melancholy, surprise, calm]
            print(f"[BSAI_IndexTTS2.5] Emotion vector: {vec}")

        return (vec,)


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
                "mp3_bitrate_MP3码率": ("INT", {"default": 192, "min": 64, "max": 320, "step": 32}),
                "output_gain": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                # 兼容旧版工作流：双语化前该输入名为 audio_音频，改名 audio 后
                # 旧 JSON 仍会传这个关键字，未加兼容会导致
                # save_audio() got an unexpected keyword argument 'audio_音频'
                "audio_音频": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("STRING", "AUDIO",)
    RETURN_NAMES = ("file_path", "audio",)
    FUNCTION = "save_audio"
    CATEGORY = "BSAI"
    OUTPUT_NODE = True

    def save_audio(self, audio, filename_prefix="BSAI_IndexTTS2_5", format="wav", mp3_bitrate_MP3码率=192, output_gain=1.0, audio_音频=None):
        if audio is None:
            audio = audio_音频
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]

        # Apply gain
        if output_gain != 1.0:
            waveform = waveform * output_gain
            # Clamp to valid range
            waveform = torch.clamp(waveform, -1.0, 1.0)

        # Generate output path (confined to ComfyUI's output dir)
        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        output_dir_real = os.path.realpath(output_dir)

        # Find unique filename
        base_name = filename_prefix
        counter = 1
        while True:
            ext = format if format != "wav" else "wav"
            filename = f"{base_name}_{counter:05d}.{ext}"
            filepath = os.path.join(output_dir, filename)
            # Containment: reject absolute paths / ".." escaping the output dir.
            if os.path.commonpath([output_dir_real, os.path.realpath(filepath)]) != output_dir_real:
                raise ValueError(
                    "filename_prefix must stay inside ComfyUI's output directory "
                    "(absolute paths and '..' are not allowed)."
                )
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
        return (filepath, {"waveform": waveform, "sample_rate": sample_rate},)


# ===========================================================================
#  Node 5: BSAI_IndexTTS2.5UnloadModel
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
    "BSAI_IndexTTS2.5LoadAudio": BSAI_IndexTTS2_5LoadAudio,
    "BSAI_IndexTTS2.5Loader": BSAI_IndexTTS2_5Loader,
    "BSAI_IndexTTS2.5Synthesis": BSAI_IndexTTS2_5Synthesis,
    "BSAI_IndexTTS2.5EmotionVector": BSAI_IndexTTS2_5EmotionVector,
    "BSAI_IndexTTS2.5SaveAudio": BSAI_IndexTTS2_5SaveAudio,
    "BSAI_IndexTTS2.5UnloadModel": BSAI_IndexTTS2_5UnloadModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSAI_IndexTTS2.5LoadAudio": "BSAI IndexTTS2.5 Load Audio",
    "BSAI_IndexTTS2.5Loader": "BSAI IndexTTS2.5 Loader",
    "BSAI_IndexTTS2.5Synthesis": "BSAI IndexTTS2.5 Synthesis",
    "BSAI_IndexTTS2.5EmotionVector": "BSAI IndexTTS2.5 Emotion Vector",
    "BSAI_IndexTTS2.5SaveAudio": "BSAI IndexTTS2.5 Save Audio",
    "BSAI_IndexTTS2.5UnloadModel": "BSAI IndexTTS2.5 Unload Model",
}
