# BSAI_ComfyUI_IndexTTS-2.5

ComfyUI custom nodes for [IndexTTS-2.5](https://github.com/index-tts/index-tts) - A Breakthrough in Multilingual, Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech.

## What's New in IndexTTS-2.5

IndexTTS-2.5 is a major upgrade over IndexTTS-2 with four key improvements:

1. **Semantic Codec Compression**: Frame rate reduced from 50Hz to 25Hz, halving sequence length and lowering inference cost
2. **Zipformer Architecture**: Replaces U-DiT backbone with more efficient Zipformer, achieving 2.28x real-time factor improvement
3. **Multilingual Support**: Chinese, English, Japanese, and Spanish with robust cross-lingual emotion transfer
4. **Reinforcement Learning (GRPO)**: Improved pronunciation accuracy and prosodic naturalness

## Features

- **Voice Cloning**: Clone any voice from a short reference audio sample
- **Zero-Shot TTS**: Generate speech in the cloned voice from text
- **Multilingual**: Support for Chinese, English, Japanese, and Spanish
- **Fast Inference Mode**: 2-10x speedup for long text with segment bucketing
- **Auto Model Download**: Automatically downloads models from ModelScope (China) or HuggingFace
- **FP16 Support**: Half-precision inference for lower VRAM usage
- **Audio Output**: Compatible with ComfyUI native audio nodes (PreviewAudio, SaveAudio, etc.)

## Installation

### Method 1: ComfyUI-Manager (Recommended)

1. Open ComfyUI-Manager in your ComfyUI
2. Search for "BSAI_ComfyUI_IndexTTS-2.5"
3. Click Install

The `install.py` script will automatically handle everything: it installs `indextts` from GitHub source, applies the transformers **4.55+/5.x** compatibility patch (`patch_indextts.py`), installs runtime dependencies, and applies the transformers 5.x compatibility shim.

### Method 2: Manual

1. Clone this repository into your `ComfyUI/custom_nodes/` directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/xm6018924/BSAI_ComfyUI_IndexTTS-2.5.git
   ```

2. Run the installation script:
   ```bash
   # Windows (ComfyUI portable)
   .\python\python.exe BSAI_ComfyUI_IndexTTS-2.5/install.py

   # Or use the batch file (auto-detects python.exe location)
   BSAI_ComfyUI_IndexTTS-2.5\install_bsai_indextts.bat
   ```

3. Restart ComfyUI

### Method 3: Manual step-by-step (if install.py fails)

If the automated installation fails, you can install manually:

```bash
# 1. Install hatchling build tool (may not be on Chinese mirrors)
pip install hatchling --index-url https://pypi.org/simple/

# 2. Install indextts from GitHub (NOT on PyPI; indextts 2.0.0)
pip install --no-deps --ignore-requires-python --no-build-isolation git+https://github.com/index-tts/index-tts.git

# 3. Install runtime dependencies (openai-whisper is REQUIRED).
#    NOTE: do NOT install keras / descript-audiotools — they are not used at
#    runtime and keras==2.9.0 conflicts with Python 3.12+.
pip install openai-whisper cn2an fugashi unidic-lite g2p_en json5 munch textstat --index-url https://pypi.org/simple/

# 4. wetext is also required, but needs a C++ toolchain (MSVC + CMake) to build
#    kaldifst on Windows/macOS. If it fails to build, that's OK — the patch below
#    makes indextts degrade gracefully (numbers/dates won't be auto-converted).
pip install wetext --index-url https://pypi.org/simple/

# 5. Apply the transformers compatibility patch (REQUIRED for transformers >= 4.55)
python BSAI_ComfyUI_IndexTTS-2.5/patch_indextts.py

# 6. Fix protobuf version (optional, defensive)
pip install "protobuf>=5.26.1,<6" --index-url https://pypi.org/simple/
```

The transformers compatibility patch (`patch_indextts.py`) and the 5.x shim
(`indextts_compat.py`) are applied automatically when ComfyUI loads the node.

## Compatibility (what the patch fixes)

`indextts` 2.0.0 (installed from `github.com/index-tts/index-tts`) was built for
older transformers. Modern ComfyUI portable builds ship **transformers ≥ 4.55**
(e.g. 4.57), where several internal symbols were removed. Without the patch you
get an `ImportError` at load time or
`AttributeError: 'GenerationConfig' object has no attribute 'forced_decoder_ids'`
at synthesis time.

`patch_indextts.py` makes indextts work on **transformers 4.55+ and 5.x** by
wrapping the removed symbols in `try/except` fallbacks (or using `getattr`):

| Symbol | Where | Fix |
|--------|-------|-----|
| `QuantizedCacheConfig` | `indextts/gpt/transformers_generation_utils.py` | try/except + dataclass placeholder |
| `_crop_past_key_values` | `indextts/gpt/transformers_generation_utils.py` | try/except + compat function |
| `NEED_SETUP_CACHE_CLASSES_MAPPING` / `QUANT_BACKEND_CLASSES_MAPPING` | `indextts/gpt/transformers_generation_utils.py` | try/except → `{}` |
| `forced_decoder_ids` | `indextts/gpt/transformers_generation_utils.py` | `getattr(..., None)` |
| `SequenceSummary` | `indextts/gpt/transformers_gpt2.py` | try/except placeholder |
| `TypicalLogitsWarper` | `indextts/utils/typical_sampling.py` | import from `transformers.generation.logits_process` |
| `wetext` import | `indextts/utils/front.py` | try/except graceful fallback (passthrough normalizer) |

The patch is **idempotent and safe**: re-running it is a no-op if already applied,
and it never fails the install if `indextts` changed upstream (it warns and skips).

> **Note on `wetext`**: if your machine has no C++ build tools (MSVC + CMake),
> `wetext` cannot compile. The patch degrades it to a pass-through normalizer —
> everything still works, only numbers/dates won't be auto-converted to spoken
> text. Install `wetext` separately if you want full normalization.

## Troubleshooting

### Installation fails with "Could not find a version that satisfies the requirement indextts"

**Cause**: `indextts` is not published on PyPI. It must be installed from GitHub source.

**Fix**: Use `install.py` or `install_bsai_indextts.bat`, or follow Method 3 above.

### Installation fails with "Cannot find command 'git'"

**Cause**: Git is not in the system PATH.

**Fix**: Add Git to PATH or specify the full path:
```bash
set PATH=C:\Program Files\Git\cmd;%PATH%
pip install --no-deps --ignore-requires-python --no-build-isolation git+https://github.com/index-tts/index-tts.git
```

### Node fails to load with "cannot import name 'OffloadedCache' from 'transformers.cache_utils'" or "'GenerationConfig' object has no attribute 'forced_decoder_ids'"

**Cause**: `indextts` 2.0.0 was written for older transformers. Starting with **transformers 4.55** (and 5.x) many internal APIs were removed: `QuantizedCacheConfig`, `SequenceSummary`, `forced_decoder_ids`, `TypicalLogitsWarper`, etc.

**Fix**: The compatibility patch (`patch_indextts.py`) handles this automatically — it is run by `install.py` / the batch file, and can also be run manually:
```bash
python BSAI_ComfyUI_IndexTTS-2.5/patch_indextts.py
```
If you still see the error:
1. Ensure `patch_indextts.py` and `indextts_compat.py` exist in the node directory
2. Restart ComfyUI (patches are applied on node load / install)
3. Or manually run: `python install.py`

### "hatchling" build dependency not found

**Cause**: The Chinese PyPI mirror (清华源) may not have `hatchling>=1.27.0`.

**Fix**: Use the official PyPI index:
```bash
pip install hatchling --index-url https://pypi.org/simple/
```

### protobuf version conflicts

**Cause**: `descript-audiotools` requires protobuf <3.20, but other packages need protobuf >=4.

**Fix**: Use protobuf 5.x (works at runtime despite the version warning):
```bash
pip install "protobuf>=5.26.1,<6"
```

### Python version incompatibility (Python 3.12+)

**Cause**: `indextts` requires Python >=3.10,<3.12, but newer ComfyUI environments use Python 3.13+.

**Fix**: Use `--ignore-requires-python` flag when installing indextts:
```bash
pip install --no-deps --ignore-requires-python --no-build-isolation git+https://github.com/index-tts/index-tts.git
```
The compatibility shim handles the transformers API differences; Python 3.13 compatibility is maintained through the shim.

## Model Download

Models are automatically downloaded on first use to `ComfyUI/models/IndexTTS2.5/`.

Download priority:
1. **Local check** - If model already exists locally, use it directly
2. **ModelScope** (preferred for China users) - `IndexTeam/IndexTTS-2.5`
3. **HuggingFace Mirror** - `https://hf-mirror.com` fallback
4. **HuggingFace Direct** - Last resort

### Manual Download

If auto-download fails, you can manually download the model:

```bash
# Via ModelScope (China)
pip install modelscope
modelscope download --model IndexTeam/IndexTTS-2.5 --local_dir ComfyUI/models/IndexTTS2.5

# Via HuggingFace
pip install huggingface_hub[cli]
huggingface-cli download IndexTeam/IndexTTS-2.5 --local-dir ComfyUI/models/IndexTTS2.5

# Via HuggingFace Mirror (China)
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download IndexTeam/IndexTTS-2.5 --local-dir ComfyUI/models/IndexTTS2.5
```

## Nodes

### BSAI IndexTTS2.5 Loader
Loads the IndexTTS-2.5 model. On first run, automatically downloads the model.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| use_fp16 | BOOLEAN | True | Use half-precision for faster inference and lower VRAM |
| device | COMBO | auto | Device: auto, cuda:0, or cpu |
| force_reload | BOOLEAN | False | Force reload model (frees VRAM first) |

**Output**: `INDEX_TTS_MODEL` - The loaded TTS model instance

### BSAI IndexTTS2.5 Synthesis
Generate speech from text using a reference audio for voice cloning.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| tts_model | INDEX_TTS_MODEL | - | Model from Loader |
| text | STRING | "Hello..." | Text to synthesize |
| reference_audio | AUDIO | - | Reference voice audio |
| use_fast_inference | BOOLEAN | True | Fast mode (2-10x speedup) |
| max_text_tokens_per_segment | INT | 100 | Max tokens per segment (fast mode) |
| max_mel_tokens | INT | 600 | Max mel tokens per generation |
| temperature | FLOAT | 1.0 | Sampling temperature |
| top_p | FLOAT | 0.8 | Top-p sampling |
| top_k | INT | 30 | Top-k sampling |
| length_penalty | FLOAT | 0.0 | Length penalty |
| num_beams | INT | 3 | Number of beams |
| repetition_penalty | FLOAT | 10.0 | Repetition penalty |
| remove_silence | BOOLEAN | True | Remove long silences |
| verbose | BOOLEAN | False | Print debug info |

**Outputs**: `audio` (AUDIO) + `status` (STRING)

### BSAI IndexTTS2.5 Save Audio
Save generated audio to disk.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| audio | AUDIO | - | Audio to save |
| filename_prefix | STRING | BSAI_IndexTTS2 | Output filename prefix |
| format | COMBO | wav | Audio format: wav, mp3, flac |
| mp3_bitrate | INT | 192 | MP3 bitrate (kbps) |
| output_gain | FLOAT | 1.0 | Output volume multiplier |

### BSAI IndexTTS2.5 Unload Model
Unload the model to free VRAM.

## Default Workflow

See `workflow_example/default_workflow.json` for a basic voice cloning workflow:

```
LoadAudio (reference) -> BSAI_IndexTTS2.5Synthesis -> PreviewAudio
                          ^                          -> BSAI_IndexTTS2.5SaveAudio
BSAI_IndexTTS2.5Loader ---'
```

## System Requirements

- **GPU**: NVIDIA GPU with at least 4GB VRAM (8GB+ recommended for FP16)
- **RAM**: At least 8GB system RAM
- **Storage**: ~5GB for model files
- **CUDA**: CUDA 12.8+ recommended
- **Python**: 3.10+

## Tips

- Use **FP16** for faster inference and lower VRAM usage
- Use **fast inference** for long text (multiple sentences)
- Keep reference audio short (10-30 seconds) for best results
- Use clean reference audio without background noise
- Supports Chinese, English, Japanese, and Spanish text

## Related Projects

- [IndexTTS](https://github.com/index-tts/index-tts) - Original IndexTTS project
- [BSAI-MiniMAX-H3-Prompt](https://github.com/BSAI-AI/BSAI-MiniMAX-H3-Prompt) - MiniMax H3 prompt optimization
- [BSAI_ComfyUI_Nodes](https://github.com/BSAI-AI/BSAI_ComfyUI_Nodes) - BSAI utility nodes

## License

MIT License - See LICENSE file for details.

## Credits

- IndexTTS-2.5 by [IndexTeam](https://github.com/index-tts) (Bilibili)
- ComfyUI by [comfyanonymous](https://github.com/comfyanonymous/ComfyUI)
