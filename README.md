# BSAI_ComfyUI_IndexTTS-2.5

ComfyUI custom nodes for [IndexTTS-2.5](https://github.com/index-tts/index-tts) - A Breakthrough in Multilingual, Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech.

## What's New in IndexTTS-2.5

IndexTTS-2.5 is a major upgrade over IndexTTS-2 with key improvements:

1. **Multilingual Support**: Chinese, English, Japanese, Spanish, and Arabic with robust cross-lingual emotion transfer
2. **Voice-Emotion Decoupling**: Independent control of speaker timbre and emotional expression
3. **Emotion Control**: Multiple emotion input modalities — reference audio, 8D emotion vector, and text-based emotion via Qwen3
4. **Speed Control**: Precise speech rate adjustment via `duration_factor` (0.5x-2.0x)
5. **Semantic Codec Compression**: Frame rate reduced from 50Hz to 25Hz, halving sequence length
6. **Zipformer Architecture**: 2.28x real-time factor improvement over IndexTTS-2
7. **Reinforcement Learning (GRPO)**: Improved pronunciation accuracy and prosodic naturalness
8. **Pinyin/Phoneme/Kana Control**: Fine-grained pronunciation control via character replacement

## Features

- **Voice Cloning**: Clone any voice from a short reference audio sample
- **Zero-Shot TTS**: Generate speech in the cloned voice from text
- **Multilingual**: Support for Chinese (ZH), English (EN), Japanese (JA), Spanish (ES), and Arabic (AR)
- **Cross-Language Synthesis**: Use a Chinese speaker's voice to synthesize English, Japanese, Spanish, or Arabic text
- **Emotion Control**:
  - **Emotion Reference Audio**: Use a separate audio to control emotion while preserving the speaker's timbre
  - **8D Emotion Vector**: Directly specify emotion intensities for [happy, angry, sad, fear, disgust, melancholy, surprise, calm]
  - **Text-Based Emotion**: Auto-generate emotion vectors from text descriptions via Qwen3 model
  - **Emotion Intensity**: Adjustable blending strength (`emo_alpha`, 0.0-1.0)
- **Speed Control**: Slow down or speed up speech via `duration_factor` (0.5x-2.0x)
- **Random Sampling**: Toggle stochastic sampling for varied outputs
- **Auto Model Download**: Automatically downloads models from ModelScope (China) or HuggingFace
- **BF16 Support**: Half-precision inference for lower VRAM usage
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

## Nodes

### 0. BSAI IndexTTS2.5 Load Audio

Load an audio file from ComfyUI's input directory. Self-contained — does not depend on ComfyUI built-in audio nodes.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| audio_file | COMBO | - | Select audio file from input directory (supports wav, mp3, flac, ogg, m4a) |

**Output**: `AUDIO` - Audio data for use with Synthesis node

> **Tip**: Upload audio files to ComfyUI's `input/` directory, or use the file upload widget in the node.

---

### 1. BSAI IndexTTS2.5 Loader

Loads the IndexTTS-2.5 model. On first run, automatically downloads the model.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| use_bf16 | BOOLEAN | True | Use BF16 half-precision for faster inference and lower VRAM |
| device | COMBO | auto | Device: auto, cuda:0, or cpu |
| use_qwen_emo | BOOLEAN | False | Load Qwen3 emotion model (required for text-based emotion control) |
| force_reload | BOOLEAN | False | Force reload model (frees VRAM first) |

> **Note**: `use_qwen_emo=True` is required when using `use_emo_text` in the Synthesis node. It loads an additional Qwen3 model for text-to-emotion inference, which increases VRAM usage. Enable it only when you need text-based emotion control.

**Output**: `INDEX_TTS_MODEL` - The loaded TTS model instance

---

### 2. BSAI IndexTTS2.5 Synthesis

Generate speech from text using a reference audio for voice cloning. Supports emotion control, speed control, and cross-language synthesis.

#### Required Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| tts_model | INDEX_TTS_MODEL | - | Model from Loader |
| text | STRING | "Hello..." | Text to synthesize |
| reference_audio | AUDIO | - | Speaker voice reference audio |
| lang | COMBO | ZH | Language: ZH, EN, JA, ES, AR, zhen |

#### Emotion Control Parameters (Optional)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| emo_audio_prompt | AUDIO | - | Emotion reference audio (voice-emotion decoupling). When connected, the speaker's timbre comes from `reference_audio` and the emotion comes from this audio |
| emo_alpha | FLOAT | 1.0 | Emotion blending strength (0.0-1.0). 1.0 = full emotion from reference, 0.0 = no emotion influence |
| emo_vector | EMO_VECTOR | - | Direct 8D emotion vector from EmotionVector node. Overrides emo_audio_prompt |
| use_emo_text | BOOLEAN | False | Auto-generate emotion from text via Qwen3 (requires use_qwen_emo=True in Loader) |
| emo_text | STRING | "" | Custom emotion description text. When empty and use_emo_text=True, uses the main text as emotion input |
| use_random | BOOLEAN | False | Enable random sampling (increases variation but may reduce voice cloning fidelity) |

#### Speed Control Parameters (Optional)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| duration_factor | FLOAT | 1.0 | Speech rate multiplier. >1.0 = slower, <1.0 = faster. Range: 0.5-2.0 |

#### Generation Parameters (Optional)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| max_text_tokens_per_segment | INT | 100 | Max tokens per segment |
| max_mel_tokens | INT | 1500 | Max mel tokens per generation |
| temperature | FLOAT | 0.8 | Sampling temperature |
| top_p | FLOAT | 0.8 | Top-p sampling |
| top_k | INT | 30 | Top-k sampling |
| length_penalty | FLOAT | 0.0 | Length penalty |
| num_beams | INT | 3 | Number of beams |
| repetition_penalty | FLOAT | 10.0 | Repetition penalty |
| do_sample | BOOLEAN | True | Use sampling instead of greedy decoding |
| verbose | BOOLEAN | False | Print debug info |

**Outputs**: `audio` (AUDIO) + `status` (STRING)

#### Emotion Control Priority

When multiple emotion inputs are provided, the priority is:

1. **emo_vector** (highest) — Direct 8D vector overrides everything
2. **use_emo_text** — Text-based emotion via Qwen3 overrides emo_audio_prompt
3. **emo_audio_prompt** — Emotion reference audio
4. **Default** — Uses speaker's voice as emotion reference (emo_alpha forced to 1.0)

---

### 3. BSAI IndexTTS2.5 Emotion Vector

Helper node to construct an 8-dimensional emotion vector for the Synthesis node.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| happy | FLOAT | 0.0 | Happiness intensity (0.0-1.0) |
| angry | FLOAT | 0.0 | Anger intensity (0.0-1.0) |
| sad | FLOAT | 0.0 | Sadness intensity (0.0-1.0) |
| fear | FLOAT | 0.0 | Fear intensity (0.0-1.0) |
| disgust | FLOAT | 0.0 | Disgust intensity (0.0-1.0) |
| melancholy | FLOAT | 0.0 | Melancholy intensity (0.0-1.0) |
| surprise | FLOAT | 0.0 | Surprise intensity (0.0-1.0) |
| calm | FLOAT | 0.0 | Calmness intensity (0.0-1.0) |
| preset | COMBO | none | Quick preset: none, happy, angry, sad, fear, disgust, melancholy, surprise, calm |

> **Tip**: When a preset is selected (not "none"), it overrides the individual emotion sliders with a single-emotion vector (value 1.0 for the selected emotion, 0.0 for others). Use "none" to manually mix multiple emotions.

**Output**: `EMO_VECTOR` - Connect to the `emo_vector` input of the Synthesis node

**Emotion Vector Order**: `[happy, angry, sad, fear, disgust, melancholy, surprise, calm]`

---

### 4. BSAI IndexTTS2.5 Save Audio

Save generated audio to disk.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| audio | AUDIO | - | Audio to save |
| filename_prefix | STRING | BSAI_IndexTTS2_5 | Output filename prefix |
| format | COMBO | wav | Audio format: wav, mp3, flac |
| mp3_bitrate | INT | 192 | MP3 bitrate (kbps) |
| output_gain | FLOAT | 1.0 | Output volume multiplier |

**Outputs**: `file_path` (STRING) + `audio` (AUDIO — pass-through for chaining)

---

### 5. BSAI IndexTTS2.5 Unload Model

Unload the model to free VRAM.

## Usage Guide

### Basic Voice Cloning

1. Add **BSAI IndexTTS2.5 Loader** node
2. Add **LoadAudio** node, load a reference voice audio (10-30 seconds, clean recording)
3. Add **BSAI IndexTTS2.5 Synthesis** node, connect `tts_model` and `reference_audio`
4. Enter text and select language
5. Connect output to **PreviewAudio** or **BSAI IndexTTS2.5 Save Audio**

### Emotion Control via Reference Audio (Voice-Emotion Decoupling)

This feature separates the speaker's timbre from the emotion:

1. Load a **speaker voice** audio (e.g., calm narration) into `reference_audio`
2. Load an **emotion reference** audio (e.g., a crying voice) into `emo_audio_prompt`
3. Adjust `emo_alpha` to control emotion intensity (0.0 = no emotion influence, 1.0 = full emotion)
4. The synthesized speech will have the speaker's voice with the emotion from the reference

### Emotion Control via 8D Vector

1. Add **BSAI IndexTTS2.5 EmotionVector** node
2. Either select a preset (e.g., "sad") or manually adjust the 8 emotion sliders
3. Connect the `emo_vector` output to the Synthesis node's `emo_vector` input
4. Use `emo_alpha` on the Synthesis node to scale the emotion intensity

**Example**: For 80% sadness, set `sad=0.8` and `emo_alpha=1.0`, or set `sad=1.0` and `emo_alpha=0.8`.

### Text-Based Emotion (Qwen3)

1. In the **Loader** node, enable `use_qwen_emo=True` (loads additional Qwen3 model)
2. In the **Synthesis** node, enable `use_emo_text=True`
3. Optionally provide a custom emotion description in `emo_text` (e.g., "You scared me! Are you a ghost?")
4. If `emo_text` is empty, the main `text` is used as the emotion input
5. Adjust `emo_alpha` to ~0.6 for natural results (too high may sound exaggerated)

### Speed Control

Use `duration_factor` to control speech rate:

- **1.0** = Normal speed
- **0.7** = Faster (30% speedup)
- **1.4** = Slower (40% slowdown)
- Range: 0.5 (2x faster) to 2.0 (2x slower)

### Cross-Language Synthesis

Use one speaker's voice across different languages:

1. Load a **Chinese speaker** audio as `reference_audio`
2. Set `lang` to the target language (EN, JA, ES, AR)
3. Enter text in the target language
4. The synthesized speech will have the Chinese speaker's timbre speaking the target language

### Pinyin/Phoneme/Kana Control

IndexTTS-2.5 supports fine-grained pronunciation control via character replacement:

```
# Chinese pinyin control
他在银<行|XING2>里<行|HANG2>走了半天

# English CMU phoneme control
He had a <minute|M IH1 . N AH0 T> to examine

# Japanese kana control
彼は料理が<上手|じょうず>だが、囲碁では<上手|うわて>に負けた
```

## Workflow Examples

The `workflow_example/` directory contains 8 ready-to-use workflow JSON files:

| File | Description |
|------|-------------|
| `default_workflow.json` | Basic voice cloning (default) |
| `01_basic_voice_cloning.json` | Basic voice cloning with updated parameters |
| `02_emotion_audio_control.json` | Voice-emotion decoupling (speaker voice + separate emotion audio, emo_alpha=0.9) |
| `03_emotion_vector_control.json` | Direct 8D emotion vector control (Sad=0.8) |
| `04_text_based_emotion.json` | Qwen3 text-based emotion (use_qwen_emo=True, emo_alpha=0.6) |
| `05_speed_control.json` | Speed comparison: normal (1.0x), fast (0.7x), slow (1.4x) |
| `06_cross_language.json` | Cross-language: Chinese voice -> English, Japanese, Spanish |
| `07_emotion_vector_presets.json` | Emotion presets comparison: Happy vs Angry |

To use a workflow:
1. Open ComfyUI
2. Drag and drop the JSON file into the ComfyUI canvas
3. Load your own reference audio in the LoadAudio nodes
4. Adjust parameters as needed
5. Click Queue Prompt

## Compatibility (what the patch fixes)

`indextts` 2.0.0 (installed from `github.com/index-tts/index-tts`) was built for
older transformers. Modern ComfyUI portable builds ship **transformers >= 4.55**
(e.g. 4.57 or 5.x), where several internal symbols were removed.

The project uses a **three-layer defense** for transformers compatibility:

1. **`indextts_compat.py`** — Class-level attribute defaults (loaded with indextts import)
2. **`BSAI_IndexTTS.py` runtime patches** — Belt-and-suspenders fallback before indextts import
3. **`patch_indextts.py` source patches** — Direct `getattr` fallbacks in indextts source files

All patches are version-agnostic: `hasattr`/`getattr` guards make them no-ops on transformers 4.x.

### Patched symbols

| Symbol | Where | Fix |
|--------|-------|-----|
| `QuantizedCacheConfig` | `transformers_generation_utils.py` | try/except + dataclass placeholder |
| `_crop_past_key_values` | `transformers_generation_utils.py` | try/except + compat function |
| `NEED_SETUP_CACHE_CLASSES_MAPPING` | `transformers_generation_utils.py` | try/except -> `{}` |
| `forced_decoder_ids` | `transformers_generation_utils.py` | `getattr(..., None)` |
| `SequenceSummary` | `transformers_gpt2.py` | try/except placeholder |
| `TypicalLogitsWarper` | `utils/typical_sampling.py` | import from `transformers.generation.logits_process` |
| `wetext` import | `utils/front.py` | try/except graceful fallback |
| `_get_non_default_generation_parameters` | `transformers_generation_utils.py` | `getattr` fallback (Fix 10) |
| `BeamSearchScorer.is_done` | `transformers_generation_utils.py` | try/except compat (Fix 11) |
| `return_legacy_cache` | `transformers_generation_utils.py` | `getattr` fallback (Fix 12) |
| `_original_object_hash` | `transformers_generation_utils.py` | `getattr` fallback (Fix 12) |
| `GenerationConfig` tensor attrs | `_compat.py` + runtime patch | Class-level defaults (Sections 24-25) |
| `PretrainedConfig._pre_quantization_dtype` | `_compat.py` + runtime patch | Class-level default |
| `PretrainedConfig.sliding_window` | `_compat.py` + runtime patch | Class-level default |

The patch is **idempotent and safe**: re-running it is a no-op if already applied,
and it never fails the install if `indextts` changed upstream (it warns and skips).

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

### Node fails to load with ImportError or AttributeError

**Cause**: `indextts` 2.0.0 was written for older transformers. Starting with **transformers 4.55** (and 5.x) many internal APIs were removed.

**Fix**: The compatibility patch (`patch_indextts.py`) handles this automatically:
```bash
python BSAI_ComfyUI_IndexTTS-2.5/patch_indextts.py
```
If you still see the error:
1. Ensure `patch_indextts.py` and `indextts_compat.py` exist in the node directory
2. Restart ComfyUI (patches are applied on node load / install)
3. Or manually run: `python install.py`

### "use_emo_text=True requires QwenEmotion" error

**Cause**: Text-based emotion control requires the Qwen3 emotion model to be loaded.

**Fix**: In the **BSAI IndexTTS2.5 Loader** node, enable `use_qwen_emo=True`. If the model is already loaded without it, enable `force_reload=True` to reload with the Qwen3 model.

### "hatchling" build dependency not found

**Cause**: The Chinese PyPI mirror may not have `hatchling>=1.27.0`.

**Fix**: Use the official PyPI index:
```bash
pip install hatchling --index-url https://pypi.org/simple/
```

### protobuf version conflicts

**Cause**: `descript-audiotools` requires protobuf <3.20, but other packages need protobuf >=4.

**Fix**: Use protobuf 5.x:
```bash
pip install "protobuf>=5.26.1,<6"
```

### Python version incompatibility (Python 3.12+)

**Cause**: `indextts` requires Python >=3.10,<3.12, but newer ComfyUI environments use Python 3.13+.

**Fix**: Use `--ignore-requires-python` flag:
```bash
pip install --no-deps --ignore-requires-python --no-build-isolation git+https://github.com/index-tts/index-tts.git
```

## Model Download

Models are automatically downloaded on first use to `ComfyUI/models/IndexTTS2.5/`.

Download priority:
1. **Local check** - If model already exists locally, use it directly
2. **ModelScope** (preferred for China users) - `IndexTeam/IndexTTS-2.5`
3. **HuggingFace Mirror** - `https://hf-mirror.com` fallback
4. **HuggingFace Direct** - Last resort

### Manual Download

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

## System Requirements

- **GPU**: NVIDIA GPU with at least 4GB VRAM (8GB+ recommended for BF16; 10GB+ if using use_qwen_emo)
- **RAM**: At least 8GB system RAM (16GB+ recommended)
- **Storage**: ~5GB for model files (+~2GB for Qwen3 emotion model if use_qwen_emo=True)
- **CUDA**: CUDA 12.8+ recommended
- **Python**: 3.10+

## Tips

- Use **BF16** for faster inference and lower VRAM usage
- Keep reference audio short (10-30 seconds) for best results
- Use clean reference audio without background noise
- For **emotion reference audio**, choose audio with strong emotional expression
- When using **text-based emotion**, set `emo_alpha` to ~0.6 for natural results
- When using **emo_vector**, `emo_alpha` scales the entire vector (e.g., 0.5 = half intensity for all emotions)
- **use_random=True** adds variation but may reduce voice cloning fidelity
- For **cross-language** synthesis, the speaker's accent may influence the output

## Related Projects

- [IndexTTS](https://github.com/index-tts/index-tts) - Original IndexTTS project
- [BSAI-MiniMAX-H3-Prompt](https://github.com/BSAI-AI/BSAI-MiniMAX-H3-Prompt) - MiniMax H3 prompt optimization
- [BSAI_ComfyUI_Nodes](https://github.com/BSAI-AI/BSAI_ComfyUI_Nodes) - BSAI utility nodes

## License

MIT License - See LICENSE file for details.

## Credits

- IndexTTS-2.5 by [IndexTeam](https://github.com/index-tts) (Bilibili)
- ComfyUI by [comfyanonymous](https://github.com/comfyanonymous/ComfyUI)
