# BSAI_ComfyUI_IndexTTS-2.5

[English](#english) | [中文](#中文)

---

<a id="english"></a>

# BSAI_ComfyUI_IndexTTS-2.5 (English)

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

## Changelog

### 2026-08-15

- **Audio Player Bar**: Replaced the simple "Play Audio" button with a full audio player bar featuring play/pause, time display (current/total), clickable progress bar, and volume toggle
- **Bilingual Parameters**: All 7 node parameters now display in both Chinese and English (e.g., `happy_开心`, `angry_愤怒`, `duration_factor_语速因子`)
- **Bilingual UI Buttons**: "Load Audio" and "Play Audio" buttons on the LoadAudio node, displayed in Chinese
- **PreviewAudio Migration**: All 8 workflow files now use ComfyUI's built-in PreviewAudio node instead of the custom BSAI PreviewAudio node. The custom BSAI PreviewAudio node has been removed
- **Bug Fixes**: Fixed JS extension loading failure (import path), fixed widget name mismatch preventing button display

### 2026-08-14

- **Bilingual Emotion Parameters**: Emotion parameters renamed to bilingual format (`happy_开心`, `angry_愤怒`, `sad_悲伤`, `fear_恐惧`, `disgust_厌恶`, `melancholy_忧郁`, `surprise_惊讶`, `calm_平静`)
- **Workflow Input Connections**: Fixed input connections in all workflow JSON files after parameter renaming
- **Silent Placeholder Audio**: Generate silent placeholder audio when reference file is missing, preventing workflow errors
- **Installer Dependencies**: Added missing omegaconf and other critical dependencies to installer

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

The transformers compatibility patch (`patch_indextts.py`) and the 5.x shim (`indextts_compat.py`) are applied automatically when ComfyUI loads the node.

## Nodes

### 0. BSAI IndexTTS2.5 Load Audio

Load an audio file from ComfyUI's input directory. Self-contained — does not depend on ComfyUI built-in audio nodes.

**UI Buttons:**
- **📁 加载音频 (Load Audio)**: Click to open a file picker and upload an audio file (wav, mp3, flac, ogg, m4a)
- **Audio Player Bar**: Displays when audio is loaded, featuring:
  - Play/pause button (▶ / ⏸)
  - Time display (current position / total duration, e.g., `0:00 / 2:00`)
  - Clickable progress bar (click to seek)
  - Volume toggle (🔊 / 🔇)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| audio_音频 | COMBO | - | Select audio file from input directory (supports wav, mp3, flac, ogg, m4a) |

**Output**: `AUDIO` - Audio data for use with Synthesis node

> **Tip**: Upload audio files to ComfyUI's `input/` directory, or use the "📁 加载音频" button on the node.

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
| happy_开心 | FLOAT | 0.0 | Happiness intensity (0.0-1.0) |
| angry_愤怒 | FLOAT | 0.0 | Anger intensity (0.0-1.0) |
| sad_悲伤 | FLOAT | 0.0 | Sadness intensity (0.0-1.0) |
| fear_恐惧 | FLOAT | 0.0 | Fear intensity (0.0-1.0) |
| disgust_厌恶 | FLOAT | 0.0 | Disgust intensity (0.0-1.0) |
| melancholy_忧郁 | FLOAT | 0.0 | Melancholy intensity (0.0-1.0) |
| surprise_惊讶 | FLOAT | 0.0 | Surprise intensity (0.0-1.0) |
| calm_平静 | FLOAT | 0.0 | Calmness intensity (0.0-1.0) |
| preset_预设 | COMBO | none | Quick preset: none, happy, angry, sad, fear, disgust, melancholy, surprise, calm |

> **Tip**: When a preset is selected (not "none"), it overrides the individual emotion sliders with a single-emotion vector (value 1.0 for the selected emotion, 0.0 for others). Use "none" to manually mix multiple emotions.

**Output**: `EMO_VECTOR` - Connect to the `emo_vector` input of the Synthesis node

**Emotion Vector Order**: `[happy, angry, sad, fear, disgust, melancholy, surprise, calm]`

---

### 4. BSAI IndexTTS2.5 Save Audio

Save generated audio to disk.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| audio_音频 | AUDIO | - | Audio to save |
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

---
---

<a id="中文"></a>

# BSAI_ComfyUI_IndexTTS-2.5 (中文)

[IndexTTS-2.5](https://github.com/index-tts/index-tts) 的 ComfyUI 自定义节点 —— 多语言、情感丰富、时长可控的自回归零样本文本转语音突破。

## IndexTTS-2.5 主要更新

IndexTTS-2.5 是对 IndexTTS-2 的重大升级，核心改进：

1. **多语言支持**：中文、英文、日语、西班牙语和阿拉伯语，支持稳健的跨语言情感迁移
2. **声音-情感解耦**：独立控制说话人音色和情感表达
3. **情感控制**：多种情感输入方式 —— 参考音频、8维情感向量、基于Qwen3的文本情感
4. **语速控制**：通过 `duration_factor` 精确调节语速（0.5倍-2.0倍）
5. **语义编解码器压缩**：帧率从50Hz降至25Hz，序列长度减半
6. **Zipformer架构**：实时率比IndexTTS-2提升2.28倍
7. **强化学习（GRPO）**：改善发音准确度和韵律自然度
8. **拼音/音素/假名控制**：通过字符替换实现细粒度发音控制

## 更新日志

### 2026-08-15

- **音频播放器栏**：将简单的"播放音频"按钮替换为完整播放器栏，包含播放/暂停、时间显示（当前/总时长）、可点击进度条、音量切换
- **中英双语参数**：全部7个节点的参数现在以中英双语显示（如 `happy_开心`、`angry_愤怒`、`duration_factor_语速因子`）
- **中英双语UI按钮**：LoadAudio节点上显示"📁 加载音频"和"▶ 播放音频"按钮
- **PreviewAudio迁移**：全部8个工作流文件改用ComfyUI内置PreviewAudio节点，替代自定义BSAI PreviewAudio节点。自定义BSAI PreviewAudio节点已移除
- **Bug修复**：修复JS扩展加载失败（import路径问题），修复widget名称不匹配导致按钮无法显示

### 2026-08-14

- **情绪参数双语化**：情绪参数改为中英对照格式（`happy_开心`、`angry_愤怒`、`sad_悲伤`、`fear_恐惧`、`disgust_厌恶`、`melancholy_忧郁`、`surprise_惊讶`、`calm_平静`）
- **工作流输入连接**：修复参数重命名后所有工作流JSON文件中的输入连接
- **静音占位音频**：参考文件缺失时生成静音占位音频，防止工作流报错
- **安装器依赖**：补充缺失的omegaconf等关键依赖

## 功能特性

- **声音克隆**：从短参考音频样本克隆任意声音
- **零样本TTS**：用克隆的声音从文本生成语音
- **多语言**：支持中文（ZH）、英文（EN）、日语（JA）、西班牙语（ES）和阿拉伯语（AR）
- **跨语言合成**：用中文说话人的声音合成英文、日语、西班牙语或阿拉伯语文本
- **情感控制**：
  - **情感参考音频**：使用单独的音频控制情感，同时保留说话人音色
  - **8维情感向量**：直接指定 [开心, 愤怒, 悲伤, 恐惧, 厌恶, 忧郁, 惊讶, 平静] 的强度
  - **基于文本的情感**：通过Qwen3模型从文本描述自动生成情感向量
  - **情感强度**：可调节混合强度（`emo_alpha`，0.0-1.0）
- **语速控制**：通过 `duration_factor` 调节语速（0.5倍-2.0倍）
- **随机采样**：切换随机采样以获得多样化输出
- **自动模型下载**：自动从ModelScope（国内）或HuggingFace下载模型
- **BF16支持**：半精度推理，降低显存占用
- **音频输出**：兼容ComfyUI内置音频节点（PreviewAudio、SaveAudio等）

## 安装

### 方法一：ComfyUI-Manager（推荐）

1. 在ComfyUI中打开ComfyUI-Manager
2. 搜索 "BSAI_ComfyUI_IndexTTS-2.5"
3. 点击安装

`install.py` 脚本会自动处理一切：从GitHub源码安装 `indextts`，应用transformers **4.55+/5.x**兼容补丁（`patch_indextts.py`），安装运行时依赖，并应用transformers 5.x兼容层。

### 方法二：手动安装

1. 将此仓库克隆到你的 `ComfyUI/custom_nodes/` 目录：
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/xm6018924/BSAI_ComfyUI_IndexTTS-2.5.git
   ```

2. 运行安装脚本：
   ```bash
   # Windows（ComfyUI便携版）
   .\python\python.exe BSAI_ComfyUI_IndexTTS-2.5/install.py

   # 或使用批处理文件（自动检测python.exe位置）
   BSAI_ComfyUI_IndexTTS-2.5\install_bsai_indextts.bat
   ```

3. 重启ComfyUI

### 方法三：逐步手动安装（install.py失败时）

如果自动安装失败，可以手动安装：

```bash
# 1. 安装hatchling构建工具（国内镜像可能没有）
pip install hatchling --index-url https://pypi.org/simple/

# 2. 从GitHub安装indextts（PyPI上没有；indextts 2.0.0）
pip install --no-deps --ignore-requires-python --no-build-isolation git+https://github.com/index-tts/index-tts.git

# 3. 安装运行时依赖（openai-whisper是必需的）
#    注意：不要安装keras / descript-audiotools —— 运行时不使用，
#    且keras==2.9.0与Python 3.12+冲突
pip install openai-whisper cn2an fugashi unidic-lite g2p_en json5 munch textstat --index-url https://pypi.org/simple/

# 4. wetext也是必需的，但在Windows/macOS上需要C++工具链（MSVC + CMake）来构建
#    kaldifst。如果构建失败也没关系 —— 下面的补丁会让indextts优雅降级
#    （数字/日期不会被自动转换）
pip install wetext --index-url https://pypi.org/simple/

# 5. 应用transformers兼容补丁（transformers >= 4.55时必需）
python BSAI_ComfyUI_IndexTTS-2.5/patch_indextts.py

# 6. 修复protobuf版本（可选，防御性）
pip install "protobuf>=5.26.1,<6" --index-url https://pypi.org/simple/
```

transformers兼容补丁（`patch_indextts.py`）和5.x兼容层（`indextts_compat.py`）在ComfyUI加载节点时会自动应用。

## 节点

### 0. BSAI IndexTTS2.5 加载音频

从ComfyUI的input目录加载音频文件。独立节点 —— 不依赖ComfyUI内置音频节点。

**UI按钮：**
- **📁 加载音频**：点击打开文件选择器上传音频文件（wav, mp3, flac, ogg, m4a）
- **音频播放器栏**：加载音频后显示，包含：
  - 播放/暂停按钮（▶ / ⏸）
  - 时间显示（当前位置 / 总时长，如 `0:00 / 2:00`）
  - 可点击进度条（点击跳转到指定位置）
  - 音量切换（🔊 / 🔇）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| audio_音频 | COMBO | - | 从input目录选择音频文件（支持wav, mp3, flac, ogg, m4a） |

**输出**：`AUDIO` - 用于合成节点的音频数据

> **提示**：将音频文件上传到ComfyUI的 `input/` 目录，或使用节点上的"📁 加载音频"按钮。

---

### 1. BSAI IndexTTS2.5 加载器

加载IndexTTS-2.5模型。首次运行时自动下载模型。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| use_bf16 | BOOLEAN | True | 使用BF16半精度，加速推理并降低显存 |
| device | COMBO | auto | 设备：auto, cuda:0, 或 cpu |
| use_qwen_emo | BOOLEAN | False | 加载Qwen3情感模型（基于文本的情感控制需要） |
| force_reload | BOOLEAN | False | 强制重新加载模型（先释放显存） |

> **注意**：使用合成节点的 `use_emo_text` 时需要 `use_qwen_emo=True`。它会加载额外的Qwen3模型用于文本到情感推理，会增加显存占用。仅在需要基于文本的情感控制时启用。

**输出**：`INDEX_TTS_MODEL` - 加载的TTS模型实例

---

### 2. BSAI IndexTTS2.5 合成

使用参考音频进行声音克隆，从文本生成语音。支持情感控制、语速控制和跨语言合成。

#### 必需参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| tts_model | INDEX_TTS_MODEL | - | 来自加载器的模型 |
| text | STRING | "Hello..." | 要合成的文本 |
| reference_audio | AUDIO | - | 说话人声音参考音频 |
| lang | COMBO | ZH | 语言：ZH, EN, JA, ES, AR, zhen |

#### 情感控制参数（可选）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| emo_audio_prompt | AUDIO | - | 情感参考音频（声音-情感解耦）。连接后，说话人音色来自 `reference_audio`，情感来自此音频 |
| emo_alpha | FLOAT | 1.0 | 情感混合强度（0.0-1.0）。1.0 = 完全使用参考情感，0.0 = 无情感影响 |
| emo_vector | EMO_VECTOR | - | 来自EmotionVector节点的8维情感向量。覆盖emo_audio_prompt |
| use_emo_text | BOOLEAN | False | 通过Qwen3从文本自动生成情感（需要加载器中use_qwen_emo=True） |
| emo_text | STRING | "" | 自定义情感描述文本。为空且use_emo_text=True时，使用主文本作为情感输入 |
| use_random | BOOLEAN | False | 启用随机采样（增加变化但可能降低声音克隆保真度） |

#### 语速控制参数（可选）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| duration_factor | FLOAT | 1.0 | 语速倍率。>1.0 = 放慢，<1.0 = 加快。范围：0.5-2.0 |

#### 生成参数（可选）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| max_text_tokens_per_segment | INT | 100 | 每段最大token数 |
| max_mel_tokens | INT | 1500 | 每次生成的最大mel token数 |
| temperature | FLOAT | 0.8 | 采样温度 |
| top_p | FLOAT | 0.8 | Top-p采样 |
| top_k | INT | 30 | Top-k采样 |
| length_penalty | FLOAT | 0.0 | 长度惩罚 |
| num_beams | INT | 3 | beam数量 |
| repetition_penalty | FLOAT | 10.0 | 重复惩罚 |
| do_sample | BOOLEAN | True | 使用采样而非贪心解码 |
| verbose | BOOLEAN | False | 打印调试信息 |

**输出**：`audio` (AUDIO) + `status` (STRING)

#### 情感控制优先级

当提供多种情感输入时，优先级为：

1. **emo_vector**（最高） — 直接8维向量覆盖一切
2. **use_emo_text** — 基于Qwen3的文本情感覆盖emo_audio_prompt
3. **emo_audio_prompt** — 情感参考音频
4. **默认** — 使用说话人声音作为情感参考（emo_alpha强制为1.0）

---

### 3. BSAI IndexTTS2.5 情感向量

辅助节点，为合成节点构建8维情感向量。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| happy_开心 | FLOAT | 0.0 | 开心强度（0.0-1.0） |
| angry_愤怒 | FLOAT | 0.0 | 愤怒强度（0.0-1.0） |
| sad_悲伤 | FLOAT | 0.0 | 悲伤强度（0.0-1.0） |
| fear_恐惧 | FLOAT | 0.0 | 恐惧强度（0.0-1.0） |
| disgust_厌恶 | FLOAT | 0.0 | 厌恶强度（0.0-1.0） |
| melancholy_忧郁 | FLOAT | 0.0 | 忧郁强度（0.0-1.0） |
| surprise_惊讶 | FLOAT | 0.0 | 惊讶强度（0.0-1.0） |
| calm_平静 | FLOAT | 0.0 | 平静强度（0.0-1.0） |
| preset_预设 | COMBO | none | 快速预设：none, happy, angry, sad, fear, disgust, melancholy, surprise, calm |

> **提示**：选择预设（非"none"）时，会用单一情感向量覆盖各滑块（选中情感值为1.0，其他为0.0）。选择"none"可手动混合多种情感。

**输出**：`EMO_VECTOR` - 连接到合成节点的 `emo_vector` 输入

**情感向量顺序**：`[happy, angry, sad, fear, disgust, melancholy, surprise, calm]`

---

### 4. BSAI IndexTTS2.5 保存音频

将生成的音频保存到磁盘。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| audio_音频 | AUDIO | - | 要保存的音频 |
| filename_prefix | STRING | BSAI_IndexTTS2_5 | 输出文件名前缀 |
| format | COMBO | wav | 音频格式：wav, mp3, flac |
| mp3_bitrate | INT | 192 | MP3比特率（kbps） |
| output_gain | FLOAT | 1.0 | 输出音量倍率 |

**输出**：`file_path` (STRING) + `audio` (AUDIO — 透传，可链式连接)

---

### 5. BSAI IndexTTS2.5 卸载模型

卸载模型以释放显存。

## 使用指南

### 基础声音克隆

1. 添加 **BSAI IndexTTS2.5 加载器** 节点
2. 添加 **加载音频** 节点，加载参考声音音频（10-30秒，干净录音）
3. 添加 **BSAI IndexTTS2.5 合成** 节点，连接 `tts_model` 和 `reference_audio`
4. 输入文本并选择语言
5. 将输出连接到 **预览音频** 或 **BSAI IndexTTS2.5 保存音频**

### 通过参考音频控制情感（声音-情感解耦）

此功能将说话人音色与情感分离：

1. 将 **说话人声音** 音频（如平静叙述）加载到 `reference_audio`
2. 将 **情感参考** 音频（如哭泣的声音）加载到 `emo_audio_prompt`
3. 调节 `emo_alpha` 控制情感强度（0.0 = 无情感影响，1.0 = 完全情感）
4. 合成的语音将具有说话人的声音和参考音频的情感

### 通过8维向量控制情感

1. 添加 **BSAI IndexTTS2.5 情感向量** 节点
2. 选择预设（如"sad"）或手动调节8个情感滑块
3. 将 `emo_vector` 输出连接到合成节点的 `emo_vector` 输入
4. 使用合成节点上的 `emo_alpha` 调节情感强度

**示例**：要80%悲伤，设置 `sad=0.8` 且 `emo_alpha=1.0`，或设置 `sad=1.0` 且 `emo_alpha=0.8`。

### 基于文本的情感（Qwen3）

1. 在 **加载器** 节点中启用 `use_qwen_emo=True`（加载额外的Qwen3模型）
2. 在 **合成** 节点中启用 `use_emo_text=True`
3. 可选在 `emo_text` 中提供自定义情感描述（如"你吓到我了！你是鬼吗？"）
4. 如果 `emo_text` 为空，则使用主 `text` 作为情感输入
5. 将 `emo_alpha` 调至约0.6以获得自然效果（过高可能显得夸张）

### 语速控制

使用 `duration_factor` 控制语速：

- **1.0** = 正常速度
- **0.7** = 加快（提速30%）
- **1.4** = 放慢（减速40%）
- 范围：0.5（快2倍）到 2.0（慢2倍）

### 跨语言合成

用一个说话人的声音跨不同语言：

1. 将 **中文说话人** 音频作为 `reference_audio` 加载
2. 将 `lang` 设置为目标语言（EN, JA, ES, AR）
3. 输入目标语言的文本
4. 合成的语音将具有中文说话人的音色说目标语言

### 拼音/音素/假名控制

IndexTTS-2.5支持通过字符替换进行细粒度发音控制：

```
# 中文拼音控制
他在银<行|XING2>里<行|HANG2>走了半天

# 英文CMU音素控制
He had a <minute|M IH1 . N AH0 T> to examine

# 日文假名控制
彼は料理が<上手|じょうず>だが、囲碁では<上手|うわて>に負けた
```

## 工作流示例

`workflow_example/` 目录包含8个即用型工作流JSON文件：

| 文件 | 说明 |
|------|------|
| `default_workflow.json` | 基础声音克隆（默认） |
| `01_basic_voice_cloning.json` | 基础声音克隆（更新参数） |
| `02_emotion_audio_control.json` | 声音-情感解耦（说话人声音 + 独立情感音频，emo_alpha=0.9） |
| `03_emotion_vector_control.json` | 直接8维情感向量控制（悲伤=0.8） |
| `04_text_based_emotion.json` | Qwen3基于文本的情感（use_qwen_emo=True，emo_alpha=0.6） |
| `05_speed_control.json` | 语速对比：正常（1.0倍）、快（0.7倍）、慢（1.4倍） |
| `06_cross_language.json` | 跨语言：中文声音 -> 英文、日文、西班牙文 |
| `07_emotion_vector_presets.json` | 情感预设对比：开心 vs 愤怒 |

使用工作流：
1. 打开ComfyUI
2. 将JSON文件拖放到ComfyUI画布中
3. 在加载音频节点中加载你自己的参考音频
4. 根据需要调整参数
5. 点击执行队列

## 兼容性（补丁修复内容）

从 `github.com/index-tts/index-tts` 安装的 `indextts` 2.0.0 是为旧版transformers构建的。
现代ComfyUI便携版自带 **transformers >= 4.55**（如4.57或5.x），其中移除了多个内部符号。

本项目使用**三层防御**确保transformers兼容性：

1. **`indextts_compat.py`** — 类级别属性默认值（随indextts导入加载）
2. **`BSAI_IndexTTS.py` 运行时补丁** — indextts导入前的双重保险回退
3. **`patch_indextts.py` 源码补丁** — 在indextts源文件中直接添加 `getattr` 回退

所有补丁都是版本无关的：`hasattr`/`getattr` 守卫使其在transformers 4.x上为空操作。

### 已补丁符号

| 符号 | 位置 | 修复方式 |
|------|------|----------|
| `QuantizedCacheConfig` | `transformers_generation_utils.py` | try/except + dataclass占位 |
| `_crop_past_key_values` | `transformers_generation_utils.py` | try/except + 兼容函数 |
| `NEED_SETUP_CACHE_CLASSES_MAPPING` | `transformers_generation_utils.py` | try/except -> `{}` |
| `forced_decoder_ids` | `transformers_generation_utils.py` | `getattr(..., None)` |
| `SequenceSummary` | `transformers_gpt2.py` | try/except占位 |
| `TypicalLogitsWarper` | `utils/typical_sampling.py` | 从 `transformers.generation.logits_process` 导入 |
| `wetext` 导入 | `utils/front.py` | try/except优雅降级 |
| `_get_non_default_generation_parameters` | `transformers_generation_utils.py` | `getattr`回退（修复10） |
| `BeamSearchScorer.is_done` | `transformers_generation_utils.py` | try/except兼容（修复11） |
| `return_legacy_cache` | `transformers_generation_utils.py` | `getattr`回退（修复12） |
| `_original_object_hash` | `transformers_generation_utils.py` | `getattr`回退（修复12） |
| `GenerationConfig` tensor属性 | `_compat.py` + 运行时补丁 | 类级别默认值（第24-25节） |
| `PretrainedConfig._pre_quantization_dtype` | `_compat.py` + 运行时补丁 | 类级别默认值 |
| `PretrainedConfig.sliding_window` | `_compat.py` + 运行时补丁 | 类级别默认值 |

补丁是**幂等且安全的**：如果已应用则重复运行为空操作，且如果 `indextts` 上游变更也不会导致安装失败（会警告并跳过）。

## 故障排除

### 安装失败："Could not find a version that satisfies the requirement indextts"

**原因**：`indextts` 未发布到PyPI。必须从GitHub源码安装。

**解决**：使用 `install.py` 或 `install_bsai_indextts.bat`，或按方法三操作。

### 安装失败："Cannot find command 'git'"

**原因**：Git不在系统PATH中。

**解决**：将Git添加到PATH或指定完整路径：
```bash
set PATH=C:\Program Files\Git\cmd;%PATH%
pip install --no-deps --ignore-requires-python --no-build-isolation git+https://github.com/index-tts/index-tts.git
```

### 节点加载失败：ImportError 或 AttributeError

**原因**：`indextts` 2.0.0 是为旧版transformers编写的。从 **transformers 4.55**（及5.x）开始，许多内部API被移除。

**解决**：兼容补丁（`patch_indextts.py`）会自动处理：
```bash
python BSAI_ComfyUI_IndexTTS-2.5/patch_indextts.py
```
如果仍然报错：
1. 确保节点目录中存在 `patch_indextts.py` 和 `indextts_compat.py`
2. 重启ComfyUI（补丁在节点加载/安装时应用）
3. 或手动运行：`python install.py`

### "use_emo_text=True requires QwenEmotion" 错误

**原因**：基于文本的情感控制需要加载Qwen3情感模型。

**解决**：在 **BSAI IndexTTS2.5 加载器** 节点中启用 `use_qwen_emo=True`。如果模型已加载但未启用此项，启用 `force_reload=True` 以重新加载Qwen3模型。

### 找不到 "hatchling" 构建依赖

**原因**：国内PyPI镜像可能没有 `hatchling>=1.27.0`。

**解决**：使用官方PyPI索引：
```bash
pip install hatchling --index-url https://pypi.org/simple/
```

### protobuf版本冲突

**原因**：`descript-audiotools` 需要protobuf <3.20，但其他包需要protobuf >=4。

**解决**：使用protobuf 5.x：
```bash
pip install "protobuf>=5.26.1,<6"
```

### Python版本不兼容（Python 3.12+）

**原因**：`indextts` 要求Python >=3.10,<3.12，但较新的ComfyUI环境使用Python 3.13+。

**解决**：使用 `--ignore-requires-python` 标志：
```bash
pip install --no-deps --ignore-requires-python --no-build-isolation git+https://github.com/index-tts/index-tts.git
```

## 模型下载

模型在首次使用时自动下载到 `ComfyUI/models/IndexTTS2.5/`。

下载优先级：
1. **本地检查** - 如果模型已存在本地，直接使用
2. **ModelScope**（国内用户优先）- `IndexTeam/IndexTTS-2.5`
3. **HuggingFace镜像** - `https://hf-mirror.com` 回退
4. **HuggingFace直连** - 最后手段

### 手动下载

```bash
# 通过ModelScope（国内）
pip install modelscope
modelscope download --model IndexTeam/IndexTTS-2.5 --local_dir ComfyUI/models/IndexTTS2.5

# 通过HuggingFace
pip install huggingface_hub[cli]
huggingface-cli download IndexTeam/IndexTTS-2.5 --local-dir ComfyUI/models/IndexTTS2.5

# 通过HuggingFace镜像（国内）
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download IndexTeam/IndexTTS-2.5 --local-dir ComfyUI/models/IndexTTS2.5
```

## 系统要求

- **GPU**：NVIDIA显卡，至少4GB显存（BF16推荐8GB+；使用use_qwen_emo推荐10GB+）
- **内存**：至少8GB系统内存（推荐16GB+）
- **存储**：约5GB模型文件（如使用use_qwen_emo则额外约2GB Qwen3情感模型）
- **CUDA**：推荐CUDA 12.8+
- **Python**：3.10+

## 使用技巧

- 使用 **BF16** 加速推理并降低显存占用
- 参考音频保持简短（10-30秒）以获得最佳效果
- 使用无背景噪音的干净参考音频
- **情感参考音频**应选择情感表达强烈的音频
- 使用 **基于文本的情感** 时，`emo_alpha` 设为约0.6以获得自然效果
- 使用 **emo_vector** 时，`emo_alpha` 缩放整个向量（如0.5 = 所有情感强度减半）
- **use_random=True** 增加变化但可能降低声音克隆保真度
- **跨语言** 合成时，说话人的口音可能影响输出

## 相关项目

- [IndexTTS](https://github.com/index-tts/index-tts) - 原始IndexTTS项目
- [BSAI-MiniMAX-H3-Prompt](https://github.com/BSAI-AI/BSAI-MiniMAX-H3-Prompt) - MiniMax H3提示词优化
- [BSAI_ComfyUI_Nodes](https://github.com/BSAI-AI/BSAI_ComfyUI_Nodes) - BSAI工具节点

## 许可证

MIT许可证 - 详见LICENSE文件

## 致谢

- IndexTTS-2.5 由 [IndexTeam](https://github.com/index-tts)（哔哩哔哩）开发
- ComfyUI 由 [comfyanonymous](https://github.com/comfyanonymous/ComfyUI) 开发
