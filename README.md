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

### Method 2: Manual

1. Clone this repository into your `ComfyUI/custom_nodes/` directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/BSAI-AI/BSAI_ComfyUI_IndexTTS-2.5.git
   ```

2. Install dependencies:
   ```bash
   # Windows (ComfyUI portable)
   .\python_embeded\python.exe -m pip install -r BSAI_ComfyUI_IndexTTS-2.5/requirements.txt

   # Or use your ComfyUI Python environment
   pip install -r BSAI_ComfyUI_IndexTTS-2.5/requirements.txt
   ```

3. Restart ComfyUI

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
