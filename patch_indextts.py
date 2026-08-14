#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_indextts.py — Apply transformers >= 4.55 / 5.x compatibility patches to the
installed `indextts` package so IndexTTS-2.5 runs on modern transformers.

WHY THIS IS NEEDED
------------------
`indextts` 2.0.0 (installed from github.com/index-tts/index-tts) was written for
older transformers. Starting with transformers 4.55 a number of internal symbols
were removed, and they were also removed in 5.x:

  - transformers.cache_utils.QuantizedCacheConfig
  - transformers.generation.candidate_generator._crop_past_key_values
  - transformers.generation.configuration_utils.NEED_SETUP_CACHE_CLASSES_MAPPING
  - transformers.generation.configuration_utils.QUANT_BACKEND_CLASSES_MAPPING
  - transformers.modeling_utils.SequenceSummary
  - transformers.TypicalLogitsWarper (moved to transformers.generation.logits_process)
  - GenerationConfig.forced_decoder_ids attribute access at runtime
  - wetext import on Windows/macOS when C++ build tools are missing

Without these patches you get ImportError at load time or
`AttributeError: 'GenerationConfig' object has no attribute 'forced_decoder_ids'`
at synthesis time.

WHAT THIS SCRIPT DOES
---------------------
Edits the 4 indextts source files that reference the removed symbols, wrapping
each import in a `try/except` fallback (or using getattr) so the exact same code
works on both transformers 4.55+ and 5.x.

It is IDEMPOTENT and SAFE:
  - Already-patched files are detected and skipped (no double-patching).
  - If the original code cannot be found (indextts changed upstream), it warns
    and skips that single patch instead of failing the whole install.
  - It never raises during normal use.

USAGE
-----
  python patch_indextts.py                 # auto-detect the indextts package
  python patch_indextts.py --indextts-dir /path/to/indextts
"""
import argparse
import os
import sys

# ---------------------------------------------------------------------------
# Patch definitions. Each entry:
#   (relative_path_inside_indextts, original_text, patched_text, description)
# The original_text is taken verbatim from indextts 2.0.0 (github main).
# ---------------------------------------------------------------------------

PATCHES = []


def _p(rel_path, orig, new, desc):
    PATCHES.append((rel_path, orig, new, desc))


# --- Fix 1: transformers.cache_utils.QuantizedCacheConfig -------------------
_p(
    "gpt/transformers_generation_utils.py",
    (
        "from transformers.cache_utils import (\n"
        "    Cache,\n"
        "    DynamicCache,\n"
        "    EncoderDecoderCache,\n"
        "    OffloadedCache,\n"
        "    QuantizedCacheConfig,\n"
        "    StaticCache,\n"
        ")\n"
    ),
    (
        "from transformers.cache_utils import (\n"
        "    Cache,\n"
        "    DynamicCache,\n"
        "    EncoderDecoderCache,\n"
        "    OffloadedCache,\n"
        "    StaticCache,\n"
        ")\n"
        "try:\n"
        "    from transformers.cache_utils import QuantizedCacheConfig\n"
        "except ImportError:\n"
        "    # transformers>=4.55 移除了 QuantizedCacheConfig；提供兼容垫片。\n"
        "    # 仅当 generation_config.cache_implementation==\"quantized\" 时才会被用到。\n"
        "    @dataclass\n"
        "    class QuantizedCacheConfig:\n"
        "        backend: str = \"quanto\"\n"
    ),
    "QuantizedCacheConfig compat",
)

# --- Fix 2: transformers.generation.candidate_generator._crop_past_key_values
_p(
    "gpt/transformers_generation_utils.py",
    (
        "from transformers.generation.candidate_generator import (\n"
        "    AssistedCandidateGenerator,\n"
        "    AssistedCandidateGeneratorDifferentTokenizers,\n"
        "    CandidateGenerator,\n"
        "    PromptLookupCandidateGenerator,\n"
        "    _crop_past_key_values,\n"
        "    _prepare_attention_mask,\n"
        "    _prepare_token_type_ids,\n"
        ")\n"
    ),
    (
        "from transformers.generation.candidate_generator import (\n"
        "    AssistedCandidateGenerator,\n"
        "    AssistedCandidateGeneratorDifferentTokenizers,\n"
        "    CandidateGenerator,\n"
        "    PromptLookupCandidateGenerator,\n"
        "    _prepare_attention_mask,\n"
        "    _prepare_token_type_ids,\n"
        ")\n"
        "try:\n"
        "    from transformers.generation.candidate_generator import _crop_past_key_values\n"
        "except ImportError:\n"
        "    # transformers>=4.55 移除了 _crop_past_key_values；提供兼容垫片（仅辅助解码路径会用到）\n"
        "    def _crop_past_key_values(model, past_key_values, maximum_length):\n"
        "        new_past = []\n"
        "        for idx in range(len(past_key_values)):\n"
        "            if past_key_values[idx] is None:\n"
        "                continue\n"
        "            new_past.append(past_key_values[idx][..., -maximum_length:, :])\n"
        "        return tuple(new_past)\n"
    ),
    "_crop_past_key_values compat",
)

# --- Fix 3: configuration_utils cache mapping dicts -------------------------
_p(
    "gpt/transformers_generation_utils.py",
    (
        "from transformers.generation.configuration_utils import (\n"
        "    NEED_SETUP_CACHE_CLASSES_MAPPING,\n"
        "    QUANT_BACKEND_CLASSES_MAPPING,\n"
        "    GenerationConfig,\n"
        "    GenerationMode,\n"
        ")\n"
    ),
    (
        "from transformers.generation.configuration_utils import (\n"
        "    GenerationConfig,\n"
        "    GenerationMode,\n"
        ")\n"
        "try:\n"
        "    from transformers.generation.configuration_utils import NEED_SETUP_CACHE_CLASSES_MAPPING\n"
        "except ImportError:\n"
        "    # transformers>=4.55 移除了该映射；仅 static/offloaded/quantized 缓存分支会用到，标准 TTS 走 DynamicCache 不进这些分支\n"
        "    NEED_SETUP_CACHE_CLASSES_MAPPING = {}\n"
        "try:\n"
        "    from transformers.generation.configuration_utils import QUANT_BACKEND_CLASSES_MAPPING\n"
        "except ImportError:\n"
        "    QUANT_BACKEND_CLASSES_MAPPING = {}\n"
    ),
    "cache mapping dicts compat",
)

# --- Fix 4: GenerationConfig.forced_decoder_ids runtime access --------------
_p(
    "gpt/transformers_generation_utils.py",
    (
        '        if generation_config.forced_decoder_ids is not None:\n'
    ),
    (
        '        if getattr(generation_config, "forced_decoder_ids", None) is not None:\n'
    ),
    "forced_decoder_ids getattr compat",
)

# --- Fix 5: transformers.modeling_utils.SequenceSummary ---------------------
_p(
    "gpt/transformers_gpt2.py",
    (
        "from transformers.modeling_utils import SequenceSummary\n"
    ),
    (
        "try:\n"
        "    from transformers.modeling_utils import SequenceSummary\n"
        "except ImportError:\n"
        "    # transformers>=4.55 移除了 SequenceSummary；IndexTTS 的 TTS 路径不依赖 GPT2 multiple-choice 头，提供占位\n"
        "    class SequenceSummary:\n"
        "        def __init__(self, *args, **kwargs):\n"
        "            raise NotImplementedError(\n"
        "                \"SequenceSummary 在当前 transformers 版本已被移除；IndexTTS 的 TTS 路径不依赖 GPT2 multiple-choice 头。\"\n"
        "            )\n"
    ),
    "SequenceSummary compat",
)

# --- Fix 6: TypicalLogitsWarper import path ---------------------------------
_p(
    "utils/typical_sampling.py",
    (
        "from transformers import TypicalLogitsWarper as BaseTypicalLogitsWarper\n"
    ),
    (
        "try:\n"
        "    from transformers.generation.logits_process import TypicalLogitsWarper as BaseTypicalLogitsWarper\n"
        "except ImportError:\n"
        "    from transformers.generation import TypicalLogitsWarper as BaseTypicalLogitsWarper\n"
    ),
    "TypicalLogitsWarper import path compat",
)

# --- Fix 7: wetext graceful fallback on Windows/macOS -----------------------
_p(
    "utils/front.py",
    (
        "            from wetext import Normalizer\n"
        "\n"
        "            self.zh_normalizer = Normalizer(remove_erhua=False, lang=\"zh\", operator=\"tn\")\n"
        "            self.en_normalizer = Normalizer(lang=\"en\", operator=\"tn\")\n"
    ),
    (
        "            try:\n"
        "                from wetext import Normalizer\n"
        "\n"
        "                self.zh_normalizer = Normalizer(remove_erhua=False, lang=\"zh\", operator=\"tn\")\n"
        "                self.en_normalizer = Normalizer(lang=\"en\", operator=\"tn\")\n"
        "            except Exception:\n"
        "                # wetext 不可用（未安装 / 本机无 C++ 构建环境导致 kaldifst 编译失败）时的兜底：\n"
        "                # 降级为原样文本归一化。代价：数字/日期等不会被自动转换为读音，但其余推理正常。\n"
        "                class _PassthroughNormalizer:\n"
        "                    def normalize(self, text):\n"
        "                        return text\n"
        "                self.zh_normalizer = _PassthroughNormalizer()\n"
        "                self.en_normalizer = _PassthroughNormalizer()\n"
        "                print(\"[BSAI] wetext 不可用，已降级为原样文本归一化（数字/日期等不会自动转换）。\")\n"
    ),
    "wetext graceful fallback",
)

# --- Fix 8: ExtensionsTrie import path (moved in transformers 5.x) -----------
# In transformers 5.x, ExtensionsTrie was moved from tokenization_utils to
# tokenization_python. This patch makes the import work on both versions.
_p(
    "gpt/transformers_generation_utils.py",
    "from transformers.tokenization_utils import ExtensionsTrie\n",
    (
        "try:\n"
        "    from transformers.tokenization_utils import ExtensionsTrie\n"
        "except ImportError:\n"
        "    # transformers 5.x 将 ExtensionsTrie 从 tokenization_utils 移至 tokenization_python\n"
        "    from transformers.tokenization_python import ExtensionsTrie\n"
    ),
    "ExtensionsTrie import path compat",
)


def find_indextts_dir(explicit=None):
    if explicit:
        return explicit
    try:
        import indextts  # type: ignore
        return os.path.dirname(indextts.__file__)
    except Exception:
        return None


def apply_one(indextts_dir, rel_path, orig, new, desc):
    path = os.path.join(indextts_dir, rel_path)
    if not os.path.exists(path):
        print(f"  [SKIP] {rel_path}: file not found")
        return "skip"
    # Read preserving the raw bytes' line endings, then normalize to "\n" so the
    # patch strings (which use "\n") match regardless of CRLF/LF in the source.
    with open(path, "r", encoding="utf-8", newline="") as f:
        raw = f.read()
    content = raw.replace("\r\n", "\n").replace("\r", "\n")

    # Already patched -> no-op
    if new in content:
        print(f"  [OK]   {rel_path}: already patched ({desc})")
        return "ok"

    # Original code not present -> indextts changed upstream; don't touch it.
    if orig not in content:
        print(
            f"  [WARN] {rel_path}: original code not found, skipping ({desc}). "
            f"indextts may have been updated upstream."
        )
        return "warn"

    content = content.replace(orig, new, 1)
    # Write back with normalized "\n" line endings (standard for Python packages).
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(content)
    print(f"  [PATCH]{rel_path}: applied ({desc})")
    return "patch"


def main():
    parser = argparse.ArgumentParser(
        description="Apply transformers compatibility patches to the installed indextts package."
    )
    parser.add_argument(
        "--indextts-dir",
        default=None,
        help="Explicit path to the installed indextts package (auto-detected if omitted).",
    )
    args = parser.parse_args()

    indextts_dir = find_indextts_dir(args.indextts_dir)
    if not indextts_dir:
        print(
            "ERROR: could not locate the indextts package.\n"
            "Make sure indextts is installed first:\n"
            "  pip install --no-deps --ignore-requires-python --no-build-isolation "
            "git+https://github.com/index-tts/index-tts.git\n"
            "Then re-run this patch."
        )
        return 1

    print(f"[patch_indextts] Target indextts package: {indextts_dir}")
    print(f"[patch_indextts] transformers: "
          f"{_transformers_version()}")
    print()

    counts = {"patch": 0, "ok": 0, "skip": 0, "warn": 0}
    for rel_path, orig, new, desc in PATCHES:
        r = apply_one(indextts_dir, rel_path, orig, new, desc)
        counts[r] += 1

    print()
    print(
        f"[patch_indextts] Done. "
        f"applied={counts['patch']} already_ok={counts['ok']} "
        f"skipped={counts['skip']} warnings={counts['warn']}"
    )
    if counts["patch"] > 0:
        print("[patch_indextts] Patches applied. A restart of ComfyUI is recommended.")
    return 0


def _transformers_version():
    try:
        import transformers  # type: ignore
        return transformers.__version__
    except Exception:
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
