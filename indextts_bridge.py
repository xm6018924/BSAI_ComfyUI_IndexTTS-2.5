"""
IndexTTS Bridge Script - runs in the isolated venv with old transformers/accelerate.

Communication protocol (line-delimited JSON over stdin/stdout):
  {"cmd": "load", "cfg_path": ..., "model_dir": ..., ...}   -> {"status": "ok"} | {"status": "error", ...}
  {"cmd": "synthesize", "spk_audio_prompt": ..., "text": ..., ...} -> {"status": "ok", "output": ...} | {"status": "error", ...}
  {"cmd": "unload"}  -> {"status": "ok"} then exit
  {"cmd": "ping"}    -> {"status": "ok", "pong": true}
"""

import sys
import os
import json
import traceback

def _send(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()

def _patch_torchaudio():
    import torch
    import numpy as np
    try:
        import torchaudio
    except ImportError:
        return
    _save_orig = torchaudio.save
    _load_orig = torchaudio.load

    def _save_patched(filepath, src, sample_rate, **kwargs):
        try:
            return _save_orig(filepath, src, sample_rate, **kwargs)
        except (ImportError, RuntimeError):
            import soundfile as sf
            wav_np = src.cpu().numpy() if hasattr(src, 'cpu') else np.array(src)
            if wav_np.ndim == 3:
                wav_np = wav_np[0]
            if wav_np.ndim == 2:
                wav_np = wav_np.T
            elif wav_np.ndim == 1:
                wav_np = wav_np.reshape(-1, 1)
            sf.write(filepath, wav_np, sample_rate)

    def _load_patched(filepath, **kwargs):
        try:
            return _load_orig(filepath, **kwargs)
        except (ImportError, RuntimeError):
            import soundfile as sf
            data, sr = sf.read(filepath)
            waveform = torch.from_numpy(data).float()
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.T
            return waveform, sr

    torchaudio.save = _save_patched
    torchaudio.load = _load_patched

def _patch_bigvgan_hub_compat():
    """huggingface_hub>=1.x no longer forwards `proxies`/`resume_download`
    to HubMixin._from_pretrained, but IndexTTS's BigVGAN old signature
    requires both as keyword-only args. Wrap _from_pretrained so they
    get defaults (None / False) and everything else passes through.

    infer_v2_5 imports `from indextts.s2mel.modules.bigvgan import bigvgan`;
    fall back to `indextts.BigVGAN` just in case."""
    import functools
    candidates = []
    try:
        from indextts.s2mel.modules.bigvgan import bigvgan
        candidates.append(("s2mel", bigvgan.BigVGAN))
    except Exception:
        pass
    if not candidates:
        try:
            from indextts.BigVGAN import bigvgan
            candidates.append(("BigVGAN", bigvgan.BigVGAN))
        except Exception:
            pass
    for name, cls in candidates:
        try:
            _orig = cls._from_pretrained
            if getattr(_orig, "_bsai_hub_patched", False):
                continue

            @functools.wraps(_orig)
            def _patched(cls, *args, proxies=None, resume_download=False, **kwargs):
                # _orig is a class-bound method (obtained via cls._from_pretrained),
                # so do NOT pass cls again.
                return _orig(*args, proxies=proxies, resume_download=resume_download, **kwargs)

            _patched._bsai_hub_patched = True
            cls._from_pretrained = classmethod(_patched)
            print(f"[IndexTTS-Bridge] bigvgan hub compat patch applied ({name})", flush=True)
        except Exception as e:
            print(f"[IndexTTS-Bridge] bigvgan patch skipped ({name}): {e}", flush=True)

def _patch_beam_scorer_finalize():
    """REPLACE finalize with a full 4.x-compatible implementation instead of
    wrapping the shim. The indextts._compat shim's original finalize only
    stitched the last single token, producing a 1-column `sequences` tensor;
    after the caller truncates the prompt
    (model_v2_5.inference_speech: output.sequences[:, trunc_index:]) that
    becomes an empty [batch, 0] tensor, which makes
    infer_v2_5.infer_generator crash with
      IndexError: index -1 is out of bounds for dimension 1 with size 0
    (swallowed by infer()'s `except IndexError: return None` -> the
    "TTS inference completed but output file not found" error).
    Site-packages `_compat.py` is patched the same way; this keeps the
    plugin self-contained and distributable."""
    try:
        import functools
        import torch
        from transformers.generation import beam_search as _bs

        _cls = getattr(_bs, "BeamSearchScorer", None)
        if _cls is None:
            print("[IndexTTS-Bridge] beam_search.BeamSearchScorer not found; patch skipped", flush=True)
            return
        if getattr(_cls, "_bsai_beam_patched", False):
            return

        def _patched(self, input_ids, final_beam_scores, final_beam_tokens,
                     final_beam_indices, pad_token_id=None, eos_token_id=None,
                     max_length=None, beam_indices=None, decoder_prompt_len=None):
            # 4.x-compatible finalize: reconstruct each beam's FULL sequence from
            # `input_ids` (beam-expanded rows) using the last-step beam indices,
            # rank by score and keep `num_beam_hyps_to_keep` best per batch.
            batch_size = len(self._beam_hyps)
            best_seqs = []
            best_scores = []
            for batch_idx in range(batch_size):
                hyps = []
                for beam_token_rank, index in enumerate(final_beam_indices[batch_idx]):
                    batch_beam_idx = batch_idx * self.group_size + index.item()
                    final_score = final_beam_scores[batch_beam_idx].item()
                    final_tokens = input_ids[batch_beam_idx]
                    beam_index = beam_indices[batch_beam_idx] if beam_indices is not None else None
                    hyps.append((final_score, final_tokens, beam_index))
                hyps.extend(self._beam_hyps[batch_idx])
                hyps = sorted(hyps, key=lambda x: x[0])
                hyps = hyps[-self.num_beam_hyps_to_keep:]
                for (score, tokens, _b_idx) in hyps:
                    best_seqs.append(tokens)
                    best_scores.append(score)
            sequences = torch.stack(best_seqs, dim=0)
            sequence_scores = torch.tensor(best_scores, device=final_beam_scores.device,
                                           dtype=final_beam_scores.dtype)
            return {
                "sequences": sequences,
                "sequence_scores": sequence_scores,
                "beam_indices": beam_indices,
            }

        _patched._bsai_beam_patched = True
        _cls.finalize = _patched
        # ConstrainedBeamSearchScorer inherits finalize from BeamSearchScorer
        print("[IndexTTS-Bridge] beam_scorer finalize compat patch applied", flush=True)
    except Exception as e:
        print(f"[IndexTTS-Bridge] beam_scorer patch skipped: {e}", flush=True)

def main():
    _patch_torchaudio()

    # ---- isolate the JSON protocol stream from model-library prints ----
    # indextts / transformers / tqdm print progress to stdout during load & infer,
    # which would corrupt the line-delimited JSON protocol. Keep a handle to the
    # real stdout for _send, then redirect python-level prints to stderr.
    _real_stdout = sys.stdout

    def _send(obj):
        _real_stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        _real_stdout.flush()

    sys.stdout = sys.stderr

    tts = None

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue

            msg = json.loads(line)
            cmd = msg.pop("cmd", None)

            if cmd == "load":
                try:
                    _patch_bigvgan_hub_compat()
                    from indextts.infer_v2_5 import IndexTTS2
                    _patch_beam_scorer_finalize()
                    if msg.get("device") == "auto":
                        msg["device"] = None  # torch >=2.13 rejects "auto"; None => library auto-selects
                    tts = IndexTTS2(**msg)
                    _send({"status": "ok"})
                except Exception as e:
                    _send({"status": "error", "msg": str(e), "tb": traceback.format_exc()})

            elif cmd == "synthesize":
                if tts is None:
                    _send({"status": "error", "msg": "Model not loaded"})
                    continue
                try:
                    _patch_beam_scorer_finalize()  # idempotent safety net
                    # normalize numeric params: transformers logits processors require
                    # isinstance(x, float) for float params and strict ints for int params;
                    # UI JSON may send ints where floats are required (e.g. 10) and vice versa
                    for _k in ("temperature", "top_p", "repetition_penalty", "length_penalty",
                               "duration_factor", "emo_alpha"):
                        if _k in msg and msg[_k] is not None:
                            msg[_k] = float(msg[_k])
                    for _k in ("top_k", "num_beams", "max_mel_tokens", "max_text_tokens_per_segment"):
                        if _k in msg and msg[_k] is not None:
                            msg[_k] = int(msg[_k])
                    tts.infer(**msg)
                    _send({"status": "ok", "output": msg.get("output_path", "")})
                except Exception as e:
                    _send({"status": "error", "msg": str(e), "tb": traceback.format_exc()})

            elif cmd == "unload":
                if tts is not None:
                    del tts
                    tts = None
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                _send({"status": "ok"})
                break

            elif cmd == "ping":
                _send({"status": "ok", "pong": True})

            else:
                _send({"status": "error", "msg": f"Unknown command: {cmd}"})

        except json.JSONDecodeError:
            _send({"status": "error", "msg": "Invalid JSON input"})
        except Exception as e:
            _send({"status": "error", "msg": str(e), "tb": traceback.format_exc()})

if __name__ == "__main__":
    main()
