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

def main():
    _patch_torchaudio()
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
                    from indextts.infer_v2_5 import IndexTTS2
                    tts = IndexTTS2(**msg)
                    _send({"status": "ok"})
                except Exception as e:
                    _send({"status": "error", "msg": str(e), "tb": traceback.format_exc()})

            elif cmd == "synthesize":
                if tts is None:
                    _send({"status": "error", "msg": "Model not loaded"})
                    continue
                try:
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
