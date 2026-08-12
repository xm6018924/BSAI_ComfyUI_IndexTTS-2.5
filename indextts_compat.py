"""
Compatibility shim for indextts to work with transformers 5.x.
Provides missing classes, functions, and modules that were removed or
relocated in transformers 5.0+.

For transformers 4.x, this module is a no-op (prints a message and exits).
Each section is independently guarded so one failure doesn't break the rest.
"""
import sys
import os
import types
import warnings
import functools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

# ===========================================================================
# Version guard: skip entirely for transformers 4.x
# ===========================================================================
try:
    import transformers
    _TF_VERSION = transformers.__version__
    _TF_MAJOR = int(_TF_VERSION.split('.')[0])
except Exception:
    _TF_VERSION = "unknown"
    _TF_MAJOR = 0

if _TF_MAJOR < 5:
    # transformers 4.x — no shim needed, indextts should work natively
    print(f"[indextts._compat] transformers {_TF_VERSION} detected — no compat shim needed (only for 5.x)")
else:
    print(f"[indextts._compat] transformers {_TF_VERSION} detected — applying compat shim for 5.x")

# ===========================================================================
# 1. cache_utils: OffloadedCache, QuantizedCacheConfig
# ===========================================================================
try:
    try:
        from transformers.cache_utils import OffloadedCache
    except ImportError:
        from transformers.cache_utils import DynamicCache

        class OffloadedCache(DynamicCache):
            """Dummy OffloadedCache for transformers 5.x compatibility."""
            pass

    try:
        from transformers.cache_utils import QuantizedCacheConfig
    except ImportError:
        @dataclass
        class QuantizedCacheConfig:
            """Dummy QuantizedCacheConfig for transformers 5.x compatibility."""
            backend: str = "quanto"
            nbits: int = 4
            axis: int = 0
            q_group_size: int = 64
            residual_length: int = 128
            device: Optional[str] = None

    # Inject into transformers.cache_utils
    import transformers.cache_utils as _cache_utils
    if not hasattr(_cache_utils, 'OffloadedCache'):
        _cache_utils.OffloadedCache = OffloadedCache
    if not hasattr(_cache_utils, 'QuantizedCacheConfig'):
        _cache_utils.QuantizedCacheConfig = QuantizedCacheConfig
except Exception as _e:
    if _TF_MAJOR >= 5:
        warnings.warn(f"[indextts._compat] Section 1 (cache_utils) failed: {_e}")

# ===========================================================================
# 2. pytorch_utils: isin_mps_friendly, find_pruneable_heads_and_indices, prune_conv1d_layer
# ===========================================================================
try:
    import transformers.pytorch_utils as _pytorch_utils

    if not hasattr(_pytorch_utils, 'isin_mps_friendly'):
        def isin_mps_friendly(input, test):
            return torch.isin(input, test)
        _pytorch_utils.isin_mps_friendly = isin_mps_friendly

    if not hasattr(_pytorch_utils, 'find_pruneable_heads_and_indices'):
        def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
            """Find heads to prune and their indices (from transformers 4.x)."""
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                mask[head] = 0
            head_index = torch.nonzero(mask.flatten()).squeeze()
            return heads, head_index
        _pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

    if not hasattr(_pytorch_utils, 'prune_conv1d_layer'):
        def prune_conv1d_layer(layer, index, dim=1):
            """Prune a Conv1D layer (from transformers 4.x)."""
            index = index.to(layer.weight.device)
            W = layer.weight.index_select(dim, index).clone()
            if layer.bias is not None:
                if dim == 1:
                    b = layer.bias[index].clone()
                else:
                    b = layer.bias
            else:
                b = None
            new_layer = type(layer)(W.shape[1], W.shape[0])
            new_layer.weight = nn.Parameter(W)
            new_layer.bias = nn.Parameter(b) if b is not None else None
            return new_layer
        _pytorch_utils.prune_conv1d_layer = prune_conv1d_layer
except Exception as _e:
    if _TF_MAJOR >= 5:
        warnings.warn(f"[indextts._compat] Section 2 (pytorch_utils) failed: {_e}")
    _pytorch_utils = None

# ===========================================================================
# 3. tokenization_utils: ExtensionsTrie
# ===========================================================================
try:
    import transformers.tokenization_utils as _tokenization_utils

    if not hasattr(_tokenization_utils, 'ExtensionsTrie'):
        class ExtensionsTrie:
            """A trie for tokenization extensions (from transformers 4.x)."""
            def __init__(self):
                self.data = {}
            def add(self, word):
                node = self.data
                for ch in word:
                    if ch not in node:
                        node[ch] = {}
                    node[ch] = node
                node[None] = word
            def split(self, text):
                """Split text using the trie."""
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
        _tokenization_utils.ExtensionsTrie = ExtensionsTrie
except Exception as _e:
    if _TF_MAJOR >= 5:
        warnings.warn(f"[indextts._compat] Section 3 (tokenization_utils) failed: {_e}")

# ===========================================================================
# 4. generation.beam_constraints module (entire module missing)
# ===========================================================================
try:
    if 'transformers.generation.beam_constraints' not in sys.modules:
        _beam_constraints = types.ModuleType('transformers.generation.beam_constraints')

        class Constraint:
            """Base constraint class (from transformers 4.x)."""
            def __init__(self):
                self.test_state = None
                self.sequence = None
            def advance(self):
                raise NotImplementedError
            def does_advance(self, token_id):
                raise NotImplementedError
            def reset(self):
                raise NotImplementedError
            def remaining(self):
                raise NotImplementedError
            def copy(self):
                raise NotImplementedError

        class PhrasalConstraint(Constraint):
            def __init__(self, token_ids):
                self.token_ids = token_ids
                self.reset()
            def advance(self):
                return self.token_ids[self.seqlen]
            def does_advance(self, token_id):
                if self.seqlen >= len(self.token_ids):
                    return False
                return token_id == self.token_ids[self.seqlen]
            def reset(self, token_ids=None):
                if token_ids is not None:
                    self.token_ids = token_ids
                self.seqlen = 0
            def remaining(self):
                return len(self.token_ids) - self.seqlen
            def copy(self):
                return PhrasalConstraint(self.token_ids.copy())

        class DisjunctiveConstraint(Constraint):
            def __init__(self, constraints):
                self.constraints = constraints
                self.current_constraint = None
                self.reset()
            def advance(self):
                return self.current_constraint.advance()
            def does_advance(self, token_id):
                for constraint in self.constraints:
                    if constraint.does_advance(token_id):
                        self.current_constraint = constraint
                        return True
                return False
            def reset(self):
                for constraint in self.constraints:
                    constraint.reset()
                self.current_constraint = None
            def remaining(self):
                if self.current_constraint is None:
                    return min(c.remaining() for c in self.constraints)
                return self.current_constraint.remaining()
            def copy(self):
                return DisjunctiveConstraint([c.copy() for c in self.constraints])

        class ConstraintListState:
            def __init__(self, constraints):
                self.constraints = constraints
                self.reset()
            def reset(self):
                for constraint in self.constraints:
                    constraint.reset()
                self.current_constraint = None
            def add(self, token_id):
                if self.current_constraint is None:
                    for constraint in self.constraints:
                        if constraint.does_advance(token_id):
                            self.current_constraint = constraint
                            break
                if self.current_constraint is not None and self.current_constraint.does_advance(token_id):
                    self.current_constraint.advance()
                    if self.current_constraint.remaining() == 0:
                        self.constraints.remove(self.current_constraint)
                        self.current_constraint = None
            def get_current_token(self):
                if self.current_constraint is None:
                    return None
                return self.current_constraint.advance()

        _beam_constraints.Constraint = Constraint
        _beam_constraints.PhrasalConstraint = PhrasalConstraint
        _beam_constraints.DisjunctiveConstraint = DisjunctiveConstraint
        _beam_constraints.ConstraintListState = ConstraintListState
        sys.modules['transformers.generation.beam_constraints'] = _beam_constraints
        import transformers.generation as _generation_mod
        _generation_mod.beam_constraints = _beam_constraints
except Exception as _e:
    if _TF_MAJOR >= 5:
        warnings.warn(f"[indextts._compat] Section 4 (beam_constraints) failed: {_e}")

# ===========================================================================
# 5. generation.beam_search module (entire module missing)
# ===========================================================================
try:
    if 'transformers.generation.beam_search' not in sys.modules:
        _beam_search = types.ModuleType('transformers.generation.beam_search')

        class BeamScorer:
            """Base beam scorer (from transformers 4.x)."""
            def process(self, *args, **kwargs):
                raise NotImplementedError
            def finalize(self, *args, **kwargs):
                raise NotImplementedError

        class BeamSearchScorer(BeamScorer):
            def __init__(self, batch_size, num_beams, device, length_penalty=1.0,
                         do_early_stopping=False, num_beam_hyps_to_keep=1,
                         num_beam_groups=1, max_length=None):
                self.batch_size = batch_size
                self.num_beams = num_beams
                self.device = device
                self.length_penalty = length_penalty
                self.do_early_stopping = do_early_stopping
                self.num_beam_hyps_to_keep = num_beam_hyps_to_keep
                self.num_beam_groups = num_beam_groups
                self.group_size = num_beams // num_beam_groups if num_beam_groups > 0 else num_beams
                self._is_init = False
                self._beam_hyps = []
                for _ in range(batch_size):
                    self._beam_hyps.append([])
            def process(self, input_ids, next_scores, next_tokens, next_indices,
                        pad_token_id=None, eos_token_id=None, beam_indices=None,
                        group_index=None, decoder_prompt_len=None):
                cur_len = input_ids.shape[-1]
                batch_size = len(self._beam_hyps)
                if not self._is_init:
                    self._is_init = True
                next_beam_scores = torch.zeros((batch_size, self.group_size), dtype=next_scores.dtype, device=next_scores.device)
                next_beam_tokens = torch.zeros((batch_size, self.group_size), dtype=next_tokens.dtype, device=next_tokens.device)
                next_beam_indices = torch.zeros((batch_size, self.group_size), dtype=next_indices.dtype, device=next_indices.device)
                for batch_idx in range(batch_size):
                    if self.do_early_stopping:
                        pass
                    beam_idx = 0
                    for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                        zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
                    ):
                        batch_beam_idx = batch_idx * self.group_size + next_index.item()
                        if beam_idx >= self.group_size:
                            break
                        next_beam_scores[batch_idx, beam_idx] = next_score
                        next_beam_tokens[batch_idx, beam_idx] = next_token
                        next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                        beam_idx += 1
                return {
                    "next_beam_scores": next_beam_scores.view(-1),
                    "next_beam_tokens": next_beam_tokens.view(-1),
                    "next_beam_indices": next_beam_indices.view(-1),
                }
            def finalize(self, input_ids, final_beam_scores, final_beam_tokens,
                         final_beam_indices, pad_token_id=None, eos_token_id=None,
                         beam_indices=None, decoder_prompt_len=None):
                batch_size = len(self._beam_hyps)
                best = []
                for batch_idx in range(batch_size):
                    if self._beam_hyps[batch_idx]:
                        sorted_hyps = sorted(self._beam_hyps[batch_idx], key=lambda x: x[0])
                        best.append(sorted_hyps[-1])
                    else:
                        best.append((final_beam_scores[batch_idx].item(),
                                     final_beam_tokens[batch_idx].unsqueeze(0)))
                sequences = torch.cat([x[1] for x in best], dim=0)
                sequence_scores = torch.tensor([x[0] for x in best], device=final_beam_scores.device)
                return sequences, sequence_scores

        class ConstrainedBeamSearchScorer(BeamSearchScorer):
            def __init__(self, *args, constraints=None, **kwargs):
                super().__init__(*args, **kwargs)
                self.constraints = constraints or []

        _beam_search.BeamScorer = BeamScorer
        _beam_search.BeamSearchScorer = BeamSearchScorer
        _beam_search.ConstrainedBeamSearchScorer = ConstrainedBeamSearchScorer
        sys.modules['transformers.generation.beam_search'] = _beam_search
        _generation_mod.beam_search = _beam_search
except Exception as _e:
    if _TF_MAJOR >= 5:
        warnings.warn(f"[indextts._compat] Section 5 (beam_search) failed: {_e}")

# ===========================================================================
# 6. generation.candidate_generator: _crop_past_key_values
# ===========================================================================
try:
    import transformers.generation.candidate_generator as _candidate_gen

    if not hasattr(_candidate_gen, '_crop_past_key_values'):
        def _crop_past_key_values(model, past_key_values, maximum_length):
            """Crop past key values to maximum length (from transformers 4.x)."""
            new_past = []
            for layer in past_key_values:
                if isinstance(layer, tuple) and len(layer) == 2:
                    new_past.append((layer[0][:, :, :maximum_length], layer[1][:, :, :maximum_length]))
                else:
                    new_past.append(layer)
            return tuple(new_past) if past_key_values is not None else past_key_values
        _candidate_gen._crop_past_key_values = _crop_past_key_values
except Exception as _e:
    if _TF_MAJOR >= 5:
        warnings.warn(f"[indextts._compat] Section 6 (candidate_generator) failed: {_e}")

# ===========================================================================
# 7. generation.configuration_utils: NEED_SETUP_CACHE_CLASSES_MAPPING, QUANT_BACKEND_CLASSES_MAPPING
# ===========================================================================
try:
    import transformers.generation.configuration_utils as _config_utils

    if not hasattr(_config_utils, 'NEED_SETUP_CACHE_CLASSES_MAPPING'):
        _config_utils.NEED_SETUP_CACHE_CLASSES_MAPPING = {}

    if not hasattr(_config_utils, 'QUANT_BACKEND_CLASSES_MAPPING'):
        _config_utils.QUANT_BACKEND_CLASSES_MAPPING = {}
except Exception as _e:
    if _TF_MAJOR >= 5:
        warnings.warn(f"[indextts._compat] Section 7 (configuration_utils) failed: {_e}")

# ===========================================================================
# 8. generation.logits_process: HammingDiversityLogitsProcessor
# ===========================================================================
try:
    import transformers.generation.logits_process as _logits_process

    if not hasattr(_logits_process, 'HammingDiversityLogitsProcessor'):
        class HammingDiversityLogitsProcessor:
            """Hamming diversity logits processor (from transformers 4.x)."""
            def __init__(self, diversity_penalty, num_beams, num_beam_groups):
                self._diversity_penalty = diversity_penalty
                self._num_beams = num_beams
                self._num_beam_groups = num_beam_groups
            def __call__(self, input_ids, scores, current_tokens, beam_group_idx):
                batch_size = current_tokens.shape[0] // self._num_beams
                group_start_idx = beam_group_idx * self._num_beams
                group_end_idx = min(group_start_idx + self._num_beams, self._num_beams * self._num_beam_groups)
                group_size = group_end_idx - group_start_idx
                if group_size <= 0:
                    return scores
                vocab_size = scores.shape[-1]
                current_tokens = current_tokens.reshape(batch_size, self._num_beams, vocab_size)
                current_tokens = current_tokens[:, group_start_idx:group_end_idx, :]
                scores_group = scores[batch_size * group_start_idx: batch_size * group_end_idx, :]
                penalty = torch.mean(current_tokens.float(), dim=1, keepdim=True) * self._diversity_penalty
                scores_group = scores_group - penalty
                scores[batch_size * group_start_idx: batch_size * group_end_idx, :] = scores_group
                return scores
        _logits_process.HammingDiversityLogitsProcessor = HammingDiversityLogitsProcessor
except Exception as _e:
    if _TF_MAJOR >= 5:
        warnings.warn(f"[indextts._compat] Section 8 (logits_process) failed: {_e}")

# ===========================================================================
# 9. modeling_utils: SequenceSummary
#    WARNING: This import can trigger torchvision errors if torch/torchvision
#    versions are mismatched. Wrapped in try/except to prevent cascade failure.
# ===========================================================================
try:
    import transformers.modeling_utils as _modeling_utils

    if not hasattr(_modeling_utils, 'SequenceSummary'):
        class SequenceSummary(nn.Module):
            """Sequence summary head (from transformers 4.x)."""
            def __init__(self, config):
                super().__init__()
                self.summary_type = getattr(config, "summary_type", "last")
                if self.summary_type == "attn":
                    self.summary = nn.Linear(config.hidden_size, 1)
                else:
                    self.summary = nn.Identity()
                if hasattr(config, "summary_use_proj") and config.summary_use_proj:
                    if hasattr(config, "summary_proj_to_labels") and config.summary_proj_to_labels and config.num_labels > 0:
                        num_classes = config.num_labels
                    else:
                        num_classes = config.hidden_size
                    self.summary = nn.Linear(config.hidden_size, num_classes)
                if hasattr(config, "summary_activation"):
                    self.activation = nn.ReLU() if config.summary_activation == "tanh" else nn.Tanh()
                else:
                    self.activation = None
                self.first_dropout = nn.Dropout(getattr(config, "summary_first_dropout", 0.1)) if hasattr(config, "summary_first_dropout") else nn.Identity()
                self.last_dropout = nn.Dropout(getattr(config, "summary_last_dropout", 0.1)) if hasattr(config, "summary_last_dropout") else nn.Identity()
            def forward(self, hidden_states, cls_index=None):
                if self.summary_type == "last":
                    output = hidden_states[:, -1]
                elif self.summary_type == "first":
                    output = hidden_states[:, 0]
                elif self.summary_type == "mean":
                    output = hidden_states.mean(dim=1)
                elif self.summary_type == "cls_index":
                    if cls_index is None:
                        cls_index = torch.full((hidden_states.shape[0],), hidden_states.shape[-1] - 1, device=hidden_states.device)
                    output = hidden_states[torch.arange(hidden_states.shape[0], device=hidden_states.device), cls_index]
                elif self.summary_type == "attn":
                    output = self.summary(hidden_states).squeeze(-1)
                else:
                    output = hidden_states
                output = self.first_dropout(output)
                output = self.summary(output)
                if self.activation is not None:
                    output = self.activation(output)
                output = self.last_dropout(output)
                return output
        _modeling_utils.SequenceSummary = SequenceSummary
except Exception as _e:
    if _TF_MAJOR >= 5:
        warnings.warn(f"[indextts._compat] Section 9 (modeling_utils) failed: {_e}")

# ===========================================================================
# 10. utils.model_parallel_utils module (entire module missing)
# ===========================================================================
try:
    if 'transformers.utils.model_parallel_utils' not in sys.modules:
        _mp_utils = types.ModuleType('transformers.utils.model_parallel_utils')

        def get_device_map(n_layers, devices, max_memory=None):
            """Get a device map for model parallelism (from transformers 4.x)."""
            if max_memory is None:
                max_memory = {}
                for i, device in enumerate(devices):
                    max_memory[i] = torch.cuda.get_device_properties(device).total_memory
            layers_per_device = n_layers // len(devices)
            device_map = {}
            for i in range(n_layers):
                device_map[i] = i // layers_per_device if i // layers_per_device < len(devices) else len(devices) - 1
            return device_map

        def assert_device_map(device_map, num_devices):
            """Assert device map is valid (from transformers 4.x)."""
            max_device = max(device_map.values()) if device_map else 0
            if max_device >= num_devices:
                raise ValueError(
                    f"The device_map contains a device {max_device} but only {num_devices} devices are available."
                )

        _mp_utils.get_device_map = get_device_map
        _mp_utils.assert_device_map = assert_device_map
        sys.modules['transformers.utils.model_parallel_utils'] = _mp_utils
        import transformers.utils as _transformers_utils
        _transformers_utils.model_parallel_utils = _mp_utils
except Exception as _e:
    if _TF_MAJOR >= 5:
        warnings.warn(f"[indextts._compat] Section 10 (model_parallel_utils) failed: {_e}")

# ===========================================================================
# 12. pytorch_utils: prune_layer (depends on section 2)
# ===========================================================================
try:
    if _pytorch_utils is not None and not hasattr(_pytorch_utils, 'prune_layer'):
        def prune_layer(layer, index, dim=None):
            """Generic prune function that delegates to prune_conv1d_layer or prune_linear_layer."""
            if hasattr(layer, 'weight') and layer.weight.dim() == 2:
                return prune_conv1d_layer(layer, index, dim=1 if dim is None else dim)
            else:
                from transformers.pytorch_utils import prune_linear_layer
                return prune_linear_layer(layer, index, dim=0 if dim is None else dim)
        _pytorch_utils.prune_layer = prune_layer
except Exception as _e:
    if _TF_MAJOR >= 5:
        warnings.warn(f"[indextts._compat] Section 12 (prune_layer) failed: {_e}")

# ===========================================================================
# 13. transformers.utils: missing constants and functions
# ===========================================================================
try:
    import transformers.utils as _tf_utils

    if not hasattr(_tf_utils, 'FLAX_WEIGHTS_NAME'):
        _tf_utils.FLAX_WEIGHTS_NAME = "flax_model.msgpack"
    if not hasattr(_tf_utils, 'TF2_WEIGHTS_NAME'):
        _tf_utils.TF2_WEIGHTS_NAME = "tf_model.h5"
    if not hasattr(_tf_utils, 'TF_WEIGHTS_NAME'):
        _tf_utils.TF_WEIGHTS_NAME = "tf_model.h5"

    if not hasattr(_tf_utils, 'download_url'):
        def download_url(url, proxies=None):
            import urllib.request
            import tempfile
            try:
                with urllib.request.urlopen(url) as response:
                    content = response.read()
                fd, path = tempfile.mkstemp()
                with os.fdopen(fd, 'wb') as f:
                    f.write(content)
                return path
            except Exception as e:
                warnings.warn(f"download_url failed: {e}")
                return None
        _tf_utils.download_url = download_url

    if not hasattr(_tf_utils, 'is_offline_mode'):
        def is_offline_mode():
            return os.environ.get("TRANSFORMERS_OFFLINE", "0").upper() in ("1", "TRUE", "YES")
        _tf_utils.is_offline_mode = is_offline_mode

    if not hasattr(_tf_utils, 'is_remote_url'):
        def is_remote_url(url):
            if url is None:
                return False
            return url.startswith("http://") or url.startswith("https://")
        _tf_utils.is_remote_url = is_remote_url

    if not hasattr(_tf_utils, 'is_safetensors_available'):
        try:
            import safetensors
            _tf_utils.is_safetensors_available = lambda: True
        except ImportError:
            _tf_utils.is_safetensors_available = lambda: False

    if not hasattr(_tf_utils, 'is_torch_sdpa_available'):
        def is_torch_sdpa_available():
            try:
                from torch.nn.functional import scaled_dot_product_attention
                return True
            except ImportError:
                return False
        _tf_utils.is_torch_sdpa_available = is_torch_sdpa_available

    if not hasattr(_tf_utils, 'DUMMY_INPUTS'):
        _tf_utils.DUMMY_INPUTS = [[[7, 6, 0, 0, 1], [1, 2, 999, 888, 777]]]

    if not hasattr(_tf_utils, 'ContextManagers'):
        from contextlib import ExitStack
        class ContextManagers:
            def __init__(self, contexts):
                self.contexts = list(contexts)
            def __enter__(self):
                self.stack = ExitStack()
                self.stack.__enter__()
                for ctx in self.contexts:
                    self.stack.enter_context(ctx)
                return self
            def __exit__(self, *args, **kwargs):
                return self.stack.__exit__(*args, **kwargs)
        _tf_utils.ContextManagers = ContextManagers

    if not hasattr(_tf_utils, 'ACCELERATE_MIN_VERSION'):
        _tf_utils.ACCELERATE_MIN_VERSION = "1.0.0"
    if not hasattr(_tf_utils, 'ADAPTER_SAFE_WEIGHTS_NAME'):
        _tf_utils.ADAPTER_SAFE_WEIGHTS_NAME = "adapter_model.safetensors"
    if not hasattr(_tf_utils, 'ADAPTER_WEIGHTS_NAME'):
        _tf_utils.ADAPTER_WEIGHTS_NAME = "adapter_model.bin"

    if not hasattr(_tf_utils, 'PushToHubMixin'):
        class PushToHubMixin:
            def push_to_hub(self, *args, **kwargs):
                raise NotImplementedError("push_to_hub not available")
        _tf_utils.PushToHubMixin = PushToHubMixin

    if not hasattr(_tf_utils, 'copy_func'):
        def copy_func(f):
            g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__,
                                    argdefs=f.__defaults__, closure=f.__closure__)
            g = functools.update_wrapper(g, f)
            g.__kwdefaults__ = f.__kwdefaults__
            return g
        _tf_utils.copy_func = copy_func

    if not hasattr(_tf_utils, 'extract_commit_hash'):
        def extract_commit_hash(resolved_file, commit_hash):
            return commit_hash
        _tf_utils.extract_commit_hash = extract_commit_hash

    if not hasattr(_tf_utils, 'has_file'):
        def has_file(path_or_repo, filename, revision=None, proxy=None, token=None):
            if os.path.isdir(path_or_repo):
                return os.path.isfile(os.path.join(path_or_repo, filename))
            return False
        _tf_utils.has_file = has_file

    if not hasattr(_tf_utils, 'is_flash_attn_2_available'):
        _tf_utils.is_flash_attn_2_available = lambda: False
    if not hasattr(_tf_utils, 'is_bitsandbytes_available'):
        _tf_utils.is_bitsandbytes_available = lambda: False
    if not hasattr(_tf_utils, 'is_optimum_available'):
        _tf_utils.is_optimum_available = lambda: False
    if not hasattr(_tf_utils, 'is_peft_available'):
        _tf_utils.is_peft_available = lambda: False
    if not hasattr(_tf_utils, 'is_torch_xla_available'):
        _tf_utils.is_torch_xla_available = lambda: False

    if not hasattr(_tf_utils, 'replace_return_docstrings'):
        def replace_return_docstrings(output_type=None, config_class=None):
            def docstring_decorator(fn):
                return fn
            return docstring_decorator
        _tf_utils.replace_return_docstrings = replace_return_docstrings

    if not hasattr(_tf_utils, 'strtobool'):
        def strtobool(value):
            value = value.lower()
            if value in ('y', 'yes', 't', 'true', 'on', '1'):
                return 1
            elif value in ('n', 'no', 'f', 'false', 'off', '0'):
                return 0
            raise ValueError(f"invalid truth value {value!r}")
        _tf_utils.strtobool = strtobool
except Exception as _e:
    if _TF_MAJOR >= 5:
        warnings.warn(f"[indextts._compat] Section 13 (transformers.utils) failed: {_e}")

# ===========================================================================
# 14-15. modeling_flax/tf_pytorch_utils (dummy modules)
# ===========================================================================
try:
    if 'transformers.modeling_flax_pytorch_utils' not in sys.modules:
        _flax_utils = types.ModuleType('transformers.modeling_flax_pytorch_utils')
        def load_flax_checkpoint_in_pytorch_model(model, flax_checkpoint_path, allow_pickle=False):
            raise NotImplementedError("Flax is not available on this platform.")
        _flax_utils.load_flax_checkpoint_in_pytorch_model = load_flax_checkpoint_in_pytorch_model
        sys.modules['transformers.modeling_flax_pytorch_utils'] = _flax_utils
except Exception:
    pass

try:
    if 'transformers.modeling_tf_pytorch_utils' not in sys.modules:
        _tf_utils_mod = types.ModuleType('transformers.modeling_tf_pytorch_utils')
        def load_tf2_checkpoint_in_pytorch_model(model, tf_checkpoint_path, config=None, allow_pickle=False):
            raise NotImplementedError("TensorFlow is not available on this platform.")
        _tf_utils_mod.load_tf2_checkpoint_in_pytorch_model = load_tf2_checkpoint_in_pytorch_model
        sys.modules['transformers.modeling_tf_pytorch_utils'] = _tf_utils_mod
except Exception:
    pass

# ===========================================================================
# 17. utils.import_utils: missing items
# ===========================================================================
try:
    import transformers.utils.import_utils as _import_utils

    if not hasattr(_import_utils, 'ENV_VARS_TRUE_VALUES'):
        _import_utils.ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
    if not hasattr(_import_utils, 'is_sagemaker_mp_enabled'):
        _import_utils.is_sagemaker_mp_enabled = lambda: False
    if not hasattr(_import_utils, 'is_torch_fx_proxy'):
        def is_torch_fx_proxy(obj):
            return False
        _import_utils.is_torch_fx_proxy = is_torch_fx_proxy
except Exception:
    pass

# ===========================================================================
# 18-23. Optional sections (all wrapped in try/except already)
# ===========================================================================
try:
    import transformers.integrations as _integrations
    if not hasattr(_integrations, 'PeftAdapterMixin'):
        class PeftAdapterMixin:
            pass
        _integrations.PeftAdapterMixin = PeftAdapterMixin
    if not hasattr(_integrations, 'deepspeed_config'):
        class deepspeed_config:
            def __init__(self, *args, **kwargs):
                pass
        _integrations.deepspeed_config = deepspeed_config
except Exception:
    pass

try:
    import transformers.loss.loss_utils as _loss_utils
    if not hasattr(_loss_utils, 'LOSS_MAPPING'):
        _loss_utils.LOSS_MAPPING = {}
except Exception:
    pass

try:
    import transformers.quantizers as _quantizers
    if not hasattr(_quantizers, 'AutoHfQuantizer'):
        class AutoHfQuantizer:
            @staticmethod
            def from_config(*args, **kwargs):
                return None
        _quantizers.AutoHfQuantizer = AutoHfQuantizer
    if not hasattr(_quantizers, 'HfQuantizer'):
        class HfQuantizer:
            pass
        _quantizers.HfQuantizer = HfQuantizer
except Exception:
    pass

try:
    import transformers.quantizers.quantizers_utils as _quant_utils2
    if not hasattr(_quant_utils2, 'get_module_from_name'):
        def get_module_from_name(model, name):
            parts = name.split('.')
            module = model
            for part in parts:
                module = getattr(module, part)
            return module
        _quant_utils2.get_module_from_name = get_module_from_name
except Exception:
    pass

try:
    import transformers.safetensors_conversion as _st_conversion
    if not hasattr(_st_conversion, 'auto_conversion'):
        def auto_conversion(*args, **kwargs):
            return None, None
        _st_conversion.auto_conversion = auto_conversion
except Exception:
    pass

try:
    import transformers.utils.hub as _hub
    if not hasattr(_hub, 'convert_file_size_to_int'):
        def convert_file_size_to_int(size):
            if isinstance(size, int):
                return size
            if size.endswith("GB"):
                return int(size[:-2]) * (1024 ** 3)
            if size.endswith("MB"):
                return int(size[:-2]) * (1024 ** 2)
            if size.endswith("KB"):
                return int(size[:-2]) * 1024
            return int(size)
        _hub.convert_file_size_to_int = convert_file_size_to_int
    if not hasattr(_hub, 'create_and_tag_model_card'):
        _hub.create_and_tag_model_card = lambda *a, **k: None
    if not hasattr(_hub, 'get_checkpoint_shard_files'):
        _hub.get_checkpoint_shard_files = lambda *a, **k: (None, None)
except Exception:
    pass

try:
    import transformers.utils.quantization_config as _quant_config
    if not hasattr(_quant_config, 'BitsAndBytesConfig'):
        class BitsAndBytesConfig:
            def __init__(self, *args, **kwargs):
                pass
        _quant_config.BitsAndBytesConfig = BitsAndBytesConfig
    if not hasattr(_quant_config, 'QuantizationMethod'):
        from enum import Enum
        class QuantizationMethod(str, Enum):
            BNB = "bnb"
            GPTQ = "gptq"
            AWQ = "awq"
            SHARDS = "shards"
            QUANTO = "quanto"
            EETQ = "eetq"
            HQQ = "hqq"
        _quant_config.QuantizationMethod = QuantizationMethod
except Exception:
    pass

if _TF_MAJOR >= 5:
    print("[indextts._compat] Compatibility shim loaded successfully for transformers 5.x")
