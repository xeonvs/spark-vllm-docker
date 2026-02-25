#!/bin/bash
set -euo pipefail

SITE=/usr/local/lib/python3.12/dist-packages

echo "Applying PR #26807: GatedDeltaNet all-mode prefix caching"

# Apply the bulk of the PR (skips test file and one failing hunk)
cd "$SITE" && patch -p1 --batch --forward --force < /workspace/mods/gdn-prefix-caching/pr26807.diff 2>&1 || true

# Keep qwen3_next.py from the installed vLLM revision and patch surgically.
# Full-file overwrite from this mod may break newer API layouts.
cd "$SITE" && python3 -c "
path = 'vllm/model_executor/models/qwen3_next.py'
with open(path) as f:
    src = f.read()

patched = False

# Newer vLLM revisions moved Attention into vllm.attention.layer.
old_import = 'from vllm.model_executor.layers.attention import Attention'
new_import = 'from vllm.attention.layer import Attention'
if old_import in src and new_import not in src:
    src = src.replace(old_import, new_import)
    patched = True
    print('Patched qwen3_next.py: Attention import compatibility')

# Remove upstream guard that blocks prefix caching for Qwen3Next.
if 'Qwen3Next currently does not support prefix caching' in src:
    lines = src.splitlines(keepends=True)
    out = []
    i = 0
    removed = False
    while i < len(lines):
        if 'assert not cache_config.enable_prefix_caching' in lines[i]:
            removed = True
            i += 1
            while i < len(lines) and ')' not in lines[i]:
                i += 1
            if i < len(lines):
                i += 1
            continue
        out.append(lines[i])
        i += 1
    if removed:
        src = ''.join(out)
        patched = True
        print('Patched qwen3_next.py: removed prefix caching assert')

# Clean up cache_config assignment if no longer used.
cache_cfg_line = '        cache_config = vllm_config.cache_config\\n'
if cache_cfg_line in src:
    trial = src.replace(cache_cfg_line, '', 1)
    if 'cache_config.' not in trial:
        src = trial
        patched = True
        print('Patched qwen3_next.py: removed unused cache_config assignment')

if patched:
    with open(path, 'w') as f:
        f.write(src)
else:
    print('qwen3_next.py already compatible')
"

# Newer vLLM revisions removed cache_config.mamba_cache_mode from CLI/config.
# Keep patched GDN backend compatible by checking enable_prefix_caching instead.
cd "$SITE" && python3 -c "
import re
path = 'vllm/v1/attention/backends/gdn_attn.py'
with open(path) as f:
    src = f.read()
patched = False

old = 'all_prefix_caching_enabled = vllm_config.cache_config.mamba_cache_mode == \"all\"'
new = 'all_prefix_caching_enabled = bool(getattr(vllm_config.cache_config, \"enable_prefix_caching\", False))'
if old in src:
    src = src.replace(old, new)
    patched = True
    print('Patched gdn_attn.py: mamba_cache_mode compatibility')

if 'BaseMambaAttentionMetadataBuilder' in src and 'from vllm.v1.attention.backends.mamba_attn import BaseMambaAttentionMetadataBuilder' not in src:
    anchor = 'from vllm.config import VllmConfig\\n'
    inject = anchor + 'from vllm.v1.attention.backends.mamba_attn import BaseMambaAttentionMetadataBuilder\\n'
    if anchor in src:
        src = src.replace(anchor, inject)
        patched = True
        print('Patched gdn_attn.py: BaseMambaAttentionMetadataBuilder import')

if 'cdiv(' in src and 'from vllm.utils.math_utils import cdiv' not in src:
    anchor = 'from vllm.config import VllmConfig\\n'
    inject = anchor + 'from vllm.utils.math_utils import cdiv\\n'
    if anchor in src:
        src = src.replace(anchor, inject)
        patched = True
        print('Patched gdn_attn.py: cdiv import')

if 'if prefix_caching_enabled:' in src and not re.search(r'(?m)^\\s*prefix_caching_enabled\\s*=', src):
    marker = '        if prefix_caching_enabled:'
    inject = '''        prefix_caching_enabled = bool(getattr(self.vllm_config.cache_config, 'enable_prefix_caching', False))
        block_size: int | None = None
        chunk_size_value: int | None = None
        if prefix_caching_enabled:
            block_size = self.kv_cache_spec.block_size
            chunk_size_value = self.chunk_size

        # APC related tensors
        state_indices_tensor: torch.Tensor | None = None
        block_idx_first_scheduled_token: torch.Tensor | None = None
        block_idx_last_computed_token: torch.Tensor | None = None
        block_idx_last_scheduled_token: torch.Tensor | None = None
        block_idx_first_scheduled_token_p: torch.Tensor | None = None
        num_computed_tokens_p: torch.Tensor | None = None
        seq_idx_p: torch.Tensor | None = None
        cu_chunk_seqlen_p: torch.Tensor | None = None
        last_chunk_indices_p: torch.Tensor | None = None
        non_spec_query_start_loc_cpu: torch.Tensor | None = None

        if prefix_caching_enabled:'''
    if marker in src:
        src = src.replace(marker, inject, 1)
        patched = True
        print('Patched gdn_attn.py: prefix_caching_enabled initialization block')

if 'assert non_spec_query_start_loc_cpu is not None' in src and 'non_spec_query_start_loc_cpu = non_spec_query_start_loc.cpu()' not in src:
    old = '            assert non_spec_query_start_loc_cpu is not None\\n'
    new = '''            assert non_spec_query_start_loc is not None
            non_spec_query_start_loc_cpu = non_spec_query_start_loc.cpu()
            assert non_spec_query_start_loc_cpu is not None
'''
    if old in src:
        src = src.replace(old, new, 1)
        patched = True
        print('Patched gdn_attn.py: non_spec_query_start_loc_cpu initialization')

if patched:
    with open(path, 'w') as f:
        f.write(src)
else:
    print('gdn_attn.py already compatible')
"

# Fix block size: make HybridAttentionMambaModelConfig check the model class
# for get_mamba_chunk_size() instead of only using ModelConfig's 2048 default.
# GDN uses chunk_size=64, not Mamba2's 2048.
cd "$SITE" && python3 -c "
path = 'vllm/model_executor/models/config.py'
with open(path) as f:
    src = f.read()

old = 'base_chunk_size = mamba_block_size or model_config.get_mamba_chunk_size()'
new = '''# Check model class for chunk_size first (e.g. GDN=64), fall back to ModelConfig default (2048)
            model_chunk_size = getattr(model_cls, 'get_mamba_chunk_size', lambda: None)()
            base_chunk_size = mamba_block_size or model_chunk_size or model_config.get_mamba_chunk_size()'''

if old in src:
    src = src.replace(old, new)
    with open(path, 'w') as f:
        f.write(src)
    print('Patched config.py: model class chunk_size lookup')
else:
    print('config.py already patched or line not found')
"

if grep -q "Qwen3Next currently does not support prefix caching" "$SITE/vllm/model_executor/models/qwen3_next.py"; then
  echo "Error: qwen3_next.py still contains prefix-caching assert after patching" >&2
  exit 1
fi

if grep -q "if prefix_caching_enabled:" "$SITE/vllm/v1/attention/backends/gdn_attn.py" && \
   ! grep -Eq "^[[:space:]]*prefix_caching_enabled[[:space:]]*=" "$SITE/vllm/v1/attention/backends/gdn_attn.py"; then
  echo "Error: gdn_attn.py uses prefix_caching_enabled but does not initialize it" >&2
  exit 1
fi

if grep -q "assert non_spec_query_start_loc_cpu is not None" "$SITE/vllm/v1/attention/backends/gdn_attn.py" && \
   ! grep -q "non_spec_query_start_loc_cpu = non_spec_query_start_loc.cpu()" "$SITE/vllm/v1/attention/backends/gdn_attn.py"; then
  echo "Error: gdn_attn.py asserts non_spec_query_start_loc_cpu but never initializes it from non_spec_query_start_loc" >&2
  exit 1
fi

if grep -q "getattr(self.vllm_config.cache_config, enable_prefix_caching" "$SITE/vllm/v1/attention/backends/gdn_attn.py"; then
  echo "Error: gdn_attn.py has malformed getattr(..., enable_prefix_caching, ...); expected quoted key" >&2
  exit 1
fi

if grep -q "cdiv(" "$SITE/vllm/v1/attention/backends/gdn_attn.py" && \
   ! grep -q "from vllm.utils.math_utils import cdiv" "$SITE/vllm/v1/attention/backends/gdn_attn.py"; then
  echo "Error: gdn_attn.py uses cdiv without import" >&2
  exit 1
fi

if grep -q "BaseMambaAttentionMetadataBuilder" "$SITE/vllm/v1/attention/backends/gdn_attn.py" && \
   ! grep -q "from vllm.v1.attention.backends.mamba_attn import BaseMambaAttentionMetadataBuilder" "$SITE/vllm/v1/attention/backends/gdn_attn.py"; then
  echo "Error: gdn_attn.py uses BaseMambaAttentionMetadataBuilder without import" >&2
  exit 1
fi

rej_count=$(find "$SITE/vllm" -name '*.rej' | wc -l)
if [ "$rej_count" -gt 0 ]; then
  echo "GDN patch warnings: $rej_count reject file(s) present under $SITE/vllm (non-fatal)"
fi

echo "GDN prefix caching patch applied"
