# Diffusion KV Reuse and Prefix Caching

Status: Draft  
Parent RFC: `diffusion_kv_cache_management.md`

## 1. High-Level Idea and Benefit

### 1.1 How This Fits Into The Existing BDE Design

The BDE RFC already defines the lower-level KV management path:

```text
DiffusionScheduler / DiffusionWorker / DiffusionModelRunner
  -> BDERequestAdapter
  -> vLLM KVCacheManager / BlockPool
  -> BlockTables / slot_mapping
  -> paged attention
```

This document defines the prefix-reuse layer on top of that path. The integration point is `BDERequestAdapter`:

```text
BDERequestAdapter
  request_id
  num_computed_tokens
  num_tokens
  block_hashes                 <- defined by this design
  skip_reading_prefix_cache    <- controlled by this design
```

The parent BDE design already reserves `block_hashes` and
`skip_reading_prefix_cache` for Phase 3 / WP-8. This design defines the
`PrefixKey` fields used to generate `block_hashes`, and the conditions under
which `skip_reading_prefix_cache` can be set to `False`.

When prefix reuse is enabled, the runtime path is:

```text
1. Diffusion request is normalized into BDERequestAdapter.
2. PrefixKey builder computes block_hashes for reusable prefix blocks.
3. KVCacheManager.get_computed_blocks(adapter) checks vLLM's prefix index.
4. Prefix-hit blocks are reused and protected by BlockPool.ref_cnt.
5. KVCacheManager.allocate_slots() allocates only missing blocks.
6. DiffusionModelRunner builds BlockTables / slot_mapping over reused + new blocks.
7. Paged attention reads/writes through the normal BDE KV path.
```

### 1.2 Scope

This v1 design supports **content-addressed cross-request KV reuse**. That means
two independent requests can share physical KV blocks when they produce the same
logical prefix `block_hashes`.

This design defines prefix identity, reuse eligibility, and CFG branch identity
for cross-request KV reuse. Lower-level KV management stays in the BDE/vLLM KV
path.

In scope for v1:

- **Text/system prompt conditioning KV.** KV produced from stable text-side
  conditioning, such as prompt tokens, system prompt tokens, or prompt-template
  outputs.
- **Reference image / first-frame conditioning KV.** KV produced from stable
  visual conditioning, such as reference images, first-frame inputs, or
  preprocessed image tokens/latents used as prefix context.
- **Committed history chunk KV.** KV for already-finalized autoregressive
  history chunks. These are generated or observed chunks that future chunks
  attend to as prefix context.
- **CFG unconditional or negative branch KV.** KV from CFG branches that are not
  the positive conditioned branch, such as unconditional branch KV or
  negative-prompt branch KV.

For all of the above, reuse is allowed only when the corresponding `PrefixKey`
fully captures the KV-affecting identity: content, model/preprocess version,
position, attention/window semantics, KV group, and CFG branch when applicable.

Session-scoped reuse is a future extension. It means a long-lived session keeps
committed KV blocks resident and reuses its existing `BlockTables` across
control steps, so it may skip re-hashing the session history and probing the
prefix index on every step.

In short, v1 should fail closed: if the adapter cannot construct a complete
`PrefixKey` for a KV block, it should not emit a reusable `block_hash` for that
block.

### 1.3 Benefit Over Current Behavior

This document focuses on the additional benefit from cross-request KV reuse, not
the general memory-management benefit of BDE. Today, diffusion models may cache
KV internally, but that reuse is usually private to one request/session or one
model-specific implementation. As a result:

- repeated requests with the same conditioning prefix or committed history
  prefix recompute KV;
- shared CFG unconditional / negative branches are hard to reuse safely;
- each model needs its own ad hoc logic for deciding whether two prefixes are
  equivalent;
- prefix-cache hit/miss behavior is not exposed through one common interface.

BDE prefix reuse turns cross-request reuse into a thin identity layer over
vLLM's existing prefix-cache mechanism: the diffusion side only has to produce
safe `block_hashes`; vLLM handles lookup and shared block lifetime.

| Concern | Current behavior | BDE prefix reuse |
| --- | --- | --- |
| Prefix identity | Custom or absent per model | explicit `PrefixKey` |
| Cross-request lookup | Usually unavailable | vLLM prefix index via `block_hashes` |
| CFG branch sharing | easy to mix branches incorrectly | branch identity is part of the key |
| Shared block lifetime | model-specific if implemented | vLLM shared blocks / `ref_cnt` |
| Observability | inconsistent | common prefix hit/reuse metrics |

The main prefix-cache benefit is therefore narrow and concrete: repeated
conditioning prefixes, committed history prefixes, and compatible CFG branches
can be reused across requests without adding a second diffusion-local prefix
index.

## 2. Proposed Implementation: PrefixKey Fields

The implementation should be a small hash builder that converts
diffusion-visible prefix metadata into deterministic vLLM block hashes.

The hash unit is one managed vLLM/BDE KV block. Each hash is computed from a
flat `DiffusionPrefixKey`:

```python
@dataclass(frozen=True)
class DiffusionPrefixKey:
    schema_version: int                 # PrefixKey schema version.

    # Model identity.
    model_id: str                       # Base model identity.
    model_revision: str | None          # Checkpoint/revision/version of the loaded weights.
    weight_adapter_key: str | None      # Weight-affecting adapters such as LoRA; None when not used.
    quantization_key: str | None        # Quantization mode if it changes model weights or produced KV.

    # Preprocessing identity.
    tokenizer_key: str | None           # Tokenizer identity for text-side inputs.
    prompt_template_key: str | None     # Chat/system/prompt template identity.
    image_processor_key: str | None     # Image/frame preprocessing config, if image inputs affect KV.
    video_processor_key: str | None     # Video preprocessing config, if video inputs affect KV.
    audio_processor_key: str | None     # Audio preprocessing config, if audio inputs affect KV.
    latent_encoder_key: str | None      # VAE/encoder identity for inputs converted to latents before KV.

    # Logical block coordinate.
    parent_block_hash: bytes | None     # Previous logical prefix block hash; None for the first visible block.
    kv_group: str                       # KV cache group name/id; "default" for single-group models.
    logical_block_index: int            # Block index in the logical prefix sequence.
    block_size: int                     # Managed KV block size used to produce this hash.
    role: str                           # Block role, e.g. conditioning, first_frame_prefix, history_chunk.
    modality: str                       # Source modality, e.g. text, image, video, audio, latent, mixed.

    # Committed-history chunk coordinate. None for non-history prefix blocks.
    chunk_index: int | None             # Committed AR chunk index.
    chunk_size: int | None              # Tokens per committed chunk; v1 requires chunk_size % block_size == 0.
    offset_in_chunk: int | None         # Token offset of this block within the committed chunk.

    # Content identity.
    content_hash: bytes                 # Hash of canonical typed model-visible content, including dtype/shape/layout.

    # Semantic identity.
    attention_key: str                  # Attention mask / visibility semantics.
    position_key: str                   # Position/RoPE/indexing semantics.
    window_key: str                     # Chunk-window/sink visibility semantics.
    timestep_key: str | None            # Denoise timestep identity if persistent KV depends on timestep.

    # CFG identity.
    cfg_branch: str                     # none, cond, negative, uncond, text_cfg, image_cfg, etc.
    branch_condition_hash: bytes | None # Hash of branch-specific condition, e.g. negative prompt/image.
    guidance_scale_key: str | None      # Present only if guidance scale changes the KV-producing forward.

    # Reuse isolation.
    cache_salt: str | None              # Reuse isolation boundary; different salts do not share.

    # Extension point.
    extra: tuple[tuple[str, str], ...] = ()  # Sorted key/value pairs for future model-specific identity.
```

The serialized key must be canonical and deterministic across processes and
ranks. The final `block_hash` can be produced by hashing the canonical
serialization.

Field notes:

- Model identity fields describe the model behavior that produces KV:
  `model_id`, `model_revision`, `weight_adapter_key`, and `quantization_key`.
- Preprocessing identity fields describe how raw request inputs become
  model-visible tokens, embeddings, latents, or conditioning tensors:
  `tokenizer_key`, `prompt_template_key`, processor keys, and
  `latent_encoder_key`.
- `cache_salt` describes the sharing boundary, not model behavior or
  preprocessing. Requests with different salts must not share prefix KV. Do not
  put `session_id` in the default key; session-scoped reuse is a separate future
  path.
- Fields that do not apply to a model should be `None`.
- `extra` should not replace the common fields above. It is only for
  model-specific identity that cannot yet be represented by the shared schema.
- `kv_group` is always part of the key. This keeps single-group models simple
  while leaving room for hybrid or multimodal models with separate KV groups.
- `parent_block_hash` chains each block to the previous logical prefix block,
  matching vLLM's prefix-cache behavior. This makes the hash represent the whole
  visible prefix up to this block, not only this block's own content.
- `logical_block_index`, `block_size`, `role`, and `modality` describe where the
  block belongs in the current BDE/vLLM logical prefix. `logical_block_index`
  may differ from the model's absolute `chunk_index` when chunk-window eviction
  or sink chunks are used.
- `chunk_index`, `chunk_size`, and `offset_in_chunk` are required only for
  committed history chunk blocks. For v1 committed history reuse, require:

```text
chunk_size % block_size == 0
```

  This makes each committed chunk decompose into whole managed KV blocks. If a
  model cannot provide this alignment, the adapter must pad/split explicitly or
  skip cross-request reuse for the affected boundary.
- `content_hash` should be computed over canonical typed model-visible content,
  not raw user payload metadata. The hashed payload should include the content
  domain, dtype, shape, layout, and deterministic values. Examples include
  post-template token ids, preprocessed image/first-frame tokens or latents,
  committed history latents, or negative-branch condition content.
- `attention_key`, `position_key`, and `window_key` capture non-content factors
  that change produced KV.
- `timestep_key` is included only if persistent KV depends on denoise timestep.
  If persistent KV is timestep-independent, leave it unset.
- CFG branch mismatch means hash mismatch. Unconditional branch sharing is
  allowed only when the branch is truly request-independent. Negative branch
  sharing is allowed only when `branch_condition_hash` matches.
- `guidance_scale_key` is unset if guidance scale only combines branch outputs
  after KV computation. It is set, or reuse is disabled, if guidance scale
  changes the KV-producing forward pass.
- `extra` is logically a map, but represented as sorted `(key, value)` pairs so
  canonical serialization cannot depend on dict insertion order. An
  implementation may accept a dict and normalize it before hashing.

## 3. Proposed Test Plan

### PrefixKey / Hash Tests

- Same model-visible prefix produces the same `block_hashes`.
- Same committed history chunk prefix produces the same `block_hashes` when
  chunk index, chunk size, offset, KV group, position, and window semantics
  match.
- Same block content under a different `parent_block_hash` produces a different
  block hash.
- Changing an earlier visible prefix block changes the hashes of later chained
  blocks.
- Changing model identity, weight adapter, quantization, tokenizer, prompt
  template, input processor, latent encoder, attention mask, position,
  chunk/window semantics, KV group, modality, or cache salt changes the hash.
- Changing model-visible content dtype, shape, layout, or value changes
  `content_hash` and therefore changes the block hash.
- Changing committed chunk index, chunk size, offset-in-chunk, or KV group
  changes the hash.
- Hash serialization is deterministic and independent of Python object id,
  request id, session id, batch row order, physical GPU rank, or physical block
  id.
- Only full reusable blocks emit hashes; partial or unsafe blocks are skipped.
- Blocks that would cross a committed chunk boundary are skipped unless the
  adapter explicitly pads or splits them into safe full blocks.
- Hashes are emitted in logical block order expected by vLLM
  `KVCacheManager.get_computed_blocks()`.

### CFG Safety Tests

- `cond`, `negative`, and `uncond` branches do not accidentally share hashes.
- Same unconditional branch across requests produces the same hash when it is
  truly request-independent.
- Different negative prompts or image conditions produce different hashes.
- Guidance scale is excluded from the key if it is applied only after branch KV
  computation.
- Guidance scale is included in the key, or reuse is disabled, if it changes the
  branch forward pass that produces KV.

### Adapter and Prefix-Cache Integration Tests

- `BDERequestAdapter.block_hashes` exposes hashes only when prefix reuse is
  enabled.
- `skip_reading_prefix_cache=True` bypasses prefix lookup.
- `skip_reading_prefix_cache=False` enables prefix lookup only when hashes are
  complete and safe.
- Prefix-hit request matches no-cache output within documented tolerance.
- Prefix-hit request allocates fewer new blocks than prefix-miss request.
- Requests with the same committed history chunk prefix hit the prefix cache
  when chunk/window identity matches.
- Requests with the same suffix chunk but different preceding visible prefix do
  not incorrectly hit the prefix cache for that suffix.
- Requests with different committed history chunk boundaries miss the prefix
  cache even if tensor shapes match.
- Shared prefix blocks have `ref_cnt > 1` while multiple requests reference
  them.
- Finishing one request does not free shared blocks still used by another
  request.
- Mixed prefix-hit / prefix-miss batching works correctly.
- Different `cache_salt` prevents sharing.

### Metrics

Expose at least:

- prefix hit rate;
- reused block count;
- newly allocated block count;
- saved prefix tokens/chunks;
- `BlockPool` occupancy;
- shared-block refcount distribution;
- cache-on/off latency;
- cache-on/off memory peak;
- cache-on/off parity result.

## 4. Future Extension: Session-Scoped KV Reuse

Session-scoped reuse should be designed separately from the initial
content-addressed prefix-cache path.

For an open realtime session, committed KV can stay resident through the same
long-lived BDE request/session lifecycle:

```text
session starts
  allocate/prefill reusable conditioning KV

for each step/chunk:
  allocate current chunk slots once
  run denoise steps using the same slot_mapping
  commit chunk
  advance num_computed_tokens
  apply chunk-window eviction

session closes or aborts
  KVCacheManager.free(adapter)
```

This path should not require a prefix-index lookup for every step in the same
session. Prefix hashes may still be useful later for restore or deduplication,
but the basic session path is lifetime-based rather than content-addressed.

Future tests for this extension:

- same session reuses committed KV blocks across steps without prefix lookup;
- `num_computed_tokens` advances only after chunk commit;
- repeated denoise steps for the same chunk allocate once and reuse the same
  `slot_mapping`;
- session close / abort calls `KVCacheManager.free(adapter)`;
- chunk-window eviction removes old chunks at chunk boundaries and protects sink
  chunks.
