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

## 2. Proposed Implementation

The implementation should be a small hash builder that converts
diffusion-visible prefix metadata into deterministic vLLM block hashes.

### 2.1 vLLM BlockHash Compatibility

`DiffusionPrefixKey` is BDE's semantic identity object. BDE serializes the key
deterministically and hashes it into vLLM's `BlockHash` byte format, then
exposes the resulting hashes through `BDERequestAdapter.block_hashes`.

As BDE reuses vLLM's prefix-cache index and `KVCacheManager.get_computed_blocks(adapter)` consumes
`adapter.block_hashes`, BDE hashes must be compatible with the same physical
lookup path.

A BDE request should not mix native token-derived block hashes with
diffusion-derived block hashes. So the serialized hash payload includes
a diffusion-specific domain and schema version

```text
domain = "bde.diffusion.prefix"
schema_version = 1
```

This separates BDE diffusion hashes from native vLLM token hashes while still
allowing both paths to reuse the same vLLM prefix-cache implementation.

### 2.2 PrefixKey Schema

The hash unit is one managed vLLM/BDE KV block. Each hash is computed from a
flat `DiffusionPrefixKey`:

```python
@dataclass(frozen=True)
class DiffusionPrefixKey:
    domain: str                         # Constant, e.g. "bde.diffusion.prefix".
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

    # Reuse isolation.
    cache_salt: str | None              # Reuse isolation boundary; different salts do not share.

    # Logical block coordinate.
    parent_block_hash: bytes | None     # Previous logical prefix block hash; None for the first visible block.
    kv_group: str                       # KV cache group name/id; "default" for single-group models.
    block_size: int                     # Managed KV block size used to produce this hash.
    role: str                           # text_conditioning, initial_observation, history_chunk, cfg_condition, etc.
    modality: str                       # Source modality, e.g. text, image, video, audio, latent, mixed.

    # Content-unit coordinate.
    logical_content_unit_index: int     # Window-aware unit index; 0 for singleton inputs.
    content_unit_size: int              # Token/latent count; v1 whole-unit reuse requires divisibility by block_size.
    offset_in_unit: int                 # Token/latent offset of this block within the content unit.

    # Content identity.
    content_digest: bytes               # Digest of the full canonical content unit, not this block's tensor bytes.

    # Semantic identity.
    attention_key: str                  # Attention mask / visibility semantics.
    position_key: str                   # Position/RoPE/indexing semantics.
    window_key: str                     # Chunk-window/sink visibility semantics.
    timestep_key: str | None            # Denoise timestep identity if persistent KV depends on timestep.

    # CFG identity.
    cfg_branch: str                     # none, cond, negative, uncond, text_cfg, image_cfg, etc.
    branch_condition_hash: bytes | None # Hash of branch-specific condition, e.g. negative prompt/image.
    guidance_scale_key: str | None      # Present only if guidance scale changes the KV-producing forward.

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
- `block_size`, `role`, and `modality` describe where the block belongs in the
  current BDE/vLLM logical prefix. Suggested v1 `role` values are
  `text_conditioning`, `initial_observation`, `history_chunk`, `action_state`,
  and `cfg_condition`. Suggested v1 `modality` values are `text`, `image`,
  `video`, `audio`, `latent`, `action`, and `mixed`.
- `logical_content_unit_index`, `content_unit_size`, and `offset_in_unit`
  describe the reusable content unit that this block belongs to. For singleton
  conditioning inputs, `logical_content_unit_index` is `0`. For committed
  history chunks or future multimodal segments, it is the model-visible,
  window-aware chunk or segment index after sink/window handling.
- For v1 reusable content units that are published as whole units, require:

```text
content_unit_size % block_size == 0
```

  This makes each reusable unit decompose into whole managed KV blocks. If a
  model cannot provide this alignment, the adapter must pad/split explicitly or
  skip cross-request reuse for the affected boundary.
- `content_digest` is computed once over the full canonical model-visible
  content unit, not over each block's tensor bytes. The digest input should
  include the typed values, dtype, shape, and layout of the content unit.
  Examples include the full token-id sequence for a text prompt, the full
  preprocessed/latent initial observation, the full finalized history chunk, or
  the full normalized CFG branch condition.
- `attention_key`, `position_key`, and `window_key` capture non-content factors
  that change produced KV.
- `timestep_key` is included only if persistent KV depends on denoise timestep.
  If persistent KV is timestep-independent, leave it unset. This is the common
  case for the parent RFC's "step-invariant conditioning KV" statement.
- CFG branch mismatch means hash mismatch. Unconditional branch sharing is
  allowed only when the branch is truly request-independent. Negative branch
  sharing is allowed only when `branch_condition_hash` matches.
- `guidance_scale_key` is unset if guidance scale only combines branch outputs
  after KV computation. It is set, or reuse is disabled, if guidance scale
  changes the KV-producing forward pass.
- `extra` is logically a map, but represented as sorted `(key, value)` pairs so
  canonical serialization cannot depend on dict insertion order. An
  implementation may accept a dict and normalize it before hashing.

### 2.3 PrefixKey To BlockHash Derivation

BDE should avoid hashing GPU KV tensors or per-block latent tensors. The
adapter should hash each reusable content unit once, then derive one vLLM
`BlockHash` for each managed KV block that belongs to that content unit.

For example, assume one finalized history chunk has exactly three managed KV
blocks:

```text
content_unit_size = block_size * 3
```

The adapter first hashes the whole chunk once:

```text
content_digest = H(full canonical committed chunk)
```

It then builds one `DiffusionPrefixKey` per block. All three keys share the
same `content_digest`, because they belong to the same finalized chunk. They
differ by `offset_in_unit`, because each block points to a different slice of
that chunk. They also differ by `parent_block_hash`, because vLLM prefix cache
matches a whole prefix chain, not an isolated local block.

```text
key_0:
  content_digest = H(full canonical committed chunk)
  offset_in_unit = 0
  parent_block_hash = None
  block_hash_0 = H(canonical_serialize(key_0))

key_1:
  content_digest = H(full canonical committed chunk)
  offset_in_unit = block_size
  parent_block_hash = block_hash_0
  block_hash_1 = H(canonical_serialize(key_1))

key_2:
  content_digest = H(full canonical committed chunk)
  offset_in_unit = block_size * 2
  parent_block_hash = block_hash_1
  block_hash_2 = H(canonical_serialize(key_2))
```

This means BDE performs one expensive content digest for the full reusable unit,
not three separate tensor-content hashes. The per-block hashes are still
different because the block offset and parent hash are different.

For non-history conditioning units, the same rule applies. A text conditioning
prefix, initial observation, or CFG branch condition can be hashed once into a
`content_digest`; each full managed KV block inside that unit then receives its
own `DiffusionPrefixKey` and vLLM `BlockHash`.

### 2.4 Prefix Cache Lifecycle

BDE prefix reuse follows vLLM's request-level prefix-cache lifecycle. Prefix
matching happens when the BDE request is admitted, through
`KVCacheManager.get_computed_blocks(adapter)`. A running denoise step should not
repeatedly probe the prefix index.

BDE owns eligibility: it decides which logical diffusion blocks may emit
reusable `block_hashes`. vLLM owns physical cache lifecycle: prefix lookup,
block refcounting, LRU eviction, and cache release remain handled by
`KVCacheManager` / `BlockPool`.

The request-level lifecycle is:

- **Window normalization:** before prefix lookup, BDE determines the
  model-visible prefix after sink/window handling. `logical_content_unit_index`,
  `position_key`, and `window_key` are derived from this visible layout.
- **Admission:** `BDERequestAdapter.block_hashes` is populated for reusable full
  prefix blocks. `KVCacheManager.get_computed_blocks()` performs the prefix
  lookup once.
- **Running / denoise:** the request reuses its assigned block ids,
  `BlockTables`, and `slot_mapping`. Tentative current-chunk KV is not
  published as reusable prefix KV.
- **Commit:** finalized, block-aligned content units may become eligible for
  future prefix reuse.
- **Finish / abort:** request-owned blocks are released through normal
  `KVCacheManager.free(adapter)` behavior.

Chunk-window eviction is a visibility/refcount transition, not a prefix-index
invalidation step. When `ChunkWindowManager` removes old chunks from a running
request's visible attention window, it releases that request's references. It
does not manually unpublish hashes from the vLLM prefix index. Physical
lifetime is governed by vLLM refcounting and LRU eviction.

Rollback is handled by publication discipline in v1. BDE should publish hashes
only for non-rollbackable committed content units. If BDE later supports
rollback after publication, it needs explicit invalidation or a commit epoch in
the `PrefixKey`.

## 3. Proposed Test Plan

### PrefixKey / Hash Tests

- BDE hashes include the diffusion domain and schema version.
- BDE requests do not mix native token-derived block hashes with
  diffusion-derived block hashes inside one request.
- The same `DiffusionPrefixKey` serializes to identical `BlockHash` bytes across
  processes and ranks.
- Same model-visible prefix produces the same `block_hashes`.
- Same committed history chunk prefix produces the same `block_hashes` when
  logical content unit index, content unit size, offset, KV group, position, and
  window semantics match.
- Same block content under a different `parent_block_hash` produces a different
  block hash.
- Changing an earlier visible prefix block changes the hashes of later chained
  blocks.
- Changing model identity, weight adapter, quantization, tokenizer, prompt
  template, input processor, latent encoder, attention mask, position,
  chunk/window semantics, KV group, modality, or cache salt changes the hash.
- Changing model-visible content dtype, shape, layout, or value changes
  `content_digest` and therefore changes the block hash.
- Same content unit with different `offset_in_unit` values produces different
  block hashes.
- Different content units with the same local block bytes do not share unless
  the full `content_digest`, offset, parent hash, and other `PrefixKey` fields
  match.
- Changing logical content unit index, content unit size, offset-in-unit, or KV
  group changes the hash.
- Hash serialization is deterministic and independent of Python object id,
  request id, session id, batch row order, physical GPU rank, or physical block
  id.
- `content_digest` is computed once per reusable content unit; per-block hashes
  are derived from the digest plus offset and parent hash rather than by hashing
  GPU tensor bytes per block.
- Only full reusable blocks emit hashes; partial or unsafe blocks are skipped.
- Blocks that would cross a content-unit boundary are skipped unless the
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
- Tentative denoise-step KV does not emit reusable `block_hashes`.
- Finalized, non-rollbackable content units may emit reusable hashes after
  commit.
- Chunk-window eviction releases the request's references but does not require
  manual unpublish from the vLLM prefix index.
- Rollbackable chunks are not published as reusable prefix blocks in v1.
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
