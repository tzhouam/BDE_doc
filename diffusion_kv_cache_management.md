# [RFC] Unified KV Cache Management for vLLM-Omni Diffusion

!!! warning "Proposal / Draft"
    This is a design proposal (RFC). It describes a subsystem that does not yet
    exist as a unified component. The per-model KV reuse paths referenced here
    (HunyuanImage3, SenseNova-U1, Bagel, NextStep) are real and ad-hoc today;
    this RFC proposes consolidating them onto vLLM's mainline KV cache stack.

- **Status:** Draft
- **Feedback period:** 1 week (minimum)
- **Owners:** Diffusion runtime
- **Related docs:**
  [Diffusion Step Execution](diffusion_step_execution.md),
  [Continuous Batching for Step-Wise Diffusion](diffusion_continuous_batching.md),
  [cache-dit](cache_dit.md), [TeaCache](teacache.md),
  [Automatic Prefix Caching in Omni Models](prefix_caching.md)
- **Upstream RFC:**
  [World Model Support (#1987)](https://github.com/vllm-project/vllm-omni/issues/1987).
  That roadmap lists "Page-attention and KV cache management for Autoregressive
  Diffusion" under *Future*; this RFC is the design for that item.

---

## Table of Contents

- [Motivation](#motivation)
- [Background: What Exists Today](#background-what-exists-today)
- [Core Decision](#core-decision)
- [Goals and Non-Goals](#goals-and-non-goals)
- [Terminology](#terminology)
- [Proposed Design](#proposed-design)
  - [Architecture Overview](#architecture-overview)
  - [Component 1: BDERequestAdapter](#component-1-bderequestadapter)
  - [Component 2: ChunkWindowSpec / ChunkWindowManager](#component-2-chunkwindowspec--chunkwindowmanager)
  - [Component 3: Scheduler admission deltas](#component-3-scheduler-admission-deltas)
  - [Component 4: Runner BlockTables + slot_mapping](#component-4-runner-blocktables--slot_mapping)
  - [Component 5: Paged attention (blockwise-causal)](#component-5-paged-attention-blockwise-causal)
  - [Configuration Surface](#configuration-surface)
  - [Request Lifecycle Integration](#request-lifecycle-integration)
  - [Chunk Window Eviction Semantics](#chunk-window-eviction-semantics)
  - [T>1 Denoise Semantics](#t1-denoise-semantics)
- [Migration of Existing Models](#migration-of-existing-models)
- [Phased Rollout](#phased-rollout)
- [Work Breakdown & Cross-Workstream Ownership](#work-breakdown--cross-workstream-ownership)
- [Alternatives Considered](#alternatives-considered)
- [Risks and Mitigations](#risks-and-mitigations)
- [Testing Plan](#testing-plan)
- [Open Questions](#open-questions)

---

## Motivation

vLLM-Omni's diffusion stack already has a mature **feature/step cache** layer
(`vllm_omni/diffusion/cache/`): TeaCache, MagCache, cache-dit (DBCache / SCM /
TaylorSeer), and a prompt-embedding cache. These all target the same goal —
**skip redundant denoise compute** by reusing block residuals or encoder
outputs across denoise steps.

What is **missing** is a managed layer for **transformer Key/Value (KV) cache**
used by the growing family of **autoregressive (AR), hybrid, and chunked
"world-model" diffusion models**. These models genuinely materialize attention
K/V tensors and reuse them — but each one does so in isolation, with hand-rolled
state living on the model module:

- **HunyuanImage3** (`models/hunyuan_image3/`): GPT-based DiT. Caches prompt /
  AR-prefix KV once (`image_kv_cache_map`, `image_kv_cache_lens`,
  `_injected_ar_kv`, `_cache_prompt_kv`) and reuses it across all denoise steps.
  TeaCache for this model is special-cased precisely because it "uses a GPT-based
  architecture with KV cache, which is incompatible with the standard hook-based
  TeaCache approach" (see `cache/teacache/backend.py`).
- **SenseNova-U1** (`models/sensenova_u1/`): uses HF `DynamicCache` with custom
  `prepare_flash_kv_cache()` / `clear_flash_kv_cache()` helpers and
  `past_key_values` threaded through `forward_und` / `forward_gen`.
- **Bagel**, **NextStep-1.1**: token-AR understanding+generation paths that also
  thread `past_key_values` / `use_cache`.
- **Chunked world-models** — the autoregressive chunk-diffusion family tracked
  by the [World Model RFC (#1987)](https://github.com/vllm-project/vllm-omni/issues/1987):
  **DreamZero** (P0 robotics world model), Matrix Game, HunYuan World, Cosmos
  WFM, LingBot-World, Genie 3. Each forward produces one video/latent chunk that
  attends **blockwise-causally** to past chunks, keeping only a **bounded window**
  of past-chunk KV alive (VGGT-style sliding replace, DreamZero-style window
  reset). This is exactly vLLM's sliding-window eviction problem, shifted from
  token granularity to chunk granularity.

### Problems with the status quo

1. **No memory accounting.** KV tensors are allocated eagerly per request and
   grow with sequence length. There is no shared budget, no
   `gpu_memory_utilization`-style sizing, and no back-pressure. Long prompts or
   high concurrency can OOM unpredictably.
2. **No cross-request reuse.** Two requests that share a conditioning prefix
   (system instruction, fixed reference image tokens, shared "negative prompt"
   for CFG) recompute identical KV every time.
3. **No cross-step contract.** The "compute conditioning KV once, reuse across
   denoise steps" idea is reinvented per model with subtly different invariants
   (`gen_timestep_scatter_index`, CFG batch layout, SP sharding).
4. **No scheduler integration.** The diffusion scheduler
   (`sched/interface.py`) tracks request lifecycle (`WAITING` / `RUNNING` /
   `PREEMPTED` / `FINISHED_*`) but has no notion of KV residency. A preempted
   request cannot release or restore KV; continuous batching cannot reason about
   KV capacity when admitting requests.
5. **No chunk-window eviction.** World-models that only keep the last `W` chunks
   alive currently free/realloc by hand. There is no allocator that understands
   "evict everything older than the window" while protecting attention sinks.

## Background: What Exists Today

| Layer | Module | Purpose | Manages KV memory? |
| --- | --- | --- | --- |
| Feature/step cache | `cache/teacache`, `cache/magcache`, `cache/cache_dit_backend.py` | Skip denoise compute via residual reuse | No |
| Prompt-embed cache | `cache/prompt_embed_cache.py` | Reuse `encode_prompt` outputs across identical prompts | No (CPU/host objects) |
| Omni prefix caching | `docs/.../prefix_caching.md` | Cache stage hidden-state / mm outputs keyed by block hash (AR stages) | Mirrors vLLM KV blocks, but for **stage outputs**, not diffusion-internal KV |
| Per-model KV reuse | `models/hunyuan_image3`, `models/sensenova_u1`, `models/bagel`, `models/nextstep_1_1` | Reuse conditioning/AR KV across denoise steps | **Yes, ad-hoc, unmanaged** |

The unified `CacheBackend` ABC (`cache/base.py`) only abstracts
`enable()` / `refresh()` / `is_enabled()` for feature caches. It deliberately
does **not** model KV memory. This RFC adds a complementary subsystem; it does
not replace the feature-cache layer.

On the **vLLM mainline** side, the V1 engine already has a battle-tested,
paged KV stack we want to reuse rather than reimplement:

- `vllm/v1/core/kv_cache_manager.py` — `KVCacheManager.allocate_slots(request,
  num_new_tokens, ...)`, `free(request)`, `get_block_ids(request_id)`,
  `get_computed_blocks(request)`. This is the single allocation entry point.
- `vllm/v1/core/block_pool.py` — `BlockPool` owns the global free queue,
  `ref_cnt`, prefix-cache index, and `null_block` placeholders.
- `vllm/v1/core/single_type_kv_cache_manager.py` — `SlidingWindowManager` with
  `get_num_skipped_tokens()` / `remove_skipped_blocks()`, dispatched from a
  `spec_manager_map` keyed by `KVCacheSpec` type.
- `vllm/v1/kv_cache_interface.py` — `SlidingWindowSpec(block_size, num_kv_heads,
  head_size, dtype, sliding_window)` and `get_kv_cache_spec_kind()` which maps
  any `SlidingWindowSpec` subclass to `KVCacheSpecKind.SLIDING_WINDOW`.
- `vllm/v1/worker/gpu/block_table.py` — `BlockTables`, which turns block ids into
  the per-token `slot_mapping` consumed by paged attention kernels.

## Core Decision

> **Reuse vLLM mainline KV management; build the missing compatibility layer in
> the diffusion engine. Do NOT build a parallel diffusion-local allocator.**

Concretely, the diffusion engine adopts vLLM's `KVCacheManager` / `BlockPool` /
`BlockTables` / paged-attention path, and we add only the glue that the
diffusion runtime is missing today:

```text
Primary plan:
  Extend the diffusion step engine so it can drive vLLM's
  KVCacheManager.allocate_slots() and BlockTables, instead of hand-rolling KV.

New compatibility layers (the deliverables of this RFC):
  1. BDERequestAdapter        — make a diffusion request look like vllm Request
  2. ChunkWindowSpec          — a SlidingWindowSpec variant (chunk granularity)
     ChunkWindowManager       — extends SlidingWindowManager, registered in
                                spec_manager_map
  3. Scheduler admission deltas — give the diffusion scheduler token-like
                                  num_new_tokens / num_computed_tokens so it can
                                  gate on KV capacity
  4. Runner BlockTables + slot_mapping integration
  5. Paged attention backend, blockwise-causal (causal=False over present blocks)
```

The previously-proposed `DiffusionKVCacheManager` (a new diffusion-local block
pool + prefix index + profiler + eviction policy) is **rejected** and moved to
[Alternatives](#alternatives-considered): it would fork a second allocator,
refcount model, prefix index, and eviction policy that we would then have to
keep in sync with vLLM forever.

## Goals and Non-Goals

### Goals

- **Reuse**, not reimplement, vLLM's `KVCacheManager`, `BlockPool`,
  `SlidingWindowManager`, `BlockTables`, and paged attention kernels.
- A `BDERequestAdapter` that satisfies the subset of the `vllm.v1.request.Request`
  contract that `KVCacheManager` actually touches (see
  [Component 1](#component-1-bderequestadapter)).
- A `ChunkWindowSpec` (subclass of `SlidingWindowSpec`) + `ChunkWindowManager`
  (subclass of `SlidingWindowManager`) for **chunk-granularity** window eviction,
  registered in vLLM's `spec_manager_map`.
- Diffusion-scheduler **admission deltas** so continuous batching can gate on KV
  capacity and free/restore on preempt/finish.
- Runner-side `BlockTables` + `slot_mapping` wiring so attention reads/writes the
  paged KV pool.
- A migration path that lets HunyuanImage3 / SenseNova-U1 / Bagel / NextStep drop
  their bespoke caches without behavior changes.

### Non-Goals

- Not changing or replacing the feature/step caches (TeaCache, MagCache,
  cache-dit). They remain orthogonal and composable.
- Not building a new diffusion-local KV allocator / block pool / prefix index.
  We reuse vLLM's (this is the explicit reversal vs the earlier draft).
- Not engaging the manager for pure DiT models with no KV (Flux, Qwen-Image,
  Wan, Z-Image, …). The adapter is simply never constructed for them.
- Not solving multi-node KV transfer in v1 (left to a later phase that can reuse
  the existing Omni connector layer + vLLM's KV connector hooks).

## Terminology

- **DiT**: Diffusion Transformer (denoiser run N times over the latent).
- **AR/hybrid model**: a model whose denoiser is a causal/GPT-style transformer
  that maintains attention KV (e.g. HunyuanImage3).
- **Chunk**: a unit of autoregressive generation in a world-model (a frame, a
  group of latent tokens). `chunk_size` tokens per chunk.
- **Chunk window**: the last `W` chunks whose KV is kept resident
  (`sliding_window = W * chunk_size`).
- **Blockwise-causal attention** (#1987 term): a chunk attends causally to all
  past chunks and bidirectionally within itself. We realize cross-chunk
  causality via block-table membership and intra-chunk bidirectionality via
  `causal=False` (see [Component 5](#component-5-paged-attention-blockwise-causal)).
- **Conditioning KV**: K/V for tokens constant across all denoise steps of a
  request (text prompt, reference-image tokens, AR prefix). Prime reuse target.
- **Denoise step**: one transformer forward over the (noisy) latent;
  `ForwardContext.denoise_step_idx` already tracks this.
- **CFG branch**: positive (conditional) vs negative (unconditional) batch rows
  when `do_classifier_free_guidance` is set.

## Proposed Design

### Architecture Overview

```
                    OmniDiffusionConfig.kv_cache_config
                                  │
        ┌─────────────────────────┼──────────────────────────┐
        │                         │                           │
  DiffusionScheduler        DiffusionWorker            DiffusionModelRunner
  (sched/*.py)              (worker/diffusion_worker)  (worker/diffusion_model_runner)
        │                         │                           │
        │ admission deltas        │ owns                      │ per-step:
        │ (num_new_tokens,        │                           │  build BlockTables
        │  num_computed_tokens)   ▼                           ▼  + slot_mapping
        │            ┌──────────────────────────────────────────────┐
        │            │   BDERequestAdapter (looks like vllm Request)  │
        └──────────▶ └──────────────────────────────────────────────┘
                                  │  passed to
                                  ▼
                ┌─────────────────────────────────────────────────────┐
                │     vLLM mainline KV stack (REUSED, not forked)       │
                │  KVCacheManager.allocate_slots() / free()             │
                │  BlockPool (free queue, ref_cnt, null_block)          │
                │  ChunkWindowManager  ⊂ SlidingWindowManager           │
                │     keyed by ChunkWindowSpec ⊂ SlidingWindowSpec      │
                └─────────────────────────────────────────────────────┘
                                  │ block ids
                                  ▼
                          BlockTables → slot_mapping
                                  │
                                  ▼
          paged attention (blockwise-causal: causal=False over present blocks)
```

Everything inside the dashed "vLLM mainline KV stack" box is **imported and
reused**. The four boxes outside it (`BDERequestAdapter`, the
`ChunkWindowSpec`/`ChunkWindowManager` registration, scheduler deltas, runner
`BlockTables`/`slot_mapping`) are the new code this RFC introduces.

### Component 1: BDERequestAdapter

`KVCacheManager.allocate_slots(request, ...)` and `get_computed_blocks(request)`
operate on a `vllm.v1.request.Request`. The diffusion engine instead has
`DiffusionRequestState` (`sched/interface.py`) wrapping an
`OmniDiffusionRequest`, which has **no** `num_computed_tokens`, `num_tokens`, or
`block_hashes`. The adapter bridges exactly the attributes the KV manager reads
— nothing more.

From reading `allocate_slots` / `get_computed_blocks`, the touched surface is:

| Attribute / method | Used by | Diffusion meaning |
| --- | --- | --- |
| `request_id` | block ownership keying | `DiffusionRequestState.request_id` |
| `num_computed_tokens` | computed-prefix length | chunk tokens already materialized (`completed_chunks * chunk_size`) |
| `num_tokens` | cache-commit cap, full-fit gate | total tokens once the current chunk lands |
| `block_hashes` | prefix-cache lookup | conditioning-prefix hash blocks; empty when caching disabled |
| `skip_reading_prefix_cache` | bypass prefix lookup | `True` until prefix reuse is enabled (Phase 3) |
| `num_preemptions` | stats only | from scheduler preempt counter |

```python
# vllm_omni/diffusion/kv_cache/adapter.py  (proposed)

class BDERequestAdapter:
    """Adapts a diffusion request to the subset of vllm Request that
    KVCacheManager.allocate_slots()/get_computed_blocks() actually reads.

    NOT a full Request: it only implements the attributes exercised by the
    KV manager. A conformance test asserts the touched attribute set does not
    drift (see Testing Plan)."""

    def __init__(self, state: DiffusionRequestState, *, chunk_size: int):
        self._state = state
        self._chunk_size = chunk_size
        self._completed_chunks = 0
        self._block_hashes: list = []          # filled only when prefix reuse on

    @property
    def request_id(self) -> str:
        return self._state.request_id

    @property
    def num_computed_tokens(self) -> int:
        return self._completed_chunks * self._chunk_size

    @property
    def num_tokens(self) -> int:
        # tokens once the in-flight chunk is committed
        return (self._completed_chunks + 1) * self._chunk_size

    @property
    def block_hashes(self):
        return self._block_hashes

    @property
    def skip_reading_prefix_cache(self) -> bool:
        return True                            # Phase 3 flips this on

    @property
    def num_preemptions(self) -> int:
        return 0

    def on_chunk_committed(self) -> None:
        self._completed_chunks += 1
```

The adapter is **not** registered as a vLLM `Request`; it is a structural
duck-type. To keep this honest, a unit test introspects which `Request`
attributes `allocate_slots` / `get_computed_blocks` reference (via a recording
proxy) and fails if the adapter is missing one.

### Component 2: ChunkWindowSpec / ChunkWindowManager

World-models keep only the last `W` chunks of KV. That is a sliding window whose
unit is a chunk, so we express it as a `SlidingWindowSpec` with
`sliding_window = W * chunk_size`, and subclass the manager to evict at chunk
boundaries.

```python
# vllm_omni/diffusion/kv_cache/chunk_window.py  (proposed)

from vllm.v1.kv_cache_interface import SlidingWindowSpec
from vllm.v1.core.single_type_kv_cache_manager import (
    SlidingWindowManager, spec_manager_map,
)

@dataclass(frozen=True, kw_only=True)
class ChunkWindowSpec(SlidingWindowSpec):
    # sliding_window (inherited) MUST equal window_chunks * chunk_size.
    chunk_size: int
    window_chunks: int
    sink_chunks: int = 0          # protected leading chunks (attention sink)
    reset_at_boundary: bool = False   # True => DreamZero window-reset semantics

    def __post_init__(self):
        super().__post_init__()
        assert self.sliding_window == self.window_chunks * self.chunk_size


class ChunkWindowManager(SlidingWindowManager):
    """Chunk-granularity eviction on top of SlidingWindowManager.

    Reuses BlockPool's free queue, ref_cnt, and null_block replacement. Only the
    'which tokens are skipped' math is overridden so eviction snaps to chunk
    boundaries and honors sink_chunks."""

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        spec: ChunkWindowSpec = self.kv_cache_spec
        sink = spec.sink_chunks * spec.chunk_size
        if spec.reset_at_boundary:
            # Window reset: at each chunk boundary everything past sink is dropped.
            completed = (num_computed_tokens // spec.chunk_size) * spec.chunk_size
            return max(0, completed - sink)
        # Sliding replace: keep last `window_chunks` chunks (+ sink).
        # Base SlidingWindowManager uses `- sliding_window + 1` to keep the
        # in-flight token's window intact. Because chunks are block-aligned and
        # we snap the skip count down to a chunk boundary, that ±1 is absorbed:
        # we never skip into a chunk that the current window still needs.
        keep = spec.sliding_window
        skipped = max(0, num_computed_tokens - keep - sink)
        # snap down to a chunk boundary so we never half-evict a chunk
        return (skipped // spec.chunk_size) * spec.chunk_size


# Register so KVCacheManager dispatches ChunkWindowSpec to ChunkWindowManager.
# Dispatch is by EXACT type (`spec_manager_map[type(kv_cache_spec)]`), so the
# subclass must be registered explicitly — inheritance alone is not enough.
spec_manager_map[ChunkWindowSpec] = ChunkWindowManager
```

Why this plugs in cleanly:

- `get_kv_cache_spec_kind()` already maps **any** `SlidingWindowSpec` subclass to
  `KVCacheSpecKind.SLIDING_WINDOW` (the `isinstance(spec, SlidingWindowSpec)`
  branch), and `SlidingWindowManager`'s cache-hit path asserts
  `SlidingWindowSpec` — both satisfied by inheritance.
- `remove_skipped_blocks()` (the eviction driver called from `allocate_slots`)
  is inherited; we only override the token-skip math. Eviction therefore still
  goes through `BlockPool`'s single free queue and `null_block` replacement — no
  parallel `_expired_pool`.
- `max_admission_blocks_per_request()` is inherited from `SlidingWindowSpec` and
  bounds pool sizing/admission to the window size, which is exactly the
  world-model invariant.

**Sink protection:** sinks are tracked as the leading `sink_chunks` blocks and
excluded from the skip count above, so `BlockPool` never reclaims them while the
request lives. (The earlier draft tracked relative indices; tracking the leading
chunk count is equivalent and avoids `null_block` index drift.)

### Component 3: Scheduler admission deltas

`KVCacheManager.allocate_slots(request, num_new_tokens, ...)` needs a
**token-like delta**. The diffusion scheduler today emits only request ids
(`DiffusionSchedulerOutput.scheduled_request_ids`) with no per-request token
count. We add the minimal accounting:

- On admit / each new chunk, the scheduler computes `num_new_tokens =
  chunk_size` (the tokens the about-to-run chunk will materialize) and the
  adapter's `num_computed_tokens` reflects already-committed chunks.
- Admission gate: before running a step, call `allocate_slots(adapter,
  num_new_tokens=chunk_size, full_sequence_must_fit=...)`. A `None` return means
  "not enough free blocks" → defer/preempt, mirroring vLLM's scheduler.
- This is bookkeeping only; it does not change diffusion's fixed step-count
  execution model.

### Component 4: Runner BlockTables + slot_mapping

After `allocate_slots` returns blocks, the runner must turn them into the
per-token `slot_mapping` the attention kernel writes/reads through. We reuse
`vllm/v1/worker/gpu/block_table.py`:

```python
# in DiffusionModelRunner, per scheduled chunk
blocks = kv_cache_manager.allocate_slots(adapter, num_new_tokens=chunk_size)
if blocks is None:
    # back-pressure: scheduler defers this request
    ...
block_ids = kv_cache_manager.get_block_ids(adapter.request_id)
bde_block_tables.append_block_ids(req_index, block_ids)     # worker-side view
slot_mapping = bde_block_tables.compute_slot_mapping(positions)
# slot_mapping + the paged kv_cache tensor go into ForwardContext / attn metadata
```

**Known gap (called out, not hand-waved):** the worker `BlockTables` view does
not automatically observe scheduler-side `null_block` replacements from
eviction. The runner must re-pull `get_block_ids()` (or apply the same skip) each
chunk and rebuild `slot_mapping`, otherwise evicted blocks would still be read.
This is a concrete work item in Phase 2, not an assumption.

### Component 5: Paged attention (blockwise-causal)

The [World Model RFC (#1987)](https://github.com/vllm-project/vllm-omni/issues/1987)
specifies the attention for autoregressive chunk diffusion as **blockwise
causal**: each forward produces one chunk that attends to **all past chunks**
(causal across chunks) while being **bidirectional within the chunk** (full
attention among the chunk's own tokens). This is exactly the mask shape we must
support, and it maps cleanly onto vLLM's paged path:

- **Across chunks (causal):** enforced structurally, not by a token mask. A
  chunk's block table contains only past + current chunks (future chunks are not
  generated yet), and `ChunkWindowManager` further trims it to the resident
  window. The kernel never sees future-chunk KV, so cross-chunk causality is
  free.
- **Within a chunk (bidirectional):** the current chunk's query tokens must see
  each other fully, so the per-forward mask over the **present** blocks is
  **non-causal** (`causal=False`). Setting `causal=True` would wrongly impose a
  triangular mask *inside* the chunk.

So "blockwise causal" decomposes into "`causal=False` over the blocks that are
present in the table." That is why the block manager (Components 2 & 4) does the
heavy lifting and the kernel runs with `causal=False`.

The diffusion `AttentionImpl.forward(query, key, value, attn_metadata)` signature
(`attention/backends/abstract.py`) takes K/V tensors directly and has **no
`kv_cache` / `block_table` parameter today**. So "reuse paged attention" is not
free — it requires a backend that:

1. accepts the paged `kv_cache` tensor + `block_table` + `slot_mapping`
   (threaded through `AttentionMetadata.extra` or a new typed field), and
2. runs with `causal=False` (`AttentionImpl.__init__` already exposes
   `causal: bool = False`).

So the v1 integration adds a **paged diffusion attention backend** (wrapping
vLLM's paged kernel) selected when `kv_cache_config.enable` is set. The existing
dense backends remain the default for pure-DiT / cache-off paths.

> KV **management** (who owns blocks, when they are evicted) is fully delegated
> to vLLM. Attention **semantics** (blockwise-causal: cross-chunk causality via
> block-table membership, intra-chunk bidirectional `causal=False`, null-hole
> handling, position ids) are a separate concern owned by the backend and are
> NOT solved by the block manager.

### Configuration Surface

Extend `OmniDiffusionConfig` (`diffusion/data.py`), which already carries
`cache_backend` / `cache_config`:

```python
@dataclass
class OmniDiffusionConfig:
    ...
    cache_backend: str = "none"                       # feature/step cache (existing)
    cache_config: DiffusionCacheConfig | dict = field(default_factory=dict)
    # NEW: transformer KV cache management (reuses vLLM stack)
    kv_cache_config: DiffusionKVCacheConfig | dict = field(default_factory=dict)

@dataclass
class DiffusionKVCacheConfig:
    enable: bool = False
    chunk_size: int = 0               # tokens per AR chunk (0 => single prefill)
    window_chunks: int | None = None  # None => full attention (no eviction)
    sink_chunks: int = 0
    reset_at_boundary: bool = False
    gpu_memory_fraction: float = 0.1
    enable_prefix_reuse: bool = False # cross-request conditioning reuse (Phase 3)
```

When `enable=False` (default), behavior is **byte-for-byte the current path** —
models keep their existing code until migrated. The RFC is strictly additive at
first.

### Request Lifecycle Integration

| Scheduler event | Action (all via the reused vLLM manager) |
| --- | --- |
| `add_request` | construct `BDERequestAdapter`; (Phase 3) compute conditioning `block_hashes` |
| `schedule` (admit) | `allocate_slots(adapter, chunk_size, full_sequence_must_fit=True)`; `None` ⇒ defer |
| each new chunk | `allocate_slots(adapter, chunk_size)`; rebuild `BlockTables`/`slot_mapping`; `adapter.on_chunk_committed()` |
| each denoise step (same chunk) | reuse the chunk's slots in place (see [T>1](#t1-denoise-semantics)); no new allocation |
| `preempt_request` | `kv_cache_manager.free(adapter)` (drop); recompute on resume |
| `finish_requests` | `kv_cache_manager.free(adapter)`; `BlockPool` decrefs shared prefix blocks |

Because diffusion runs a **fixed, known** number of steps per chunk and
conditioning KV is step-invariant, the common case is: allocate per chunk, evict
old chunks via `ChunkWindowManager`, never grow unbounded.

### Chunk Window Eviction Semantics

Two configurable strategies, both implemented as the `get_num_skipped_tokens`
override in [Component 2](#component-2-chunkwindowspec--chunkwindowmanager):

- **Sliding replace (VGGT-style):** keep the last `window_chunks` chunks (plus
  `sink_chunks`). When chunk `k` is admitted with `num_computed_tokens =
  k*chunk_size` already computed, blocks for chunks older than the window are
  skipped and returned to `BlockPool`'s free queue. Eviction snaps to chunk
  boundaries so a chunk is never half-evicted.
- **Window reset (DreamZero-style):** at each chunk boundary, everything past the
  sink is dropped; the window restarts. Modeled by computing the completed-chunk
  prefix and skipping all of it beyond the sink.

Both rely on the fact that `remove_skipped_blocks()` runs from inside
`allocate_slots()` **before** `get_num_blocks_to_allocate()`, so freed blocks are
available for the new chunk in the same step.

### T>1 Denoise Semantics

When a chunk is denoised over `T > 1` steps, each step is a forward over the
**same** chunk tokens, not new tokens:

- `num_computed_tokens` advances by `chunk_size` only **after** the chunk is
  fully denoised (`on_chunk_committed()` is called once per chunk, not once per
  denoise step).
- Within the chunk's `T` steps, attention writes K/V into the **same** allocated
  slots (in-place update), not appended new slots. So `allocate_slots` is called
  once per chunk, and steps `1..T-1` reuse the chunk's `slot_mapping`.

This matches HunyuanImage3's "compute once, reuse across steps" behavior, now
expressed against the paged pool.

## Migration of Existing Models

Strictly opt-in and incremental. For each model:

1. Add a thin adapter that, when `od_config.kv_cache_config.enable` is true,
   routes K/V through the paged backend + `BlockTables` instead of the model's
   own cache object; otherwise falls back to today's code.
2. Validate numerical parity (cache-on vs cache-off) on the model's reference
   prompts.
3. Delete the bespoke state once parity holds.

Priority order:

1. **HunyuanImage3** — richest hand-rolled KV reuse; maps to "single chunk,
   step-invariant conditioning KV", smallest window logic. Biggest win.
2. **SenseNova-U1** — `DynamicCache` + custom flash KV helpers.
3. **Chunked world-models** — exercise `ChunkWindowManager` sliding/reset paths.
4. **Bagel**, **NextStep-1.1** — token-AR paths.

Pure DiT models (Flux, Qwen-Image, Wan, Z-Image, Hunyuan-Video, LTX2, …) are
**unaffected**: no KV, adapter never constructed.

## Phased Rollout

- **Phase 0 — Scaffolding (additive, no behavior change).**
  Add `kv_cache/` package (`adapter.py`, `chunk_window.py`),
  `DiffusionKVCacheConfig`, config plumbing, `ForwardContext` field for paged KV
  binding. Register `ChunkWindowSpec → ChunkWindowManager` in `spec_manager_map`.
  Default disabled. CI green with no model changes.
- **Phase 1 — Adapter + single-chunk reuse.**
  `BDERequestAdapter`, drive `KVCacheManager.allocate_slots()` for a single
  conditioning chunk, runner `BlockTables`/`slot_mapping`, paged backend with
  `causal=False`. Migrate **HunyuanImage3**; prove parity + memory boundedness.
- **Phase 2 — Chunk window eviction + scheduler deltas.**
  `ChunkWindowManager` sliding/reset, scheduler admission deltas, `null_block`
  re-sync in the worker `BlockTables` view. Migrate a chunked world-model.
- **Phase 3 — Cross-request prefix reuse + CFG sharing.**
  Flip `skip_reading_prefix_cache` off; compute conditioning `block_hashes`;
  reuse vLLM's prefix index + `ref_cnt`. Migrate SenseNova-U1 / Bagel / NextStep.
- **Phase 4 (optional) — CPU spill + cross-node KV** via vLLM's KV connector and
  the Omni connector layer.

Each phase is independently shippable behind the disabled-by-default flag.

## Work Breakdown & Cross-Workstream Ownership

This RFC is the design for the
[World Model RFC (#1987)](https://github.com/vllm-project/vllm-omni/issues/1987)
*Future* item **"Page-attention and KV cache management for Autoregressive
Diffusion."** It does not stand alone: its first real consumer is **DreamZero**
(the Stage-1 P0 robotics world model), and it shares seams with several in-flight
#1987 workstreams. This section decomposes the work into packages (WP) with
explicit dependencies and handoff interfaces so it can be parallelized.

### #1987 workstreams (for cross-reference)

| ID | Workstream | #1987 references |
| --- | --- | --- |
| **A** | CFG parallel refactor | #2063, #2160, #2078, #2423 |
| **B** | Multiturn stateful session management | Stage-1 P0 |
| **C** | DreamZero model integration | #2162 |
| **D** | Realtime OpenPI API server | #3673 |
| **E** | Performance (PP / stream batch, DiT caching, quant) | #2280 |
| **F** | **Page-attention & KV cache management** | **this RFC** |

### Work packages

| WP | Scope | Phase | Depends on | Handoff interface | Owner area |
| --- | --- | --- | --- | --- | --- |
| **WP-0** | `kv_cache/` package, `DiffusionKVCacheConfig`, `ForwardContext` paged-KV field, register `ChunkWindowSpec→ChunkWindowManager` | 0 | — | `DiffusionKVCacheConfig`, `ForwardContext.kv_cache_state` | **F** |
| **WP-1** | `BDERequestAdapter` (duck-type `Request` surface) + conformance test | 1 | **B** (chunk accounting from session state) | `num_computed_tokens` / `num_tokens` / `block_hashes` contract | **F** + **B** |
| **WP-2** | `ChunkWindowSpec` / `ChunkWindowManager`, `spec_manager_map` registration | 2 | vLLM core review | exact-type spec registration, `get_num_skipped_tokens` | **F** + vLLM core |
| **WP-3** | Scheduler admission deltas (`num_new_tokens = chunk_size`) | 2 | **A** (CFG batch layout), `per_request_scheduler` (#2078) | per-step token delta to `allocate_slots` | **F** + diffusion scheduler |
| **WP-4** | Runner `BlockTables` + `slot_mapping`, `null_block` re-sync | 1–2 | **A** (block table after CFG/SP shard) | `ForwardContext` paged binding | **F** + diffusion worker |
| **WP-5** | Paged attention backend, blockwise-causal (`causal=False`) | 1 | attention backend owners | `AttentionMetadata` paged fields (`kv_cache` / `block_table` / `slot_mapping`) | **F** + attention |
| **WP-6** | **DreamZero** migration + parity gate (sliding-replace / window-reset) | 2 | WP-2, WP-4, WP-5; **C** (#2162) | parity vs cache-off; window config | **C** (DreamZero owner) |
| **WP-7** | Multiturn session KV lifetime ([Open Q6](#open-questions)): `free()` per-session vs per-turn, preempt during open stream | 2–3 | WP-1; **B**, **D** (#3673) | session→adapter lifetime, free granularity | **B** + **D** |
| **WP-8** | Cross-request prefix reuse + CFG sharing | 3 | WP-1, WP-2; **A** (CFG branch identity) | `block_hashes` + `PrefixKey(CFG branch)` | **F** + **A** |

### Coordination / handoff points to agree before coding

1. **With A (CFG parallel):** `BlockTables` / `slot_mapping` are per-rank and must
   be built **after** CFG/SP sharding; the CFG branch identity must be encodable
   for prefix reuse (WP-8). Agree where in the CFG-parallel forward the paged KV
   binding is injected.
2. **With B (session management):** B owns the long-lived request the adapter
   wraps. It must expose committed-chunk count (→ `num_computed_tokens`) and
   decide whether KV is freed per turn or per session (WP-7).
3. **With C (DreamZero):** first real consumer + parity target; picks
   sliding-replace vs window-reset, `chunk_size`, `window_chunks`, `sink_chunks`.
4. **With D (realtime API):** the multiturn loop decides when chunks are admitted
   / committed and when an aborted stream releases KV.
5. **With vLLM core:** registering a new `SlidingWindowSpec` subclass in
   `spec_manager_map` (WP-2) needs upstream sign-off.

> Owner-area columns are **proposed**, by functional surface, not assignments.
> Confirm with the #1987 assignees (@TKONIY, @bowieshi, @asukaqaq-s, @cherhh,
> @amy-why-3459) before committing names.

## Alternatives Considered

1. **Build a diffusion-local `DiffusionKVCacheManager` (the earlier draft).**
   A new block pool, prefix index, refcount model, profiler, and eviction policy
   living entirely in `vllm_omni/diffusion/kv_cache/`. **Rejected.** It would
   maintain a *second* allocator/refcount/prefix/eviction stack in parallel with
   vLLM's, doubling the surface to test and keep in sync, and would not benefit
   from vLLM's prefix-cache, hybrid-allocator, and KV-connector improvements over
   time. The only argument for it was "DiT shape differs from token decode" — but
   that difference is fully absorbed by `BDERequestAdapter` +
   `ChunkWindowManager`, without forking the allocator.
2. **Only swap the attention backend; leave KV management to the models.**
   Insufficient: the diffusion engine has *no* Request / scheduler / BlockTables
   contracts for KV at all. Changing only the attention backend still leaves
   memory accounting, eviction, and admission unsolved. The compatibility layer
   (Components 1–4) is the actual work; the backend (Component 5) is necessary
   but not sufficient.
3. **Keep per-model caches; just add a memory guard.**
   Lowest effort, but leaves five divergent implementations, no cross-request
   reuse, no scheduler awareness, and no chunk-window eviction. Rejected as a
   long-term answer.
4. **Fold KV management into the existing `CacheBackend` ABC.**
   `CacheBackend` models `enable/refresh` for feature caches and intentionally
   ignores memory. Overloading it conflates "skip compute" with "own memory".
   Kept as separate, composable subsystems.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| `BDERequestAdapter` drifts from the `Request` surface vLLM actually reads | Conformance test records attribute access in `allocate_slots`/`get_computed_blocks` and fails on any unimplemented attribute |
| Worker `BlockTables` view misses scheduler `null_block` eviction → reads stale KV | Re-pull `get_block_ids()` and rebuild `slot_mapping` each chunk; explicit Phase 2 work item + test |
| Paged backend missing (today's `AttentionImpl.forward` has no block_table arg) | Component 5 adds a paged backend with `causal=False`; dense backend stays default for cache-off |
| Off-by-one in chunk eviction (half-evicted chunk) | `get_num_skipped_tokens` snaps to chunk boundary; unit tests for both sliding/reset at boundaries |
| Sink chunks reclaimed by `BlockPool` | sinks excluded from skip count; ref held for request lifetime; test |
| `ChunkWindowSpec` not dispatched by manager map | Explicit `spec_manager_map[ChunkWindowSpec] = ChunkWindowManager`; assert at startup |
| Numerical drift vs current per-model paths | Parity gate (cache-on vs cache-off) per model before deleting bespoke code |
| SP/TP correctness for sharded KV | Adapter is per-rank; block tables computed after latent sharding; reuse existing SP hooks |

## Testing Plan

- **Adapter conformance:** record `Request` attribute access inside
  `KVCacheManager.allocate_slots` / `get_computed_blocks`; assert
  `BDERequestAdapter` implements exactly that set.
- **Manager dispatch:** assert `ChunkWindowSpec` resolves to `ChunkWindowManager`
  via `spec_manager_map` and `get_kv_cache_spec_kind() == SLIDING_WINDOW`.
- **Eviction unit tests:** sliding-replace and window-reset, including
  chunk-boundary off-by-one cases and sink protection; verify evicted blocks
  return to `BlockPool` free queue and `null_block` appears in the table.
- **slot_mapping integrity:** after eviction, runner `slot_mapping` references no
  evicted/`null_block` slot.
- **T>1 reuse:** allocate once per chunk; steps `1..T-1` reuse the same slots
  (assert no new blocks allocated).
- **Parity:** HunyuanImage3 / SenseNova-U1 reference prompts produce identical
  output with KV manager on vs off (bit-exact or documented tolerance).
- **Memory:** sustained high-concurrency run stays within declared
  `gpu_memory_fraction`; chunk-window models plateau at `window_chunks` blocks.
- **Scheduler:** `allocate_slots` returning `None` correctly defers admission and
  recovers when blocks free; preempt/resume round-trip preserves output.
- **Benchmarks:** extend `benchmarks/diffusion/` to report pool occupancy,
  prefix hit-rate, and tokens/s with cache on/off.

## Open Questions

1. **Where to thread the paged `kv_cache` + `block_table` into the diffusion
   attention path** — a new typed field on `AttentionMetadata`, or via `extra`?
   (Leaning typed field for clarity, with backends ignoring it when absent.)
2. **Block size for DiT prompts.** Conditioning is short (tens–hundreds of
   tokens). Coarse `block_size` vs `chunk_size` alignment — should `block_size`
   divide `chunk_size`?
3. **Prefix hashing granularity (Phase 3).** Hash raw token ids (cheaper, aligns
   with vLLM) or post-`encode_prompt` embeddings (precise for models with prompt
   preprocessing)?
4. **CFG sharing safety.** Are there models whose unconditional branch
   conditioning is *not* request-independent (true-CFG per-request negatives)?
   Those must opt out of cross-request reuse.
5. **Hybrid-allocator interaction.** If a model mixes full-attention and
   chunk-window layers, do we rely on vLLM's hybrid KV cache groups, or restrict
   v1 to a single group?
6. **Multiturn session KV lifetime (#1987 P0).** The World Model RFC's realtime
   loop accumulates context across turns (observation → prediction → repeat). Does
   the chunk window persist KV across the whole session (one long-lived
   `BDERequestAdapter` per session), or is each turn a fresh admission? This
   determines whether `free()` happens per turn or per session, and how preemption
   interacts with an open realtime stream.

---

### CC List

(Fill in maintainers of `vllm_omni/diffusion/` scheduler, worker, cache, and the
HunyuanImage3 / SenseNova-U1 model owners, plus vLLM core KV cache owners for the
`spec_manager_map` registration.)
