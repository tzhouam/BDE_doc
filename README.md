# BDE_doc

Design docs for the **Block Diffusion Engine (BDE)** KV cache work in vLLM-Omni.

## Contents

- [Unified KV Cache Management for vLLM-Omni Diffusion](diffusion_kv_cache_management.md) —
  RFC for reusing vLLM mainline KV management (`KVCacheManager` / `BlockPool` /
  `BlockTables` / paged attention) in the diffusion engine via thin compatibility
  layers, with chunk-window eviction for autoregressive world models.

## Related

- Upstream: [vLLM-Omni World Model Support (#1987)](https://github.com/vllm-project/vllm-omni/issues/1987).
  This RFC is the design for that roadmap's *Future* item
  "Page-attention and KV cache management for Autoregressive Diffusion."
