# SAM2

- github source: https://github.com/facebookresearch/sam2

## 1. Add Sparse Prompt Padding-Mask Attention Support

Files:

- `sam2/modeling/sam/prompt_encoder.py`
- `sam2/modeling/sam/mask_decoder.py`
- `sam2/modeling/sam/transformer.py`

What changed:

- Extended `PromptEncoder.forward(...)` with optional
  `enable_dummy_boxes`, defaulting to `True` to preserve upstream SAM2
  behavior.
- `PromptEncoder` remains responsible only for prompt embedding. Sparse-prompt
  padding masks are now assembled outside the prompt encoder so MDSTL can
  manage fixed-slot batching explicitly.
- Extended `MaskDecoder.forward(...)` and `predict_masks(...)` with optional
  `sparse_key_padding_mask`.
- `MaskDecoder` now prepends a valid prefix for SAM output tokens and then
  concatenates the sparse-prompt padding mask before entering the two-way
  transformer.
- Extended `TwoWayTransformer`, `TwoWayAttentionBlock`, `Attention`, and
  `RoPEAttention` with a prompt-token key-padding-mask path separate from the
  existing memory mask path.

Why:

- MDSTL stage-5 needs fixed-slot sparse prompt batching with a prompt
  attention mask, instead of relying on extra `label == -1` tokens as fake
  padding.
- Native SAM2 `label == -1` tokens are semantically meaningful
  `not_a_point_embed` tokens, not ignorable padding. Without an explicit prompt
  key-padding mask, padding to a shared `P_max` changes decoder behavior.
- Keeping native dummy-point insertion as an explicit `PromptEncoder`
  compatibility switch avoids forcing MDSTL to reimplement `boxes is None`
  dummy-point behavior at the integration boundary.

Risk / behavior notes:

- Prompt padding-mask semantics intentionally follow standard PyTorch
  key-padding-mask convention:
  - `True = padding`
  - `False = valid`
- This differs from the existing local `memory_key_padding_mask` patch, where
  SAM2-compatible memory masks still use:
  - `True = valid`
  - `False = padding`
- The prompt mask is only applied where sparse prompt tokens act as keys/values:
  - sparse-token self-attention
  - image-to-token cross-attention
- The prompt mask is not applied to token-to-image attention, because that path
  attends over image tokens rather than sparse prompt tokens.
- Existing call sites remain source-compatible because all new mask arguments
  are optional, and old callers that ignore prompt padding continue to get the
  previous behavior.
- Low-level attention in `sam2/modeling/sam/transformer.py` now keeps only the
  standard `key_padding_mask` interface, with:
  - `True = padding`
  - `False = valid`
- The memory-side wrapper in `sam2/modeling/memory_attention.py` still accepts
  `memory_key_padding_mask` with SAM-compatible semantics:
  - `True = valid`
  - `False = padding`
  and flips it once before calling the shared low-level attention module.
