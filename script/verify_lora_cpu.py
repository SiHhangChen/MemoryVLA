"""
verify_lora_cpu.py

CPU verification of the MemoryVLA LoRA integration (tasks #9/#10/#11).

Part A -- `vla/lora.py` in isolation (tiny LlamaForCausalLM):
    * LoraLinear math: at injection time output == base (zero-init lora_B); after perturbing
      adapters the low-rank path matches the closed-form formula.
    * Requires-grad flags: base weight frozen, lora_A/lora_B trainable.
    * State-dict compatibility: `...self_attn.q_proj.weight` keys preserved; lora_A/lora_B
      added; full-model `load_state_dict(strict=True)` round-trip works after re-injection.

Part B -- real CogACT-Large checkpoint + `MemoryVLA.from_pretrained(use_lora=True)` on CPU:
    * Adapters are injected over the loaded base Llama (32 layers x 4 proj = 128 modules).
    * Base weights load from the checkpoint; adapter tensors are freshly zero-initialized.
    * `freeze_backbones("align")` keeps only adapters (plus projector/memory/action head)
      trainable inside the LLM; `trainable_module_keys` includes `vlm.llm_backbone`.
    * `save_checkpoint`-style trainable filtering keeps exactly the adapter tensors under the
      `llm_backbone` prefix (frozen base weights excluded).
    * LoRA-resume round-trip: a checkpoint carrying only adapter keys is re-loaded through
      `from_pretrained` and the adapter values match.

Run with the `memvla` env python; no GPU required (torch.load is redirected to CPU).
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("/data1/workspace/chensihang/membench/policy/MemoryVLA")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

# ----------------------------------------------------------------------------------------------
# Part A -- vla/lora.py in isolation
# ----------------------------------------------------------------------------------------------
def part_a_tiny_llama() -> None:
    from transformers import LlamaConfig, LlamaForCausalLM

    from vla.lora import LoraLinear, has_lora, inject_lora_llm, set_lora_trainable

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    llm = LlamaForCausalLM(cfg)

    # --- LoraLinear math (zero-init => output == base) ---
    base = nn.Linear(64, 64)
    base.weight.data.normal_(0, 0.02)
    base.bias.data.normal_(0, 0.02)
    lora = LoraLinear(base, r=4, lora_alpha=8.0, lora_dropout=0.0)
    x = torch.randn(3, 64)
    lora.eval()
    with torch.no_grad():
        out_base = F.linear(x, base.weight, base.bias)
        out_lora = lora(x)
    assert torch.allclose(out_base, out_lora, atol=1e-6), "zero-init lora_B must reproduce base output"

    # --- requires_grad flags ---
    assert lora.weight.requires_grad is False, "base weight must be frozen"
    assert lora.bias.requires_grad is False
    assert lora.lora_A.requires_grad and lora.lora_B.requires_grad, "adapters must be trainable"

    # --- low-rank path closed-form ---
    with torch.no_grad():
        lora.lora_B.data = torch.ones_like(lora.lora_B)
        out = lora(x)
    expected = F.linear(x, base.weight, base.bias) + (x @ lora.lora_A.t()) @ lora.lora_B.t() * lora.scaling
    assert torch.allclose(out, expected, atol=1e-6), "LoRA path closed-form mismatch"
    print("[Part A] LoraLinear math + grad flags OK")

    # --- injection over a tiny Llama: output bit-identical to base (zero lora_B) ---
    ids = torch.randint(0, 128, (1, 16))
    with torch.no_grad():
        logits_base = llm(ids).logits.detach().clone()
    n_inj = inject_lora_llm(llm, r=4, lora_alpha=8.0, lora_dropout=0.0)
    assert n_inj == cfg.num_hidden_layers * 4, f"expected 8 injected, got {n_inj}"
    assert has_lora(llm)
    layer0 = llm.model.layers[0].self_attn
    for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert isinstance(getattr(layer0, p), LoraLinear), f"{p} not wrapped"
    # idempotency
    n_again = inject_lora_llm(llm, r=4, lora_alpha=8.0, lora_dropout=0.0)
    assert n_again == 0, "re-injection must be a no-op"
    with torch.no_grad():
        logits_lora = llm(ids).logits.detach()
    assert torch.allclose(logits_base, logits_lora, atol=1e-6), "injection must not change output at init"
    # trainable switching
    n = set_lora_trainable(llm, trainable=False)
    assert n == n_inj * 2, "set_lora_trainable count mismatch"
    assert all(
        not p.requires_grad for name, p in llm.named_parameters() if "lora_" in name
    ), "set_lora_trainable(False) must freeze all adapters"
    set_lora_trainable(llm, trainable=True)
    print(f"[Part A] injection over tiny Llama OK ({n_inj} modules, idempotent, output-preserving)")

    # --- state-dict round trip ---
    sd = llm.state_dict()
    for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert f"model.layers.0.self_attn.{p}.weight" in sd, "base key must be preserved"
        assert f"model.layers.0.self_attn.{p}.lora_A" in sd, "lora_A key missing"
    llm2 = LlamaForCausalLM(cfg)
    inject_lora_llm(llm2, r=4, lora_alpha=8.0, lora_dropout=0.1)
    llm2.load_state_dict(sd, strict=True)  # strict => all injected keys line up
    for k, v in sd.items():
        if "lora_" in k:
            assert torch.equal(v, llm2.state_dict()[k]), f"round-trip mismatch at {k}"
    print("[Part A] state-dict round-trip (strict) OK")


# ----------------------------------------------------------------------------------------------
# Part B -- real CogACT checkpoint + MemoryVLA.from_pretrained(use_lora=True) on CPU
# ----------------------------------------------------------------------------------------------
def part_b_cogact_lora() -> None:
    import torch as _t
    # Redirect torch.load to CPU so the non-bf16 path of `from_pretrained` works without a GPU.
    _orig_load = _t.load
    _t.load = lambda *a, **k: _orig_load(*a, **{**k, "map_location": "cpu"})

    from prismatic.models.backbones.llm.llama2 import LLAMA2_MODELS
    from prismatic.models.materialize import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform
    from vla import MemoryVLA

    # Point the Llama backbone at the local copy (offline; HF Hub is unreachable here).
    LLAMA2_MODELS["llama2-7b-pure"]["hf_hub_path"] = str(ROOT / "pretrained" / "Llama-2-7b-hf")

    # Real vision backbone (offline timm weights under pretrained/) + empty Llama-2 (config only,
    #   weights are filled from the checkpoint by `from_pretrained`).
    print(">>> Building real DinoSigLIP vision backbone (CPU) ...", flush=True)
    vision_backbone, _ = get_vision_backbone_and_transform("dinosiglip-vit-so-224px", "resize-naive")
    print(f">>> Building Llama-2-7b (empty config, no weight download) ...", flush=True)
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer("llama2-7b-pure", inference_mode=True)

    # Patch resize side-effect: tokenizer pad = <PAD>, embeds already padded by backbone __init__.
    assert tokenizer.pad_token_id is not None and tokenizer.pad_token_id != tokenizer.eos_token_id

    model_id = "prism-dinosiglip-224px+7b"
    ckpt = ROOT / "pretrained" / "CogACT-Large" / "checkpoints" / "CogACT-Large.pt"

    print(">>> MemoryVLA.from_pretrained(use_lora=True, action_dim=13) ...", flush=True)
    vla = MemoryVLA.from_pretrained(
        ckpt,
        model_id,
        vision_backbone,
        llm_backbone,
        arch_specifier="no-align+fused-gelu-mlp",
        freeze_weights=False,
        action_dim=13,
        future_action_window_size=15,
        use_lora=True,
        lora_r=16,
        lora_alpha=32.0,
        lora_dropout=0.1,
        dataloader_type="group",
        group_size=8,
    )

    # --- injection happened on the loaded base ---
    from vla.lora import LoraLinear
    n_lora = sum(1 for m in vla.vlm.llm_backbone.llm.modules() if isinstance(m, LoraLinear))
    assert n_lora == 32 * 4, f"expected 128 LoraLinear modules, got {n_lora}"
    assert vla.use_lora
    # base weights loaded from checkpoint (spot-check embed_tokens); freeze happens later via
    #   `freeze_backbones("align")` (freeze_weights=False mirrors the training-time load path).
    sd_ckpt = torch.load(ckpt, map_location="cpu")["model"]
    emb_now = vla.vlm.llm_backbone.llm.model.embed_tokens.weight.detach()
    assert torch.allclose(emb_now, sd_ckpt["llm_backbone"]["llm.model.embed_tokens.weight"]), "base embed not loaded"
    # fresh adapters: lora_B all zero
    lora_Bs = [p for name, p in vla.named_parameters() if name.endswith("lora_B")]
    assert len(lora_Bs) == 128 and all(torch.count_nonzero(p) == 0 for p in lora_Bs), "lora_B must be zero-init"
    print(f"[Part B] from_pretrained(use_lora=True) OK: {n_lora} adapters, base weights loaded, lora_B zero")

    # --- freeze_backbones("align") keeps adapters trainable ---
    vla.freeze_backbones("align")
    assert not vla.vlm.vision_backbone.dino_featurizer.patch_embed.proj.weight.requires_grad
    assert vla.vlm.projector.parameters().__next__().requires_grad, "projector must stay trainable"
    q0 = vla.vlm.llm_backbone.llm.model.layers[0].self_attn.q_proj
    assert not q0.weight.requires_grad and q0.lora_A.requires_grad and q0.lora_B.requires_grad
    n_llm_trainable = sum(
        1 for p in vla.vlm.llm_backbone.parameters() if p.requires_grad
    )
    assert n_llm_trainable == 128 * 2, f"only adapters trainable in LLM, got {n_llm_trainable}"
    assert "vlm.llm_backbone" in vla.trainable_module_keys
    print(f"[Part B] freeze_backbones('align') OK: {n_llm_trainable} trainable LLM tensors (all adapters)")

    # --- save_checkpoint-style trainable filter keeps exactly the adapters under llm_backbone ---
    # NB: `state_dict()` returns detached tensors, so the filter is keyed on parameter NAMES
    #   (via `named_parameters`, which preserves requires_grad) -- exactly like `save_checkpoint`.
    full = vla.state_dict()
    frozen_param_names = {n for n, p in vla.named_parameters() if not p.requires_grad}
    filtered = {k: v for k, v in full.items() if k not in frozen_param_names}
    llm_keys = [k for k in filtered if k.startswith("vlm.llm_backbone.")]
    assert len(llm_keys) == 128 * 2, f"expected 256 adapter tensors, got {len(llm_keys)}"
    assert all(".lora_" in k for k in llm_keys), "frozen base weights must not leak into checkpoint"
    print(f"[Part B] trainable filter OK: {len(llm_keys)} adapter tensors under `llm_backbone`")

    # --- LoRA-resume round-trip through from_pretrained ---
    resume_state = {
        "projector": sd_ckpt["projector"],                    # strict load
        "llm_backbone": {k.removeprefix("vlm.llm_backbone."): v.clone() for k, v in filtered.items()
                         if k.startswith("vlm.llm_backbone.")},  # only adapter keys
    }
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_ckpt = Path(f.name)
    torch.save({"model": resume_state}, tmp_ckpt)

    vision_backbone2, _ = get_vision_backbone_and_transform("dinosiglip-vit-so-224px", "resize-naive")
    llm_backbone2, _ = get_llm_backbone_and_tokenizer("llama2-7b-pure", inference_mode=True)
    vla2 = MemoryVLA.from_pretrained(
        tmp_ckpt,
        model_id,
        vision_backbone2,
        llm_backbone2,
        arch_specifier="no-align+fused-gelu-mlp",
        freeze_weights=True,
        action_dim=13,
        future_action_window_size=15,
        use_lora=True,
        lora_r=16,
        lora_alpha=32.0,
        lora_dropout=0.1,
        dataloader_type="group",
        group_size=8,
    )
    for k, v in resume_state["llm_backbone"].items():
        got = vla2.vlm.llm_backbone.state_dict()[k]
        assert torch.equal(got, v), f"resume adapter mismatch at {k}"
    tmp_ckpt.unlink()
    print("[Part B] LoRA-resume round-trip OK (adapter values restored through from_pretrained)")

    print("\n>>> ALL LoRA CPU VERIFICATIONS PASSED ✔")


if __name__ == "__main__":
    part_a_tiny_llama()
    part_b_cogact_lora()
