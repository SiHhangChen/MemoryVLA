"""
vla/lora.py

Lightweight, dependency-free Low-Rank Adaptation (LoRA) for the Llama-2 LLM backbone.

Injects `LoraLinear` modules *in place* over the attention projections (q/k/v/o) of every
decoder layer. Design goals (see also the plan):

  * Keep the module tree intact: `llm` remains a `transformers.LlamaForCausalLM` and each
    projection slot (`self_attn.q_proj` etc.) is still a submodule of `LlamaDecoderLayer`,
    so FSDP's `transformer_auto_wrap_policy` (wrap on `LlamaDecoderLayer`) and the existing
    `HFCausalLLMBackbone` contract keep working without any changes.
  * Preserve state-dict key compatibility: `LoraLinear` re-registers the *same* `weight`/
    `bias` `Parameter` objects, so base-checkpoint `llm_backbone.load_state_dict(...)`
    still sees `...self_attn.q_proj.weight` and loads fine.
  * Zero-initialize `lora_B`: at injection time the LoRA path contributes exactly zero, so
    model output is bit-identical to the base model; training only *deviates* as the
    adapters learn.
  * Standard scaling: `scaling = lora_alpha / r`.
"""

from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Default projections to adapt inside each Llama self-attention block.
DEFAULT_TARGET_MODULES: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


class LoraLinear(nn.Module):
    """
    In-place LoRA wrapper around a base `nn.Linear`.

    The base `weight`/`bias` are frozen and reused (by reference) so that the pre-trained
    weights remain loaded and the `state_dict` keys match the original checkpoint layout.
    Only `lora_A` (r x in) and `lora_B` (out x r) are trainable; `lora_B` starts at zero so
    `forward()` equals the frozen base linear until training updates the adapters.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError(f"LoraLinear expects an `nn.Linear`, got `{type(base_linear)}`.")
        if not (isinstance(r, int) and r > 0):
            raise ValueError(f"LoRA rank `r` must be a positive int, got {r!r}.")

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r
        self.lora_alpha = float(lora_alpha)
        self.scaling = self.lora_alpha / r

        # Re-register the base parameters (by reference) so state-dict keys are preserved.
        # They are frozen: only the LoRA adapters below receive gradients.
        self.weight = base_linear.weight
        self.bias = base_linear.bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

        # Low-rank adapter matrices.
        #   lora_A: (r, in_features)  -- initialized small (randn * 0.01)
        #   lora_B: (out_features, r) -- zero-init => output == base at injection time
        self.lora_A = nn.Parameter(torch.randn(r, self.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        self.lora_dropout = nn.Dropout(p=lora_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = F.linear(x, self.weight, self.bias)
        # LoRA path: x (..., in) -> (..., r) -> (..., out), scaled by alpha / r.
        result = result + (self.lora_dropout(x) @ self.lora_A.t()) @ self.lora_B.t() * self.scaling
        return result


def inject_lora_llm(
    llm: nn.Module,
    r: int = 16,
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.1,
    target_modules: Iterable[str] = DEFAULT_TARGET_MODULES,
) -> int:
    """
    Replace the attention projection modules of every decoder layer with `LoraLinear`.

    Expects `llm` to expose `llm.model.layers[i].self_attn.<proj>` (standard
    `LlamaForCausalLM` / `transformers` layout). Returns the number of modules injected
    (for logging).
    """
    n_injected = 0
    layers = getattr(getattr(llm, "model", None), "layers", None)
    if layers is None:
        raise ValueError(
            f"`inject_lora_llm` expects an HF CausalLM with `.model.layers`, got `{type(llm)}`."
        )

    for layer in layers:
        self_attn = getattr(layer, "self_attn", None)
        if self_attn is None:
            continue
        for proj_name in target_modules:
            base_linear = getattr(self_attn, proj_name, None)
            if base_linear is None or isinstance(base_linear, LoraLinear):
                continue
            setattr(
                self_attn,
                proj_name,
                LoraLinear(
                    base_linear,
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                ),
            )
            n_injected += 1

    return n_injected


def set_lora_trainable(llm: nn.Module, trainable: bool = True) -> int:
    """Set `requires_grad` on every `lora_A` / `lora_B` parameter in the injected modules."""
    n_params = 0
    for module in llm.modules():
        if isinstance(module, LoraLinear):
            module.lora_A.requires_grad_(trainable)
            module.lora_B.requires_grad_(trainable)
            n_params += 2
    return n_params


def has_lora(llm: nn.Module) -> bool:
    """True if any `LoraLinear` module is present under `llm`."""
    return any(isinstance(m, LoraLinear) for m in llm.modules())
