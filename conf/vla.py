from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path
from typing import Optional, Union
from draccus import ChoiceRegistry


@dataclass
class VLAConfig(ChoiceRegistry):
    # fmt: off
    vla_id: str                                     # Unique VLA Policy ID that fully specifies a configuration variant
    base_vlm: Union[str, Path]                      # Base VLM as ID/Path to Run Directory (e.g., `prism-dinosiglip+7b`)
    freeze_vision_backbone: bool                    # Freeze Vision Backbone Parameters (akin to pretraining)
    freeze_llm_backbone: bool                       # Freeze LLM Backbone parameters
    unfreeze_last_llm_layer: bool                   # Unfreeze final layer of LLM (only takes effect if LLM is frozen)

    # Data Mixture Parameters
    data_mix: str                                   # Open-X Embodiment Dataset =>> Unique Mixture ID (e.g., `bridge`)
    shuffle_buffer_size: int                        # Size of Shuffle Buffer (100K for Bridge, 1M for OXE)

    # Optimization Parameters
    epochs: int                                     # Epochs to Run (in case `max_steps` is not specified)
    max_steps: Optional[int]                        # [Optional] Max Gradient Steps to Run (overrides `epochs`)

    expected_world_size: int                        # Expected # of GPUs =>> allows us to gate training on hardware
    global_batch_size: int                          # Global Batch Size (divided across processes / world size)
    per_device_batch_size: int                      # Per-Device Batch Size (per-process / individual GPU)
                                                    #   =>> # of accumulation steps is auto-computed

    learning_rate: float                            # Peak Learning Rate (`lr_scheduler_type` sets warmup/decay)
    weight_decay: float                             # Weight Decay for AdamW Optimizer
    max_grad_norm: float                            # Max Grad Norm (for global gradient clipping)
    lr_scheduler_type: str                          # LR Scheduler (usually: "constant" | "linear-warmup+cosine-decay")
    warmup_ratio: float                             # Fraction of Steps to Warmup (for warmup LR schedulers)

    train_strategy: str                             # Train Strategy (default "fsdp-full-shard")

    # Enable Gradient/Activation Checkpointing (for the LLM Backbone)
    enable_gradient_checkpointing: bool = True      # Enable Gradient/Activation Checkpointing during Training

    # Mixed Precision Training via Torch Native AMP (`autocast`)
    enable_mixed_precision_training: bool = True    # Enable Traditional BF16 Mixed Precision
    reduce_in_full_precision: bool = True           # Accumulate/Reduce All-Gather Gradients in FP32 Full Precision

    # fmt: on


# === OpenVLA Training Configurations ===
# = [8 GPU] Fast Iteration =>> SigLIP 224px + Bridge =
@dataclass
class Exp_SigLIP_224px_Bridge(VLAConfig):
    vla_id: str = "siglip-224px+mx-bridge"
    base_vlm: Union[str, Path] = "siglip-224px+7b"

    freeze_vision_backbone: bool = False
    freeze_llm_backbone: bool = False
    unfreeze_last_llm_layer: bool = False

    # Data Mixture Parameters
    data_mix: str = "bridge"
    shuffle_buffer_size: int = 256_000

    # Optimization Parameters
    epochs: int = 100
    max_steps: Optional[int] = None

    expected_world_size: int = 8
    global_batch_size: int = 256
    per_device_batch_size: int = 32

    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "constant"
    warmup_ratio: float = 0.0

    train_strategy: str = "fsdp-full-shard"


# === CogACT-VLA Pretraining Configs ===
@dataclass
class Exp_CogACT_OXE_Magic_Soup_Plus_Minus(Exp_SigLIP_224px_Bridge):
    vla_id: str = "prism-dinosiglip-224px+oxe+diffusion"
    base_vlm: Union[str, Path] = "prism-dinosiglip-224px+7b"

    # data_mix: str = "oxe_magic_soup_plus"
    data_mix: str = "oxe_magic_soup_plus_minus"
    shuffle_buffer_size: int = 250_000
    expected_world_size: int = 16
    global_batch_size: int = 256
    per_device_batch_size: int = 16
    max_grad_norm: float = 1.0
    learning_rate: float = 2e-5

    epochs: int = 100


# === MemoryVLA Fine-Tuning on WA01 (LeRobot v3, 13-dim actions, 2x A100) ===
# LoRA fine-tune: vision + Llama base frozen (`align` stage); the LLM is trained only through
#   injected LoRA adapters (q/k/v/o_proj, r=16/alpha=32) -- see `MemoryVLA.enable_lora()`.
@dataclass
class Exp_CogACT_WA01(Exp_CogACT_OXE_Magic_Soup_Plus_Minus):
    vla_id: str = "prism-dinosiglip-224px+wa01+diffusion"
    base_vlm: Union[str, Path] = "prism-dinosiglip-224px+7b"

    # LoRA =>> freeze vision + LLM base (stage resolves to "align"); only adapters + projector +
    #   perception memory + diffusion action head are trainable.
    freeze_vision_backbone: bool = True
    freeze_llm_backbone: bool = True
    unfreeze_last_llm_layer: bool = False

    data_mix: str = "wa01"
    shuffle_buffer_size: int = 0
    expected_world_size: int = 2
    global_batch_size: int = 32          # 2 GPUs x 8/device x 2 grad-accumulation steps
    per_device_batch_size: int = 8       # MUST equal `group_size` so each batch is one memory group
    epochs: int = 3
    max_steps: Optional[int] = 6000      # ~0.5 epoch over WA01 (414K frames @ batch 32)
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.03


# === MemoryVLA Fine-Tuning on WR03 (LeRobot v3, 13-dim actions, 4x A100) ===
# Same hybrid mobile-manipulator action space as WA01 (`membench.pandaomron_hybrid13.v2`,
# action_dim == 13), same AV1 224x224 left agentview camera. LoRA fine-tune identical to WA01.
@dataclass
class Exp_CogACT_WR03(Exp_CogACT_WA01):
    vla_id: str = "prism-dinosiglip-224px+wr03+diffusion"
    data_mix: str = "wr03"
    expected_world_size: int = 4       # 4 GPUs


# === MemoryVLA Fine-Tuning on TS03 (LeRobot v3, 13-dim actions) ===
# Same hybrid mobile-manipulator action space (`membench.pandaomron_hybrid13.v2`,
# action_dim == 13), same AV1 224x224 left agentview camera. LoRA fine-tune identical to WA01/WR03.
# 200 seeds / 181493 frames, "add exactly N sugar cubes" memory task. Actual #GPUs is
# overridden by the launch script via `--vla.expected_world_size`.
@dataclass
class Exp_CogACT_TS03(Exp_CogACT_WR03):
    vla_id: str = "prism-dinosiglip-224px+ts03+diffusion"
    data_mix: str = "ts03"


# === Define a VLA Registry Enum for Reference & Validation ===
@unique
class VLARegistry(Enum):
    # Sanity Check Configurations =>> BridgeV2
    SIGLIP_224PX_MX_BRIDGE = Exp_SigLIP_224px_Bridge

    # === CogACT-VLA Pretraining Configs ===
    EXP_COGACT_OXE_MAGIC_SOUP_PLUS_MINUS = Exp_CogACT_OXE_Magic_Soup_Plus_Minus

    # === MemoryVLA Fine-Tuning on WA01 (LeRobot v3, 13-dim) ===
    EXP_COGACT_WA01 = Exp_CogACT_WA01

    # === MemoryVLA Fine-Tuning on WR03 (LeRobot v3, 13-dim) ===
    EXP_COGACT_WR03 = Exp_CogACT_WR03

    # === MemoryVLA Fine-Tuning on TS03 (LeRobot v3, 13-dim) ===
    EXP_COGACT_TS03 = Exp_CogACT_TS03

    @property
    def vla_id(self) -> str:
        return self.value.vla_id


# Register VLAs in Choice Registry
for vla_variant in VLARegistry:
    VLAConfig.register_subclass(vla_variant.vla_id, vla_variant.value)
