"""
verify_integration_cpu.py

Real-component CPU integration test for the MemoryVLA WA01 (LeRobot v3) training path.
Loads the *real* Llama-2 tokenizer + DinoSigLIP vision backbone + PurePromptBuilder,
then runs `get_vla_dataset_and_collator(data_format="lerobot", ...)` and checks that
a few transformed samples + a collated batch have the expected shapes/keys.

Does NOT load the 7B LLM weights or use a GPU (CPU only).
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/workspace/chensihang/membench/policy/MemoryVLA")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from transformers import LlamaTokenizerFast  # noqa: E402

from prismatic.models.backbones.llm.prompting import PurePromptBuilder  # noqa: E402
from prismatic.models.materialize import get_vision_backbone_and_transform  # noqa: E402
from prismatic.util.data_utils import PaddedCollatorForActionPrediction  # noqa: E402
from vla.materialize import get_vla_dataset_and_collator  # noqa: E402


def main() -> None:
    # ------------------------------------------------------------------ 1) Tokenizer
    # Same as `LLaMa2LLMBackbone` (`llama2-7b-pure` -> meta-llama/Llama-2-7b-hf), loaded locally.
    print(">>> [1/4] Loading real Llama-2 tokenizer ...", flush=True)
    tok_dir = ROOT / "pretrained" / "Llama-2-7b-hf"
    tokenizer = LlamaTokenizerFast.from_pretrained(str(tok_dir), use_fast=True)
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})  # mirrors LLaMa2LLMBackbone.__init__
    tokenizer.padding_side = "right"
    print(f"    tokenizer: pad_id={tokenizer.pad_token_id}, vocab={len(tokenizer)}", flush=True)

    # ------------------------------------------------------------------ 2) Vision backbone + transform
    # Same ids as `Prism_7B_DINOSigLIP_224px` config (vision_backbone_id, image_resize_strategy).
    print(">>> [2/4] Loading DinoSigLIP vision backbone (offline) ...", flush=True)
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        "dinosiglip-vit-so-224px", "resize-naive"
    )
    print(
        f"    vision_dim={vision_backbone.embed_dim}, res={vision_backbone.default_image_resolution}",
        flush=True,
    )

    # ------------------------------------------------------------------ 3) Real materialize path
    print(">>> [3/4] Building dataset via `get_vla_dataset_and_collator(data_format='lerobot')` ...", flush=True)
    dataset, action_tokenizer, collator = get_vla_dataset_and_collator(
        data_root_dir=Path("/data1/workspace/chensihang/membench/data/WA01"),
        data_mix="wa01",
        image_transform=image_transform,
        tokenizer=tokenizer,
        prompt_builder_fn=PurePromptBuilder,
        default_image_resolution=vision_backbone.default_image_resolution,
        future_action_window_size=15,
        data_format="lerobot",
        group_size=8,
        train=False,  # deterministic: no episode shuffle
        shuffle_buffer_size=0,
    )
    print(f"    dataset len (grouped) = {len(dataset)}", flush=True)
    stats = dataset.dataset_statistics["wa01"]["action"]
    print(f"    q01/q99/mean/std shapes: {stats['q01'].shape}", flush=True)
    assert stats["q01"].shape == (13,), f"action stats not 13-dim: {stats['q01'].shape}"
    assert len(dataset) == 413760, f"unexpected grouped length {len(dataset)}"

    # ------------------------------------------------------------------ 4) Sample + collate
    print(">>> [4/4] Sampling frames and collating a batch ...", flush=True)
    it = iter(dataset)
    instances = []
    for _ in range(8):
        frame = next(it)
        keys = sorted(frame.keys())
        acts = frame["actions"]
        am = frame["action_masks"]
        pv = frame["pixel_values"]
        assert isinstance(frame["episode_ids"], np.ndarray) and frame["episode_ids"].shape == (1,), frame[
            "episode_ids"
        ]
        assert acts.shape == (16, 13), f"actions.shape={acts.shape}"
        assert am.shape == (16,), f"action_masks.shape={am.shape}"
        assert tuple(pv["dino"].shape) == (3, 224, 224), f"pixel_values dino shape {pv['dino'].shape}"
        assert tuple(pv["siglip"].shape) == (3, 224, 224), f"pixel_values siglip shape {pv['siglip'].shape}"
        assert frame["input_ids"].ndim == 1 and frame["input_ids"].shape[0] > 0
        assert tuple(am) != (16,) or acts.min() >= -1.0 - 1e-4 and acts.max() <= 1.0 + 1e-4
        # Normalized action range check (active dims clipped to [-1,1])
        assert acts[am].min() >= -1.0 - 1e-4 and acts[am].max() <= 1.0 + 1e-4, acts[am].minmax()
        instances.append(frame)
        print(
            f"    frame[{_}] ep={int(frame['episode_ids'][0])} ts={frame['timesteps'][0]:>5d} "
            f"acts={acts.shape} mask_active={int(am.sum())}/16 in_ids={frame['input_ids'].shape[0]}",
            flush=True,
        )

    batch = collator(instances)
    print(">>> collated batch keys:", sorted(batch.keys()), flush=True)
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"    {k}: {tuple(v.shape)} {v.dtype}", flush=True)
        else:
            print(f"    {k}: {type(v).__name__}", flush=True)

    assert batch["actions"].shape == (8, 16, 13), batch["actions"].shape
    assert batch["action_masks"].shape == (8, 16), batch["action_masks"].shape
    assert batch["input_ids"].ndim == 2 and batch["input_ids"].shape[0] == 8
    assert batch["timesteps"].shape == (8,), batch["timesteps"].shape
    assert batch["episode_ids"].shape == (8,), batch["episode_ids"].shape
    assert "pixel_values" in batch and batch["pixel_values"]["dino"].shape[0] == 8
    assert batch["labels"].shape == batch["input_ids"].shape
    assert tuple(batch["attention_mask"].shape) == batch["input_ids"].shape

    print(">>> ALL INTEGRATION CHECKS PASSED (CPU, real components) ✔", flush=True)


if __name__ == "__main__":
    main()
