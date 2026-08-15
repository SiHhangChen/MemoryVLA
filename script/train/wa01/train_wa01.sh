#!/bin/bash
# ==============================================================================
# MemoryVLA 在 WA01 (LeRobot v3, 13-dim actions) 上的 LoRA 微调启动脚本
# ------------------------------------------------------------------------------
# 决策(已确认):
#   * 仅 LLM 注入 LoRA (Llama-2-7B q/k/v/o_proj, r=16, alpha=32, dropout=0.1)
#   * 纯视觉语言输入 (不用 observation.state, 保持 CogACT 架构)
#   * 动作头按 13 维重建 (WA01 mobile manipulator)
#   * 冻结: vision + Llama base (stage="align")
#   * 训练: LoRA adapters + projector + perception memory + DiT-L diffusion head
# ------------------------------------------------------------------------------
# 硬件: 2x A100 80GB | 环境: memvla (torch 2.2 / tf 2.15 / py3.10)
#
# 用法:
#   bash script/train/wa01/train_wa01.sh             # 正式训练 (80000 步)
#   SMOKE_TEST=1 bash script/train/wa01/train_wa01.sh # 2 卡冒烟测试 (5 步)
#   RESUME=1 bash script/train/wa01/train_wa01.sh    # 从最新 checkpoint 续训
#   RESUME=1 RESUME_CKPT=<path> bash ...             # 从指定 checkpoint 续训
#   GPU_IDS="0,1" bash ...                            # 指定 GPU
# ==============================================================================
set -euo pipefail

# --- 强制完全离线 (所有预训练权重均在本地 `pretrained/`, HF Hub 不可达) ----------
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# --- NCCL 看门狗超时兜底 (默认 1800s; 主进程视频解码偶发 stall 时避免误杀) -------
export NCCL_TIMEOUT=3600
export NCCL_DEBUG=WARN

# --- 定位仓库根目录 (脚本位于 script/train/wa01/ 下) --------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
echo ">>> Repo root: $REPO_ROOT"

# --- 激活 conda 环境 -----------------------------------------------------------
ENV_NAME="${ENV_NAME:-memvla}"
if [[ "$(command -v conda)" != "" ]]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
fi
PYTHON_BIN="$(command -v python)"
echo ">>> Using python: $PYTHON_BIN"
"$PYTHON_BIN" -c "import torch, av, pyarrow; print('>>> env ok:', torch.__version__)"

# --- 关键路径校验 --------------------------------------------------------------
DATA_DIR="/data1/workspace/chensihang/membench/data/WA01"
[[ -e "$DATA_DIR" ]] || { echo "ERROR: 路径不存在 -> $DATA_DIR"; exit 1; }

# --- Checkpoint: 从零开始 vs 断点续训 -----------------------------------------
RESUME="${RESUME:-0}"
is_resume=False
if [[ "$RESUME" == "1" ]]; then
    # 续训: 优先 RESUME_CKPT, 否则自动取该 run 下最新的 checkpoint
    CKPT="${RESUME_CKPT:-$(ls -t runs/memvla_wa01/checkpoints/*.pt 2>/dev/null | head -1)}"
    [[ -n "$CKPT" && -e "$CKPT" ]] || {
        echo "ERROR: 未找到可续训的 checkpoint (请用 RESUME_CKPT=<path> 或确认 runs/memvla_wa01/checkpoints/*.pt)"
        exit 1
    }
    # 从文件名解析 step/epoch (与 train.py 的 re.search("step-(\d+)-" / "epoch-(\d+)-") 一致)
    resume_step=""
    resume_epoch=""
    [[ "$CKPT" =~ step-([0-9]+)- ]] && resume_step="$((10#${BASH_REMATCH[1]}))"
    [[ "$CKPT" =~ epoch-([0-9]+)- ]] && resume_epoch="$((10#${BASH_REMATCH[1]}))"
    if [[ -z "$resume_step" || -z "$resume_epoch" ]]; then
        echo "ERROR: checkpoint 文件名不合法 (需要 step-NNNNN-epoch-N- 格式): $CKPT"
        exit 1
    fi
    is_resume=True
    echo ">>> [RESUME] 从 $CKPT 续训 (step=$resume_step, epoch=$resume_epoch)"
else
    CKPT="./pretrained/CogACT-Large/checkpoints/CogACT-Large.pt"
    [[ -e "$CKPT" ]] || { echo "ERROR: 路径不存在 -> $CKPT"; exit 1; }
    echo ">>> [FROM SCRATCH] 从预训练权重 $CKPT 启动"
fi

# --- 运行参数 -------------------------------------------------------------------
GPU_IDS="${GPU_IDS:-6,7}"
n_gpu=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
bs=8
global_bs=32                     # 8/device x 2 GPU x 2 梯度累积 => grad_accum = 32//8//2 = 2
run_id="memvla_wa01"
# 单视频解码硬超时(秒): 超过则放弃该 episode 并跳过, 防止 AV1 解码挂死拖垮整个训练 (默认 120s)
DECODE_TIMEOUT="${DECODE_TIMEOUT:-120}"

if [[ "${SMOKE_TEST:-0}" == "1" ]]; then
    echo ">>> [SMOKE TEST] 2 卡冒烟验证 (5 步)"
    max_steps=5
    save_interval=1000000
    run_id="memvla_wa01--smoke"
else
    echo ">>> [FULL RUN] 正式训练 (80000 步, 梯度累积 2)"
    max_steps=80000
    save_interval=10000
fi

echo ">>> GPU=$GPU_IDS  n_gpu=$n_gpu  bs=$bs  global_bs=$global_bs  max_steps=$max_steps"

# --- 启动训练 -------------------------------------------------------------------
TRAIN_ARGS=(
  --pretrained_checkpoint "$CKPT"
  --vla.type prism-dinosiglip-224px+wa01+diffusion
  --vla.expected_world_size "${n_gpu}"
  --vla.per_device_batch_size "${bs}"
  --vla.max_steps "${max_steps}"
  --vla.global_batch_size "${global_bs}"
  --vla.learning_rate 2e-5
  --data_root_dir "$DATA_DIR"
  --run_root_dir ./runs
  --run_id "${run_id}"
  --is_resume "${is_resume}"
  --decode_timeout "${DECODE_TIMEOUT}"
  --action_dim 13
  --data_format lerobot
  --dataloader_type group
  --group_size "${bs}"
  --future_action_window_size 15
  --repeated_diffusion_steps 4
  --image_aug False
  --use_lora True
  --lora_r 16
  --lora_alpha 32
  --lora_dropout 0.1
  --lora_target_modules "[q_proj, k_proj, v_proj, o_proj]"
  --trackers "[jsonl]"
  --save_interval "${save_interval}"
  --seed 42
)
if [[ "$RESUME" == "1" ]]; then
    TRAIN_ARGS+=(--resume_step "${resume_step}" --resume_epoch "${resume_epoch}")
fi

CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun --nproc_per_node="${n_gpu}" train.py "${TRAIN_ARGS[@]}"
