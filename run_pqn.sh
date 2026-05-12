#!/bin/bash

set -euo pipefail

# ------------------------------------------------------------
# Config for interactive run
# ------------------------------------------------------------
THREADS=36
export WANDB_MODE=offline

# ------------------------------------------------------------
# Basic logging
# ------------------------------------------------------------
echo "Run started on: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"

mkdir -p logs
mkdir -p checkpoints

# ------------------------------------------------------------
# Environment setup
# ------------------------------------------------------------
source ~/.bashrc
conda activate CarRacing

echo "Python: $(which python)"
python --version

echo "CUDA visible devices: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

# ------------------------------------------------------------
# Thread settings
# ------------------------------------------------------------
export OMP_NUM_THREADS=${THREADS}
export MKL_NUM_THREADS=${THREADS}
export OPENBLAS_NUM_THREADS=${THREADS}
export NUMEXPR_NUM_THREADS=${THREADS}
export PYTHONUNBUFFERED=1

# ------------------------------------------------------------
# Training command
# ------------------------------------------------------------
python main_carracing_pqn_discrete_roadmask_full96.py \
  --env-id CarRacing-v3 \
  --seed 1 \
  --cuda \
  --track \
  --num-envs 36 \
  --num-steps 256 \
  --total-timesteps 20000000 \
  --learning-rate 2e-4 \
  --anneal-lr \
  --gamma 0.99 \
  --num-minibatches 4 \
  --update-epochs 2 \
  --max-grad-norm 0.5 \
  --start-e 1.0 \
  --end-e 0.01 \
  --exploration-fraction 0.10 \
  --q-lambda 0.65 \
  --save-path checkpoints/carracing_pqn_discrete_roadmask_seed1_20M.agent.pt

echo "Run finished on: $(date)"