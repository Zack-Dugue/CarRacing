#!/bin/bash
#
#SBATCH --job-name=carracing-ppo-a100
#SBATCH --partition=batch
#SBATCH --account=aalaeddini_391_0001
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus=1
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
THREADS=128

# ------------------------------------------------------------
# Basic logging
# ------------------------------------------------------------
echo "Job started on: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unset}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-unset}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

mkdir -p logs
mkdir -p checkpoints

# ------------------------------------------------------------
# Environment setup
# ------------------------------------------------------------
source ~/.bashrc

module load miniforge

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

# Cleaner logs.
export PYTHONUNBUFFERED=1

# Use offline W&B unless you know compute nodes have internet.

# ------------------------------------------------------------
# Training command
# ------------------------------------------------------------
python main.py \
  --env-id CarRacing-v3 \
  --seed 1 \
  --cuda \
  --num-envs 256 \
  --num-steps 256 \
  --total-timesteps 10000000 \
  --learning-rate 3e-4 \
  --anneal-lr \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --num-minibatches 8 \
  --update-epochs 4 \
  --clip-coef 0.075 \
  --ent-coef 0.02 \
  --vf-coef 0.1 \
  --max-grad-norm 0.5 \
  --save-path checkpoints/carracing_seed_bigger_lr.agent.pt

echo "Job finished on: $(date)"