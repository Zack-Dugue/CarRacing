#!/bin/bash
#
#SBATCH --job-name=carracing-ppo
#SBATCH --partition=gpu-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --account=aalaeddini_391_0001

set -euo pipefail

# ------------------------------------------------------------
# Basic logging
# ------------------------------------------------------------
echo "Job started on: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Working directory: $(pwd)"

mkdir -p logs
mkdir -p checkpoints

# ------------------------------------------------------------
# Environment setup
# ------------------------------------------------------------
# Change this if your conda install is somewhere else.
source ~/.bashrc

# Change this to your actual conda env name.
conda activate CarRacing

echo "Python: $(which python)"
python --version

echo "CUDA visible devices: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

# ------------------------------------------------------------
# Thread settings
# ------------------------------------------------------------
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Helpful for cleaner logs.
export PYTHONUNBUFFERED=1

# ------------------------------------------------------------
# Training command
# ------------------------------------------------------------
python main_with_checkpoint_save.py \
  --env-id CarRacing-v3 \
  --seed 1 \
  --cuda True \
  --num-envs 8 \
  --num-steps 128 \
  --total-timesteps 10000000 \
  --learning-rate 2.5e-4 \
  --anneal-lr True \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --num-minibatches 4 \
  --update-epochs 4 \
  --clip-coef 0.1 \
  --ent-coef 0.01 \
  --vf-coef 0.5 \
  --max-grad-norm 0.5

echo "Job finished on: $(date)"