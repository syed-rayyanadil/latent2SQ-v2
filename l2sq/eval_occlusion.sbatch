#!/bin/bash
#SBATCH --job-name=eval_occlusion
#SBATCH --gres=gpu:1
#SBATCH --constraint="VRAM:48G"
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/occlusion_%j.log
#SBATCH --error=logs/occlusion_%j.log

set -eo pipefail

mkdir -p logs runs/occlusion_eval
source ~/miniconda3/etc/profile.d/conda.sh
conda activate l2sq
module load cuda/12.1.0
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export HF_HOME=/storage/user/hdo/hf_cache
export HF_HUB_OFFLINE=1

python -u -m l2sq.eval_occlusion \
    --parallel ~/latent2SQ/runs/chair_L1_logscale/decoder.pt \
    --flow ~/rayyan/latent2SQ-v2/runs/flow_tnorm/decoder.pt \
    --out runs/occlusion_eval/results.csv
