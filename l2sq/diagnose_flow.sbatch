#!/bin/bash
#SBATCH --job-name=diagnose_flow
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40"
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/diagnose_%j.log
#SBATCH --error=logs/diagnose_%j.log

set -eo pipefail

mkdir -p logs
source ~/miniconda3/etc/profile.d/conda.sh
conda activate l2sq
module load cuda/12.1
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export HF_HOME=/storage/user/hdo/hf_cache

python -u -m l2sq.diagnose_flow runs/flow_3000/decoder.pt
