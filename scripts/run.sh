#!/bin/bash
#SBATCH --job-name=8_merge_outputs.py
#SBATCH --output=8_merge_outputs.py.out
#SBATCH --error=8_merge_outputs.py.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --account=pi-ycleong

# Load environment
module load python/anaconda-2023.09 cuda/11.7 gcc/10.2.0

# Put HuggingFace & Torch cache to scratch (NOT $HOME)
export HF_HOME=/scratch/midway3/${USER}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export TORCH_HOME=$HF_HOME

# Slurm runs from job dir; use absolute path so we find the script and .env
PROJECT_ROOT="/home/xpan02/topic_recurrence"
SCRIPT_DIR="$PROJECT_ROOT/scripts"
ENV_FILE="$PROJECT_ROOT/.env"

# -------- Run clustering script --------
cd "$PROJECT_ROOT" || exit 1
python3 "$SCRIPT_DIR/8_merge_outputs.py" #--env "$ENV_FILE"
