#!/bin/bash
#SBATCH --job-name=deg_enj
#SBATCH --output=deg_enj.out
#SBATCH --error=deg_enj.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --account=macs40123

# Load environment
module load python/anaconda-2023.09 cuda/11.7 gcc/10.2.0

# Put HuggingFace & Torch cache to scratch (NOT $HOME)
export HF_HOME=/scratch/midway3/${USER}/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME     # optional backward compatibility
export TORCH_HOME=$HF_HOME             # also route torch weights there

mkdir -p $HF_HOME

# Good practice: echo environment to log
echo "Using GPU on $(hostname)"
echo "HF cache dir: $HF_HOME"

# -------- Run Embedding Script --------
python3 6_outdegree_merge_enjoy.py