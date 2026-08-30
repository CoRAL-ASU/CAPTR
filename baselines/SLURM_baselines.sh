#!/bin/bash
#
#SBATCH --verbose
#SBATCH --output=./partial_baseline.log
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:2
#SBATCH --mem=100GB
#SBATCH --constraint="gpu_mem:80gb"

echo "load cuda"
module load cuda

#######################################
export HF_HOME="/storage/ukp/work/englaender/Master_Thesis/master-thesis-h-star/"
export VLLM_CONFIG_ROOT="/storage/ukp/work/englaender/Master_Thesis/master-thesis-h-star/.config/vllm"
export VLLM_CACHE_ROOT="/storage/ukp/work/englaender/Master_Thesis/master-thesis-h-star/.cache/vllm"
export HOME="/storage/ukp/work/englaender/Master_Thesis/master-thesis-h-star/"
CONDA_BASE_PATH="/storage/ukp/work/englaender/miniconda"
source "${CONDA_BASE_PATH}/etc/profile.d/conda.sh"
conda activate "/storage/ukp/work/englaender/miniconda/envs/hstar/"
#######################################

echo "loaded conda. start script"

python3 partial_input_baseline.py
# python3 captions_baseline.py --debug
# python3 interleaved_baseline.py --debug
# python3 oracle_entity_replaced_baseline.py --debug
