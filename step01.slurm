#!/bin/bash
#SBATCH --job-name=ll-step01
#SBATCH --time=08:00:00
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --output logs/ll-step01-log-%J.txt
#SBATCH --mem=40G
#SBATCH --gpus=1
 
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /net/tscratch/people/plgjaneliasz/conda_lens
srun python step01_extract_attns.py --num-gpus 1 --auth-token "$HF_TOKEN" --model-name meta-llama/Llama-2-7b-chat-hf --data-path data/cnndm.parquet --output-path outputs/step01_cnndm_test.pt --data-type hallu-ds
conda deactivate
