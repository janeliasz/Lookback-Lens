#!/bin/bash
#SBATCH --job-name=ll-step04
#SBATCH --time=12:00:00
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --output logs/ll-step04-log-%J.txt
#SBATCH --mem=40G
#SBATCH --gpus=1
 
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /net/tscratch/people/plgjaneliasz/conda_lens
srun python step04_run_decoding.py --num_gpus 1 --auth_token "$HF_TOKEN" --model_name meta-llama/Llama-2-7b-chat-hf --data_path data/cnndm.parquet --output_path outputs/cnndm-lookback-decoding.jsonl --do_sample --guiding_classifier classifiers/classifier_anno-nq-7b_sliding_window_8.pkl --chunk_size 8 --num_candidates 8 --data_type hallu-ds
conda deactivate
