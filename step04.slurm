#!/bin/bash
#SBATCH --job-name=ll-step04
#SBATCH --time=01:00:00
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --output logs/ll-step04-log-%J.txt
#SBATCH --mem=40G
#SBATCH --gpus=1
 
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /net/tscratch/people/plgjaneliasz/conda_lens
srun python step04_run_decoding.py --debug --parallel --total_shard 4 --shard_id 0 --num_gpus 1 --auth_token "$HF_TOKEN" --model_name meta-llama/Llama-2-7b-chat-hf --data_path data/hallu-ds.parquet --output_path outputs/output-hallu-ds-lookback-decoding.jsonl --do_sample --guiding_classifier classifiers/classifier_anno-nq-7b_sliding_window_8.pkl --chunk_size 4 --num_candidates 4
conda deactivate
