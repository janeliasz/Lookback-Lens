#!/bin/bash
#SBATCH --job-name=ll-step02
#SBATCH --time=06:00:00
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --output logs/ll-step02-log-%J.txt
#SBATCH --mem=20G
#SBATCH --gpus=1
 
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /net/tscratch/people/plgjaneliasz/conda_lens
srun python step02_eval_gpt4o.py --hyp outputs/step01_nq.pt --ref data/nq_df.parquet --out outputs/anno-nq_deb.jsonl --limit 10
conda deactivate
