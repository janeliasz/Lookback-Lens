#!/bin/bash
#SBATCH --job-name=ll-step03
#SBATCH --time=01:00:00
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --output logs/ll-step03-log-%J.txt
#SBATCH --mem=20G
#SBATCH --gpus=1
 
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /net/tscratch/people/plgjaneliasz/conda_lens
srun python step03_lookback_lens.py --anno_1 step03_inputs/anno-full-without_xsum.jsonl --lookback_ratio_1 step03_inputs/step01_full-without_xsum.pt --anno_2 step03_inputs/anno-xsum.jsonl --lookback_ratio_2 step03_inputs/step01_xsum.pt --sliding_window 8 --clf_output_path outputs/clfs/clf_full_without_xsum.pkl --results_output_path outputs/clf_results.jsonl
conda deactivate
