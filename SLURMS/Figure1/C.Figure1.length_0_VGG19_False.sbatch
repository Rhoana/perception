#!/bin/bash
#SBATCH -p seas_dgx1
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --gres=gpu
#SBATCH --array=3-4
#SBATCH --mem=20000
#SBATCH -t 3-00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haehn@seas.harvard.edu
#SBATCH -o /n/home05/haehn/Projects/CP/SLURMS/Figure1/LOGS/%A_%a.out
#SBATCH -e /n/home05/haehn/Projects/CP/SLURMS/Figure1/LOGS/%A_%a.err

# Print this sub-job's task ID
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

source new-modules.sh

module load Anaconda/5.0.1-fasrc01

cd /n/home05/haehn/Projects/CP/EXP/

source activate CP

python run_regression.py C.Figure1.length 0 VGG19 False $SLURM_ARRAY_TASK_ID

# end
exit 0;
