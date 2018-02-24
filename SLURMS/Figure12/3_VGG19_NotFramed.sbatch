#!/bin/bash
#SBATCH -p cox
#SBATCH -n 1 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --gres=gpu
#SBATCH --mem=128000
#SBATCH -t 3-12:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haehn@seas.harvard.edu
#SBATCH -o /n/home05/haehn/Projects/CP/SLURMS/Figure12/LOGS/%N.%j.out
#SBATCH -e /n/home05/haehn/Projects/CP/SLURMS/Figure12/LOGS/%N.%j.err

source new-modules.sh

module load Anaconda/5.0.1-fasrc01

cd /n/home05/haehn/Projects/CP/EXP/

source activate CP

python bulk_experiments.py Figure12 3 VGG19 NotFramed

# end
exit 0;
