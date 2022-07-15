#!/bin/bash
#SBATCH -J halo_mcmc_script
#SBATCH --partition=short
#SBATCH --time=16:00:00
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=1-10%10
#SBATCH --output=back_logs/%A-%a.out
#SBATCH --error=back_logs/%A-%a.err
#SBATCH --mail-user=nvanalfen2@gmail.com
#SBATCH --mail-type=ALL

#module load discovery anaconda3/3.7
source /home/nvanalfen/miniconda3/bin/activate
conda activate alignments

python back_split_halo_mcmc_script.py $SLURM_ARRAY_TASK_ID
