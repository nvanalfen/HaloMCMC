#!/bin/bash
#SBATCH -J illustris_FT
#SBATCH --partition=short
#SBATCH --time=22:00:00
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --cores-per-socket=10
#SBATCH --array=1-10%10
#SBATCH --output=full_total_logs/%A-%a.out
#SBATCH --error=full_total_logs/%A-%a.err
#SBATCH --mail-user=nvanalfen2@gmail.com
#SBATCH --mail-type=ALL

#module load discovery anaconda3/3.7
source /home/nvanalfen/miniconda3/bin/activate
conda activate alignments

python illustris_halo_mcmc_script.py $SLURM_ARRAY_TASK_ID variables/full_total.txt

