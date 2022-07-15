#!/bin/bash
#SBATCH -J halo_mcmc_script
#SBATCH --partition=short
#SBATCH --time=23:00:00
#SBATCH --mem=100G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --cores-per-socket=5
#SBATCH --array=1-10%10
#SBATCH --output=front_logs/%A-%a.out
#SBATCH --error=front_logs/%A-%a.err
#SBATCH --mail-user=nvanalfen2@gmail.com
#SBATCH --mail-type=ALL

#module load discovery anaconda3/3.7
source /home/nvanalfen/miniconda3/bin/activate
conda activate alignments

python split_halo_mcmc_script.py $SLURM_ARRAY_TASK_ID variables/censat_front.txt
