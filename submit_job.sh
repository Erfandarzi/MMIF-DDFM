#!/bin/bash
#SBATCH --partition=chip-gpu     # Specify the partition or queue to use
#SBATCH --account=chip
#SBATCH --time=05-10:00:00         # Set the maximum run time (here, 3 hours)
#SBATCH --job-name=test_unet       # Set the job name
#SBATCH --output=test_unet_%j.out  # Name the output file (including job ID)
#SBATCH --error=test_unet_%j.err   # Name the error file (including job ID)
#SBATCH --nodes=1                  # Request one node
#SBATCH --cpus-per-task=4          # Request 4 CPUs per task
#SBATCH --mem=10G                  # Request 10 GB of memory
#SBATCH --gres=gpu:TITAN_RTX:1     # Request one Titan RTX GPU
JOB_ID=$SLURM_JOB_ID

echo "Starting the job script"
module load anaconda3 || echo "Failed to load anaconda3"
module load cuda || echo "Failed to load cuda"
echo "Modules loaded"

source ~/.bashrc
echo "Bashrc sourced"

conda activate lmaunet || echo "Failed to activate conda environment"
echo "Environment activated"

echo "Running Python script"
python sample.py --method GEM >> test_unet_${JOB_ID}.out 2>&1
echo "Python script completed"

 