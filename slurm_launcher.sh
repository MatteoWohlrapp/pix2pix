#!/usr/bin/bash

#SBATCH -J "mri_inr"   # job name
#SBATCH --time=1-00:00:00   # walltime
#SBATCH --output=/vol/aimspace/projects/practical_SoSe24/mri_inr/matteo/logs/train_%A.out  # Standard output of the script (Can be absolute or relative path)
#SBATCH --error=/vol/aimspace/projects/practical_SoSe24/mri_inr/matteo/logs/train_%A.err  # Standard error of the script
#SBATCH --mem=64G
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16   # number of processor cores (i.e. tasks)
#SBATCH --gres=gpu:1  # replace 0 with 1 if gpu needed
#SBATCH --qos=master-queuesave

# load python module
. "/opt/anaconda3/etc/profile.d/conda.sh"

# activate corresponding environment
conda deactivate
conda activate pix

cd "/vol/aimspace/projects/practical_SoSe24/mri_inr/matteo/code/pix2pix"

python train.py --dataroot ../../data/UCSF-PDGM/ --name pix2pix-test --model pix2pix --dataset_mode custom --direction BtoA --display_freq 10000 --print_freq 10000 --input_nc 1 --output_nc 1 --use_wandb --val_dataroot ../../data/UCSF-PDGM/ --batch_size 400 --gpu_ids 0 --netG unet_128