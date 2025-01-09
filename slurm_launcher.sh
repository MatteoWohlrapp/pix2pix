#!/usr/bin/bash

#SBATCH -J "mri_inr"   # job name
#SBATCH --time=2-00:00:00   # walltime
#SBATCH --output=/vol/aimspace/projects/practical_SoSe24/mri_inr/matteo/logs/train_%A.out  # Standard output of the script (Can be absolute or relative path)
#SBATCH --error=/vol/aimspace/projects/practical_SoSe24/mri_inr/matteo/logs/train_%A.err  # Standard error of the script
#SBATCH --mem=64G
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16   # number of processor cores (i.e. tasks)
#SBATCH --qos=master-queuesave
#SBATCH --gres=gpu:A100:1              # Request 1 specific GPU (e.g., A100)

# load python module
. "/opt/anaconda3/etc/profile.d/conda.sh"

# activate corresponding environment
conda deactivate
conda activate pix

cd "/vol/aimspace/projects/practical_SoSe24/mri_inr/matteo/code/pix2pix"

python train.py --dataroot /vol/miltank/datasets/CheXpert --name pix2pix-chex06 --model pix2pix --dataset_mode chex --direction AtoB --display_freq 250 --print_freq 250 --input_nc 1 --output_nc 1 --use_wandb --val_dataroot /vol/miltank/datasets/CheXpert --batch_size 32 --gpu_ids 0 --netG unet_128 --csv_path /vol/aimspace/projects/practical_SoSe24/mri_inr/matteo/data/CheXpert/chex-metadata.csv --photon_count 10000