import wandb
import subprocess
import os
from options.train_options import TrainOptions
import torch
import numpy as np
from data import create_dataset
from models import create_model
import time
from skimage.metrics import peak_signal_noise_ratio as psnr

def train_model():
    # Initialize wandb with sweep config
    wandb.init()
    
    # Get the base options
    opt = TrainOptions().parse()
    
    # Override options with sweep parameters
    opt.lr = wandb.config.learning_rate
    opt.beta1 = wandb.config.beta1
    opt.batch_size = wandb.config.batch_size
    opt.netG = wandb.config.generator_architecture
    opt.netD = wandb.config.discriminator_architecture
    opt.n_epochs = 20  # Reduced number of epochs for quick evaluation
    opt.norm = wandb.config.norm_type
    
    # Create smaller training dataset
    opt.number_of_samples = 2000  # Limit dataset size
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print('Using subset of training images = %d' % dataset_size)
    
    # Create validation dataset
    train_dataroot = opt.dataroot
    opt.dataroot = opt.val_dataroot
    opt.phase = 'val'
    val_dataset = create_dataset(opt, train=False)
    print('Number of validation images = %d' % len(val_dataset))
    opt.dataroot = train_dataroot
    opt.phase = 'train'
    
    # Create and setup model
    model = create_model(opt)
    model.setup(opt)
    
    best_psnr = 0
    total_iters = 0
    
    # Training loop
    for epoch in range(opt.n_epochs):
        epoch_start_time = time.time()
        for i, data in enumerate(dataset):
            total_iters += opt.batch_size
            
            # Training step
            model.set_input(data)
            model.optimize_parameters()
            
            # Log training metrics
            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                metrics = {f"train/{k}": v for k, v in losses.items()}
                wandb.log(metrics, step=total_iters)
        
        # Validation step
        model.eval()
        val_psnr = 0
        n_val = 0
        
        for i, data in enumerate(val_dataset):
            model.set_input(data)
            with torch.no_grad():
                model.forward()
            
            # Calculate PSNR using scikit-image
            fake = model.fake_B.cpu().numpy()
            real = model.real_B.cpu().numpy()
            val_psnr += psnr(real[0], fake[0], data_range=1.0)
            n_val += 1
            
            # Log first batch of validation images
            if i == 0:
                model.compute_visuals()
                visuals = model.get_current_visuals()
                for label, image in visuals.items():
                    wandb.log({f"val_images/{label}": wandb.Image(image)}, step=total_iters)
        
        avg_psnr = val_psnr / n_val
        wandb.log({"val/psnr": avg_psnr}, step=total_iters)
        
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
        
        print(f'End of epoch {epoch} / {opt.n_epochs} \t PSNR: {avg_psnr:.2f}')
        model.train()
    
    # Log final best PSNR
    wandb.log({"best_psnr": best_psnr})

# Define the sweep configuration
sweep_configuration = {
    'method': 'random',
    'metric': {'name': 'best_psnr', 'goal': 'maximize'},
    'parameters': {
        'learning_rate': {'values': [0.0002, 0.0001, 0.00005]},
        'beta1': {'values': [0.5, 0.75, 0.8]},
        'batch_size': {'values': [16]},
        'generator_architecture': {'values': ['unet_128', 'unet_256']},
        'discriminator_architecture': {'values': ['basic']},
        'norm_type': {'values': ['batch', 'instance']}
    }
}

if __name__ == '__main__':
    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_configuration, project="pix2pix-hyperparameter-search-01-23")
    
    # Run the sweep
    wandb.agent(sweep_id, train_model, count=10)  # Will try 10 different combinations 