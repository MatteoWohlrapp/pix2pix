"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
import wandb
import torch
import os
from models.networks import UnetGenerator
from fairness.fairness_loss import FairnessLoss
from tqdm import tqdm
from fairness.classification_model import TGradeBCEClassifier, TTypeBCEClassifier
from fairness.resnet_classification_network import ResNetClassifierNetwork


def load_classifier_models(dataset_mode, tum, device):
    if dataset_mode == "chex":
        classifier_path = "/vol/miltank/projects/practical_SoSe24/mri_inr/matteo/code/torchxrayvision/chex-densenet-class_norm01-best.pt" if tum else "/lotterlab/users/matteo/code/torchxrayvision/output/class_norm01/chex-densenet-class_norm01-best.pt"
        classifier = torch.load(classifier_path, map_location=device)
        for param in classifier.parameters():
            param.requires_grad = False
        return classifier
    elif dataset_mode == "ucsf":
        ttype_path = "" if tum else "/lotterlab/users/matteo/models/ttype/checkpoints/ttype_test_20240925_113544_epoch_1_best.pth"
        tgrade_path = "" if tum else "/lotterlab/users/matteo/models/tgrade/checkpoints/tgrade_test_20240925_113550_epoch_1_best.pth"
        paths = {"TGradeBCEClassifier": tgrade_path, "TTypeBCEClassifier": ttype_path}
        task_models = {}
        for name, path in paths.items():
            if name == "TGradeBCEClassifier":
                classifier = TGradeBCEClassifier()
            elif name == "TTypeBCEClassifier":
                classifier = TTypeBCEClassifier()
            classifier = classifier.to(device)

            network = ResNetClassifierNetwork(num_classes=classifier.num_classes
                                                , resnet_version="resnet18")
            
            network = network.to(device)
            classifier.set_network(network)
            classifier.load_state_dict(torch.load(path, map_location=device))
            for param in classifier.parameters():
                param.requires_grad = False
            task_models[name] = classifier

        def apply_task_models(x):
            first_output = task_models["TGradeBCEClassifier"](x)
            second_output = task_models["TTypeBCEClassifier"](x)
            return torch.cat((first_output, second_output), dim=1)
        return apply_task_models

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    gpu_ids = opt.gpu_ids
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
    # Initialize wandb
    if opt.use_wandb:
        wandb.init(
            project=opt.wandb_project_name,
            name=opt.name,
            config={
                # Model parameters
                "model": opt.model,
                "netG": opt.netG,
                "netD": opt.netD,
                "gan_mode": opt.gan_mode,
                # Dataset parameters
                "batch_size": opt.batch_size,
                "sampling_mask": opt.sampling_mask if hasattr(opt, 'sampling_mask') else None,
                "number_of_samples": opt.number_of_samples if hasattr(opt, 'number_of_samples') else None,
                # Training parameters
                "n_epochs": opt.n_epochs,
                "n_epochs_decay": opt.n_epochs_decay,
                "lr": opt.lr,
                "beta1": opt.beta1,
            }
        )

    dataset = create_dataset(opt)  # create training dataset
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)

    # Create validation dataset if val_dataroot is provided
    if opt.val:
        val_dataset = create_dataset(opt, train=False)
        print('The number of validation images = %d' % len(val_dataset))

    model = create_model(opt)
    state_dict = torch.load(opt.model_g_path, map_location=torch.device(model.device))
    fixed_state_dict = {'module.' + k: v for k, v in state_dict.items()}
    model.netG.load_state_dict(fixed_state_dict)

    if opt.model_d_path:
        state_dict = torch.load(opt.model_d_path, map_location=torch.device(model.device))
        fixed_state_dict = {'module.' + k: v for k, v in state_dict.items()}
        model.netD.load_state_dict(fixed_state_dict)

    model.setup(opt)
    total_iters = 0

    tum = opt.tum

    classifier_models = load_classifier_models(opt.dataset_mode, tum, model.device)
    #fairness_loss = FairnessLoss(classifier_models, opt.fairness_lambda)
    #fairness_loss.to(model.device)

    #model.fairness_loss = fairness_loss
    #model.lambda_fairness = opt.lambda_fairness

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        
        # Create progress bar for the epoch
        pbar = tqdm(enumerate(dataset), 
                   total=len(dataset)//opt.batch_size,
                   desc=f'Epoch {epoch}/{opt.n_epochs + opt.n_epochs_decay}',
                   leave=True)
        
        for i, data in pbar:
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            
            # Training step
            model.set_input(data)
            model.optimize_parameters()

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                
                # Log training metrics to wandb
                if opt.use_wandb:
                    metrics = {f"train/{k}": v for k, v in losses.items()}
                    metrics.update({
                        "epoch": epoch,
                        "iteration": total_iters,
                        "time/data_loading": t_data,
                        "time/forward_backward": t_comp
                    })
                    wandb.log(metrics, step=total_iters)
                
                # Print to console
                message = f'(epoch: {epoch}, iters: {epoch_iter}, time: {t_comp:.3f}, data: {t_data:.3f}) '
                message += ' '.join(f'{k}: {v:.3f}' for k, v in losses.items())
                print(message)

            # Log training images occasionally
            if total_iters % (opt.print_freq * 10) == 0:
                if opt.use_wandb:
                    model.compute_visuals()
                    visuals = model.get_current_visuals()
                    for label, image in visuals.items():
                        if len(image.shape) == 4:  # If batch dimension exists
                            n_images = min(image.shape[0], 5)
                            image = image[:n_images]
                        wandb.log({f"train_images/{label}": wandb.Image(image)}, 
                                step=total_iters)

            if total_iters % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                model.save_networks('latest')

            iter_data_time = time.time()

            # Update progress bar description with losses if needed
            if total_iters % opt.print_freq == 0:
                pbar.set_postfix({k: f'{v:.3f}' for k, v in losses.items()})

        # Validation step
        if opt.val and epoch % opt.val_freq == 0:
            model.eval()
            val_losses = {}
            val_iter = 0
            
            for i, data in enumerate(val_dataset):
                model.set_input(data)
                with torch.no_grad():
                    model.forward()
                losses = model.get_current_losses()
                
                # Accumulate validation losses
                for k, v in losses.items():
                    val_losses[k] = val_losses.get(k, 0) + v
                val_iter += 1
                
                # Log first batch of validation images
                if i == 0 and opt.use_wandb:
                    model.compute_visuals()
                    visuals = model.get_current_visuals()
                    for label, image in visuals.items():
                        if len(image.shape) == 4:  # If batch dimension exists
                            n_images = min(image.shape[0], 5)
                            image = image[:n_images]
                        wandb.log({f"val_images/{label}": wandb.Image(image)}, 
                                step=total_iters)
            
            # Average and log validation losses
            val_losses = {k: v / val_iter for k, v in val_losses.items()}
            if opt.use_wandb:
                metrics = {f"val/{k}": v for k, v in val_losses.items()}
                wandb.log(metrics, step=total_iters)
            
            # Print validation losses
            message = f'Validation (epoch: {epoch}) '
            message += ' '.join(f'{k}: {v:.3f}' for k, v in val_losses.items())
            print(message)
            
            model.train()

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()

    if opt.use_wandb:
        wandb.finish()
