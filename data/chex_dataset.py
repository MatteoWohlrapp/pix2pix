import os
import pathlib
from typing import Optional
from skimage.transform import radon, iradon, resize
from skimage.io import imread
import numpy as np
import polars as pl
import torch
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset

def min_max_slice_normalization(scan: torch.Tensor) -> torch.Tensor:
    scan_min = scan.min()
    scan_max = scan.max()
    if scan_max == scan_min:
        return scan
    normalized_scan = (scan - scan_min) / (scan_max - scan_min)
    return normalized_scan

class ChexDataset(BaseDataset):
    """Dataset for X-ray reconstruction using CycleGAN."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--photon_count', type=float, default=1e5, help='photon count for Poisson noise')
        parser.add_argument('--number_of_samples', type=int, default=None, help='number of samples to use (0 for all)')
        parser.add_argument('--seed', type=int, default=31415, help='random seed')
        parser.add_argument('--csv_path', type=str, required=True, help='path to the metadata CSV file')
        return parser

    def __init__(self, opt, train=True):
        BaseDataset.__init__(self, opt)
        self.data_root = pathlib.Path(opt.dataroot)
        self.csv_path = pathlib.Path(opt.csv_path)
        self.number_of_samples = opt.number_of_samples if hasattr(opt, 'number_of_samples') else None
        self.seed = opt.seed if hasattr(opt, 'seed') else 31415
        self.split = opt.phase
        self.photon_count = opt.photon_count if hasattr(opt, 'photon_count') else 1e5
        print(f"Using {self.photon_count} photons")
        self.train = train
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Convert numpy to tensor
            transforms.Lambda(min_max_slice_normalization),
            transforms.Lambda(lambda x: x.float())  # Ensure float32
        ])

    def _load_metadata(self):
        """Load metadata for the dataset from CSV file."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.csv_path}")
            
        df = pl.read_csv(self.csv_path)
        
        if self.train:
            df = df.filter(pl.col("split") == "train_recon")
        else:
            df = df.filter(pl.col("split") == "val_recon")
            
        if self.number_of_samples is not None and self.number_of_samples > 0:
            df = df.sample(n=self.number_of_samples, seed=self.seed)
            
        return df

    def apply_bowtie_filter(self,sinogram):
        """
        Apply a bowtie filter to the Sinogram.

        Parameters:
        - sinogram: 2D numpy array of the Sinogram.

        Returns:
        - filtered_sinogram: Sinogram with the bowtie filter applied.
        """
        rows, cols = sinogram.shape
        profile = np.linspace(0.05, 1.0, cols // 2)
        filter_profile = np.concatenate([profile[::-1], profile])[:cols]
        return sinogram * filter_profile[np.newaxis, :]

    def process_image(self, image: np.ndarray) -> tuple:
        """
        Process the X-ray image with controllable noise levels.
        
        Parameters:
        - image: 2D numpy array of the original image
        - photon_count: Number of photons (lower = more noise)
        
        Returns:
        - reconstructed_image: Reconstructed image after processing
        - metrics: Dictionary with noise metrics
        """
        # Normalize input image to [0,1] range
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        
        # Steps 2-5: Same as before
        theta = np.linspace(0., 180., max(image.shape), endpoint=False)
        sinogram = radon(image, theta=theta, circle=False)
        filtered_sinogram = self.apply_bowtie_filter(sinogram)
        
        max_val = np.max(filtered_sinogram)
        scaled_sinogram = (filtered_sinogram / max_val) * self.photon_count
        noisy_sinogram = np.random.poisson(scaled_sinogram).astype(float)
        noisy_sinogram = (noisy_sinogram / self.photon_count) * max_val

        reconstructed_padded_image = iradon(noisy_sinogram, theta=theta, filter_name='hann', circle=False)
        reconstructed_image = resize(reconstructed_padded_image, image.shape, mode='reflect', anti_aliasing=True)
        
        # Normalize reconstructed image to [0,1] range
        reconstructed_image = (reconstructed_image - np.min(reconstructed_image)) / (np.max(reconstructed_image) - np.min(reconstructed_image))

        return reconstructed_image

    def __getitem__(self, index):
        row = self.metadata.row(index, named=True)
        
        # Load the original image
        image_path = os.path.join(self.data_root, row["Path"])
        original_image = imread(image_path, as_gray=True).astype(np.float32)  # Convert to float32
        original_image = min_max_slice_normalization(original_image)
        original_image = resize(original_image, (256, 256), anti_aliasing=True)
        
        # Process the image before any resizing
        degraded_image = self.process_image(original_image)
        
        # Apply transforms to both images
        if self.transform:
            original_tensor = self.transform(original_image)
            degraded_tensor = self.transform(degraded_image)
        
        return {
            'A': degraded_tensor,    # degraded image
            'B': original_tensor,    # original image
            'A_paths': image_path,
            'B_paths': image_path
        }

    def __len__(self):
        return len(self.metadata)

    def get_patient_data(self, patient_id):
        patient_slices_metadata = self.metadata.filter(pl.col("PatientID") == patient_id)
        patient_slices_metadata = patient_slices_metadata.sort("slice_id")

        if len(patient_slices_metadata) == 0:
            print(f"No slices found for PatientID={patient_id}")
            return []

        slices = []
        for row_idx in range(len(patient_slices_metadata)):
            row = patient_slices_metadata.row(row_idx, named=True)
            
            # Load the original image
            image_path = os.path.join(self.data_root, row["Path"])
            original_image = imread(image_path, as_gray=True)
            
            # Process the image before any resizing
            degraded_image = self.process_image(original_image)
            
            # Apply transforms to both images
            if self.transform:
                original_tensor = self.transform(original_image)
                degraded_tensor = self.transform(degraded_image)
            
            slices.append((degraded_tensor, original_tensor))

        return slices 