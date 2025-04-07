import numpy as np
import warnings
import tensorflow as tf
from tqdm import tqdm

class AdaMultiLossesNorm:
    def __init__(self):
        pass

    def compute_losses(self, dataset=None, batch_size=None, *loss_functions):
        """
        Compute multiple losses from the dataset or from provided functions.

        Args:
            dataset (iterable, optional): Dataset containing inputs for loss computation.
            *loss_functions (callable): Loss functions that take (free, motion) as input.

        Returns:
            list of np.arrays: Computed losses.
        """
        losses = []
        
        if dataset is not None:
            for motion_data, free in tqdm(dataset):
                if isinstance(motion_data, tuple) and len(motion_data) == 3:
                    _, motion, _ = motion_data
                else:
                    motion = motion_data if not isinstance(motion_data, tuple) else motion_data[0]

                if free.shape[0] != batch_size or motion.shape[0] != batch_size:
                    print(f"Wrong shape detected free shape = {free.shape} while motion shape = {motion.shape}")
                    continue
                
                computed_losses = [loss_fn(free, motion).numpy() for loss_fn in loss_functions]
                losses.append([np.mean(loss) for loss in computed_losses])
        
        return np.array(losses).T  # Transpose to group by loss type
    
    def compute_normalized_weights_and_biases(self, *losses):
        """
        Implements Adaptive Multi-Losses Normalization for any number of loss functions.

        Args:
            *losses (list of lists or np.arrays): Variable number of loss lists/arrays.

        Returns:
            total_loss (float): The final computed total loss.
            weights (list of floats): Adjusted weights for each loss.
            biases (list of floats): Adjusted biases for each loss.
        """
        
        # Convert input losses to numpy arrays
        losses = [np.array(loss) for loss in losses]
        num_losses = len(losses)
        
        if num_losses < 1:
            raise ValueError("At least one loss function must be provided.")
        
        # Compute means and standard deviations
        means = [np.mean(loss) for loss in losses]
        stds = [np.std(loss, ddof=0) for loss in losses]
        
        # Check for zero standard deviation and set to a small number if needed
        stds = [s if s > 1e-8 else 1e-8 for s in stds]
        if any(s == 1e-8 for s in stds):
            warnings.warn("Standard deviation of one or more losses was zero and has been set to a small number.")
        
        # Select a reference loss (first one by default)
        ref_std = stds[0]
        ref_mean = means[0]
        
        # Compute weights and biases
        weights = [ref_std / s for s in stds]
        biases = [ref_mean - w * m for w, m in zip(weights, means)]
        
        return num_losses, weights, biases