import math
import argparse
import tensorflow as tf
import numpy as np
import json
from tqdm import tqdm
from tensorflow.keras.callbacks import CSVLogger, LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

from src.utils.losses import Losses 
from networks.wat_stacked_unets import WATStackedUNets
from networks.stacked_unets import StackedUNets
from src.data.dataloader import DataLoader
from src.utils.adaptive_losses import AdaMultiLossesNorm

# Initialize loss object
loss_and_metric = Losses()
ada_multi_losses_norm = AdaMultiLossesNorm()

def load_hyperparameters(config_path):
    """Load hyperparameters from a JSON configuration file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def update_hyperparameters_with_args(config, args):
    """Update hyperparameters with the provided command line arguments."""
    if args.dataset:
        config['dataset'] = args.dataset
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    if args.epochs:
        config['epochs'] = args.epochs
    if args.height:
        config['height'] = args.height
    if args.width:
        config['width'] = args.width
    if args.split_ratio:
        config['data_loader']['split_ratio'] = args.split_ratio
    if args.view:
        config['data_loader']['view'] = args.view
    if args.crop is not None:
        config['data_loader']['crop'] = args.crop
    if args.split_json_path:
        config['data_loader']['split_json_path'] = args.split_json_path
    if args.checkpoint_path:
        config['checkpoint_path'] = args.checkpoint_path
    return config

def load_loss_functions(loss_function_names):
    """Load loss functions dynamically based on the names from the config."""
    available_losses = {
        "ssim_loss": loss_and_metric.ssim_loss,
        "perceptual_loss": loss_and_metric.perceptual_loss,
        "mae_loss": loss_and_metric.mae_loss
    }
    
    losses = []
    for loss_name in loss_function_names:
        if loss_name in available_losses:
            losses.append(available_losses[loss_name])
        else:
            raise ValueError(f"Loss function {loss_name} is not recognized.")
    
    return losses

def load_model_architecture(model_architecture_name):
    """Load loss functions dynamically based on the names from the config."""
    available_models = {
        "stacked_unets": StackedUNets,
        "wat_stacked_unets": WATStackedUNets 
        # Add Generative networks here
        }
    
    for model_name in available_models:
        if model_name in model_architecture_name:
            return available_models[model_name]
        else:
            raise ValueError(f"Model {model_name} is not recognized.")
        

def exponential_lr(epoch, LEARNING_RATE):
    """Learning rate scheduler function."""
    if epoch < 10:
        return LEARNING_RATE
    else:
        return LEARNING_RATE * math.exp(0.1 * (10 - epoch))  # lr decreases exponentially by a factor of 10
    
def load_data_loader(dataset_path, batch_size, data_ids, data_loader_config):
    """Load and split data using DataLoader based on configuration."""
    try:
        data_loader = DataLoader(
            data_path=dataset_path,
            split_ratio=data_loader_config['split_ratio'],
            view=data_loader_config['view'],
            data_id=data_ids[dataset_path],
            crop=data_loader_config['crop'],
            batch_size=batch_size,
            split_json_path=data_loader_config['split_json_path']
        )
        data_loader.split_data()
        return data_loader
    except KeyError:
        raise ValueError(f"Dataset path {dataset_path} not found in data_ids.")
    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}")

def train(config):
    """Train the model."""
    print('---------------------------------')
    print('Model Training ...')
    print('---------------------------------')

    HEIGHT = config['height']
    WIDTH = config['width']
    LEARNING_RATE = config['learning_rate']
    BATCH_SIZE = config['batch_size']
    NB_EPOCH = config['epochs']
    WEIGHTS_PATH = config['weights_path']
    CHECKPOINT_PATH = config['checkpoint_path']
    
    try:
        # Load loss functions from config
        input_losses = load_loss_functions(config['loss_functions'])

        # Select model architecture
        model_architecture = load_model_architecture(config['model_architecture'])
        
        # Initialize model
        model = model_architecture.Correction_Multi_input(HEIGHT, WIDTH)
        
        # Load data using data loader config
        data_loader_config = config['data_loader']
        data_loader = load_data_loader(config['dataset'], BATCH_SIZE, config['data_ids'], data_loader_config)
        train_dataset = data_loader.generator('train', enable_SAP=config['enable_SAP'])
        validation_dataset = data_loader.generator('validation')

        if len(input_losses) > 1:

            # Calculate adaptive weights and biases
            try:
                losses = ada_multi_losses_norm.compute_losses(train_dataset, BATCH_SIZE, *input_losses)
                n_loss, w_comb, b_comb = ada_multi_losses_norm.compute_normalized_weights_and_biases(*losses)

                print(f"Number of losses: {n_loss}")
                print(f"Weight (w_comb): {w_comb}")
                print(f"Bias (b_comb): {b_comb}")
            except ValueError as e:
                print(f"Error: {e}")
                # Fallback to default values if adaptive normalization fails
                w_comb = [1] * len(input_losses)  # Ensure w_comb has the correct length
                b_comb = [0] * len(input_losses)  # Ensure b_comb has the correct length

            def total_loss(w_comb, b_comb, n_loss):
                """Wrapper function to create a total_loss function with dynamic w_comb and b_comb."""
                def loss(y_true, y_pred):
                    """Custom loss function combining perceptual and SSIM losses."""
                    losses = []
                    for i, loss_fn in enumerate(input_losses):
                        # Compute each loss (e.g., perceptual and SSIM)
                        current_loss = loss_fn(y_true, y_pred)
                        
                        # Apply the weight and bias for the current loss
                        scaled_loss = current_loss * w_comb[i]
                        adjusted_loss = scaled_loss + b_comb[i]
                        losses.append(adjusted_loss)
                    
                    total = sum(losses) / n_loss  # Normalize by the number of losses
                    return total

                return loss
        else:
            total_loss = input_losses[0]

        # Compile model with updated total_loss function
        model.compile(
            loss=total_loss(w_comb, b_comb),  # Pass w_comb and b_comb to total_loss
            optimizer=Adam(learning_rate=LEARNING_RATE),
            metrics=[loss_and_metric.ssim_score, 'mse', loss_and_metric.psnr]
        )

        # Load from checkpoint if specified
        if CHECKPOINT_PATH:
            try:
                model = load_model(CHECKPOINT_PATH)
                print(f"Model loaded from checkpoint: {CHECKPOINT_PATH}")
            except Exception as e:
                print(f"Warning: Could not load model from checkpoint: {CHECKPOINT_PATH}. Error: {e}")

        # Callbacks
        csv_logger = CSVLogger(f'{WEIGHTS_PATH}_Loss_Acc.csv', append=True, separator=',')
        reduce_lr = LearningRateScheduler(lambda epoch: exponential_lr(epoch, LEARNING_RATE))
        checkpoint_path = f'{WEIGHTS_PATH}WAT_style_stacked_{{epoch:02d}}_val_loss_{{val_loss:.4f}}.h5'
        model_checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=False,
            save_weights_only=False,
            mode='min',
            verbose=1
        )

        # Train model
        history = model.fit(
            train_dataset,
            epochs=NB_EPOCH,
            verbose=1,
            validation_data=validation_dataset,
            callbacks=[csv_logger, reduce_lr, model_checkpoint]
        )
        print("Training completed successfully.")
    except Exception as e:
        print(f"Error during model training: {e}")

def main():
    """Main function to parse arguments and execute training."""
    parser = argparse.ArgumentParser(description="Train WAT U-Net model.")
    parser.add_argument('-c', '--config', type=str, required=True, help="Path to the JSON configuration file.")
    parser.add_argument('-d', '--dataset', type=str, help="Path to the dataset.")
    parser.add_argument('--batch_size', type=int, help="Batch size for training.")
    parser.add_argument('--learning_rate', type=float, help="Learning rate for training.")
    parser.add_argument('--epochs', type=int, help="Number of epochs.")
    parser.add_argument('--height', type=int, help="Height of the input images.")
    parser.add_argument('--width', type=int, help="Width of the input images.")
    parser.add_argument('--split_ratio', type=float, nargs=3, help="Train, validation, test split ratio.")
    parser.add_argument('--view', type=str, help="View for data (e.g., Axial).")
    parser.add_argument('--crop', type=bool, help="Whether to crop the images.")
    parser.add_argument('--split_json_path', type=str, help="Path to the split json file.")
    parser.add_argument('--checkpoint_path', type=str, help="Path to the checkpoint for resuming training.")
    args = parser.parse_args()

    # Load and update config
    config = load_hyperparameters(args.config)
    config = update_hyperparameters_with_args(config, args)

    # Train the model
    train(config)

if __name__ == "__main__":
    main()
