import sys
import os
import math
import argparse
import tensorflow as tf
import numpy as np
import json
import logging
from tqdm import tqdm
from tensorflow.keras.callbacks import CSVLogger, LearningRateScheduler, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

from src.utils.losses import Losses 
from networks.wat_stacked_unets import WATStackedUNets
from networks.stacked_unets import StackedUNets
from src.data.dataloader import DataLoader
from src.utils.adaptive_losses import AdaMultiLossesNorm

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize loss object
loss_and_metric = Losses()
ada_multi_losses_norm = AdaMultiLossesNorm()

def load_hyperparameters(config_path):
    """Load hyperparameters from a JSON configuration file."""
    logging.info("Loading configuration file...")
    with open(config_path, 'r') as f:
        config = json.load(f)
    logging.info("Hyperparameters loaded successfully.")
    return config

def load_loss_functions(loss_function_names):
    """Load loss functions dynamically based on the names from the config."""
    logging.info("Loading loss functions...")
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
    
    logging.info("Loss functions loaded successfully.")
    return losses

def load_model_architecture(model_architecture_name):
    """Load model architecture dynamically based on the name from the config."""
    logging.info(f"Loading model architecture: {model_architecture_name}...")
    available_models = {
        "stacked_unets": StackedUNets(),
        "wat_stacked_unets": WATStackedUNets()
    }
    
    if model_architecture_name in available_models:
        logging.info(f"Model architecture '{model_architecture_name}' loaded successfully.")
        return available_models[model_architecture_name]
    else:
        raise ValueError(f"Model '{model_architecture_name}' is not recognized.")
        

def exponential_lr(epoch, LEARNING_RATE):
    """Learning rate scheduler function."""
    if epoch < 10:
        return LEARNING_RATE
    else:
        return LEARNING_RATE * math.exp(0.1 * (10 - epoch))
    
def load_data_loader(dataset_path, batch_size, data_ids, data_loader_config):
    """Load and split data using DataLoader based on configuration."""
    logging.info("Initializing DataLoader...")
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
        logging.info("DataLoader initialized and data split successfully.")
        return data_loader
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def train(config):
    """Train the model."""
    logging.info('Starting model training...')
    
    HEIGHT = config['height']
    WIDTH = config['width']
    LEARNING_RATE = config['learning_rate']
    BATCH_SIZE = config['batch_size']
    NB_EPOCH = config['epochs']
    WEIGHTS_PATH = config['weights_path']
    CHECKPOINT_PATH = config['checkpoint_path']
    
    try:
        input_losses = load_loss_functions(config['loss_functions'])
        model_architecture = load_model_architecture(config['model_architecture'])
        
        model = model_architecture.Correction_Multi_input(HEIGHT, WIDTH)
        
        data_loader_config = config['data_loader']
        data_loader = load_data_loader(config['dataset'], BATCH_SIZE, config['data_ids'], data_loader_config)
        
        train_dataset = data_loader.generator('train', enable_SAP=config['enable_SAP'])
        validation_dataset = data_loader.generator('validation')
        
        losses = ada_multi_losses_norm.compute_losses(train_dataset, BATCH_SIZE, *input_losses)
        n_loss, w_comb, b_comb = ada_multi_losses_norm.compute_normalized_weights_and_biases(*losses)
        
        def total_loss(y_true, y_pred):
            losses = [fn(y_true, y_pred) * w_comb[i] + b_comb[i] for i, fn in enumerate(input_losses)]
            return sum(losses) / n_loss

        model.compile(
            loss=total_loss,
            optimizer=Adam(learning_rate=LEARNING_RATE),
            metrics=[loss_and_metric.ssim_score, 'mse', loss_and_metric.psnr]
        )

        if CHECKPOINT_PATH:
            model = load_model(CHECKPOINT_PATH, custom_objects={'total_loss': total_loss})

        log_dir = os.path.join(WEIGHTS_PATH, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        callbacks = [
            CSVLogger(f'{WEIGHTS_PATH}_Loss_Acc.csv'),
            LearningRateScheduler(lambda epoch: exponential_lr(epoch, LEARNING_RATE)),
            ModelCheckpoint(f'{WEIGHTS_PATH}model_best.h5', save_best_only=True, monitor='val_loss'),
            TensorBoard(log_dir=log_dir)
        ]

        model.fit(
            train_dataset,
            epochs=NB_EPOCH,
            validation_data=validation_dataset,
            callbacks=callbacks
        )
        
        logging.info("Training completed successfully.")
        
        test_dataset = data_loader.generator('test')
        model.evaluate(test_dataset)
        
    except Exception as e:
        logging.error(f"Error during model training: {e}")

def main():
    """Main function to parse arguments and execute training."""
    parser = argparse.ArgumentParser(description="Train WAT U-Net model.")
    parser.add_argument('-c', '--config', type=str, required=True, help="Path to the JSON configuration file.")
    parser.add_argument('-d', '--dataset', type=str, help="Path to the dataset.")
    args = parser.parse_args()

    config = load_hyperparameters(args.config)
    
    if args.dataset:
        config['dataset'] = args.dataset

    train(config)

if __name__ == "__main__":
    main()