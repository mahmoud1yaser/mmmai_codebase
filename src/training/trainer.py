import sys
sys.path.append("/kaggle/working/mmmai_codebase/")
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
from networks.wat_unet import WATUNet
from src.data.dataloader import DataLoader
from src.utils.adaptive_losses import AdaMultiLossesNorm

# Initialize loss object
loss_and_metric = Losses()
ada_multi_losses_norm = AdaMultiLossesNorm()

def setup_logging(log_dir):
    """Set up logging to file and console."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'training.log')
    
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to avoid duplication
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create file handler and add it to logger
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Create console handler and add it to logger
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def load_hyperparameters(config_path):
    """Load hyperparameters from a JSON configuration file."""
    logging.info("Loading configuration file...")
    with open(config_path, 'r') as f:
        config = json.load(f)
    logging.info("Hyperparameters loaded successfully.")
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
    
    logging.info("Loss functions loaded successfully.")
    return losses

def load_model_architecture(model_architecture_name):
    """Load model architecture dynamically based on the name from the config."""
    available_models = {
        "stacked_unets": StackedUNets(),
        "wat_stacked_unets": WATStackedUNets(),
        "wat_unet": WATUNet(),
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
    print("Initializing DataLoader...")
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
        print("DataLoader initialized and data split successfully.")
        return data_loader
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def train(config):
    """Train the model."""    
    HEIGHT = config['height']
    WIDTH = config['width']
    LEARNING_RATE = config['learning_rate']
    BATCH_SIZE = config['batch_size']
    NB_EPOCH = config['epochs']
    WEIGHTS_PATH = config['weights_path']
    CHECKPOINT_PATH = config['checkpoint_path']
    SAVE_BEST = config["callbacks"]["model_checkpoint"]["filename_pattern"]
    START_EPOCH = config["start_epoch"]
    SKIP_AMLN = config["skip_ada_multi_losses_norm"]

    # Create weights directory if it doesn't exist
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    
    # Setup logging
    log_dir = os.path.join(WEIGHTS_PATH, "logs")
    setup_logging(log_dir)
    
    try:
        logging.info("Loading loss functions...")
        input_losses = load_loss_functions(config['loss_functions'])
        
        logging.info(f"Loading model architecture: {config['model_architecture']}...")
        model_architecture = load_model_architecture(config['model_architecture'])
        
        model = model_architecture.Correction_Multi_input(HEIGHT, WIDTH)
        
        logging.info("Initializing DataLoader...")
        data_loader_config = config['data_loader']
        data_loader = load_data_loader(config['dataset'], BATCH_SIZE, config['data_ids'], data_loader_config)
        
        train_dataset = data_loader.generator('train', enable_SAP=config['enable_SAP'])
        validation_dataset = data_loader.generator('validation')
        
        if len(input_losses) > 1:
            if not SKIP_AMLN:
                logging.info("Starting AdaMultiLossesNorm computation...")
                losses = ada_multi_losses_norm.compute_losses(train_dataset, BATCH_SIZE, *input_losses)
                n_loss, w_comb, b_comb = ada_multi_losses_norm.compute_normalized_weights_and_biases(*losses)
            else:
                logging.info("Skipping AdaMultiLossesNorm computation.")
                n_loss = len(input_losses)
                w_comb = [1.0] * n_loss
                b_comb = [0.0] * n_loss
        else:
            n_loss = 1
            w_comb = [1.0]
            b_comb = [0.0]

        logging.info(f"Number of losses: {n_loss}")
        logging.info(f"Weights: {w_comb}")
        logging.info(f"Biases: {b_comb}")
        
        def total_loss(y_true, y_pred):
            losses = [fn(y_true, y_pred) * w_comb[i] + b_comb[i] for i, fn in enumerate(input_losses)]
            return sum(losses) / n_loss

        model.compile(
            loss=total_loss,
            optimizer=Adam(learning_rate=LEARNING_RATE),
            metrics=[loss_and_metric.ssim_score, 'mse', loss_and_metric.psnr]
        )

        if CHECKPOINT_PATH:
            logging.info(f"Loading checkpoint from {CHECKPOINT_PATH}")
            model = load_model(CHECKPOINT_PATH, custom_objects={'total_loss': total_loss})
        else:
            START_EPOCH = 0
            logging.info("No checkpoint provided, starting from scratch.")
            
        callbacks = [
            CSVLogger(f'{WEIGHTS_PATH}_Loss_Acc.csv'),
            LearningRateScheduler(lambda epoch: exponential_lr(epoch, LEARNING_RATE)),
            ModelCheckpoint(f'{WEIGHTS_PATH}{SAVE_BEST}.h5', save_best_only=True, monitor='val_loss'),
            TensorBoard(log_dir=log_dir)
        ]

        logging.info("Starting model training...")
        model.fit(
            train_dataset,
            epochs=NB_EPOCH,
            validation_data=validation_dataset,
            callbacks=callbacks,
            initial_epoch=START_EPOCH
        )
        
        logging.info("Training completed successfully.")
        
        logging.info("Evaluating model on test dataset...")
        test_dataset = data_loader.generator('test')
        model.evaluate(test_dataset)
        
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise

def main():
    """Main function to parse arguments and execute training."""
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument('-c', '--config', type=str, required=True, help="Path to the JSON configuration file.")
    parser.add_argument('-d', '--dataset', type=str, help="Path to the dataset.")
    args = parser.parse_args()

    config = load_hyperparameters(args.config)
    
    if args.dataset:
        config['dataset'] = args.dataset

    train(config)

if __name__ == "__main__":
    main()