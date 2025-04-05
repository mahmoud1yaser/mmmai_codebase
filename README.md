# mmmai_codebase

## Overview

This repository provides the code for training and testing a deep learning model on MRI data, particularly focused on motion artifact correction and segmentation. You can configure and train the model using hyperparameters specified in a JSON configuration file. The codebase is modular, allowing you to easily add new networks and loss functions for experimentation.

## Training

To train the model, follow the steps below:

1. **Configure Hyperparameters**:
   Edit the `configs/hyperparameters.json` file to specify your dataset and training settings. Example configuration:

   ```json
   {
     "dataset": "/kaggle/input/mmmai-simulated-data/ds004795-download",  # Path to dataset
     "data_ids": {  # Edit paths for datasets based on your environment
       "/kaggle/input/mmmai-simulated-data/ds004795-download": "Motion_Simulated", 
       "/kaggle/input/mmmai-regist-data/MR-ART-Regist": "Motion",
       "/kaggle/input/brats-motion-data/new_Brats_motion_data": "BraTS"
     },
     "loss_functions": [  # Specify loss functions to be used (you can also add or remove losses)
       "ssim_loss",
       "perceptual_loss"
     ],
     "model_architecture": "wat_stacked_unets",  # Select the model architecture (e.g., wat_stacked_unets or stacked_unets)
     "checkpoint_path": null,  # Optional: Provide the path to a checkpoint if resuming training
     "epochs": 10,  # Number of epochs for training
     "batch_size": 10,  # Batch size for training
     "enable_SAP": true, # set enable_SAP = false if using other generative models
     "start_epoch":0, # set only in case checkpoint_path is given
     "weights_path": "trained_models/", # output folder path

   }
   ```

2. **Customizing the Configuration**:
   - **Dataset**: Specify the path to the dataset. You may need to adjust the paths based on your environment.
   - **Data IDs**: List the corresponding dataset identifiers and paths.
   - **Loss Functions**: Choose from the implemented loss functions. You can also add new ones in the `utils/losses.py` file.
   - **Model Architecture**: Select the model architecture to train (currently supports `wat_stacked_unets` and `stacked_unets`).
   - **Epochs and Batch Size**: Adjust the number of epochs and batch size for training according to your hardware and task.

---

## Setting Up Environment

### 1. **Create and Activate Conda Environment**

To create a Conda environment with Python 3.10.12, run the following command:

```bash
conda create -n myenv python=3.10.12
```

Activate the environment:

```bash
conda activate myenv
```

### 2. **Upgrade pip** (Optional but recommended)

```bash
python -m pip install --upgrade pip
```

### 3. **Install Required Packages**

Install Packages:

```bash
pip install -r requirements.txt
```

### 4. **Verify the Installation**

Run the following command to verify your installations:

```bash
python -c "import tensorflow as tf; import numpy as np; import pandas as pd; import os; print('TensorFlow version:', tf.__version__); print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))"
```

---

## Adding a New Network

To add a new network to the training pipeline:

1. **Implement the Network**:
   - Create a new Python file under the `networks/` directory.
   - Define your model architecture in the new file.

2. **Update the Trainer**:
   - Open `src/training/trainer.py`.
   - Import your new network class at the top of the file.
   - Modify the `load_model_architecture` function to include your new network. For example:

   ```python
   from networks.new_network import NewNetwork

   def load_model_architecture(model_architecture_name):
       available_models = {
           "stacked_unets": StackedUNets,
           "wat_stacked_unets": WATStackedUNets,
           "new_network": NewNetwork  # Add your new network here
       }
       return available_models.get(model_architecture_name, None)
   ```

3. **Update Configuration**:
   - Edit `configs/config.json` to include your new network under the `model_architecture` key.

---

## Adding a New Loss Function

To add a new loss function:

1. **Implement the Loss**:
   - Add a new loss function in the `utils/losses.py` file. Ensure it accepts the arguments `y_true` and `y_pred`:

   ```python
   def new_loss(y_true, y_pred):
       # Define your custom loss here
       return loss_value
   ```

2. **Update the Trainer**:
   - Open `src/training/trainer.py`.
   - Modify the `load_loss_functions` function to include your new loss:

   ```python
   from utils.losses import new_loss

   def load_loss_functions(loss_function_names):
       available_losses = {
           "ssim_loss": loss_and_metric.ssim_loss,
           "perceptual_loss": loss_and_metric.perceptual_loss,
           "new_loss": new_loss  # Add your new loss here
       }
       return [available_losses[loss_name] for loss_name in loss_function_names]
   ```

3. **Update Configuration**:
   - Edit `configs/config.json` and add your new loss to the `loss_functions` list:

   ```json
   "loss_functions": [
     "ssim_loss",
     "perceptual_loss",
     "new_loss"  # Add your new loss here
   ]
   ```

---

## Training the Model

1. Once the configuration is set and any custom networks or losses are added, run the training script:

   ```bash
   python src/training/trainer.py --config configs/config.json
   ```

2. The script will automatically load the dataset, model, and loss functions from the configuration file and begin training.

---
## Monitoring Training with TensorBoard

The training process automatically logs metrics and visualizations to TensorBoard for real-time monitoring.

### Metrics Tracked
- **Losses**: Training & validation loss for each configured loss function
- **Metrics**: SSIM, PSNR, and other evaluation metrics
- **Learning Rate**: Current learning rate value (especially useful with schedulers)
- **Images**: Sample inputs, targets, and predictions (logged periodically)

### Launching TensorBoard

1. During or after training, run:
   ```bash
   tensorboard --logdir=logs/tensorboard
2. Open your browser to
   ```bash
   http://localhost:6006/
### Launching TensorBoard on Kaggle
Using Kaggle, you can't access localhost, so that we can use ngrok
1. Download ngrok 
   ```bash
   !wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
   !tar xf ./ngrok-v3-stable-linux-amd64.tgz -C /usr/local/bin
2. Authenticate ngrok using your access token
   ```bash
   !ngrok authtoken [Your access token]
3. Launches TensorBoard and Ngrok simultaneously using Python's multiprocessing:

   ```python
   import os
   import multiprocessing
 
   pool = multiprocessing.Pool(processes = 10)
   results_of_processes = [pool.apply_async(os.system, args=(cmd, ), callback = None )
                        for cmd in [
                        f"tensorboard --logdir /kaggle/working/logs --load_fast=false --host 0.0.0.0 --port 6006 &",
                        "/usr/local/bin/ngrok http 6006 &"
                        ]]
4. Retrieves the public Ngrok URL from a locally running Ngrok instance:
   ```bash
   ! curl -s http://localhost:4040/api/tunnels | python3 -c \
    "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"

## Additional Notes

- **Checkpointing**: You can specify a `checkpoint_path` to resume training from a previously saved model.
- **Callbacks**: The training script supports several callbacks including CSV logging, learning rate scheduling, model checkpointing, and TensorBoard integration.
- **Dataset Paths**: Ensure the dataset paths in the `data_ids` section are correct for your environment. For example, when running on Kaggle, use the appropriate dataset input paths.