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
     "enable_SAP": true # set enable_SAP = false if using other generative models
   }
   ```

2. **Customizing the Configuration**:
   - **Dataset**: Specify the path to the dataset. You may need to adjust the paths based on your environment.
   - **Data IDs**: List the corresponding dataset identifiers and paths.
   - **Loss Functions**: Choose from the implemented loss functions. You can also add new ones in the `utils/losses.py` file.
   - **Model Architecture**: Select the model architecture to train (currently supports `wat_stacked_unets` and `stacked_unets`).
   - **Epochs and Batch Size**: Adjust the number of epochs and batch size for training according to your hardware and task.

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
   - Edit `configs/hyperparameters.json` to include your new network under the `model_architecture` key.

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
   - Edit `configs/hyperparameters.json` and add your new loss to the `loss_functions` list:

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
   python src/training/trainer.py --config configs/hyperparameters.json
   ```

2. The script will automatically load the dataset, model, and loss functions from the configuration file and begin training.

---

## Additional Notes

- **Checkpointing**: You can specify a `checkpoint_path` to resume training from a previously saved model.
- **Callbacks**: The training script supports several callbacks including CSV logging, learning rate scheduling, and model checkpointing.
- **Dataset Paths**: Ensure the dataset paths in the `data_ids` section are correct for your environment. For example, when running on Kaggle, use the appropriate dataset input paths.

---

## License

This code is released under the [MIT License](LICENSE).