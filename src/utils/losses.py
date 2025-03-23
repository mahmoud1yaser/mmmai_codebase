import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

class Losses():
    def __init__(self):
        self.perceptual_models = self.init_vgg16_model()

    def init_vgg16_model(self):
        """
        Initialize a pre-trained VGG16 model for feature extraction.

        Args:
            perceptual_layer_name: Name of the layer to extract features from.

        Returns:
            Pre-trained VGG16 model with specified layer for feature extraction.

        This function loads a pre-trained VGG16 model with ImageNet weights and removes the top
        classification layers. It then extracts the specified layer for feature extraction and
        freezes the model's layers to prevent further training.

        """
        # Load pre-trained VGG16 model without the top classification layers
        vgg_model = VGG16(include_top=False, weights='imagenet', input_shape=(256, 256, 3))

        # Extract the specified layer from the VGG16 model
        perceptual_model_conv1 = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('block1_conv1').output)
        perceptual_model_conv2 = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('block2_conv1').output)
        perceptual_model_conv3 = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('block3_conv1').output)

        # Freeze the layers in the perceptual model so they are not trained further
        for perceptual_model in [perceptual_model_conv1,perceptual_model_conv2,perceptual_model_conv3]:
            for layer in perceptual_model.layers:
                layer.trainable = False
            
        print("VGG16 Model Initialized")
        return perceptual_model_conv1, perceptual_model_conv2, perceptual_model_conv3

    # Initialize VGG16 model for feature extraction


    def perceptual_loss(self,y_true,y_pred):
        """
        Custom loss function for perceptual loss.

        Args:
            y_true: Ground truth image(s).
            y_pred: Predicted image(s).

        Returns:
            Normalized reduced perceptual loss.

        This function defines a custom loss for training neural networks. It converts single-channel
        images to RGB, preprocesses them for VGG16, and extracts features from a specified layer
        using a pre-trained VGG16 model. It then calculates the mean squared error between the features
        of the true and predicted images. The loss is normalized and reduced to a single scalar value.

        """
        # Extract perceptual models
        perceptual_model_conv1, perceptual_model_conv2, perceptual_model_conv3 = self.perceptual_models

        # Convert single-channel images to RGB
        y_true_rgb = tf.repeat(y_true, 3, axis=-1)
        y_pred_rgb = tf.repeat(y_pred, 3, axis=-1)

        # Preprocess images for VGG16
        y_true_processed = tf.keras.applications.vgg16.preprocess_input(y_true_rgb)
        y_pred_processed = tf.keras.applications.vgg16.preprocess_input(y_pred_rgb)

        # Extract features from specified layer for true and predicted images
        features_true_conv1 = perceptual_model_conv1(y_true_processed)
        features_pred_conv1 = perceptual_model_conv1(y_pred_processed)
    
        # Extract features from specified layer for true and predicted images
        features_true_conv2 = perceptual_model_conv2(y_true_processed)
        features_pred_conv2 = perceptual_model_conv2(y_pred_processed)
    
        # Extract features from specified layer for true and predicted images
        features_true_conv3 = perceptual_model_conv3(y_true_processed)
        features_pred_conv3 = perceptual_model_conv3(y_pred_processed)
        
        # Calculate L2 loss
        mse_conv1 = tf.reduce_mean(tf.keras.losses.mse(features_true_conv1, features_pred_conv1))
        mse_conv2 = tf.reduce_mean(tf.keras.losses.mse(features_true_conv2, features_pred_conv2))
        mse_conv3 = tf.reduce_mean(tf.keras.losses.mse(features_true_conv3, features_pred_conv3))
        
        total_loss = 0.65*mse_conv1 + 0.3*mse_conv2 + 0.05*mse_conv3

        return total_loss

    def ssim_loss(self,y_true,y_pred):
        score = tf.image.ssim(
        y_true,
        y_pred,
        max_val=1.0
        )
        
        loss = (1-score)/2
        return loss
    
    def mae_loss(self, y_true, y_pred):
        """MAE Loss: Computes the mean absolute error between the true and predicted images."""
        loss = tf.reduce_mean(tf.abs(y_true - y_pred))
        return loss

    def psnr(self,y_true,y_pred):
        return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))

    def ssim_score(self,y_true,y_pred):
        score = tf.image.ssim(
        y_true,
        y_pred,
        max_val=1.0
        )
        return score