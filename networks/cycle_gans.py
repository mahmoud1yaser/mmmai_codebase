import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU, ReLU, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import random

class CycleGAN:
    def __init__(self, img_width=256, img_height=256, output_channels=1, lambda_cycle=10):
        self.img_width = img_width
        self.img_height = img_height
        self.output_channels = output_channels
        self.lambda_cycle = lambda_cycle

        # Initialize generators and discriminators
        self.generator_motion = self.unet_generator(output_channels, norm_type='instancenorm')
        self.generator_free = self.unet_generator(output_channels, norm_type='instancenorm')
        self.discriminator_motion = self.discriminator(norm_type='instancenorm', target=False)
        self.discriminator_free = self.discriminator(norm_type='instancenorm', target=False)

        # Loss functions
        self.loss_obj = BinaryCrossentropy(from_logits=True)

        # Optimizers
        self.generator_motion_optimizer = Adam(2e-4, beta_1=0.5)
        self.generator_free_optimizer = Adam(2e-4, beta_1=0.5)
        self.discriminator_motion_optimizer = Adam(2e-4, beta_1=0.5)
        self.discriminator_free_optimizer = Adam(2e-4, beta_1=0.5)

    def unet_generator(self, output_channels, norm_type='batchnorm'):
        down_stack = [
            self.downsample(64, 4, norm_type, apply_norm=False),
            self.downsample(128, 4, norm_type),
            self.downsample(256, 4, norm_type),
            self.downsample(512, 4, norm_type),
            self.downsample(512, 4, norm_type),
            self.downsample(512, 4, norm_type),
            self.downsample(512, 4, norm_type),
            self.downsample(512, 4, norm_type),
        ]

        up_stack = [
            self.upsample(512, 4, norm_type, apply_dropout=True),
            self.upsample(512, 4, norm_type, apply_dropout=True),
            self.upsample(512, 4, norm_type, apply_dropout=True),
            self.upsample(512, 4, norm_type),
            self.upsample(256, 4, norm_type),
            self.upsample(128, 4, norm_type),
            self.upsample(64, 4, norm_type),
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = Conv2DTranspose(
            output_channels, 4, strides=2,
            padding='same', kernel_initializer=initializer,
            activation='tanh')

        concat = Concatenate()

        inputs = Input(shape=[None, None, 1])
        x = inputs

        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = concat([x, skip])

        x = last(x)

        return Model(inputs=inputs, outputs=x)

    def discriminator(self, norm_type='batchnorm', target=True):
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = Input(shape=[None, None, 1], name='input_image')
        x = inp

        if target:
            tar = Input(shape=[None, None, 1], name='target_image')
            x = Concatenate()([inp, tar])

        down1 = self.downsample(64, 4, norm_type, False)(x)
        down2 = self.downsample(128, 4, norm_type)(down1)
        down3 = self.downsample(256, 4, norm_type)(down2)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)
        conv = Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(zero_pad1)

        if norm_type.lower() == 'batchnorm':
            norm1 = BatchNormalization()(conv)
        elif norm_type.lower() == 'instancenorm':
            norm1 = InstanceNormalization()(conv)

        leaky_relu = LeakyReLU()(norm1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)

        last = Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)

        if target:
            return Model(inputs=[inp, tar], outputs=last)
        else:
            return Model(inputs=inp, outputs=last)

    def downsample(self, filters, size, norm_type='batchnorm', apply_norm=True):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))

        if apply_norm:
            if norm_type.lower() == 'batchnorm':
                result.add(BatchNormalization())
            elif norm_type.lower() == 'instancenorm':
                result.add(InstanceNormalization())

        result.add(LeakyReLU())

        return result

    def upsample(self, filters, size, norm_type='batchnorm', apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))

        if norm_type.lower() == 'batchnorm':
            result.add(BatchNormalization())
        elif norm_type.lower() == 'instancenorm':
            result.add(InstanceNormalization())

        if apply_dropout:
            result.add(Dropout(0.5))

        result.add(ReLU())

        return result

    def discriminator_loss(self, real, generated):
        real_loss = self.loss_obj(tf.ones_like(real), real)
        generated_loss = self.loss_obj(tf.zeros_like(generated), generated)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss * 0.5

    def generator_loss(self, generated):
        return self.loss_obj(tf.ones_like(generated), generated)

    def calc_cycle_loss(self, real_image, cycled_image):
        loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
        return self.lambda_cycle * loss1

    def identity_loss(self, real_image, same_image):
        loss = tf.reduce_mean(tf.abs(real_image - same_image))
        return self.lambda_cycle * 0.5 * loss

    @tf.function
    def train_step(self, real_x, real_y):
        with tf.GradientTape(persistent=True) as tape:
            fake_y = self.generator_free(real_x, training=True)
            cycled_x = self.generator_motion(fake_y, training=True)

            fake_x = self.generator_motion(real_y, training=True)
            cycled_y = self.generator_free(fake_x, training=True)

            same_x = self.generator_motion(real_x, training=True)
            same_y = self.generator_free(real_y, training=True)

            disc_real_x = self.discriminator_free(real_x, training=True)
            disc_real_y = self.discriminator_motion(real_y, training=True)

            disc_fake_x = self.discriminator_free(fake_x, training=True)
            disc_fake_y = self.discriminator_motion(fake_y, training=True)

            gen_g_loss = self.generator_loss(disc_fake_y)
            gen_f_loss = self.generator_loss(disc_fake_x)

            total_cycle_loss = self.calc_cycle_loss(real_x, cycled_x) + self.calc_cycle_loss(real_y, cycled_y)

            total_gen_g_loss = gen_g_loss + total_cycle_loss + self.identity_loss(real_y, same_y)
            total_gen_f_loss = gen_f_loss + total_cycle_loss + self.identity_loss(real_x, same_x)

            disc_x_loss = self.discriminator_loss(disc_real_x, disc_fake_x)
            disc_y_loss = self.discriminator_loss(disc_real_y, disc_fake_y)

        generator_g_gradients = tape.gradient(total_gen_g_loss, self.generator_free.trainable_variables)
        generator_f_gradients = tape.gradient(total_gen_f_loss, self.generator_motion.trainable_variables)

        discriminator_x_gradients = tape.gradient(disc_x_loss, self.discriminator_free.trainable_variables)
        discriminator_y_gradients = tape.gradient(disc_y_loss, self.discriminator_motion.trainable_variables)

        self.generator_motion_optimizer.apply_gradients(zip(generator_g_gradients, self.generator_free.trainable_variables))
        self.generator_free_optimizer.apply_gradients(zip(generator_f_gradients, self.generator_motion.trainable_variables))

        self.discriminator_motion_optimizer.apply_gradients(zip(discriminator_x_gradients, self.discriminator_free.trainable_variables))
        self.discriminator_free_optimizer.apply_gradients(zip(discriminator_y_gradients, self.discriminator_motion.trainable_variables))

    def generate_images(self, model, test_input):
        prediction = model(test_input)

        plt.figure(figsize=(12, 12))

        display_list = [test_input[0], prediction[0]]
        title = ['Input Image', 'Predicted Image']

        for i in range(2):
            plt.subplot(1, 2, i+1)
            plt.title(title[i])
            plt.imshow(display_list[i] * 0.5 + 0.5, cmap='gray')
            plt.axis('off')
        plt.show()

    def train(self, train_dataset, validation_dataset, epochs, checkpoint_path):
        ckpt = tf.train.Checkpoint(
            generator_motion=self.generator_motion,
            generator_free=self.generator_free,
            discriminator_motion=self.discriminator_motion,
            discriminator_free=self.discriminator_free,
            generator_motion_optimizer=self.generator_motion_optimizer,
            generator_free_optimizer=self.generator_free_optimizer,
            discriminator_motion_optimizer=self.discriminator_motion_optimizer,
            discriminator_free_optimizer=self.discriminator_free_optimizer
        )

        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')

        with open('metrics.csv', mode='w', newline='') as metrics_file:
            metrics_writer = csv.writer(metrics_file)
            metrics_writer.writerow(['Epoch', 'mse', 'psnr', 'ssim_score', "val_mse", "val_psnr", "val_ssim_score"])
            metrics_file.flush()

            for epoch in range(epochs):
                ssim_values = []
                psnr_values = []
                mse_values = []
                val_ssim_values = []
                val_psnr_values = []
                val_mse_values = []
                n = 0
                flag = True

                for motion, free in train_dataset:
                    self.train_step(motion, free)
                    prediction = self.generator_free(motion, training=False)
                    mse_values.append(self.mse_score(free, prediction))
                    ssim_values.append(self.ssim_score(free, prediction))
                    psnr_values.append(self.psnr(free, prediction))
                    n += 1

                    if n % 100 == 0:
                        avg_ssim = tf.reduce_mean(ssim_values)
                        avg_psnr = tf.reduce_mean(psnr_values)
                        avg_mse = tf.reduce_mean(mse_values)
                        print(f"Step {n + 1}, SSIM: {avg_ssim}, PSNR: {avg_psnr}, MSE: {avg_mse}")

                avg_ssim = tf.reduce_mean(ssim_values)
                avg_psnr = tf.reduce_mean(psnr_values)
                avg_mse = tf.reduce_mean(mse_values)

                for motion_val, free_val in validation_dataset:
                    if flag:
                        r = random.randint(0, 10)
                        if r > 8:
                            selected_image = motion_val
                            flag = False

                    val_prediction = self.generator_free(motion_val, training=False)
                    val_mse_values.append(self.mse_score(free_val, val_prediction))
                    val_ssim_values.append(self.ssim_score(free_val, val_prediction))
                    val_psnr_values.append(self.psnr(free_val, val_prediction))

                avg_ssim_val = tf.reduce_mean(val_ssim_values)
                avg_psnr_val = tf.reduce_mean(val_psnr_values)
                avg_mse_val = tf.reduce_mean(val_mse_values)

                print(f"========================== {epoch + 1} Metrics ==========================")
                print(f"Epoch {epoch + 1}, SSIM: {avg_ssim}, PSNR: {avg_psnr}, MSE: {avg_mse}, SSIM_Val: {avg_ssim_val}, PSNR_Val: {avg_psnr_val}, MSE_Val: {avg_mse_val}")

                metrics_writer.writerow([epoch + 1, avg_mse.numpy(), avg_psnr.numpy(), avg_ssim.numpy(), avg_mse_val.numpy(), avg_psnr_val.numpy(), avg_ssim_val.numpy()])
                metrics_file.flush()

                self.generate_images(self.generator_free, selected_image)

                with tf.keras.utils.custom_object_scope({'InstanceNormalization': InstanceNormalization}):
                    self.generator_free.save(f"{checkpoint_path}/Cycle_GAN_GeneratorFree{epoch + 1}.h5")
                    self.generator_motion.save(f"{checkpoint_path}/Cycle_GAN_GeneratorMotion{epoch + 1}.h5")
                    self.discriminator_motion.save(f"{checkpoint_path}/Cycle_GAN_DiscriminatorMotion{epoch + 1}.h5")
                    self.discriminator_free.save(f"{checkpoint_path}/Cycle_GAN_DiscriminatorFree{epoch + 1}.h5")
class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5, trainable=True, **kwargs):
        super(InstanceNormalization, self).__init__(trainable=trainable, **kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True)

        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset