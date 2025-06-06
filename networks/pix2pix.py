import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import numpy as np
import os
import pathlib
import time
import datetime
from matplotlib import pyplot as plt
from IPython import display
import nibabel as nib
from skimage import exposure
import random
class Pix2Pix:
    def __init__(self):
        self.LAMBDA = 100
        self.generator = self.Generator()
        self.discriminator = self.Discriminator()
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def downsample(self, filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                   kernel_initializer=initializer, use_bias=False))
        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())
        result.add(tf.keras.layers.LeakyReLU())
        return result

    def upsample(self, filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))
        result.add(tf.keras.layers.BatchNormalization())
        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))
        result.add(tf.keras.layers.ReLU())
        return result

    def Generator(self):
        inputs = tf.keras.layers.Input(shape=[256, 256, 1])

        down_stack = [
            self.downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
            self.downsample(128, 4),  # (batch_size, 64, 64, 128)
            self.downsample(256, 4),  # (batch_size, 32, 32, 256)
            self.downsample(512, 4),  # (batch_size, 16, 16, 512)
            self.downsample(512, 4),  # (batch_size, 8, 8, 512)
            self.downsample(512, 4),  # (batch_size, 4, 4, 512)
            self.downsample(512, 4),  # (batch_size, 2, 2, 512)
            self.downsample(512, 4),  # (batch_size, 1, 1, 512)
        ]

        up_stack = [
            self.upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
            self.upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
            self.upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
            self.upsample(512, 4),  # (batch_size, 16, 16, 1024)
            self.upsample(256, 4),  # (batch_size, 32, 32, 512)
            self.upsample(128, 4),  # (batch_size, 64, 64, 256)
            self.upsample(64, 4),  # (batch_size, 128, 128, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(1, 4,
                                               strides=2,
                                               padding='same',
                                               kernel_initializer=initializer,
                                               activation='tanh')  # (batch_size, 256, 256, 3)

        x = inputs

        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = skips[:-1]
        skips.reverse()
        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate(axis=-1)([x, skip])

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)

    def generator_loss(self, disc_generated_output, gen_output, target):
        gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        total_gen_loss = gan_loss + (self.LAMBDA * l1_loss)
        return total_gen_loss, gan_loss, l1_loss

    def Discriminator(self):
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[256, 256, 1], name='input_image')
        tar = tf.keras.layers.Input(shape=[256, 256, 1], name='target_image')

        x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

        down1 = self.downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
        down2 = self.downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
        down3 = self.downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                      kernel_initializer=initializer,
                                      use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

        last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                      kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

        return tf.keras.Model(inputs=[inp, tar], outputs=last)

    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
        total_disc_loss = real_loss + generated_loss
        return total_disc_loss

    @tf.function
    def train_step(self, input_image, target):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator(input_image, training=True)

            disc_real_output = self.discriminator([input_image, target], training=True)
            disc_generated_output = self.discriminator([input_image, gen_output], training=True)

            gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(disc_generated_output, gen_output, target)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))

    def generate_images(self, model, test_input, tar):
        prediction = model(test_input, training=True)
        plt.figure(figsize=(15, 15))

        display_list = [test_input[0], tar[0], prediction[0]]
        title = ['Input Image', 'Ground Truth', 'Predicted Image']

        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.title(title[i])
            plt.imshow(display_list[i] * 0.5 + 0.5, cmap='gray')
            plt.axis('off')
        plt.show()

# Example usage:
# data_loader = DataLoader(data_path, split_ratio, batch_size)
# pix2pix = Pix2Pix()
# train_dataset = data_loader.generator('train')
# for motion, free in train_dataset:
#     pix2pix.train_step(motion, free)