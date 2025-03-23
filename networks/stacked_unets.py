import keras
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import numpy as np
import pywt

class StackedUNets:
    def __init__(self):
        self.IMAGE_ORDERING_CHANNELS_LAST = "channels_last"
        self.IMAGE_ORDERING_CHANNELS_FIRST = "channels_first"

        # Default IMAGE_ORDERING = channels_last
        self.IMAGE_ORDERING = self.IMAGE_ORDERING_CHANNELS_LAST

        if self.IMAGE_ORDERING == 'channels_first':
            self.MERGE_AXIS = 1
        elif self.IMAGE_ORDERING == 'channels_last':
            self.MERGE_AXIS = -1

    # # Define the Haar filters
    # def haar_filter(self):
    #     low_pass = tf.constant([1, 1], dtype=tf.float32) / tf.math.sqrt(2.0)
    #     high_pass = tf.constant([1, -1], dtype=tf.float32) / tf.math.sqrt(2.0)
    #     return low_pass, high_pass

    # # Function to apply 1D filter across the width dimension
    # def apply_filter_1d(self, inputs, filter_kernel, stride):
    #     """
    #     Applies a 1D filter across the width dimension.
    #     """
    #     in_channels = inputs.shape[-1]
    #     out_channels = in_channels  # Ensure output channels match input channels
        
    #     # Adjust the reshape operation to match filter_kernel's shape
    #     filter_kernel = tf.reshape(filter_kernel, [2, 1, 1, 1])  # [filter_height, filter_width, in_channels, out_channels]
    #     filter_kernel = tf.tile(filter_kernel, [1, 1, in_channels, out_channels])  # Tile to match input channels
        
    #     return tf.nn.conv2d(inputs, filter_kernel, strides=[1, 1, stride, 1], padding='SAME', data_format='NHWC')

    # # Function to perform 1D wavelet transform
    # def wavelet_transform_1d(self, inputs):
    #     low_pass, high_pass = self.haar_filter()
        
    #     # Apply filters manually
    #     low = self.apply_filter_1d(inputs, low_pass, stride=2)
    #     high = self.apply_filter_1d(inputs, high_pass, stride=2)
        
    #     return low, high

    # # Function to perform 2D wavelet transform
    # def wavelet_transform_2d(self, inputs):
    #     low, high = self.wavelet_transform_1d(inputs)
    #     low_low, low_high = self.wavelet_transform_1d(tf.transpose(low, [0, 2, 1, 3]))
    #     high_low, high_high = self.wavelet_transform_1d(tf.transpose(high, [0, 2, 1, 3]))
    #     return tf.transpose(low_low, [0, 2, 1, 3]), tf.transpose(low_high, [0, 2, 1, 3]), tf.transpose(high_low, [0, 2, 1, 3]), tf.transpose(high_high, [0, 2, 1, 3])

    # # Function to perform multi-level wavelet transform
    # def multi_level_wavelet_transform(self, inputs, levels=3):
    #     wavelet_coeffs_1 = []
    #     wavelet_coeffs_2 = []
    #     wavelet_coeffs_3 = []
        
    #     for level in range(levels):
    #         low_low, low_high, high_low, high_high = self.wavelet_transform_2d(inputs)
    #         if level == 0:
    #             wavelet_coeffs_1.append([low_high, high_low, high_high])
    #         elif level == 1:
    #             wavelet_coeffs_2.append([low_high, high_low, high_high])
    #         elif level == 2:
    #             wavelet_coeffs_3.append([low_low, low_high, high_low, high_high])

    #         inputs = low_low  # Proceed to the next level with the low-frequency component
        
    #     wat1 = tf.convert_to_tensor(wavelet_coeffs_1)
    #     wat2 = tf.convert_to_tensor(wavelet_coeffs_2)
    #     wat3 = tf.convert_to_tensor(wavelet_coeffs_3)
        
    #     wat1 = tf.squeeze(wat1, axis=0)
    #     wat1 = tf.transpose(wat1, perm=[1, 2, 3, 4, 0])

    #     wat2 = tf.squeeze(wat2, axis=0)
    #     wat2 = tf.transpose(wat2, perm=[1, 2, 3, 4, 0])

    #     wat3 = tf.squeeze(wat3, axis=0)
    #     wat3 = tf.transpose(wat3, perm=[1, 2, 3, 4, 0])
        
    #     # Get the shape of the input tensor
    #     wat1_shape = tf.shape(wat1)
    #     # Calculate the new shape for reshaping
    #     wat1_shape = tf.concat([wat1_shape[:-2], [wat1_shape[-2] * wat1_shape[-1]]], axis=0)
    #     # Reshape the tensor
    #     wat1 = tf.reshape(wat1, wat1_shape)
        
    #     # Get the shape of the input tensor
    #     wat2_shape = tf.shape(wat2)
    #     # Calculate the new shape for reshaping
    #     wat2_shape = tf.concat([wat2_shape[:-2], [wat2_shape[-2] * wat2_shape[-1]]], axis=0)
    #     # Reshape the tensor
    #     wat2 = tf.reshape(wat2, wat2_shape)
        
        
    #     # Get the shape of the input tensor
    #     wat3_shape = tf.shape(wat3)
    #     # Calculate the new shape for reshaping
    #     wat3_shape = tf.concat([wat3_shape[:-2], [wat3_shape[-2] * wat3_shape[-1]]], axis=0)
    #     # Reshape the tensor
    #     wat3 = tf.reshape(wat3, wat3_shape)
        
    #     return wat1, wat2, wat3

        
    # def wat_3(self, inputs):
    #     wat1, wat2, wat3 = self.multi_level_wavelet_transform(inputs) 
        
    #     # Explicitly set the shapes
    #     batch_size = inputs.shape[0]
    #     height, width = inputs.shape[1], inputs.shape[2]
    #     channels = inputs.shape[3]

    #     wat1.set_shape((batch_size, height // 2, width // 2, channels * 3))
    #     wat2.set_shape((batch_size, height // 4, width // 4, channels * 3))
    #     wat3.set_shape((batch_size, height // 8, width // 8, channels * 4))
            
    #     return wat1, wat2, wat3


    # def wat_layer_1(self, x, wat):
    #     watp_prod = Conv2D(16*(2**1), (3, 3), data_format=self.IMAGE_ORDERING, padding='same')(wat)
    #     watp_prod = Activation('relu')(watp_prod)
    #     watp_prod = Conv2D(32*(2**1), (3, 3), data_format=self.IMAGE_ORDERING, padding='same')(watp_prod)
        
    #     watp_sum = Conv2D(16*(2**1), (3, 3), data_format=self.IMAGE_ORDERING, padding='same')(wat)
    #     watp_sum = Activation('relu')(watp_sum)
    #     watp_sum = Conv2D(32*(2**1), (3, 3), data_format=self.IMAGE_ORDERING, padding='same')(watp_sum)
        
    #     x = multiply([x, watp_prod])
    #     x = Add()([x, watp_sum])    
    #     x = Activation('relu')(x)
    #     return x
        

    # def wat_layer_2(self, x, wat):
    #     watp_prod = Conv2D(16*(2**2), (3, 3), data_format=self.IMAGE_ORDERING, padding='same')(wat)
    #     watp_prod = Activation('relu')(watp_prod)
    #     watp_prod = Conv2D(32*(2**2), (3, 3), data_format=self.IMAGE_ORDERING, padding='same')(watp_prod)
        
    #     watp_sum = Conv2D(16*(2**2), (3, 3), data_format=self.IMAGE_ORDERING, padding='same')(wat)
    #     watp_sum = Activation('relu')(watp_sum)
    #     watp_sum = Conv2D(32*(2**2), (3, 3), data_format=self.IMAGE_ORDERING, padding='same')(watp_sum)
        
    #     x = multiply([x, watp_prod])
    #     x = Add()([x, watp_sum])    
    #     x = Activation('relu')(x)
    #     return x


    # def wat_layer_3(self, x, wat):
    #     watp_prod = Conv2D(16*(2**3), (3, 3), data_format=self.IMAGE_ORDERING, padding='same')(wat)
    #     watp_prod = Activation('relu')(watp_prod)
    #     watp_prod = Conv2D(32*(2**3), (3, 3), data_format=self.IMAGE_ORDERING, padding='same')(watp_prod)
        
    #     watp_sum = Conv2D(16*(2**3), (3, 3), data_format=self.IMAGE_ORDERING, padding='same')(wat)
    #     watp_sum = Activation('relu')(watp_sum)
    #     watp_sum = Conv2D(32*(2**3), (3, 3), data_format=self.IMAGE_ORDERING, padding='same')(watp_sum)
        
    #     x = multiply([x, watp_prod])
    #     x = Add()([x, watp_sum])    
    #     x = Activation('relu')(x)
    #     return x


    # CBAM --------------------------------------------
    # Convolutional Block Attention Module(CBAM) block
    def cbam_block(self, cbam_feature, ratio=8):
        cbam_feature = self.channel_attention(cbam_feature, ratio)
        cbam_feature = self.spatial_attention(cbam_feature)
        return cbam_feature

    def channel_attention(self, input_feature, ratio=8):
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        channel = input_feature.shape[channel_axis]  # input_feature._keras_shape[channel_axis]

        shared_layer_one = Dense(channel//ratio,
                                activation='relu',
                                kernel_initializer='he_normal',
                                use_bias=True,
                                bias_initializer='zeros')
        shared_layer_two = Dense(channel,
                                kernel_initializer='he_normal',
                                use_bias=True,
                                bias_initializer='zeros')

        avg_pool = GlobalAveragePooling2D()(input_feature)
        avg_pool = Reshape((1,1,channel))(avg_pool)
        assert avg_pool.shape[1:] == (1,1,channel)
        avg_pool = shared_layer_one(avg_pool)
        assert avg_pool.shape[1:] == (1,1,channel//ratio)
        avg_pool = shared_layer_two(avg_pool)
        assert avg_pool.shape[1:] == (1,1,channel)

        max_pool = GlobalMaxPooling2D()(input_feature)
        max_pool = Reshape((1,1,channel))(max_pool)
        assert max_pool.shape[1:] == (1,1,channel)
        max_pool = shared_layer_one(max_pool)
        assert max_pool.shape[1:] == (1,1,channel//ratio)
        max_pool = shared_layer_two(max_pool)
        assert max_pool.shape[1:] == (1,1,channel)

        cbam_feature = Add()([avg_pool,max_pool])
        cbam_feature = Activation('sigmoid')(cbam_feature)

        if K.image_data_format() == "channels_first":
            cbam_feature = Permute((3, 1, 2))(cbam_feature)

        return multiply([input_feature, cbam_feature])

    def spatial_attention(self, input_feature):
        kernel_size = 7

        if K.image_data_format() == "channels_first":
            channel = input_feature._keras_shape[1]
            cbam_feature = Permute((2,3,1))(input_feature)
        else:
            channel = input_feature.shape[-1]
            cbam_feature = input_feature

        avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
        assert avg_pool.shape[-1] == 1
        max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
        assert max_pool.shape[-1] == 1
        concat = Concatenate(axis=3)([avg_pool, max_pool])
        assert concat.shape[-1] == 2
        cbam_feature = Conv2D(filters = 1,
                        kernel_size=kernel_size,
                        strides=1,
                        padding='same',
                        activation='sigmoid',
                        kernel_initializer='he_normal',
                        use_bias=False)(concat)
        assert cbam_feature.shape[-1] == 1

        if K.image_data_format() == "channels_first":
            cbam_feature = Permute((3, 1, 2))(cbam_feature)

        return multiply([input_feature, cbam_feature])
        
    def UNet(self, img_input):
        k1 = 32
        k2 = 64
        k3 = 128
        k4 = 256
        
        # watp1, watp2, watp3 = self.wat_3(img_input)
        
        # Block 1 in Contracting Path
        conv1 = Conv2D(k1, (3, 3), data_format=self.IMAGE_ORDERING,padding='same', dilation_rate=1)(img_input)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation(tf.nn.leaky_relu)(conv1)
        #conv1 = Dropout(0.2)(conv1)
        conv1 = Conv2D(k1, (3, 3), data_format=self.IMAGE_ORDERING, padding='same', dilation_rate=1)(conv1)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation(tf.nn.leaky_relu)(conv1)

        conv1 = self.cbam_block(conv1)    # Convolutional Block Attention Module(CBAM) block

        o = AveragePooling2D((2, 2), strides=(2, 2))(conv1)

        # Block 2 in Contracting Path
        conv2 = Conv2D(k2, (3, 3), data_format=self.IMAGE_ORDERING, padding='same', dilation_rate=1)(o)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation(tf.nn.leaky_relu)(conv2)
    #     conv2 = Dropout(0.2)(conv2)
        conv2 = Conv2D(k2, (3, 3), data_format=self.IMAGE_ORDERING, padding='same', dilation_rate=1)(conv2)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation(tf.nn.leaky_relu)(conv2)
        
        # conv2 = self.wat_layer_1(conv2, watp1)

        conv2 = self.cbam_block(conv2)    # Convolutional Block Attention Module(CBAM) block

        o = AveragePooling2D((2, 2), strides=(2, 2))(conv2)

        # Block 3 in Contracting Path
        conv3 = Conv2D(k3, (3, 3), data_format=self.IMAGE_ORDERING, padding='same', dilation_rate=1)(o)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation(tf.nn.leaky_relu)(conv3)
        #conv3 = Dropout(0.2)(conv3)
        conv3 = Conv2D(k3, (3, 3), data_format=self.IMAGE_ORDERING, padding='same', dilation_rate=1)(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation(tf.nn.leaky_relu)(conv3)

        # conv3 = self.wat_layer_2(conv3, watp2)

        conv3 = self.cbam_block(conv3)    # Convolutional Block Attention Module(CBAM) block

        o = AveragePooling2D((2, 2), strides=(2, 2))(conv3)

        # Transition layer between contracting and expansive paths:
        conv4 = Conv2D(k4, (3, 3), data_format=self.IMAGE_ORDERING, padding='same', dilation_rate=1)(o)
        conv4 = BatchNormalization()(conv4)
        conv4 = Activation(tf.nn.leaky_relu)(conv4)
        #conv4 = Dropout(0.2)(conv4)
        conv4 = Conv2D(k4, (3, 3), data_format=self.IMAGE_ORDERING, padding='same', dilation_rate=1)(conv4)
        conv4 = BatchNormalization()(conv4)
        conv4 =Activation(tf.nn.leaky_relu)(conv4)

        # conv4 = self.wat_layer_3(conv4, watp3)

        conv4 = self.cbam_block(conv4)    # Convolutional Block Attention Module(CBAM) block

        # Block 1 in Expansive Path
        up1 = UpSampling2D((2, 2), data_format=self.IMAGE_ORDERING)(conv4)
        up1 = concatenate([up1, conv3], axis=self.MERGE_AXIS)
        deconv1 =  Conv2D(k3, (3, 3), data_format=self.IMAGE_ORDERING, padding='same', dilation_rate=1)(up1)
        deconv1 = BatchNormalization()(deconv1)
        deconv1 = Activation(tf.nn.leaky_relu)(deconv1)
        #deconv1 = Dropout(0.2)(deconv1)
        deconv1 =  Conv2D(k3, (3, 3), data_format=self.IMAGE_ORDERING, padding='same', dilation_rate=1)(deconv1)
        deconv1 = BatchNormalization()(deconv1)
        deconv1 = Activation(tf.nn.leaky_relu)(deconv1)

        deconv1 = self.cbam_block(deconv1)    # Convolutional Block Attention Module(CBAM) block

        # Block 2 in Expansive Path
        up2 = UpSampling2D((2, 2), data_format=self.IMAGE_ORDERING)(deconv1)
        up2 = concatenate([up2, conv2], axis=self.MERGE_AXIS)
        deconv2 = Conv2D(k2, (3, 3), data_format=self.IMAGE_ORDERING, padding='same', dilation_rate=1)(up2)
        deconv2 = BatchNormalization()(deconv2)
        deconv2 = Activation(tf.nn.leaky_relu)(deconv2)
        #deconv2 = Dropout(0.2)(deconv2)
        deconv2 = Conv2D(k2, (3, 3), data_format=self.IMAGE_ORDERING, padding='same', dilation_rate=1)(deconv2)
        deconv2 = BatchNormalization()(deconv2)
        deconv2 = Activation(tf.nn.leaky_relu)(deconv2)

        deconv2 = self.cbam_block(deconv2)    # Convolutional Block Attention Module(CBAM) block

        # Block 3 in Expansive Path
        up3 = UpSampling2D((2, 2), data_format=self.IMAGE_ORDERING)(deconv2)
        up3 = concatenate([up3, conv1], axis=self.MERGE_AXIS)
        deconv3 = Conv2D(k1, (3, 3), data_format=self.IMAGE_ORDERING, padding='same', dilation_rate=1)(up3)
        deconv3 = BatchNormalization()(deconv3)
        deconv3 = Activation(tf.nn.leaky_relu)(deconv3)
        #deconv3 = Dropout(0.2)(deconv3)
        deconv3 = Conv2D(k1, (3, 3), data_format=self.IMAGE_ORDERING, padding='same', dilation_rate=1)(deconv3)
        deconv3 = BatchNormalization()(deconv3)
        deconv3 = Activation(tf.nn.leaky_relu)(deconv3)

        deconv3 = self.cbam_block(deconv3)    # Convolutional Block Attention Module(CBAM) block

        output = Conv2D(1, (3, 3), data_format=self.IMAGE_ORDERING, padding='same')(deconv3)
        # 	output = Activation('sigmoid')(output)
        output = Activation('sigmoid')(output)
        return output


    def Correction_Multi_input(self, input_height, input_width):
        assert input_height % 32 == 0
        assert input_width % 32 == 0

    #   UNET
        img_input_1 = Input(shape=(input_height, input_width, 1))
        img_input_2 = Input(shape=(input_height, input_width, 1))
        img_input_3 = Input(shape=(input_height, input_width, 1))
    # 	kk = 32
        kk = 64
        conv1 = Conv2D(kk, (3, 3), data_format=self.IMAGE_ORDERING,padding='same', dilation_rate=1)(img_input_1) # dilation_rate=6
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv2 = Conv2D(kk, (3, 3), data_format=self.IMAGE_ORDERING,padding='same', dilation_rate=1)(img_input_2) # dilation_rate=6
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        conv3 = Conv2D(kk, (3, 3), data_format=self.IMAGE_ORDERING,padding='same', dilation_rate=1)(img_input_3) # dilation_rate=6
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation('relu')(conv3)

        input_concat = concatenate([conv1, conv2, conv3], axis=self.MERGE_AXIS)  #conv4

        ## Two Stacked Nets:
        pred_1  = self.UNet(input_concat)
        input_2 = concatenate([input_concat, pred_1], axis=self.MERGE_AXIS)
        pred_2  = self.UNet(input_2) #

        model = Model(inputs=[img_input_1,img_input_2,img_input_3], outputs=pred_2)

        return model