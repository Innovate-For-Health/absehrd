import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class medgan(object):
    def __init__(self,
                 data_type='binary',
                 input_dim=615,
                 embedding_dim=128,
                 random_dim=128,
                 generator_dims=(128, 128),
                 discriminator_dims=(256, 128, 1),
                 compress_dims=(),
                 decompress_dims=(),
                 bn_decay=0.99,
                 l2scale=0.001):
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.generator_dims = list(generator_dims) + [embedding_dim]
        self.random_dim = random_dim
        self.data_type = data_type

        if data_type == 'binary':
            self.ae_activation = tf.nn.tanh
        else:
            self.ae_activation = tf.nn.relu

        self.generator_activation = tf.nn.relu
        self.discriminator_activation = tf.nn.relu
        self.discriminator_dims = discriminator_dims
        self.compress_dims = list(compress_dims) + [embedding_dim]
        self.decompress_dims = list(decompress_dims) + [input_dim]
        self.bn_decay = bn_decay
        self.l2scale = l2scale
        
    def build_autoencoder(input_dim,latent_dim):
        inputs = keras.Input(shape=input_dim)
        x = layers.Dense(16, activation="relu")(inputs)
        encoder = keras.Model(encoder_inputs, x, name="encoder")
        
      
        