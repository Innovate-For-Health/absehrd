"""
Description: generic generative adversarial network template.
Source: 
Modified by: Haley Hunter-Zinck
Date: July 3, 2020
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import initializers
from tensorflow.keras import regularizers

# regularizers.l2(l=self.l2scale)

class medgan:
    
    def __init__(self,
                 data_type='binary',
                 input_dim=615,
                 embedding_dim=128,
                 random_dim=128,
                 generator_dims=(128, 128),
                 discriminator_dims=(256, 128),
                 compress_dims=(),
                 decompress_dims=(),
                 bn_decay=0.99,
                 l2scale=0.001,
                 dropout_rate=0.2):
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
        self.dropout_rate = dropout_rate
        self.init_fxn = initializers.GlorotUniform(seed=None)
        
        self.opt = tf.keras.optimizers.Adam(1e-4)
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        
            
    def build_encoder(self):
        model = tf.keras.Sequential()
        for units in enumerate(self.compress_dims):
            model.add(layers.Dense(units=units, 
                                   activation=self.ae_activation, 
                                   kernel_initializer=self.init_fxn, 
                                   kernal_regularizer=regularizers.l2(l=self.l2scale)))
        return model
    
    def build_decoder(self):
        model = tf.keras.Sequential()
        for units in enumerate(self.decompress_dims):
            model.add(tf.keras.layers.Dense(units=units, 
                                            activation=self.ae_activation, 
                                            kernel_initializer=self.init_fxn,
                                            kernal_regularizer=regularizers.l2(l=self.l2scale)))
        
        if self.data_type == 'binary':
            model.add(layers.Dense(unit=self.embedding_dim, 
                                   activation='sigmoid',
                                   kernal_regularizer=regularizers.l2(l=self.l2scale)))
        else:
            model.add(layers.Dense(unit=self.embedding_dim, 
                                   activation='relu',
                                   kernal_regularizer=regularizers.l2(l=self.l2scale)))
        
        return model
    
    def build_autoencoder(self):
        model = tf.keras.Sequential()
        model.add(self.encoder)
        model.add(self.decoder)
        return model
            
    def build_generator(self):
        model = tf.keras.Sequential()
        for units in enumerate(self.discriminator_dims):
            model.add(layers.BatchNormalization(momentum=self.bn_decay, scale=True))
            model.add(tf.keras.layers.Dense(units=units, 
                                            activation=self.generator_activation, 
                                            kernel_initializer=self.init_fxn,
                                            kernal_regularizer=regularizers.l2(l=self.l2scale)))
        model.add(layers.BatchNormalization(momentum=self.bn_decay, scale=True))
        
        if self.data_type == 'binary':
            model.add(layers.Dense(unit=self.embedding_dim, 
                                   activation='tanh',
                                   kernal_regularizer=regularizers.l2(l=self.l2scale)))
        else:
            model.add(layers.Dense(unit=self.embedding_dim, 
                                   activation='relu',
                                   kernal_regularizer=regularizers.l2(l=self.l2scale)))
            
        return model
    
    def build_discriminator(self):
        model = tf.keras.Sequential()
        for units in enumerate(self.discriminator_dims):
            model.add(tf.keras.layers.Dense(units=units, 
                                            activation=self.discriminator_activation, 
                                            kernel_initializer=self.init_fxn,
                                            kernal_regularizer=regularizers.l2(l=self.l2scale)))
            model.add(tf.keras.layers.Dropout(rate=self.dropout_rate))
        
        if self.data_type == 'binary':
            model.add(layers.Dense(units=1, 
                                   activation='sigmoid',
                                   kernal_regularizer=regularizers.l2(l=self.l2scale)))
        else:
            model.add(layers.Dense(units=1, 
                                   activation='relu',
                                   kernal_regularizer=regularizers.l2(l=self.l2scale)))
        
        return model
    
    def loss_autoencoder(x,z):
        reconstruction_error = tf.reduce_mean(tf.square(tf.subtract(z, x)))
        return reconstruction_error
       
    def loss_discriminator(real_output, synth_output):
        real_loss = losses.BinaryCrossentropy(tf.ones_like(real_output), real_output)
        fake_loss = losses.BinaryCrossentropy(tf.zeros_like(synth_output), synth_output)
        total_loss = real_loss + fake_loss
        return total_loss
    
    def loss_discriminator(synth_output):
        return losses.BinaryCrossentropy(tf.ones_like(synth_output), synth_output)
    
    def train_ae(self, x):
        model = self.build_autoencoder()
        with tf.GradientTape() as tape:
            gradients = tape.gradient(self.loss_autoencoder(model, x), model.trainable_variables)
            gradient_variables = zip(gradients, model.trainable_variables)
            self.opt.apply_gradients(gradient_variables)
        return model
    
    def train_step(self, x, batch_size):
        noise = tf.random.normal([batch_size, self.embedding_dim])
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            synth_x_cont = self.generator(noise, training=True)
            synth_x_disc = self.decoder(synth_x_cont)
            
            real_output = self.discriminator(x, training = True)
            synth_output = self.discriminator(synth_x_disc, training=True)
            
            gen_loss = self.loss_generator(synth_output)
            disc_loss = self.loss_discriminator(real_output, synth_output)
            
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        
        self.opt.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.opt.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        
    def train(self, x, n_epoch, batch_size):
        
        self.train_ae(x=x)
        
        for i in range(n_epoch):
            self.train_step(x=x, batch_size=batch_size)
            
    def generate(self, n_sample):
        noise = tf.random.normal([n_sample, self.embedding_dim])
        synth_x = self.generator(noise, training=False)
        return(synth_x)
            
            
        
        