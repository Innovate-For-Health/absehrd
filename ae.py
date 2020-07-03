"""
Description: generic autoencoder template
Source: https://towardsdatascience.com/implementing-an-autoencoder-in-tensorflow-2-0-5e86126e9f7
Modified by: Haley Hunter-Zinck
Date: July 3, 2020
Note: 
    init_fxn = 'he_uniform'
    act_fxn = tf.nn.relu
"""

import tensorflow as tf

class encoder(tf.keras.layers.Layer):
    
    def __init__(self, input_dim, latent_dim, act_fxn, init_fxn):
        super(encoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.act_fxn = act_fxn
        self.init_fxn = init_fxn
        self.hidden_layer = tf.keras.layers.Dense(units = latent_dim, activation = act_fxn, kernel_initializer = init_fxn)
        self.output_layer = tf.keras.layers.Dense(units = latent_dim,activation = act_fxn)
      
    def __call__(self, x):
        activation = self.hidden_layer(x)
        return self.output_layer(activation)

class decoder(tf.keras.layers.Layer):
    
    def __init__(self, input_dim, latent_dim, act_fxn, init_fxn):
        super(decoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.act_fxn = act_fxn
        self.init_fxn = init_fxn
        self.hidden_layer = tf.keras.layers.Dense(units=latent_dim, activation=act_fxn, kernel_initializer=init_fxn)
        self.output_layer = tf.keras.layers.Dense(units=input_dim, activation=act_fxn)

    def __call__(self, h):
        activation = self.hidden_layer(h)
        return self.output_layer(activation)

class autoencoder(tf.keras.Model):
    
    def __init__(self, input_dim, latent_dim, act_fxn, init_fxn):
        super(autoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.act_fxn = act_fxn
        self.init_fxn = init_fxn
        self.encoder = encoder(input_dim = input_dim, latent_dim = latent_dim, act_fxn = act_fxn, init_fxn = init_fxn)
        self.decoder = decoder(input_dim = input_dim, latent_dim = latent_dim, act_fxn = act_fxn, init_fxn = init_fxn)

    def __call__(self, x):
        h = self.encoder(x)
        z = self.decoder(h)
        return z

    def loss(model, x):
        reconstruction_error = tf.reduce_mean(tf.square(tf.subtract(model(x), x)))
        return reconstruction_error
    
    def train(loss, model, opt, x):
        with tf.GradientTape() as tape:
            gradients = tape.gradient(loss(model, x), model.trainable_variables)
            gradient_variables = zip(gradients, model.trainable_variables)
            opt.apply_gradients(gradient_variables)
            
    def get_decoder(self):
        return self.decoder
    
    def get_encoder(self):
        return self.encoder
            
