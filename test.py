"""
Description: test various subclasses.
Author: Haley Hunter-Zinck
Date: July 3, 2020
use %reset in terminal to clear all variables
"""


# autoencoder  template
import tensorflow as tf
from ae import autoencoder
import numpy as np

# load raw dataset
x = np.load('/Users/haleyhunter-zinck/Documents/workspace/synth/structured/output/nhamcs_2011.npy')


# train and print loss for each epoch and step
input_dim = x.shape[1]
latent_dim = 100
init_fxn = 'he_uniform'
act_fxn = tf.nn.relu
n_epoch = 10
learning_rate = 0.01
opt = tf.optimizers.Adam(learning_rate = learning_rate)
batch_size = 256

# process dataset
training_dataset = tf.data.Dataset.from_tensor_slices(x.astype('float32'))
training_dataset = training_dataset.batch(batch_size)
training_dataset = training_dataset.shuffle(x.shape[0])
training_dataset = training_dataset.prefetch(batch_size * 4)

# train autoencoder
ae_model = autoencoder(input_dim = input_dim, latent_dim = latent_dim, act_fxn = act_fxn, init_fxn = init_fxn)
for epoch in range(n_epoch):
    for step, batch_features in enumerate(training_dataset):
        autoencoder.train(autoencoder.loss, ae_model, opt, batch_features)
        loss_values = autoencoder.loss(ae_model, batch_features)
        tf.summary.scalar('loss', loss_values, step=step)
        
        
# translate with decoder
ae_decoder = ae_model.get_decoder()
mu = 0
sigma = 1
n = latent_dim
h = np.asmatrix(np.random.normal(mu, sigma, n))
z = ae_decoder(h)
print(z)
