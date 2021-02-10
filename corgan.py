""" Adaptation of CorGAN: Correlation-Capturing Convolutional Generative 
Adversarial Networks for Generating Synthetic Healthcare Records.

Torfi, A., & Fox, E. A. (2020). COR-GAN: Correlation-Capturing
Convolutional Neural Networks for Generating Synthetic Healthcare Records.
https://arxiv.org/abs/2001.09346

https://github.com/astorfi/cor-gan
"""

import os
import multiprocessing
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

# absehrd modules
from synthesizer import Synthesizer

class Dataset:
    """Data representation for COR-GAN.
    """

    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data
        self.sample_size = data.shape[0]
        self.feature_size = data.shape[1]

    def return_data(self):
        """Return the data matrix.

        Returns
        -------
        array_like
            Array of data.

        """
        return self.data

    def __len__(self):
        """Return the sample size of the data matrix.

        Returns
        -------
        int
            Number of rows in the dataset.

        """
        return len(self.data)

    def __getitem__(self, idx):
        """Return the sample in the dataset at a given index.

        Parameters
        ----------
        idx : int
            Index of data sample

        Returns
        -------
        Tensor
            Row of the dataset as a pytorch Tensor.
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data[idx]
        sample = np.clip(sample, 0, 1)
        if self.transform:
            pass
        return torch.from_numpy(sample)

class Autoencoder(nn.Module):
    """Autoencoder model for translating the output of the generator.
    """

    def __init__(self, feature_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.Tanh())
        self.decoder = nn.Sequential(nn.Linear(128, feature_size)
                                     , nn.Sigmoid())
    def forward(self, x):
        """Get restored version of the data sample through the autoencoder.

        Parameters
        ----------
        x : array_like
            Input array to the autoencoder.

        Returns
        -------
        array_like
            Output array from the autoencoder.

        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def decode(self, x):
        """Get restored version from hidden layer.

        Parameters
        ----------
        x : array_like
            Input Array to the decoder layer of the autoencoder.

        Returns
        -------
        array_like
            Output array from the decoder layer of the autoencoder.

        """
        x = self.decoder(x)
        return x

class Generator(nn.Module):
    """ Generator model for generative adversarial network of Cor-GAN.
    """

    def __init__(self, latent_dim):
        super().__init__()
        self.gen_dim = 128
        self.linear1 = nn.Linear(latent_dim, self.gen_dim)
        self.bn1 = nn.BatchNorm1d(self.gen_dim, eps=0.001, momentum=0.01)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(latent_dim, self.gen_dim)
        self.bn2 = nn.BatchNorm1d(self.gen_dim, eps=0.001, momentum=0.01)
        self.activation2 = nn.Tanh()
        self.latent_dim=latent_dim

    def forward(self, x):
        """ Get output from generator model.

        Parameters
        ----------
        x : array_like
            Input array to the generator.

        Returns
        -------
        array_like
            Output array from the generator.

        """

        # Layer 1
        residual = x
        temp = self.activation1(self.bn1(self.linear1(x)))
        out1 = temp + residual
        # Layer 2
        residual = out1
        temp = self.activation2(self.bn2(self.linear2(out1)))
        out2 = temp + residual
        return out2

class Discriminator(nn.Module):
    """ Discriminator model for generative adversarial network of Cor-GAN.
    """

    def __init__(self, minibatch_averaging, feature_size):
        super().__init__()
        # Discriminator's parameters
        self.dis_dim = 256
        # The minibatch averaging setup
        ma_coef = 1
        if minibatch_averaging:
            ma_coef = ma_coef * 2
        self.model = nn.Sequential(
            nn.Linear(ma_coef * feature_size, self.dis_dim),
            nn.ReLU(True),
            nn.Linear(self.dis_dim, int(self.dis_dim)),
            nn.ReLU(True),
            nn.Linear(self.dis_dim, int(self.dis_dim)),
            nn.ReLU(True),
            nn.Linear(int(self.dis_dim), 1)
        )
        self.minibatch_averaging = minibatch_averaging
        self.feature_size=feature_size

    def forward(self, x):
        """ Get output from discriminator model.

        Parameters
        ----------
        x : array_like
            Input array to the discriminator.

        Returns
        -------
        array_like
            Input array from the discriminator.

        """

        if self.minibatch_averaging:
            x_mean = torch.mean(x, 0).repeat(x.shape[0], 1)
            x = torch.cat((x, x_mean), 1)
        output = self.model(x)
        return output

class Corgan(Synthesizer):
    """ Cor-GAN: generative advesarial network plus autoencoder for
        generating synthetic binary data.
    """
    
    def __init__(self, debug=False, n_cpu=1):
        """
        Create a CorGAN structure.

        Parameters
        ----------
        debug : bool, optional
            If True, output debugging messages; otherwise, do not print
            debugging messages. The default is False.
        n_cpu : int, optional
            Number of CPUs to use during training. The default is 1.

        Returns
        -------
        None.

        """

        self.debug=debug
        self.n_cpu = n_cpu

    def autoencoder_loss(self, x_output, y_target, epsilon = 1e-12):
        """Calculate loss function for autoencoder model.

        Parameters
        ----------
        x_output : array_like
            Output from the autoencoder.
        y_target : array_like
            Input to the autoencoder.
        epsilon : float, optional
            Buffer value to avoid take log of 0. The default is 1e-12.

        Returns
        -------
        loss : TYPE
            DESCRIPTION.

        """
        term = y_target * torch.log(x_output + epsilon) + \
            (1. - y_target) * torch.log(1. - x_output + epsilon)
        loss = torch.mean(-torch.sum(term, 1), 0)
        return loss

    def discriminator_accuracy(self, predicted, y_true):
        """Calculate accuracy of the discriminator model.

        Parameters
        ----------
        predicted : array_like
            Array of predicted values output by the discriminator.
        y_true : array_like
            Array of actual values for the corresponding predicted values.

        Returns
        -------
        accuracy : float
            Accuracy for the corresponding predicted and true values.

        """
        total = y_true.size(0)
        correct = (torch.abs(predicted - y_true) <= 0.5).sum().item()
        accuracy = 100.0 * correct / total
        return accuracy

    def weights_init(self, m_layer):
        """Initialize weights of a neural network layer.

        Parameters
        ----------
        m_layer : torch.nn.Module
            pytorch layer of a neural network.

        Returns
        -------
        None.

        """
        classname = m_layer.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m_layer.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m_layer.weight.data, 1.0, 0.02)
            nn.init.constant_(m_layer.bias.data, 0)
        if isinstance(m_layer, nn.Linear):
            torch.nn.init.xavier_uniform_(m_layer.weight)
            m_layer.bias.data.fill_(0.01)

    def train(self, x,
              n_epochs_pretrain=100,
              n_epochs=100,
              frac_trn=0.8,
              batch_size=512,
              sample_interval=100,
              latent_dim=128,
              lr=0.001,
              b1=0.9,
              b2=0.999,
              weight_decay=0.0001,
              n_iter_D=5,
              minibatch_averaging=0,
              clamp_lower=-0.01,
              clamp_upper=0.01,
              epoch_time_show=1,
              epoch_save_model_freq=100,
              path_checkpoint='corgan_ckpts',
              prefix_checkpoint='ckpt'):
        """Train the Cor-GAN model.

        Parameters
        ----------
        x : array_like
            2D array of data to train the generator.
        n_epochs_pretrain : int
            Number of epochs to use during training of the autoencoder.
        n_epochs : int
            Number of epochs to use during training of the GAN.
        frac_trn : float
            Fraction of data to use for training; remaining used validation
            during the training procedure.
        batch_size : int
            DESCRIPTION.
        sample_interval : TYPE
            DESCRIPTION.
        latent_dim : TYPE
            DESCRIPTION.
        lr : TYPE
            DESCRIPTION.
        b1 : TYPE
            DESCRIPTION.
        b2 : TYPE
            DESCRIPTION.
        weight_decay : TYPE
            DESCRIPTION.
        n_iter_D : TYPE
            DESCRIPTION.
        minibatch_averaging : TYPE
            DESCRIPTION.
        clamp_lower : TYPE
            DESCRIPTION.
        clamp_upper : TYPE
            DESCRIPTION.
        epoch_time_show : TYPE
            DESCRIPTION.
        epoch_save_model_freq : TYPE
            DESCRIPTION.
        path_checkpoint : TYPE
            DESCRIPTION.
        prefix_checkpoint : TYPE
            DESCRIPTION.
        

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        # train and test split
        idx = np.random.permutation(len(x))
        idx_trn, idx_tst = idx[:int(frac_trn * len(x))], idx[int(frac_trn * len(x)):]
        x_trn = x[idx_trn,:]
        x_tst = x[idx_tst,:]
        x_trn = x_trn.astype(np.float32)
        x_tst = x_tst.astype(np.float32)

        # adapt batch-size to small datasets
        if len(x_tst) < batch_size or len(x_trn) < batch_size:
            if self.debug:
                print('\nWarning: decreasing batch size from',
                      batch_size, 'to', min(len(x_tst), len(x_trn)),
                      'due to small dataset.', flush=True)
            batch_size = min(len(x_tst), len(x_trn))

        # Train data loader
        dataset_train_object = Dataset(data=x_trn, transform=False)
        sampler_random = torch.utils.data.sampler.RandomSampler(data_source=dataset_train_object,
                                                               replacement=True)
        d_trn = DataLoader(dataset_train_object, batch_size=batch_size,
                                      shuffle=False, num_workers=0,
                                      drop_last=True, sampler=sampler_random)

        # Test data loader
        dataset_test_object = Dataset(data=x_tst, transform=False)
        sampler_random = torch.utils.data.sampler.RandomSampler(data_source=dataset_test_object,
                                                               replacement=True)
        d_tst = DataLoader(dataset_test_object, batch_size=batch_size,
                                     shuffle=False, num_workers=0,
                                     drop_last=True, sampler=sampler_random)

        # Initialize generator and discriminator
        generator_model = Generator(latent_dim=latent_dim)
        discriminator_model = Discriminator(minibatch_averaging=minibatch_averaging,
                                           feature_size=x.shape[1])
        autoencoder_model = Autoencoder(feature_size=x.shape[1])
        autoencoder_decoder = autoencoder_model.decoder

        # Define cuda tensors
        tensor_obj = torch.FloatTensor
        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1

        max_cpu = multiprocessing.cpu_count()
        if max_cpu > 1 and self.n_cpu > 1:
            self.n_cpu = min(max_cpu, self.n_cpu)
            if self.debug:
                print('Using', self.n_cpu, 'of', max_cpu, 'CPUs', flush=True)
            generator_model = nn.DataParallel(generator_model, list(range(self.n_cpu)))
            discriminator_model = nn.DataParallel(discriminator_model, list(range(self.n_cpu)))
            autoencoder_model = nn.DataParallel(autoencoder_model, list(range(self.n_cpu)))
            autoencoder_decoder = nn.DataParallel(autoencoder_decoder, list(range(self.n_cpu)))

        # Weight initialization
        generator_model.apply(self.weights_init)
        discriminator_model.apply(self.weights_init)
        autoencoder_model.apply(self.weights_init)

        # Optimizers
        g_params = [{'params': generator_model.parameters()},
                    {'params': autoencoder_decoder.parameters(), 'lr': 1e-4}]
        optimizer_g = torch.optim.Adam(g_params, lr=lr, betas=(b1, b2), weight_decay=weight_decay)
        optimizer_d = torch.optim.Adam(discriminator_model.parameters(), lr=lr, betas=(b1, b2),
                                       weight_decay=weight_decay)
        optimizer_a = torch.optim.Adam(autoencoder_model.parameters(), lr=lr, betas=(b1, b2),
                                       weight_decay=weight_decay)

        timer_pre = tqdm(range(n_epochs_pretrain), desc='Pre-training', 
                         unit=' epochs', disable=(not self.debug))
        for epoch_pre in timer_pre:
            for i, samples in enumerate(d_trn):

                # Configure input
                real_samples = Variable(samples.type(tensor_obj))

                # Generate a batch of images
                recons_samples = autoencoder_model(real_samples)

                # Loss measures generator's ability to fool the discriminator
                a_loss = self.autoencoder_loss(recons_samples, real_samples)

                # # Reset gradients
                optimizer_a.zero_grad()

                a_loss.backward()
                optimizer_a.step()

                if self.debug:
                    batches_done = epoch_pre * len(x_trn) + i
                    if batches_done % sample_interval == 0:
                        msg = '[A loss: %.3f]' % (a_loss.item())
                        timer_pre.set_postfix_str(s=msg, refresh=False)

        gen_iterations = 0
        timer = tqdm(range(n_epochs), desc='Training', unit=' epochs', disable=(not self.debug))
        for epoch in timer:
            for i, samples in enumerate(d_trn):

                # Adversarial ground truths
                valid = Variable(tensor_obj(samples.shape[0]).fill_(1.0),
                                 requires_grad=False)
                fake = Variable(tensor_obj(samples.shape[0]).fill_(0.0),
                                requires_grad=False)

                # Configure input
                real_samples = Variable(samples.type(tensor_obj))

                # Sample noise as generator input
                arr_z = torch.randn(samples.shape[0], latent_dim)

                # reset requires_grad
                for param in discriminator_model.parameters():
                    param.requires_grad = False

                # Zero grads
                optimizer_g.zero_grad()

                # Generate a batch of images
                fake_samples = generator_model(arr_z)

                # uncomment if there is no autoencoder
                fake_samples = autoencoder_decoder(fake_samples)

                # Loss measures generator's ability to fool the discriminator
                err_g = torch.mean(discriminator_model(fake_samples).view(-1))
                err_g.backward(mone)

                optimizer_g.step()
                gen_iterations += 1

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # reset requires_grad
                for param in discriminator_model.parameters():
                    param.requires_grad = True

                # train the discriminator n_iter_D times
                j = 0
                while j < n_iter_D:
                    j += 1

                    # clamp parameters to a cube
                    for param in discriminator_model.parameters():
                        param.data.clamp_(clamp_lower, clamp_upper)

                    # reset gradients of discriminator
                    optimizer_d.zero_grad()

                    err_d_real = torch.mean(discriminator_model(real_samples).view(-1))
                    err_d_real.backward(mone)

                    # Measure discriminator's ability to classify real from generated samples
                    # The detach() method constructs a new view on a tensor which is declared
                    # not to need gradients, i.e., it is to be excluded from further tracking of
                    # operations, and therefore the subgraph involving this view is not recorded.
                    # Refer to http://www.bnikolic.co.uk/blog/pytorch-detach.html.

                    err_d_fake = torch.mean(discriminator_model(fake_samples.detach()).view(-1))
                    err_d_fake.backward(one)
                    err_d = -(err_d_real - err_d_fake)

                    # Optimizer step
                    optimizer_d.step()

            with torch.no_grad():

                # Variables
                real_samples_test = next(iter(d_tst))
                real_samples_test = Variable(real_samples_test.type(tensor_obj))
                arr_z = torch.randn(samples.shape[0], latent_dim)

                # Generator
                fake_samples_test_temp = generator_model(arr_z)
                fake_samples_test = autoencoder_decoder(fake_samples_test_temp)

                # Discriminator
                # torch.sigmoid() is needed as the discriminator outputs are
                #   logits without any sigmoid.
                out_real_test = discriminator_model(real_samples_test).view(-1)
                accuracy_real_test = self.discriminator_accuracy(torch.sigmoid(out_real_test),
                                                                 valid)

                out_fake_test = discriminator_model(fake_samples_test.detach()).view(-1)
                accuracy_fake_test = self.discriminator_accuracy(torch.sigmoid(out_fake_test), fake)

                # Test autoencoder
                reconst_samples_test = autoencoder_model(real_samples_test)
                a_loss_test = self.autoencoder_loss(reconst_samples_test, real_samples_test)

            if self.debug:
                msg = 'TRAIN: [Loss_D: %.3f] [Loss_G: %.3f] [Loss_D_real: %.3f] [Loss_D_fake %.3f]' \
                      % (err_d.item(), err_g.item(), err_d_real.item(),err_d_fake.item())
                msg = msg+' | TEST: [A loss: %.2f] [real accuracy: %.2f] [fake accuracy: %.2f]' \
                    % (a_loss_test.item(), accuracy_real_test, accuracy_fake_test)
                timer.set_postfix_str(s=msg, refresh=False)

            # End of epoch
            #epoch_end = time.time()

            parameter_dict = {'model':'corgan',
                                  'latent_dim':latent_dim,
                                  'feature_size':x.shape[1],
                                  'batch_size':batch_size,
                                  'n_cpu':self.n_cpu,
                                  'minibatch_averaging':minibatch_averaging}

            if (epoch + 1) % epoch_save_model_freq == 0 or (epoch + 1) == n_epochs:

                if not os.path.isdir(path_checkpoint):
                    os.mkdir(path_checkpoint)

                torch.save({
                    'epoch': epoch + 1,
                    'Generator_state_dict': generator_model.state_dict(),
                    'Discriminator_state_dict': discriminator_model.state_dict(),
                    'Autoencoder_state_dict': autoencoder_model.state_dict(),
                    'Autoencoder_Decoder_state_dict': autoencoder_decoder.state_dict(),
                    'optimizer_G_state_dict': optimizer_g.state_dict(),
                    'optimizer_D_state_dict': optimizer_d.state_dict(),
                    'optimizer_A_state_dict': optimizer_a.state_dict(),
                    'parameter_dict':parameter_dict
                }, os.path.join(path_checkpoint, prefix_checkpoint + \
                                ".model_epoch_%d.pth" % (epoch + 1)))

        return {'epoch': epoch + 1,
                    'Generator_state_dict': generator_model.state_dict(),
                    'Discriminator_state_dict': discriminator_model.state_dict(),
                    'Autoencoder_state_dict': autoencoder_model.state_dict(),
                    'Autoencoder_Decoder_state_dict': autoencoder_decoder.state_dict(),
                    'optimizer_G_state_dict': optimizer_g.state_dict(),
                    'optimizer_D_state_dict': optimizer_d.state_dict(),
                    'optimizer_A_state_dict': optimizer_a.state_dict(),
                    'parameter_dict':parameter_dict}

    def generate(self, model, n_gen):
        """Generate samples for the Cor-GAN generator.

        Parameters
        ----------
        model : dict or str
            Dictionary representing trained GAN model or a file name.
        n_gen : int
            Number of samples to generate.

        Returns
        -------
        gen_samples : array_like
            2D array of samples generated.

        """

        device = torch.device("cpu")

        # Loading the checkpoint
        checkpoint=model
        if isinstance(model, str):
            checkpoint = torch.load(model)

        # parameters
        latent_dim = checkpoint['parameter_dict']['latent_dim']
        feature_size = checkpoint['parameter_dict']['feature_size']
        batch_size = checkpoint['parameter_dict']['batch_size']
        n_cpu = checkpoint['parameter_dict']['n_cpu']

        # Setup model
        generator_model = Generator(latent_dim=latent_dim)
        autoencoder_model = Autoencoder(feature_size=feature_size)
        autoencoder_decoder = autoencoder_model.decoder
        if multiprocessing.cpu_count() > 1 and n_cpu > 1:
            n_cpu = min(multiprocessing.cpu_count(), n_cpu)
            generator_model = nn.DataParallel(generator_model, list(range(n_cpu)))
            autoencoder_model = nn.DataParallel(autoencoder_model, list(range(n_cpu)))
            autoencoder_decoder = nn.DataParallel(autoencoder_decoder, list(range(n_cpu)))

        # Load models
        generator_model.load_state_dict(checkpoint['Generator_state_dict'])
        autoencoder_model.load_state_dict(checkpoint['Autoencoder_state_dict'])
        autoencoder_decoder.load_state_dict(checkpoint['Autoencoder_Decoder_state_dict'])

        # insert weights [required]
        generator_model.eval()
        autoencoder_model.eval()
        autoencoder_decoder.eval()

        # number of patches to generate
        gen_samples = np.zeros((n_gen, feature_size))
        n_batches = int(n_gen / batch_size) + 1

        for i in range(n_batches):

            batch_size_itr = min(batch_size, n_gen - i * batch_size)
            arr_z = torch.randn(batch_size_itr, latent_dim, device=device)
            gen_samples_tensor = generator_model(arr_z)
            gen_samples_decoded = autoencoder_decoder(gen_samples_tensor)

            idx_max = min(i * batch_size + batch_size, n_gen)
            gen_samples[i * batch_size:idx_max, :] = gen_samples_decoded.cpu().data.numpy()

        return gen_samples
