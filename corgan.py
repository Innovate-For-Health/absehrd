import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

# for cpu count
import multiprocessing

# custom
from generator_framework import GeneratorFramework

class Dataset:
    def __init__(self, data, transform=None):
        # Transform
        self.transform = transform
        # load data here
        self.data = data
        self.sampleSize = data.shape[0]
        self.featureSize = data.shape[1]
        
    def return_data(self):
        return self.data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data[idx]
        sample = np.clip(sample, 0, 1)
        if self.transform:
           pass
        return torch.from_numpy(sample)

class Autoencoder(nn.Module):
    
    def __init__(self, feature_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.Tanh())
        self.decoder = nn.Sequential(nn.Linear(128, feature_size)
                                     , nn.Sigmoid())
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def decode(self, x):
        x = self.decoder(x)
        return x


class Generator(nn.Module):
    
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.genDim = 128
        self.linear1 = nn.Linear(latent_dim, self.genDim)
        self.bn1 = nn.BatchNorm1d(self.genDim, eps=0.001, momentum=0.01)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(latent_dim, self.genDim)
        self.bn2 = nn.BatchNorm1d(self.genDim, eps=0.001, momentum=0.01)
        self.activation2 = nn.Tanh()
        self.latent_dim=latent_dim
    
    def forward(self, x):
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
    
    def __init__(self, minibatch_averaging, feature_size):
        super(Discriminator, self).__init__()
        # Discriminator's parameters
        self.disDim = 256
        # The minibatch averaging setup
        ma_coef = 1
        if minibatch_averaging:
            ma_coef = ma_coef * 2
        self.model = nn.Sequential(
            nn.Linear(ma_coef * feature_size, self.disDim),
            nn.ReLU(True),
            nn.Linear(self.disDim, int(self.disDim)),
            nn.ReLU(True),
            nn.Linear(self.disDim, int(self.disDim)),
            nn.ReLU(True),
            nn.Linear(int(self.disDim), 1)
        )
        self.minibatch_averaging = minibatch_averaging
        self.feature_size=feature_size
    
    def forward(self, x):
        if self.minibatch_averaging:
            ### minibatch averaging ###
            x_mean = torch.mean(x, 0).repeat(x.shape[0], 1)  # Average over the batch
            x = torch.cat((x, x_mean), 1)  # Concatenation
        # Feeding the model
        output = self.model(x)
        return output


class corgan(GeneratorFramework):
    
    def generator_loss(self, y_fake, y_true, epsilon = 1e-12):
        return -0.5 * torch.mean(torch.log(y_fake + epsilon))
    
    def autoencoder_loss(self, x_output, y_target, epsilon = 1e-12):
        term = y_target * torch.log(x_output + epsilon) + (1. - y_target) * torch.log(1. - x_output + epsilon)
        loss = torch.mean(-torch.sum(term, 1), 0)
        return loss
    
    def discriminator_loss(self, outputs, labels):
        loss = torch.mean((1 - labels) * outputs) - torch.mean(labels * outputs)
        return loss
        
    def discriminator_accuracy(self, predicted, y_true):
        total = y_true.size(0)
        correct = (torch.abs(predicted - y_true) <= 0.5).sum().item()
        accuracy = 100.0 * correct / total
        return accuracy
    
    def sample_transform(self, sample):
        sample[sample >= 0.5] = 1
        sample[sample < 0.5] = 0
        return sample
    
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
    
    def train(self, x, 
              n_cpu=1, 
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
              prefix_checkpoint='ckpt',
              debug=False):
        
        # train and test split
        idx = np.random.permutation(len(x))
        idx_trn, idx_tst = idx[:int(frac_trn * len(x))], idx[int(frac_trn * len(x)):]
        x_trn = x[idx_trn,:]
        x_tst = x[idx_tst,:]
        x_trn = x_trn.astype(np.float32)
        x_tst = x_tst.astype(np.float32)
        
        # adapt batch-size to small datasets
        if len(x_tst) < batch_size or len(x_trn) < batch_size:
            if debug:
                print('\nWarning: decreasing batch size from',
                      batch_size, 'to', min(len(x_tst), len(x_trn)),
                      'due to small dataset.')
            batch_size = min(len(x_tst), len(x_trn))
        
        # Train data loader
        dataset_train_object = Dataset(data=x_trn, transform=False)
        samplerRandom = torch.utils.data.sampler.RandomSampler(data_source=dataset_train_object, replacement=True)
        d_trn = DataLoader(dataset_train_object, batch_size=batch_size,
                                      shuffle=False, num_workers=0, 
                                      drop_last=True, sampler=samplerRandom)
        
        # Test data loader
        dataset_test_object = Dataset(data=x_tst, transform=False)
        samplerRandom = torch.utils.data.sampler.RandomSampler(data_source=dataset_test_object, replacement=True)
        d_tst = DataLoader(dataset_test_object, batch_size=batch_size,
                                     shuffle=False, num_workers=0, 
                                     drop_last=True, sampler=samplerRandom)

        
        # Initialize generator and discriminator
        generatorModel = Generator(latent_dim=latent_dim)
        discriminatorModel = Discriminator(minibatch_averaging=minibatch_averaging, feature_size=x.shape[1])
        autoencoderModel = Autoencoder(feature_size=x.shape[1])
        autoencoderDecoder = autoencoderModel.decoder
        
        # Define cuda Tensors
        Tensor = torch.FloatTensor
        one = torch.tensor(1, dtype=torch.float)
        mone = one * -1
    
        if multiprocessing.cpu_count() > 1 and n_cpu > 1:
            n_cpu = min(multiprocessing.cpu_count(), n_cpu)
            if debug:
                print("Let's use", n_cpu, "CPUs!")
            generatorModel = nn.DataParallel(generatorModel, list(range(n_cpu)))
            discriminatorModel = nn.DataParallel(discriminatorModel, list(range(n_cpu)))
            autoencoderModel = nn.DataParallel(autoencoderModel, list(range(n_cpu)))
            autoencoderDecoder = nn.DataParallel(autoencoderDecoder, list(range(n_cpu)))
    
        # Weight initialization
        generatorModel.apply(self.weights_init)
        discriminatorModel.apply(self.weights_init)
        autoencoderModel.apply(self.weights_init)
    
        # Optimizers
        g_params = [{'params': generatorModel.parameters()},
                    {'params': autoencoderDecoder.parameters(), 'lr': 1e-4}]
        optimizer_G = torch.optim.Adam(g_params, lr=lr, betas=(b1, b2), weight_decay=weight_decay)
        optimizer_D = torch.optim.Adam(discriminatorModel.parameters(), lr=lr, betas=(b1, b2),
                                       weight_decay=weight_decay)
        optimizer_A = torch.optim.Adam(autoencoderModel.parameters(), lr=lr, betas=(b1, b2),
                                       weight_decay=weight_decay)
    
        for epoch_pre in range(n_epochs_pretrain):
            for i, samples in enumerate(d_trn):
    
                # Configure input
                real_samples = Variable(samples.type(Tensor))
    
                # Generate a batch of images
                recons_samples = autoencoderModel(real_samples)
    
                # Loss measures generator's ability to fool the discriminator
                a_loss = self.autoencoder_loss(recons_samples, real_samples)
    
                # # Reset gradients (if you uncomment it, it would be a mess. Why?!!!!!!!!!!!!!!!)
                optimizer_A.zero_grad()
    
                a_loss.backward()
                optimizer_A.step()
    
                if debug:
                    batches_done = epoch_pre * len(x_trn) + i
                    if batches_done % sample_interval == 0:
                        print(
                            "[Epoch %d/%d of pretraining] [Batch %d/%d] [A loss: %.3f]"
                            % (epoch_pre + 1, n_epochs_pretrain, i, len(x), a_loss.item())
                            , flush=True)
    
        gen_iterations = 0
        for epoch in range(n_epochs):
            epoch_start = time.time()
            for i, samples in enumerate(d_trn):
    
                # Adversarial ground truths
                valid = Variable(Tensor(samples.shape[0]).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(samples.shape[0]).fill_(0.0), requires_grad=False)
    
                # Configure input
                real_samples = Variable(samples.type(Tensor))
    
                # Sample noise as generator input
                z = torch.randn(samples.shape[0], latent_dim)
    
                for p in discriminatorModel.parameters():  # reset requires_grad
                    p.requires_grad = False
    
                # Zero grads
                optimizer_G.zero_grad()
    
                # Generate a batch of images
                fake_samples = generatorModel(z)
    
                # uncomment if there is no autoencoder
                fake_samples = autoencoderDecoder(fake_samples)
    
                # Loss measures generator's ability to fool the discriminator
                errG = torch.mean(discriminatorModel(fake_samples).view(-1))
                errG.backward(mone)
    
                # read more at https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/4
                optimizer_G.step()
                gen_iterations += 1
    
                # ---------------------
                #  Train Discriminator
                # ---------------------
    
                for p in discriminatorModel.parameters():  # reset requires_grad
                    p.requires_grad = True
    
                # train the discriminator n_iter_D times
                j = 0
                while j < n_iter_D:
                    j += 1
    
                    # clamp parameters to a cube
                    for p in discriminatorModel.parameters():
                        p.data.clamp_(clamp_lower, clamp_upper)
    
                    # reset gradients of discriminator
                    optimizer_D.zero_grad()
    
                    errD_real = torch.mean(discriminatorModel(real_samples).view(-1))
                    errD_real.backward(mone)
    
                    # Measure discriminator's ability to classify real from generated samples
                    # The detach() method constructs a new view on a tensor which is declared
                    # not to need gradients, i.e., it is to be excluded from further tracking of
                    # operations, and therefore the subgraph involving this view is not recorded.
                    # Refer to http://www.bnikolic.co.uk/blog/pytorch-detach.html.
    
                    errD_fake = torch.mean(discriminatorModel(fake_samples.detach()).view(-1))
                    errD_fake.backward(one)
                    errD = -(errD_real - errD_fake)
    
                    # Optimizer step
                    optimizer_D.step()
    
            with torch.no_grad():
    
                # Variables
                real_samples_test = next(iter(d_tst))
                real_samples_test = Variable(real_samples_test.type(Tensor))
                z = torch.randn(samples.shape[0], latent_dim)
    
                # Generator
                fake_samples_test_temp = generatorModel(z)
                fake_samples_test = autoencoderDecoder(fake_samples_test_temp)
    
                # Discriminator
                # torch.sigmoid() is needed as the discriminator outputs are logits without any sigmoid.
                out_real_test = discriminatorModel(real_samples_test).view(-1)
                accuracy_real_test = self.discriminator_accuracy(torch.sigmoid(out_real_test), valid)
    
                out_fake_test = discriminatorModel(fake_samples_test.detach()).view(-1)
                accuracy_fake_test = self.discriminator_accuracy(torch.sigmoid(out_fake_test), fake)
    
                # Test autoencoder
                reconst_samples_test = autoencoderModel(real_samples_test)
                a_loss_test = self.autoencoder_loss(reconst_samples_test, real_samples_test)
    
            if debug:
                print('TRAIN: [Epoch %d/%d] [Batch %d/%d] Loss_D: %.3f Loss_G: %.3f Loss_D_real: %.3f Loss_D_fake %.3f'
                      % (epoch + 1, n_epochs, i, len(x_trn),
                         errD.item(), errG.item(), errD_real.item(), errD_fake.item()), flush=True)
                print("TEST: [Epoch %d/%d] [Batch %d/%d] [A loss: %.2f] [real accuracy: %.2f] [fake accuracy: %.2f]"
                    % (epoch + 1, n_epochs, i, len(x_trn),
                       a_loss_test.item(), accuracy_real_test,
                       accuracy_fake_test), flush=True)
    
            # End of epoch
            epoch_end = time.time()
            
            if debug:
                if epoch_time_show:
                    print("It has been {0} seconds for this epoch".format(round(epoch_end - epoch_start,2)), flush=True)
        
            parameter_dict = {'latent_dim':latent_dim, 
                                  'feature_size':x.shape[1],
                                  'batch_size':batch_size,
                                  'n_cpu':n_cpu}
        
            if (epoch + 1) % epoch_save_model_freq == 0 or (epoch + 1) == n_epochs:
                # Refer to https://pytorch.org/tutorials/beginner/saving_loading_models.html
                if not os.path.isdir(path_checkpoint):
                    os.mkdir(path_checkpoint)
                    
                torch.save({
                    'epoch': epoch + 1,
                    'Generator_state_dict': generatorModel.state_dict(),
                    'Discriminator_state_dict': discriminatorModel.state_dict(),
                    'Autoencoder_state_dict': autoencoderModel.state_dict(),
                    'Autoencoder_Decoder_state_dict': autoencoderDecoder.state_dict(),
                    'optimizer_G_state_dict': optimizer_G.state_dict(),
                    'optimizer_D_state_dict': optimizer_D.state_dict(),
                    'optimizer_A_state_dict': optimizer_A.state_dict(),
                    'parameter_dict':parameter_dict
                }, os.path.join(path_checkpoint, prefix_checkpoint + ".model_epoch_%d.pth" % (epoch + 1)))
                
        return {'epoch': epoch + 1,
                    'Generator_state_dict': generatorModel.state_dict(),
                    'Discriminator_state_dict': discriminatorModel.state_dict(),
                    'Autoencoder_state_dict': autoencoderModel.state_dict(),
                    'Autoencoder_Decoder_state_dict': autoencoderDecoder.state_dict(),
                    'optimizer_G_state_dict': optimizer_G.state_dict(),
                    'optimizer_D_state_dict': optimizer_D.state_dict(),
                    'optimizer_A_state_dict': optimizer_A.state_dict(),
                    'parameter_dict':parameter_dict}
    
    # model may be a file name to the checkpoint or a model dictionary object
    def generate(self, model, n_gen):
    
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
        generatorModel = Generator(latent_dim=latent_dim)
        autoencoderModel = Autoencoder(feature_size=feature_size)
        autoencoderDecoder = autoencoderModel.decoder
        if multiprocessing.cpu_count() > 1 and n_cpu > 1:
            n_cpu = min(multiprocessing.cpu_count(), n_cpu)
            generatorModel = nn.DataParallel(generatorModel, list(range(n_cpu)))
            autoencoderModel = nn.DataParallel(autoencoderModel, list(range(n_cpu)))
            autoencoderDecoder = nn.DataParallel(autoencoderDecoder, list(range(n_cpu)))
    
        # Load models
        generatorModel.load_state_dict(checkpoint['Generator_state_dict'])
        autoencoderModel.load_state_dict(checkpoint['Autoencoder_state_dict'])
        autoencoderDecoder.load_state_dict(checkpoint['Autoencoder_Decoder_state_dict'])
    
        # insert weights [required]
        generatorModel.eval()
        autoencoderModel.eval()
        autoencoderDecoder.eval()
            
        # number of patches to generate
        gen_samples = np.zeros((n_gen, feature_size))
        n_batches = int(n_gen / batch_size) + 1
        
        for i in range(n_batches):
           
            batch_size_itr = min(batch_size, n_gen - i * batch_size)
            z = torch.randn(batch_size_itr, latent_dim, device=device)
            gen_samples_tensor = generatorModel(z)
            gen_samples_decoded = autoencoderDecoder(gen_samples_tensor)
            
            idx_max = min(i * batch_size + batch_size, n_gen)
            gen_samples[i * batch_size:idx_max, :] = gen_samples_decoded.cpu().data.numpy()
            
            # Check to see if there is any nan
            assert (gen_samples[i, :] != gen_samples[i, :]).any() == False
            
        return gen_samples
    
    
     
