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

    def buildAutoencoder(self, x_input):
        decodeVariables = {}
        with tf.variable_scope('autoencoder', regularizer=l2_regularizer(self.l2scale)):
            tempVec = x_input
            tempDim = self.inputDim
            i = 0
            for compressDim in self.compressDims:
                W = tf.get_variable('aee_W_'+str(i), shape=[tempDim, compressDim])
                b = tf.get_variable('aee_b_'+str(i), shape=[compressDim])
                tempVec = self.aeActivation(tf.add(tf.matmul(tempVec, W), b))
                tempDim = compressDim
                i += 1
    
            i = 0
            for decompressDim in self.decompressDims[:-1]:
                W = tf.get_variable('aed_W_'+str(i), shape=[tempDim, decompressDim])
                b = tf.get_variable('aed_b_'+str(i), shape=[decompressDim])
                tempVec = self.aeActivation(tf.add(tf.matmul(tempVec, W), b))
                tempDim = decompressDim
                decodeVariables['aed_W_'+str(i)] = W
                decodeVariables['aed_b_'+str(i)] = b
                i += 1
            W = tf.get_variable('aed_W_'+str(i), shape=[tempDim, self.decompressDims[-1]])
            b = tf.get_variable('aed_b_'+str(i), shape=[self.decompressDims[-1]])
            decodeVariables['aed_W_'+str(i)] = W
            decodeVariables['aed_b_'+str(i)] = b

            if self.dataType == 'binary':
                x_reconst = tf.nn.sigmoid(tf.add(tf.matmul(tempVec,W),b))
                loss = tf.reduce_mean(-tf.reduce_sum(x_input * tf.log(x_reconst + 1e-12) + (1. - x_input) * tf.log(1. - x_reconst + 1e-12), 1), 0)
            else:
                x_reconst = tf.nn.relu(tf.add(tf.matmul(tempVec,W),b))
                loss = tf.reduce_mean((x_input - x_reconst)**2)
            
        return loss, decodeVariables

    def buildGenerator(self, x_input, bn_train):
        tempVec = x_input
        tempDim = self.randomDim
        with tf.variable_scope('generator', regularizer=l2_regularizer(self.l2scale)):
            for i, genDim in enumerate(self.generatorDims[:-1]):
                W = tf.get_variable('W_'+str(i), shape=[tempDim, genDim])
                h = tf.matmul(tempVec,W)
                h2 = batch_norm(h, decay=self.bnDecay, scale=True, is_training=bn_train, updates_collections=None)
                h3 = self.generatorActivation(h2)
                tempVec = h3 + tempVec
                tempDim = genDim
            W = tf.get_variable('W'+str(i), shape=[tempDim, self.generatorDims[-1]])
            h = tf.matmul(tempVec,W)
            h2 = batch_norm(h, decay=self.bnDecay, scale=True, is_training=bn_train, updates_collections=None)

            if self.dataType == 'binary':
                h3 = tf.nn.tanh(h2)
            else:
                h3 = tf.nn.relu(h2)

            output = h3 + tempVec
        return output
    
    def buildGeneratorTest(self, x_input, bn_train):
        tempVec = x_input
        tempDim = self.randomDim
        with tf.variable_scope('generator', regularizer=l2_regularizer(self.l2scale)):
            for i, genDim in enumerate(self.generatorDims[:-1]):
                W = tf.get_variable('W_'+str(i), shape=[tempDim, genDim])
                h = tf.matmul(tempVec,W)
                h2 = batch_norm(h, decay=self.bnDecay, scale=True, is_training=bn_train, updates_collections=None, trainable=False)
                h3 = self.generatorActivation(h2)
                tempVec = h3 + tempVec
                tempDim = genDim
            W = tf.get_variable('W'+str(i), shape=[tempDim, self.generatorDims[-1]])
            h = tf.matmul(tempVec,W)
            h2 = batch_norm(h, decay=self.bnDecay, scale=True, is_training=bn_train, updates_collections=None, trainable=False)

            if self.dataType == 'binary':
                h3 = tf.nn.tanh(h2)
            else:
                h3 = tf.nn.relu(h2)

            output = h3 + tempVec
        return output
    
    def getDiscriminatorResults(self, x_input, keepRate, reuse=False):
        batchSize = tf.shape(x_input)[0]
        inputMean = tf.reshape(tf.tile(tf.reduce_mean(x_input,0), [batchSize]), (batchSize, self.inputDim))
        tempVec = tf.concat([x_input, inputMean], 1)
        tempDim = self.inputDim * 2
        with tf.variable_scope('discriminator', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            for i, discDim in enumerate(self.discriminatorDims[:-1]):
                W = tf.get_variable('W_'+str(i), shape=[tempDim, discDim])
                b = tf.get_variable('b_'+str(i), shape=[discDim])
                h = self.discriminatorActivation(tf.add(tf.matmul(tempVec,W),b))
                h = tf.nn.dropout(h, keepRate)
                tempVec = h
                tempDim = discDim
            W = tf.get_variable('W', shape=[tempDim, 1])
            b = tf.get_variable('b', shape=[1])
            y_hat = tf.squeeze(tf.nn.sigmoid(tf.add(tf.matmul(tempVec, W), b)))
        return y_hat
    
    def buildDiscriminator(self, x_real, x_fake, keepRate, decodeVariables, bn_train):
        #Discriminate for real samples
        y_hat_real = self.getDiscriminatorResults(x_real, keepRate, reuse=False)

        #Decompress, then discriminate for real samples
        tempVec = x_fake
        i = 0
        for _ in self.decompressDims[:-1]:
            tempVec = self.aeActivation(tf.add(tf.matmul(tempVec, decodeVariables['aed_W_'+str(i)]), decodeVariables['aed_b_'+str(i)]))
            i += 1

        if self.dataType == 'binary':
            x_decoded = tf.nn.sigmoid(tf.add(tf.matmul(tempVec, decodeVariables['aed_W_'+str(i)]), decodeVariables['aed_b_'+str(i)]))
        else:
            x_decoded = tf.nn.relu(tf.add(tf.matmul(tempVec, decodeVariables['aed_W_'+str(i)]), decodeVariables['aed_b_'+str(i)]))

        y_hat_fake = self.getDiscriminatorResults(x_decoded, keepRate, reuse=True)

        loss_d = -tf.reduce_mean(tf.log(y_hat_real + 1e-12)) - tf.reduce_mean(tf.log(1. - y_hat_fake + 1e-12))
        loss_g = -tf.reduce_mean(tf.log(y_hat_fake + 1e-12))

        return loss_d, loss_g, y_hat_real, y_hat_fake

     
    def generateData(self,
                     nSamples=100,
                     modelFile='model',
                     batchSize=100,
                     outFile='out'):
        x_dummy = tf.placeholder('float', [None, self.inputDim])
        _, decodeVariables = self.buildAutoencoder(x_dummy)
        x_random = tf.placeholder('float', [None, self.randomDim])
        bn_train = tf.placeholder('bool')
        x_emb = self.buildGeneratorTest(x_random, bn_train)
        tempVec = x_emb
        i = 0
        for _ in self.decompressDims[:-1]:
            tempVec = self.aeActivation(tf.add(tf.matmul(tempVec, decodeVariables['aed_W_'+str(i)]), decodeVariables['aed_b_'+str(i)]))
            i += 1

        if self.dataType == 'binary':
            x_reconst = tf.nn.sigmoid(tf.add(tf.matmul(tempVec, decodeVariables['aed_W_'+str(i)]), decodeVariables['aed_b_'+str(i)]))
        else:
            x_reconst = tf.nn.relu(tf.add(tf.matmul(tempVec, decodeVariables['aed_W_'+str(i)]), decodeVariables['aed_b_'+str(i)]))

        np.random.seed(1234)
        saver = tf.train.Saver()
        outputVec = []
        burn_in = 1000
        with tf.Session() as sess:
            saver.restore(sess, modelFile)
            print('burning in')
            for i in range(burn_in):
                randomX = np.random.normal(size=(batchSize, self.randomDim))
                output = sess.run(x_reconst, feed_dict={x_random:randomX, bn_train:True})

            print('generating')
            nBatches = int(np.ceil(float(nSamples)) / float(batchSize))
            for i in range(nBatches):
                randomX = np.random.normal(size=(batchSize, self.randomDim))
                output = sess.run(x_reconst, feed_dict={x_random:randomX, bn_train:False})
                outputVec.extend(output)

        outputMat = np.array(outputVec)
        np.save(outFile, outputMat)
    
    def calculateDiscAuc(self, preds_real, preds_fake):
        preds = np.concatenate([preds_real, preds_fake], axis=0)
        labels = np.concatenate([np.ones((len(preds_real))), np.zeros((len(preds_fake)))], axis=0)
        auc = roc_auc_score(labels, preds)
        return auc
    
    def calculateDiscAccuracy(self, preds_real, preds_fake):
        total = len(preds_real) + len(preds_fake)
        hit = 0
        for pred in preds_real: 
            if pred > 0.5: hit += 1
        for pred in preds_fake: 
            if pred < 0.5: hit += 1
        acc = float(hit) / float(total)
        return acc

    def train(self,
              dataPath='data',
              modelPath='',
              outPath='out',
              nEpochs=500,
              discriminatorTrainPeriod=2,
              generatorTrainPeriod=1,
              pretrainBatchSize=100,
              batchSize=1000,
              pretrainEpochs=100,
              saveMaxKeep=0):
        x_raw = tf.placeholder('float', [None, self.inputDim])
        x_random= tf.placeholder('float', [None, self.randomDim])
        keep_prob = tf.placeholder('float')
        bn_train = tf.placeholder('bool')

        loss_ae, decodeVariables = self.buildAutoencoder(x_raw)
        x_fake = self.buildGenerator(x_random, bn_train)
        loss_d, loss_g, y_hat_real, y_hat_fake = self.buildDiscriminator(x_raw, x_fake, keep_prob, decodeVariables, bn_train)
        trainX, validX = self.loadData(dataPath)

        t_vars = tf.trainable_variables()
        ae_vars = [var for var in t_vars if 'autoencoder' in var.name]
        d_vars = [var for var in t_vars if 'discriminator' in var.name]
        g_vars = [var for var in t_vars if 'generator' in var.name]

        all_regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        optimize_ae = tf.train.AdamOptimizer().minimize(loss_ae + sum(all_regs), var_list=ae_vars)
        optimize_d = tf.train.AdamOptimizer().minimize(loss_d + sum(all_regs), var_list=d_vars)
        decodeVariablesValues = list(decodeVariables.values())
        optimize_g = tf.train.AdamOptimizer().minimize(loss_g + sum(all_regs), var_list=g_vars+decodeVariablesValues)

        initOp = tf.global_variables_initializer()

        nBatches = int(np.ceil(float(trainX.shape[0]) / float(batchSize)))
        saver = tf.train.Saver(max_to_keep=saveMaxKeep)
        logFile = outPath + '.log'

        with tf.Session() as sess:
            if modelPath == '': sess.run(initOp)
            else: saver.restore(sess, modelPath)
            nTrainBatches = int(np.ceil(float(trainX.shape[0])) / float(pretrainBatchSize))
            nValidBatches = int(np.ceil(float(validX.shape[0])) / float(pretrainBatchSize))

            if modelPath== '':
                for epoch in range(pretrainEpochs):
                    idx = np.random.permutation(trainX.shape[0])
                    trainLossVec = []
                    for i in range(nTrainBatches):
                        batchX = trainX[idx[i*pretrainBatchSize:(i+1)*pretrainBatchSize]]
                        _, loss = sess.run([optimize_ae, loss_ae], feed_dict={x_raw:batchX})
                        trainLossVec.append(loss)
                    idx = np.random.permutation(validX.shape[0])
                    validLossVec = []
                    for i in range(nValidBatches):
                        batchX = validX[idx[i*pretrainBatchSize:(i+1)*pretrainBatchSize]]
                        loss = sess.run(loss_ae, feed_dict={x_raw:batchX})
                        validLossVec.append(loss)
                    validReverseLoss = 0.
                    buf = 'Pretrain_Epoch:%d, trainLoss:%f, validLoss:%f, validReverseLoss:%f' % (epoch, np.mean(trainLossVec), np.mean(validLossVec), validReverseLoss)
                    print(buf)
                    self.print2file(buf, logFile)

            idx = np.arange(trainX.shape[0])
            for epoch in range(nEpochs):
                d_loss_vec= []
                g_loss_vec = []
                for i in range(nBatches):
                    for _ in range(discriminatorTrainPeriod):
                        batchIdx = np.random.choice(idx, size=batchSize, replace=False)
                        batchX = trainX[batchIdx]
                        randomX = np.random.normal(size=(batchSize, self.randomDim))
                        _, discLoss = sess.run([optimize_d, loss_d], feed_dict={x_raw:batchX, x_random:randomX, keep_prob:1.0, bn_train:False})
                        d_loss_vec.append(discLoss)
                    for _ in range(generatorTrainPeriod):
                        randomX = np.random.normal(size=(batchSize, self.randomDim))
                        _, generatorLoss = sess.run([optimize_g, loss_g], feed_dict={x_raw:batchX, x_random:randomX, keep_prob:1.0, bn_train:True})
                        g_loss_vec.append(generatorLoss)

                idx = np.arange(len(validX))
                nValidBatches = int(np.ceil(float(len(validX)) / float(batchSize)))
                validAccVec = []
                validAucVec = []
                for i in range(nBatches):
                    batchIdx = np.random.choice(idx, size=batchSize, replace=False)
                    batchX = validX[batchIdx]
                    randomX = np.random.normal(size=(batchSize, self.randomDim))
                    preds_real, preds_fake, = sess.run([y_hat_real, y_hat_fake], feed_dict={x_raw:batchX, x_random:randomX, keep_prob:1.0, bn_train:False})
                    validAcc = self.calculateDiscAccuracy(preds_real, preds_fake)
                    validAuc = self.calculateDiscAuc(preds_real, preds_fake)
                    validAccVec.append(validAcc)
                    validAucVec.append(validAuc)
                buf = 'Epoch:%d, d_loss:%f, g_loss:%f, accuracy:%f, AUC:%f' % (epoch, np.mean(d_loss_vec), np.mean(g_loss_vec), np.mean(validAccVec), np.mean(validAucVec))
                print(buf)
                self.print2file(buf, logFile)
                savePath = saver.save(sess, outPath, global_step=epoch)
        print(savePath)


