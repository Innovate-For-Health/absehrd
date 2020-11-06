import numpy as np
import torch
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm
import matplotlib.pyplot as plt

# sehrd
from preprocessor import Preprocessor
from validator import Validator

class MLP(torch.nn.Module):
    """Multilayer perceptron model.

    Attributes
    ----------
    input_size : int
        Number of nodes of the input layer
    hidden_size : int
        Number of nodes of the hidden layer
    fc1 : torch.nn
        fill
    relu : torch.nn
        activation function between first and hidden layer
    fc2 : torch.nn
        fill
    sigmoid : torch.nn
        activation function between hidden and output layer
    """

    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        """Get output from MLP.

        Parameters
        ----------
        x : array_like
            Input to the MLP.

        Returns
        -------
        output : array_like
            Output from the MLP.

        """
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output

class Realism(Validator):
    """Validates the realism properties of the synthetic data.

    Attributes
    ----------
    delim : str
        Delimiter for features names and feature values.
    """

    def __init__(self):
        self.delim = '__'

    def validate_univariate(self, arr_r, arr_s, header):
        """Calculate feature frequencies of each matrix

        Parameters
        ----------
        r : TYPE
            DESCRIPTION.
        s : TYPE
            DESCRIPTION.
        header : TYPE
            DESCRIPTION.

        Returns
        -------
        dict
            DESCRIPTION.

        """

        frq_r = np.zeros(shape=arr_r.shape[1])
        for j in range(arr_r.shape[1]):
            frq_r[j] = np.mean(arr_r[:,j])

        frq_s = np.zeros(shape=arr_s.shape[1])
        for j in range(arr_s.shape[1]):
            frq_s[j] = np.mean(arr_s[:,j])

        return {'frq_r':frq_r, 'frq_s':frq_s,
                'header_r':header, 'header_s':header}

    def validate_effect(self, arr_r, arr_s, header, outcome):
        """Calculate effect sizes of logistic regression model.

        Parameters
        ----------
        arr_r : TYPE
            DESCRIPTION.
        arr_s : TYPE
            DESCRIPTION.
        header : TYPE
            DESCRIPTION.
        outcome : TYPE
            DESCRIPTION.

        Returns
        -------
        dict
            DESCRIPTION.

        """

        max_iter = 1000000
        l1_ratio = None
        penalty = 'l2'
        solver = 'lbfgs'

        x_r = arr_r
        x_s = arr_s

        idx_outcome = np.where(header == outcome)
        if len(idx_outcome) == 0:
            idx_outcome = np.where(header == outcome+self.delim+outcome)
        y_r = np.reshape(np.round(np.reshape(x_r[:,idx_outcome],
                    newshape=(len(x_r),1))).astype(int), len(x_r))
        y_s = np.reshape(np.round(np.reshape(x_s[:,idx_outcome],
                    newshape=(len(x_s),1))).astype(int), len(x_s))

        x_r = np.delete(x_r, idx_outcome, axis=1)
        x_s = np.delete(x_s, idx_outcome, axis=1)

        reg_r = LogisticRegression(max_iter=max_iter, solver=solver,
                penalty=penalty, l1_ratio=l1_ratio).fit(X=x_r, y=y_r)
        reg_s = LogisticRegression(max_iter=max_iter, solver=solver,
                penalty=penalty, l1_ratio=l1_ratio).fit(X=x_s, y=y_s)

        effect_r = self.scale(reg_r.coef_)
        effect_s = self.scale(reg_s.coef_)

        return {'effect_r':effect_r, 'effect_s':effect_s,
                'header_r':header, 'header_s':header}

    def validate_prediction(self, x_synth, y_synth, x_real, y_real,
                            do_gan_train, n_epoch=5, model_type='mlp', debug=False):
        """Validate the synthetic dataset in the GAN-train and GAN-test
        framework.

        Parameters
        ----------
        x_synth : TYPE
            DESCRIPTION.
        y_synth : TYPE
            DESCRIPTION.
        x_real : TYPE
            DESCRIPTION.
        y_real : TYPE
            DESCRIPTION.
        do_gan_train : TYPE
            DESCRIPTION.
        n_epoch : TYPE, optional
            DESCRIPTION. The default is 5.
        model_type : TYPE, optional
            DESCRIPTION. The default is 'mlp'.
        debug : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        dict
            DESCRIPTION.

        """

        if (sum(y_synth) == 0 or sum(y_synth) == len(y_synth)) and do_gan_train:
            print('Error: synthetic outcome is constant')
            return None

        if (sum(y_real) == 0 or sum(y_real) == len(y_real)) and not do_gan_train:
            print('Error: real outcome is constant')
            return None

        if do_gan_train:
            x_train = x_synth
            y_train = y_synth
            x_test = x_real
            y_test = y_real
        else:
            x_train = x_real
            y_train = y_real
            x_test = x_synth
            y_test = y_synth

        if model_type == 'mlp':

            x_train = torch.FloatTensor(x_train)
            y_train = torch.FloatTensor(y_train)
            x_test = torch.FloatTensor(x_test)
            y_test = torch.FloatTensor(y_test)

            model = MLP(input_size=x_synth.shape[1], hidden_size=256)

            criterion = torch.nn.BCELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

            model.eval()
            p_tst = model(x_test)
            before_train = criterion(p_tst.squeeze(), y_test)

            if debug:
                print('Test loss before training' , before_train.item())

            model.train()
            for epoch in range(n_epoch):
                optimizer.zero_grad()
                p_trn = model(x_train)
                loss = criterion(p_trn.squeeze(), y_train)

                if debug:
                    print('Epoch {}: train loss: {}'.format(epoch, loss.item()))

                loss.backward()
                optimizer.step()

            model.eval()
            p_tst = model(x_test).detach().cpu().numpy()

        elif model_type == 'lr':

            model = LogisticRegression(max_iter=1000,
                                       solver='liblinear',
                                       penalty='l1').fit(X=x_train, y=y_train)
            p_tst = model.predict_proba(x_test)[:,1]

        if do_gan_train:
            roc = metrics.roc_curve(y_true=y_real, y_score=p_tst)
            auc = metrics.roc_auc_score(y_true=y_real, y_score=p_tst)
        else:
            roc = metrics.roc_curve(y_true=y_synth, y_score=p_tst)
            auc = metrics.roc_auc_score(y_true=y_synth, y_score=p_tst)

        return {'mode':model, 'p':p_tst, 'roc':roc, 'auc':auc}

    def gan_train(self, x_synth, y_synth, x_real, y_real, n_epoch=5,
                  model_type='mlp', debug=False):
        """Train a predictive model with synthetic data, test in a real
        dataset, and report predictive performance.

        Parameters
        ----------
        x_synth : TYPE
            DESCRIPTION.
        y_synth : TYPE
            DESCRIPTION.
        x_real : TYPE
            DESCRIPTION.
        y_real : TYPE
            DESCRIPTION.
        n_epoch : TYPE, optional
            DESCRIPTION. The default is 5.
        model_type : TYPE, optional
            DESCRIPTION. The default is 'mlp'.
        debug : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        return self.validate_prediction(x_synth, y_synth, x_real, y_real,
                                   do_gan_train=True, n_epoch=n_epoch,
                                   model_type=model_type, debug=debug)

    def gan_test(self, x_synth, y_synth, x_real, y_real, n_epoch=5,
                 model_type='mlp', debug=False):
        """Train a predictive model with real data, test in a synthetic
        dataset, and report predictive performance.

        Parameters
        ----------
        x_synth : TYPE
            DESCRIPTION.
        y_synth : TYPE
            DESCRIPTION.
        x_real : TYPE
            DESCRIPTION.
        y_real : TYPE
            DESCRIPTION.
        n_epoch : TYPE, optional
            DESCRIPTION. The default is 5.
        model_type : TYPE, optional
            DESCRIPTION. The default is 'mlp'.
        debug : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        return self.validate_prediction(x_synth, y_synth, x_real, y_real,
                                   do_gan_train=False, n_epoch=n_epoch,
                                   model_type=model_type, debug=debug)

    def gan_train_test(self, mat_f_r_trn, mat_f_r_tst, mat_f_s, header, outcome, n_epoch=5,
                 model_type='lr'):
        """Conduct GAN-train and GAN-test validation framework.

        Parameters
        ----------
        r_trn : TYPE
            DESCRIPTION.
        r_tst : TYPE
            DESCRIPTION.
        s : TYPE
            DESCRIPTION.
        col_names : TYPE
            DESCRIPTION.
        outcome : TYPE
            DESCRIPTION.
        n_epoch : TYPE, optional
            DESCRIPTION. The default is 5.
        model_type : TYPE, optional
            DESCRIPTION. The default is 'mlp'.

        Returns
        -------
        dict
            DESCRIPTION.

        """
        
        # preprocess
        pre = Preprocessor(missing_value)
        met_f_r = pre.get_metadata(x = mat_f_r_trn, header=header)
        obj_d_r_trn = pre.get_discretized_matrix(arr=mat_f_r_trn,
                                                 meta=met_f_r,
                                                 header=header,
                                                 debug=False)
        obj_d_r_tst = pre.get_discretized_matrix(arr=mat_f_r_tst,
                                                 meta=met_f_r,
                                                 header=header,
                                                 debug=False)
        obj_d_s = pre.get_discretized_matrix(arr=mat_f_s,
                                                 meta=met_f_r,
                                                 header=header,
                                                 debug=False)
        
        # extract
        r_trn = obj_d_r_trn['x']
        r_tst = obj_d_r_tst['x']
        s_all = obj_d_s['x']

        # split synthetic dataset
        frac_train = len(r_trn) / (len(r_trn) + len(r_tst))
        n_subset_s = round(len(s_all) * frac_train)
        idx_trn = np.random.choice(len(s_all), n_subset_s, replace=False)
        idx_tst = np.setdiff1d(range(len(s_all)), idx_trn)
        s_trn = s_all[idx_trn,:]
        s_tst = s_all[idx_tst,:]

        # extract outcome for prediction tests
        idx_outcome = np.where(col_names == outcome)
        y_r_trn = np.reshape(np.round(np.reshape(r_trn[:,idx_outcome],
                            newshape=(len(r_trn),1))).astype(int), len(r_trn))
        y_r_tst = np.reshape(np.round(np.reshape(r_tst[:,idx_outcome],
                            newshape=(len(r_tst),1))).astype(int), len(r_tst))
        y_s_trn = np.reshape(np.round(np.reshape(s_trn[:,idx_outcome],
                            newshape=(len(s_trn),1))).astype(int), len(s_trn))
        y_s_tst = np.reshape(np.round(np.reshape(s_tst[:,idx_outcome],
                            newshape=(len(s_tst),1))).astype(int), len(s_tst))

        # extract features for prediction tests
        x_r_trn = np.delete(r_trn, idx_outcome, axis=1)
        x_r_tst = np.delete(r_tst, idx_outcome, axis=1)
        x_s_trn = np.delete(s_trn, idx_outcome, axis=1)
        x_s_tst = np.delete(s_tst, idx_outcome, axis=1)

        # conduct res gan-train, gan-test comparisons
        res_gan_real = self.gan_train(x_synth=x_r_trn, y_synth=y_r_trn,
                                         x_real=x_r_tst, y_real=y_r_tst,
                                         n_epoch=n_epoch,
                                         model_type=model_type)
        res_gan_train = self.gan_train(x_synth=x_s_trn, y_synth=y_s_trn,
                                      x_real=x_r_tst, y_real=y_r_tst,
                                      n_epoch=n_epoch,
                                      model_type=model_type)
        res_gan_test = self.gan_test(x_synth=x_s_tst, y_synth=y_s_tst,
                                    x_real=x_r_tst, y_real=y_r_tst,
                                    n_epoch=n_epoch,
                                    model_type=model_type)

        return {'gan_real':res_gan_real, 'gan_train':res_gan_train, 'gan_test':res_gan_test}

    def kl_divergence(self, pdf_p, pdf_q):
        """Kullback–Leibler divergence.

        Parameters
        ----------
        pdf_p : TYPE
            DESCRIPTION.
        pdf_q : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return np.sum(np.where(np.logical_and(pdf_p != 0, pdf_q != 0),
                               pdf_p * np.log(pdf_p / pdf_q), 0))

    def validate_feature(self, r_feat, s_feat, var_type,
                         categorical_metric = 'euclidean',
                         numerical_metric = 'kl'):
        """Conduct feature level validation.

        Parameters
        ----------
        r_feat : array_like
            Array of real data
        s_feat : array_like
            Array of synthetic data
        var_type : str
            Modal type of data features
        categorical_metric : str
            Metric for calculating distance between real and synthetic
            categorical feature arrays.

        Returns
        -------
        float
            Distance between real and synthetic data feature.
        """

        dist = None

        if var_type in ('constant','binary','categorical'):

            uniq_vals = np.unique(r_feat)
            r_frq = np.zeros(shape=uniq_vals.shape)
            s_frq = np.zeros(shape=uniq_vals.shape)

            for i in range(len(uniq_vals)):
                r_frq[i] = np.count_nonzero(r_feat == uniq_vals[i])
                s_frq[i] = np.count_nonzero(s_feat == uniq_vals[i])

            r_frq = r_frq / len(r_feat)
            s_frq = s_frq / len(s_feat)

            if categorical_metric == 'euclidean':
                dist = np.linalg.norm(r_frq-s_frq)

        elif var_type in ('continuous','count'):

            if numerical_metric == 'kl':

                r_pdf = norm.pdf(r_feat.astype(float))
                s_pdf = norm.pdf(s_feat.astype(float))
                dist = self.kl_divergence(r_pdf, s_pdf)

        return dist

    def feature_frequency(self, mat_f_r_trn, mat_f_r_tst, mat_f_s, header, missing_value):
        """Frequency of all features in each matrix.

        Parameters
        ----------
        mat_f_r_trn : TYPE
            DESCRIPTION.
        mat_f_r_tst : TYPE
            DESCRIPTION.
        mat_f_s : TYPE
            DESCRIPTION.
        header : TYPE
            DESCRIPTION.
        missing_value : TYPE
            DESCRIPTION.

        Returns
        -------
        dict
            DESCRIPTION.

        """

        # preprocess
        pre = Preprocessor(missing_value)
        met_f_r = pre.get_metadata(arr = mat_f_r_trn, header=header)
        obj_d_r_trn = pre.get_discretized_matrix(arr=mat_f_r_trn,
                                                 meta=met_f_r,
                                                 header=header)
        obj_d_r_tst = pre.get_discretized_matrix(arr=mat_f_r_tst,
                                                 meta=met_f_r,
                                                 header=header)
        obj_d_s = pre.get_discretized_matrix(arr=mat_f_s,
                                                 meta=met_f_r,
                                                 header=header)

        # compare r_trn and r_tst with s
        res_trn = self.validate_univariate(arr_r=obj_d_r_trn['x'],
                                           arr_s=obj_d_s['x'],
                                           header=header)
        res_tst = self.validate_univariate(arr_r=obj_d_r_tst['x'],
                                           arr_s=obj_d_s['x'],
                                           header=header)

        # combine results
        return {'frq_r_trn':res_trn['frq_r'], 'frq_s_trn':res_trn['frq_s'],
                'frq_r_tst':res_tst['frq_r'], 'frq_s_tst':res_tst['frq_s'],
                'header':obj_d_r_trn['header']}

    def feature_effect(self, mat_f_r_trn, mat_f_r_tst, mat_f_s, header, outcome, missing_value):
        """Effect size of all features in real and synthetic data matrices.

        Parameters
        ----------
        mat_f_r_trn : TYPE
            DESCRIPTION.
        mat_f_r_tst : TYPE
            DESCRIPTION.
        mat_f_s : TYPE
            DESCRIPTION.
        header : TYPE
            DESCRIPTION.
        outcome : TYPE
            DESCRIPTION.
        missing_value : TYPE
            DESCRIPTION.

        Returns
        -------
        dict
            DESCRIPTION.

        """

        # preprocess
        pre = Preprocessor(missing_value)
        met_f_r = pre.get_metadata(arr = mat_f_r_trn, header=header)
        obj_d_r_trn = pre.get_discretized_matrix(arr=mat_f_r_trn,
                                                 meta=met_f_r,
                                                 header=header)
        obj_d_r_tst = pre.get_discretized_matrix(arr=mat_f_r_tst,
                                                 meta=met_f_r,
                                                 header=header)
        obj_d_s = pre.get_discretized_matrix(arr=mat_f_s,
                                                 meta=met_f_r,
                                                 header=header)

        # compare r_trn and r_tst with s
        res_trn = self.validate_effect(arr_r=obj_d_r_trn['x'],
                                           arr_s=obj_d_s['x'],
                                           header=header,
                                           outcome=outcome)
        res_tst = self.validate_effect(arr_r=obj_d_r_tst['x'],
                                           arr_s=obj_d_s['x'],
                                           header=header,
                                           outcome=outcome)

        # combine results
        return {'effect_r_trn':res_trn['effect_r'],
                'effect_s_trn':res_trn['effect_s'],
                'effect_r_tst':res_tst['effect_r'],
                'effect_s_tst':res_tst['effect_s']}

    def plot(self, res, analysis, file_pdf):
        """Plot the results of a realism validation analysis.

        Parameters
        ----------
        res : TYPE
            DESCRIPTION.
        analysis : TYPE
            DESCRIPTION.
        file_pdf : TYPE
            DESCRIPTION.

        Returns
        -------
        bool
            DESCRIPTION.

        """

        fontsize = 6

        fig = plt.figure()

        if analysis == 'feature_frequency':

            plt.plot([0,1],[0,1], color="gray", linestyle='--')
            plt.scatter(res['frq_r_trn'], res['frq_s_trn'], label='Train')
            plt.scatter(res['frq_r_tst'], res['frq_s_tst'], label='Test')
            for i in range(len(res['header'])):
                plt.text(x=res['frq_r_trn'][i], y=res['frq_s_trn'][i],
                         s=res['header'][i])
            plt.xlabel('Real feature frequency', fontsize=fontsize)
            plt.ylabel('Synthetic feature frequency', fontsize=fontsize)
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.tick_params(axis='x', labelsize=fontsize)
            plt.tick_params(axis='y', labelsize=fontsize)
            plt.legend(fontsize=fontsize)

        elif analysis == 'feature_effect':

            plt.plot([0,1],[0,1], color="gray", linestyle='--')
            plt.scatter(res['frq_r'], res['frq_s'], label='Importance')
            plt.xlabel('Real', fontsize=fontsize)
            plt.ylabel('Synthetic', fontsize=fontsize)
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.tick_params(axis='x', labelsize=fontsize)
            plt.tick_params(axis='y', labelsize=fontsize)
            plt.legend(fontsize=fontsize)

        elif analysis == 'gan_train_test':

            plt.plot(res['roc'][0], res['roc'][1], label="Real")
            plt.plot(res['roc'][0], res['roc'][1], label="GAN-train")
            plt.plot(res['roc'][0], res['roc'][1], label="GAN-test")
            plt.plot([0,1],[0,1], color="gray", linestyle='--')
            plt.tick_params(axis='x', labelsize=fontsize)
            plt.tick_params(axis='y', labelsize=fontsize)
            plt.legend(fontsize=fontsize)
            plt.xlabel('False positive rate', fontsize=fontsize)
            plt.ylabel('True positive rate', fontsize=fontsize)

        fig.savefig(file_pdf, bbox_inches='tight')
        return True

    def summarize(self, res, analysis, n_decimal=2):
        """Create a summary of a realism validation analysis.

        Parameters
        ----------
        res : TYPE
            DESCRIPTION.
        analysis : TYPE
            DESCRIPTION.
        n_decimal : TYPE, optional
            DESCRIPTION. The default is 2.

        Returns
        -------
        msg : TYPE
            DESCRIPTION.

        """

        msg = '\nSummary:'

        if analysis == 'feature_frequency':
            corr_trn = np.corrcoef(x=res['frq_r_trn'], y=res['frq_s_trn'])[0,1]
            msg = msg + '\nFrequency correlation (train): ' + str(np.round(corr_trn, n_decimal))
            corr_tst = np.corrcoef(x=res['frq_r_tst'], y=res['frq_s_tst'])[0,1]
            msg = msg + '\nFrequency correlation (test): ' + str(np.round(corr_tst, n_decimal))

        elif analysis == 'feature_effect':

            corr_trn = np.corrcoef(x=res['effect_r_trn'], y=res['effect_s_trn'])[0,1]
            msg = msg + '\nImportance correlation: ' + str(np.round(corr_trn, n_decimal))
            corr_tst = np.corrcoef(x=res['effect_r_tst'], y=res['effect_s_tst'])[0,1]
            msg = msg + '\nImportance correlation: ' + str(np.round(corr_tst, n_decimal))

        elif analysis == 'gan_train_test':
            msg = msg + '\nRealism assessment: ' + \
                    '\n  > Real AUC: ' + \
                    str(np.round(res['gan_real']['auc'], n_decimal)) + \
                    '\n  > GAN-train AUC: ' + \
                    str(np.round(res['gan_train']['auc'], n_decimal)) + \
                    '\n  > GAN-test AUC: ' + \
                    str(np.round(res['gan_test']['auc'], n_decimal))
        else:
            msg = 'Warning: summary message for analysis \'' + analysis + \
            '\' not currently implemented in realism::summarize().'

        return msg
