import numpy as np
from sklearn import metrics
from sklearn.neighbors import DistanceMetric
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# sehrd modules
from validator import Validator
from corgan import Corgan
from corgan import Discriminator

class Privacy(Validator):
    """Validates the privacy preserving properties of the synthetic data.
    """

    def distance(self, arr1, arr2, metric='euclidean'):
        """Calculate a distance metric between two vectors.

        Parameters
        ----------
        arr1 : array_like
            Numeric array.
        arr2 : array_like
            Another numeric array.
        metric : str
            Label for distance metric to calculate.

        Returns
        -------
        float
            Distance between the two arrays.
        """

        dist = DistanceMetric.get_metric(metric)

        if len(arr1.shape) == 1:
            arr1 = np.reshape(arr1, newshape=(1,len(arr1)))
        if len(arr2.shape) == 1:
            arr2 = np.reshape(arr2, newshape=(1,len(arr2)))

        return dist.pairwise(X=arr1, Y=arr2)

    def nearest_neighbors(self, arr1, arr2=None, metric='euclidean'):
        """Calculate a nearest neighbor distance.

        Parameters
        ----------
        arr1 : array_like
            Numeric array.
        arr2 : array_like
            Another numeric array.
        metric : str
            Label for distance metric to calculate.

        Returns
        -------
        array_like
            Array of the distance to the nearest neighbor for each
            column of arr1.
        """

        d_min = np.full(shape=len(arr1), fill_value=float('inf'))

        if arr2 is None:

            for i in range(len(arr1)):
                d_i = np.full(shape=len(arr1), fill_value=float('inf'))

                for j in range(len(arr1)):
                    if i != j:
                        d_i[j] = self.distance(arr1[i,:], arr1[j,:], metric)

                d_min[i] = np.min(d_i)

        else:

            for i in range(len(arr1)):
                d_i = np.full(shape=len(arr2), fill_value=float('inf'))

                for j in range(len(arr2)):
                    d_i[j] = self.distance(arr1[i,:], arr2[j,:], metric)

                d_min[i] = np.min(d_i)

        return d_min

    def assess_memorization(self, x_real, x_synth, metric='euclidean'):
        """Calculate the distribution of nearest neighbors.

        Parameters
        ----------
        x_real : array_like
            Array of real .
        x_synth : array_like
            DESCRIPTION.
        metric : str, optional
            DESCRIPTION. The default is 'euclidean'.

        Returns
        -------
        dict
            DESCRIPTION.

        """

        # real to real
        nn_real = self.nearest_neighbors(arr1=x_real, metric=metric)

        # real to synth
        nn_synth = self.nearest_neighbors(arr1=x_real, arr2=x_synth, metric=metric)

        # real to probabilistically sampled
        x_prob = np.full(shape=x_real.shape, fill_value=0)
        for j in range(x_real.shape[1]):
            x_prob[:,j] = np.random.binomial(n=1, p=np.mean(x_real[:,j]), size=x_real.shape[0])
        nn_prob = self.nearest_neighbors(arr1=x_real, arr2=x_prob, metric=metric)

        # real to noise
        x_rand = np.random.randint(low=0, high=2, size=x_real.shape)
        nn_rand = self.nearest_neighbors(arr1=x_real, arr2=x_rand, metric=metric)

        return {'real':nn_real, 'synth':nn_synth, 'prob':nn_prob, 'rand':nn_rand, 'metric':metric}

    def membership_inference_hayes(self, r_trn, r_tst, s_all, n_cpu):
        """Membership inference scenario as in Hayes et al. 2018.

        Parameters
        ----------
        r_trn : TYPE
            DESCRIPTION.
        r_tst : TYPE
            DESCRIPTION.
        s_all : TYPE
            DESCRIPTION.
        n_cpu : TYPE
            DESCRIPTION.

        Returns
        -------
        dict
            DESCRIPTION.

        """

        cor = Corgan()

        # evaluation set
        x_all = np.row_stack((r_tst,r_trn))
        y_all = np.append(np.zeros(len(r_tst)), np.ones(len(r_trn)))

        # train shadow GAN
        gan_shadow = cor.train(x=s_all, n_cpu=n_cpu)

        # load shadow discriminator
        minibatch_averaging = gan_shadow['parameter_dict']['minibatch_averaging']
        feature_size = gan_shadow['parameter_dict']['feature_size']
        d_shadow = Discriminator(minibatch_averaging=minibatch_averaging,
                                 feature_size=feature_size)
        d_shadow.load_state_dict(gan_shadow['Discriminator_state_dict'])
        d_shadow.eval()

        # calculate probabilities from shadow discriminator
        p_all = d_shadow(x_all)

        roc = metrics.roc_curve(y_true=y_all, y_score=p_all)
        auc = metrics.roc_auc_score(y_true=y_all, y_score=p_all)

        return {'prob': p_all, 'label':y_all, 'roc':roc, 'auc':auc}

    def membership_inference_torfi(self, r_trn, r_tst, s_all):
        """Membership inference scenario as in Torfi et al. 2020.

        Parameters
        ----------
        r_trn : TYPE
            DESCRIPTION.
        r_tst : TYPE
            DESCRIPTION.
        s_all : TYPE
            DESCRIPTION.

        Returns
        -------
        dict
            DESCRIPTION.

        """

        # store all pairwise distances
        d_trn = cdist(r_trn, s_all, metric='cosine').min(axis=1)
        d_tst = cdist(r_tst, s_all, metric='cosine').min(axis=1)

        # scale distances to get 'probabilities'
        p_trn = self.scale(d_trn, invert=True)
        p_tst = self.scale(d_tst, invert=True)

        # calculate performance metrics
        p_all = np.append(p_trn, p_tst)
        y_all = np.append(np.ones(len(p_trn)), np.zeros(len(p_tst)))
        roc = metrics.roc_curve(y_true=y_all, y_score=p_all)
        auc = metrics.roc_auc_score(y_true=y_all, y_score=p_all)

        return {'prob': p_all, 'label':y_all, 'roc':roc, 'auc':auc}

    def membership_inference(self, r_trn, r_tst, s_all, mi_type='hayes', n_cpu=1):
        """Membership inference wrapper function.

        Parameters
        ----------
        r_trn : TYPE
            DESCRIPTION.
        r_tst : TYPE
            DESCRIPTION.
        s_all : TYPE
            DESCRIPTION.
        mi_type : TYPE, optional
            DESCRIPTION. The default is 'hayes'.
        n_cpu : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        if mi_type == 'hayes':
            return self.membership_inference_hayes(r_trn, r_tst, s_all, n_cpu=n_cpu)

        if mi_type == 'torfi':
            return self.membership_inference_torfi(r_trn, r_tst, s_all)

        return None

    def plot(self, res, analysis, file_pdf):
        """Plot the results of a privacy validation analysis.

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

        if analysis == 'nearest_neighbors':

            # plot distributions of distances
            plt.hist((res['real'], res['synth'],
                      res['prob'], res['rand']),
                     bins=30,
                     label = ['Real-real','Real-synthetic','Real-probabilistic','Real-random'])
            plt.xlabel(res['metric'].capitalize()+' distance', fontsize=fontsize)
            plt.ylabel('Number of samples', fontsize=fontsize)
            plt.tick_params(axis='x', labelsize=fontsize)
            plt.tick_params(axis='y', labelsize=fontsize)
            plt.legend(fontsize=fontsize)

        elif analysis == 'membership_inference':

            # plot ROC curve
            plt.plot(res['roc'][0], res['roc'][1], label="Real")
            plt.plot([0,1],[0,1], color="gray", linestyle='--')
            plt.tick_params(axis='x', labelsize=fontsize)
            plt.tick_params(axis='y', labelsize=fontsize)
            plt.xlabel('False positive rate', fontsize=fontsize)
            plt.ylabel('True positive rate', fontsize=fontsize)
            plt.title('AUC = ' + res['auc'])

        else:
            msg = 'Warning: plot for analysis \'' + analysis + \
                '\' not currently implemented in privacy::plot().'
            print(msg)

        plt.show()
        fig.savefig(file_pdf, bbox_inches='tight')
        return True

    def summarize(self, res, analysis, n_decimal=2):
        """Create a summary of a privacy validation analysis.

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

        msg = ''
        n_decimal = 2

        if analysis == 'nearest_neighbors':
            msg = 'Mean nearest neighbor distance: ' + \
                    '  > Real-real: ' + \
                    str(np.round(np.mean(res['real']),n_decimal)) + \
                    '  > Real-synthetic: ' + \
                    str(np.round(np.mean(res['synth']),n_decimal)) + \
                    '  > Real-probabilistic: ' + \
                    str(np.round(np.mean(res['prob']),n_decimal)) + \
                    '  > Real-random: ' + \
                    str(np.round(np.mean(res['rand']),n_decimal))

        elif analysis == 'membership_inference':
            msg = 'AUC for auxiliary-synthetic: ' + res['auc_as'] + \
                'AUC for real train-test: ' + res['auc_rr']
        else:
            msg = 'Warning: summary message for analysis \'' + analysis + \
            '\' not currently implemented in privacy::summarize().'

        return msg
