import numpy as np
from sklearn import metrics
from sklearn.neighbors import DistanceMetric
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
from tqdm import tqdm

# sehrd modules
from validator import Validator
from corgan import Corgan
from corgan import Discriminator
from preprocessor import Preprocessor

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
            d_all = dist.squareform(dist.pdist(arr1, metric=metric))
            np.fill_diagonal(d_all, np.inf)
            d_min = d_all.min(axis=1)
        else:
            d_min = dist.cdist(arr1, arr2, metric=metric).min(axis=1)

        return d_min

    def assess_memorization(self, mat_f_r, mat_f_s, missing_value, header,
                            metric='euclidean', debug=False):
        """Calculate the distribution of nearest neighbors.

        Parameters
        ----------
        mat_f_r : array_like
            Realistically formatted matrix of real data.
        x_synth : array_like
            Realistically formatted matrix of synthetic data.
        metric : str, optional
            Distance metric label. The default is 'euclidean'.

        Returns
        -------
        dict
            DESCRIPTION.

        """
        
        if debug:
            newline = '\n'
        
        # preprocess
        pre = Preprocessor(missing_value)
        met_f_r = pre.get_metadata(arr = mat_f_r, header=header)
        obj_d_r = pre.get_discretized_matrix(arr=mat_f_r,
                                                 meta=met_f_r,
                                                 header=header)
        obj_d_s = pre.get_discretized_matrix(arr=mat_f_s,
                                                 meta=met_f_r,
                                                 header=header)
        x_real = obj_d_r['x']
        x_synth = obj_d_s['x']


        # real train to real train
        if debug:
            print(newline+'Real - real:', flush=True)
        nn_real = self.nearest_neighbors(arr1=x_real, metric=metric)

        # real to synth
        if debug:
            print(newline+'Real - synthetic:', flush=True)
        nn_synth = self.nearest_neighbors(arr1=x_real, arr2=x_synth, metric=metric)

        # real to probabilistically sampled
        if debug:
            print(newline+'Real - probabilistic:', flush=True)
        x_prob = np.full(shape=x_real.shape, fill_value=0)
        for j in range(x_real.shape[1]):
            x_prob[:,j] = np.random.binomial(n=1, p=np.mean(x_real[:,j]), size=x_real.shape[0])
        nn_prob = self.nearest_neighbors(arr1=x_real, arr2=x_prob, metric=metric)

        # real to noise
        if debug:
            print(newline+'Real - noise:', flush=True)        
        x_rand = np.random.randint(low=0, high=2, size=x_real.shape)
        nn_rand = self.nearest_neighbors(arr1=x_real, arr2=x_rand, metric=metric)

        return {'real':nn_real, 
                'synth':nn_synth, 
                'prob':nn_prob, 
                'rand':nn_rand, 
                'metric':metric,
                'analysis':'nearest_neighbors'}

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
        gan_shadow = cor.train(x=s_all, n_cpu=n_cpu, debug=True)

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

        return {'prob': p_all, 
                'label':y_all, 
                'roc':roc, 
                'auc':auc,
                'analysis':'membership_inference'}

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
        d_trn = dist.cdist(r_trn, s_all, metric='cosine').min(axis=1)
        d_tst = dist.cdist(r_tst, s_all, metric='cosine').min(axis=1)

        # scale distances to get 'probabilities'
        p_all = self.scale(np.append(d_trn, d_tst), invert=True)

        # calculate performance metrics
        y_all = np.append(np.ones(len(r_trn)), np.zeros(len(r_tst)))
        roc = metrics.roc_curve(y_true=y_all, y_score=p_all)
        auc = metrics.roc_auc_score(y_true=y_all, y_score=p_all)

        return {'prob': p_all, 
                'label':y_all, 
                'roc':roc, 
                'auc':auc,
                'analysis':'membership_inference'}

    def membership_inference(self, mat_f_r_trn, mat_f_r_tst, mat_f_s, header, 
                       missing_value, mi_type='torfi', n_cpu=1):
        """Membership inference wrapper function.

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
        mi_type : TYPE, optional
            DESCRIPTION. The default is 'torfi'.
        n_cpu : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        # preprocess
        pre = Preprocessor(missing_value)
        met_f_r = pre.get_metadata(arr = mat_f_r_trn, header=header)
        obj_d_r_trn = pre.get_discretized_matrix(arr=mat_f_r_trn,
                                                 meta=met_f_r,
                                                 header=header, 
                                                 require_missing=True)
        obj_d_r_tst = pre.get_discretized_matrix(arr=mat_f_r_tst,
                                                 meta=met_f_r,
                                                 header=header, 
                                                 require_missing=True)
        obj_d_s = pre.get_discretized_matrix(arr=mat_f_s,
                                                 meta=met_f_r,
                                                 header=header, 
                                                 require_missing=True)
        
        r_trn = obj_d_r_trn['x']
        r_tst = obj_d_r_tst['x']
        s_all = obj_d_s['x']


        """
        if mi_type == 'hayes':
            return self.membership_inference_hayes(r_trn, r_tst, s_all, n_cpu=n_cpu)
        """

        if mi_type == 'torfi':
            return self.membership_inference_torfi(r_trn, r_tst, s_all)

        return None

    def plot(self, res, file_pdf=None, n_decimal=2, fontsize=14):
        """Plot the results of a privacy validation analysis.

        Parameters
        ----------
        res : TYPE
            DESCRIPTION.
        file_pdf : TYPE, optional
            If specified, plot is saved to a PDF file at the given path; 
            otherwise, plotted to standard out.
        n_decimal: int
            Number of decimal places to print for numeric text.
        fontsize: int
            Size of text for plot title, axis labels, and legends.

        Returns
        -------
        bool
            DESCRIPTION.

        """

        fig = plt.figure()

        if res['analysis'] == 'nearest_neighbors':

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

        elif res['analysis'] == 'membership_inference':

            # plot ROC curve
            plt.plot(res['roc'][0], res['roc'][1], label="Real")
            plt.plot([0,1],[0,1], color="gray", linestyle='--')
            plt.tick_params(axis='x', labelsize=fontsize)
            plt.tick_params(axis='y', labelsize=fontsize)
            plt.xlabel('False positive rate', fontsize=fontsize)
            plt.ylabel('True positive rate', fontsize=fontsize)
            plt.title('AUC = ' + str(np.round(res['auc'], n_decimal)))

        else:
            msg = 'Warning: plot for analysis \'' + res['analysis'] + \
                '\' not currently implemented in privacy::plot().'
            print(msg)

        if file_pdf is None:
            plt.show()
        else:
            fig.savefig(file_pdf, bbox_inches='tight')
            
        return True

    def summarize(self, res, n_decimal=2):
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

        msg = '\nSummary of '+res['analysis']+':'
        newline = '\n  > '

        if res['analysis'] == 'nearest_neighbors':
            msg = msg + '\n(note: average nearest neighbor distance)' + \
                    newline + 'Real-real:             ' + \
                    str(np.round(np.mean(res['real']), n_decimal)) + \
                    newline + 'Real-synthetic:        ' + \
                    str(np.round(np.mean(res['synth']), n_decimal)) + \
                    newline + 'Real-probabilistic:    ' + \
                    str(np.round(np.mean(res['prob']), n_decimal)) + \
                    newline + 'Real-random:           ' + \
                    str(np.round(np.mean(res['rand']), n_decimal))

        elif res['analysis'] == 'membership_inference':
            msg = msg + newline + 'Attack AUC: ' + \
                str(np.round(res['auc'], n_decimal))
        else:
            msg = 'Warning: summary message for analysis \'' + res['analysis'] + \
            '\' not currently implemented in privacy::summarize().'

        return msg
