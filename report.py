import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from realism import realism
from privacy import privacy
from sklearn.linear_model import LogisticRegression

class report(object):
    
    def __init__(self, missing_value):
        self.missing_value=missing_value
    
    def make_report(self, r_trn, r_tst, s, col_names, file_pdf, outcome=None,  
                          dist_metric='euclidean', n_epoch=5, model_type='mlp',
                          report_type='prediction', n_nn_sample=100):
        
        # check user input
        if outcome is None and (report_type=='prediction' or report_type=='description'):
            print('\nError: outcome must be specified for prediction or description report.')
            return False
        if outcome is not None and len(np.where(col_names==outcome)) == 0:
            print('\nError: outcome ', outcome, ' not a recognized feature.')
            return False
        if len(col_names) != r_trn.shape[1]:
            print('\nError: length of col_names must match length of r')
            return False
        if r_trn.shape[1] != s.shape[1]:
            print('\nError: number of columns in r and s must match')
            return False
        
        rea = realism(self.missing_value)
        pri = privacy()
                
        # extract features and outcome for prediction tests
        idx_outcome = np.where(col_names == outcome)
        y_r_trn = np.reshape(np.round(np.reshape(r_trn[:,idx_outcome], newshape=(len(r_trn),1))).astype(int), len(r_trn))
        y_r_tst = np.reshape(np.round(np.reshape(r_tst[:,idx_outcome], newshape=(len(r_tst),1))).astype(int), len(r_tst))
        y_s = np.reshape(np.round(np.reshape(s[:,idx_outcome], newshape=(len(s),1))).astype(int), len(s))
        x_r_trn = np.delete(r_trn, idx_outcome, axis=1)
        x_r_tst = np.delete(r_tst, idx_outcome, axis=1)
        x_s = np.delete(s, idx_outcome, axis=1)
        
        # univariate
        res_uni = rea.validate_univariate(r_tst, s, col_names, discretized=True)
        corr_uni = np.corrcoef(x=res_uni['frq_r'], y=res_uni['frq_s'])[0,1]
        
        # nearest neighbors
        idx_r = np.random.randint(low=0, high=len(r_trn), size=min((len(r_trn), n_nn_sample)))
        idx_s = np.random.randint(low=0, high=len(s), size=min((len(s), n_nn_sample)))
        res_nn = pri.assess_memorization(r_trn[idx_r,:], s[idx_s,:], metric=dist_metric)
        
        # real-real, gan-train, gan-test
        if report_type == 'prediction' or report_type == 'description':
            res_gan_real = rea.gan_train(x_synth=x_r_trn, y_synth=y_r_trn, 
                                         x_real=x_r_tst, y_real=y_r_tst, 
                                         n_epoch=n_epoch, 
                                         model_type=model_type)
            res_gan_train = rea.gan_train(x_synth=x_s, y_synth=y_s, 
                                          x_real=x_r_tst, y_real=y_r_tst, 
                                          n_epoch=n_epoch, 
                                          model_type=model_type)
            res_gan_test = rea.gan_test(x_synth=x_s, y_synth=y_s, 
                                        x_real=x_r_tst, y_real=y_r_tst, 
                                        n_epoch=n_epoch, 
                                        model_type=model_type)
            
            if res_gan_real is None or res_gan_train is None or res_gan_test is None:
                return False
            
        # regression
        if report_type == 'description':
            reg_r = LogisticRegression(max_iter=10000).fit(X=x_r_tst, y=y_r_tst)
            reg_s = LogisticRegression(max_iter=10000).fit(X=x_s, y=y_s)
            coef_r = (reg_r.coef_ - np.min(reg_r.coef_)) / (np.max(reg_r.coef_) - np.min(reg_r.coef_))
            coef_s = (reg_s.coef_ - np.min(reg_s.coef_)) / (np.max(reg_s.coef_) - np.min(reg_s.coef_))
        
        with PdfPages(file_pdf) as pdf:
            
            fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2)
            #fig.suptitle("Prediction report")
            
            fontsize = 6
            color = 'gray'
            x_buffer = 0.1
            y_buffer = 0.075
            m_buffer = 1.5
            n_decimal = 2
            
            if report_type == 'prediction': 
                ax0.set_title('Prediction report')
            elif report_type == 'description':
                ax0.set_title('Description report')
            
            msgs = ['Real training data: '+str(r_trn.shape),
                    'Real testing data: '+str(r_tst.shape),
                    'Synthetic: '+str(s.shape),
                    'Frequency correlation: '+str(np.round(corr_uni,n_decimal)),
                    'Mean nearest neighbor distance: ',
                    '  > Real-real: '+str(np.round(np.mean(res_nn['real']),n_decimal)),
                    '  > Real-synthetic: '+str(np.round(np.mean(res_nn['synth']),n_decimal)),
                    '  > Real-probabilistic: '+str(np.round(np.mean(res_nn['prob']),n_decimal)),
                    '  > Real-random: '+str(np.round(np.mean(res_nn['rand']),n_decimal)),
                    'Realism assessment: ',
                    '  > Real AUC: '+str(np.round(res_gan_real['auc'],n_decimal)),
                    '  > GAN-train AUC: '+str(np.round(res_gan_train['auc'],n_decimal)),
                    '  > GAN-test AUC: '+str(np.round(res_gan_test['auc'],n_decimal))]

            ax0.set_xticks([])
            ax0.set_yticks([])
            for i in range(len(msgs)):
                ax0.text(x_buffer, 1+y_buffer/2-y_buffer*(i+m_buffer), 
                     msgs[i], fontsize=fontsize, color=color)
            
            ax1.plot([0,1],[0,1], color="gray", linestyle='--')
            ax1.scatter(res_uni['frq_r'], res_uni['frq_s'], label='Frequency')
            if(report_type == 'description'):
                ax1.scatter(coef_r, coef_s, label='Importance')
            ax1.set_xlabel('Real', fontsize=fontsize)
            ax1.set_ylabel('Synthetic', fontsize=fontsize)
            ax1.set_xlim([0, 1])
            ax1.set_ylim([0, 1])
            ax1.tick_params(axis='x', labelsize=fontsize)
            ax1.tick_params(axis='y', labelsize=fontsize)
            ax1.legend(fontsize=fontsize)
            
            # memorization
            ax2.hist((res_nn['real'], res_nn['synth'], 
                      res_nn['prob'], res_nn['rand']),
                     bins=30, 
                     label = ['Real-real','Real-synthetic','Real-probabilistic','Real-random'])
            ax2.set_xlabel(dist_metric.capitalize()+' distance', fontsize=fontsize)
            ax2.set_ylabel('Number of samples', fontsize=fontsize)
            ax2.tick_params(axis='x', labelsize=fontsize)
            ax2.tick_params(axis='y', labelsize=fontsize)
            ax2.legend(fontsize=fontsize)
           
            # gan-train,etc. 
            ax3.plot(res_gan_real['roc'][0], res_gan_real['roc'][1], label="Real")
            ax3.plot(res_gan_train['roc'][0], res_gan_train['roc'][1], label="GAN-train")
            ax3.plot(res_gan_test['roc'][0], res_gan_test['roc'][1], label="GAN-test")
            ax3.plot([0,1],[0,1], color="gray", linestyle='--')
            ax3.tick_params(axis='x', labelsize=fontsize)
            ax3.tick_params(axis='y', labelsize=fontsize)
            ax3.legend(fontsize=fontsize)
            ax3.set_xlabel('False positive rate', fontsize=fontsize)
            ax3.set_ylabel('True positive rate', fontsize=fontsize)
            
            plt.tight_layout()
            pdf.savefig()
            plt.close(fig)
        
        return True
    
    def prediction_report(self, r_trn, r_tst, s, col_names, outcome, file_pdf, 
                          dist_metric='euclidean', n_epoch=5, model_type='mlp',
                          n_nn_sample=100):
        
        return self.make_report(r_trn=r_trn,
                                r_tst=r_tst,
                                s=s, 
                                col_names=col_names, 
                                file_pdf=file_pdf, 
                                outcome=outcome, 
                                dist_metric=dist_metric, 
                                n_epoch=n_epoch,
                                model_type=model_type,
                                n_nn_sample=n_nn_sample,
                                report_type='prediction')
        
    
    def description_report(self, r_trn, r_tst, s, col_names, outcome, file_pdf, 
                          dist_metric='euclidean', n_epoch=5, model_type='mlp',
                          n_nn_sample=100):
        
        return self.make_report(r_trn=r_trn,
                                r_tst=r_tst,
                                s=s, 
                                col_names=col_names, 
                                file_pdf=file_pdf, 
                                outcome=outcome, 
                                dist_metric=dist_metric, 
                                n_epoch=n_epoch, 
                                model_type=model_type,
                                n_nn_sample=n_nn_sample,
                                report_type='description')
    