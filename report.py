import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from realism import realism
from privacy import privacy
from sklearn.linear_model import LogisticRegression

class report(object):
    
    def make_report(r, s, col_names, file_pdf, outcome=None,  
                          dist_metric='euclidean', n_epoch=5, 
                          type='prediction'):
        
        # check user input
        if outcome is None and (type=='prediction' or type=='description'):
            print('\nError: outcome must be specified for prediction or description report.')
            return False
        if outcome is not None and len(np.where(col_names==outcome)) == 0:
            print('\nError: outcome ', outcome, ' not a recognized feature.')
            return False
        if len(col_names) != r.shape[1]:
            print('\nError: length of col_names must match length of r')
            return False
        if r.shape[1] != s.shape[1]:
            print('\nError: number of columns in r and s must match')
            return False
                
        # extract features and outcome for prediction tests
        idx_outcome = np.where(col_names == outcome)
        y_r = np.round(np.reshape(r[:,idx_outcome], newshape=(len(r),1))).astype(int)
        y_s = np.round(np.reshape(s[:,idx_outcome], newshape=(len(r),1))).astype(int)
        x_r = np.delete(r, idx_outcome, axis=1)
        x_s = np.delete(s, idx_outcome, axis=1)
        
        # univariate
        res_uni = realism.validate_univariate(r, s, col_names)
        corr_uni = np.corrcoef(x=res_uni['frq_r'], y=res_uni['frq_s'])[0,1]
        
        # nearest neighbors
        res_nn = privacy.assess_memorization(privacy, r, s, metric=dist_metric)
        
        # real-real, gan-train, gan-test
        if type == 'prediction' or type == 'description':
            res_gan_real = realism.gan_train(realism, x_r, y_r, x_r, y_r, n_epoch=n_epoch)
            res_gan_train = realism.gan_train(realism, x_s, y_s, x_r, y_r, n_epoch=n_epoch)
            res_gan_test = realism.gan_test(realism, x_s, y_s, x_r, y_r, n_epoch=n_epoch)
            
            if res_gan_real is None or res_gan_train is None or res_gan_test is None:
                return False
            
        # regression
        if type == 'description':
            reg_r = LogisticRegression().fit(X=x_r, y=y_r)
            reg_s = LogisticRegression().fit(X=x_s, y=y_s)
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
            
            msgs = ['Real: '+str(r.shape),
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
            ax0.set_title('Prediction report')
            for i in range(len(msgs)):
                ax0.text(x_buffer, 1-y_buffer*(i+m_buffer), 
                     msgs[i], fontsize=fontsize, color=color)
            
            ax1.plot([0,1],[0,1], color="gray", linestyle='--')
            ax1.scatter(res_uni['frq_r'], res_uni['frq_s'], label='Frequency')
            if(type == 'description'):
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
    
    def prediction_report(self, r, s, col_names, outcome, file_pdf, 
                          dist_metric='euclidean', n_epoch=5):
        
        return self.make_report(r=r, 
                                s=s, 
                                col_names=col_names, 
                                file_pdf=file_pdf, 
                                outcome=outcome, 
                                dist_metric=dist_metric, 
                                n_epoch=n_epoch, 
                                type='prediction')
        
    
    def description_report(self, r, s, col_names, outcome, file_pdf, 
                          dist_metric='euclidean', n_epoch=5):
        
        return self.make_report(r=r, 
                                s=s, 
                                col_names=col_names, 
                                file_pdf=file_pdf, 
                                outcome=outcome, 
                                dist_metric=dist_metric, 
                                n_epoch=n_epoch, 
                                type='description')
    
    def clustering_report(self, r, s, col_names, file_pdf, 
                          dist_metric='euclidean', n_epoch=5):
        
        return self.make_report(r=r, 
                                s=s, 
                                col_names=col_names, 
                                file_pdf=file_pdf, 
                                outcome=None, 
                                dist_metric=dist_metric, 
                                n_epoch=n_epoch, 
                                type='clustering')
    