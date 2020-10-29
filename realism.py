import numpy as np
import torch
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm
import matplotlib.pyplot as plt

# sehrd
from preprocessor import preprocessor
from validator import Validator

class mlp(torch.nn.Module):
        def __init__(self, input_size, hidden_size):
            super(mlp, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(self.hidden_size, 1)
            self.sigmoid = torch.nn.Sigmoid()
            
        def forward(self, x):
            hidden = self.fc1(x)
            relu = self.relu(hidden)
            output = self.fc2(relu)
            output = self.sigmoid(output)
            return output

class realism(Validator):
    
    def __init__(self):
         self.delim = '__'
    
    def validate_univariate(self, d_r, d_s, header):
                
        frq_r = np.zeros(shape=d_r['x'].shape[1])
        for j in range(d_r['x'].shape[1]):
            frq_r[j] = np.mean(d_r['x'][:,j])
        
        frq_s = np.zeros(shape=d_s['x'].shape[1])
        for j in range(d_s['x'].shape[1]):
            frq_s[j] = np.mean(d_s['x'][:,j])
            
        return {'frq_r':frq_r, 'frq_s':frq_s, 
                'header_r':d_r['header'], 'header_s':d_s['header']}
    
    def validate_effect(self, d_r, d_s, outcome):
        
        max_iter = 1000000
        l1_ratio = None
        penalty == 'l2'
        solver = 'lbfgs'

        x_r = d_r['x']
        x_s = d_s['x']
        
        idx_outcome = np.where(d_r['header'] == outcome)
        if len(idx_outcome) == 0:
            idx_outcome = np.where(d_r['header'] == outcome+delim+outcome)
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
        
        coef_r = self.scale(reg_r.coef_)
        coef_s = self.scale(reg_s.coef_)

        return {'effect_r':effect_r, 'effect_s':effect_s, 
                'header_r':d_r['header'], 'header_s':d_s['header']}
        
    
    def validate_prediction(self, x_synth, y_synth, x_real, y_real, 
                            do_gan_train, n_epoch=5, model_type='mlp', debug=False):
        
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
            
            model = mlp(input_size=x_synth.shape[1], hidden_size=256)
            
            criterion = torch.nn.BCELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
            
            model.eval()
            p = model(x_test)
            before_train = criterion(p.squeeze(), y_test)
            
            if debug:
                print('Test loss before training' , before_train.item())
            
            model.train()
            for epoch in range(n_epoch):
                optimizer.zero_grad()
                p = model(x_train)
                loss = criterion(p.squeeze(), y_train)
               
                if debug:
                    print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
                    
                loss.backward()
                optimizer.step()
            
            model.eval()
            p = model(x_test).detach().cpu().numpy()
        
        elif model_type == 'lr':
            
            model = LogisticRegression(max_iter=1000, solver='liblinear', penalty='l1').fit(X=x_train, y=y_train)
            p = model.predict_proba(x_test)[:,1]
       
        if do_gan_train:
            roc = metrics.roc_curve(y_true=y_real, y_score=p)
            auc = metrics.roc_auc_score(y_true=y_real, y_score=p)
        else:
            roc = metrics.roc_curve(y_true=y_synth, y_score=p)
            auc = metrics.roc_auc_score(y_true=y_synth, y_score=p)
        
        return {'mode':model, 'p':p, 'roc':roc, 'auc':auc}
    
    def gan_train(self, x_synth, y_synth, x_real, y_real, n_epoch=5, 
                  model_type='mlp', debug=False):
        
        return self.validate_prediction(x_synth, y_synth, x_real, y_real, 
                                   do_gan_train=True, n_epoch=n_epoch, 
                                   model_type=model_type, debug=debug)
    
    def gan_test(self, x_synth, y_synth, x_real, y_real, n_epoch=5, 
                 model_type='mlp', debug=False):
        
        return self.validate_prediction(x_synth, y_synth, x_real, y_real, 
                                   do_gan_train=False, n_epoch=n_epoch, 
                                   model_type=model_type, debug=debug)
    
    def gan_train_test(self, r_trn, r_tst, s, col_names, outcome, n_epoch=5, 
                 model_type='mlp'):
        
        # split synthetic dataset
        frac_train = len(r_trn) / (len(r_trn) + len(r_tst))
        n_subset_s = round(len(s) * frac_train)
        idx_trn = np.random.choice(len(s), n_subset_s, replace=False)
        idx_tst = np.setdiff1d(range(len(s)), idx_trn)
        s_trn = s[idx_trn,:]
        s_tst = s[idx_tst,:]

        # extract outcome for prediction tests
        idx_outcome = np.where(col_names == outcome)
        y_r_trn = np.reshape(np.round(np.reshape(r_trn[:,idx_outcome], newshape=(len(r_trn),1))).astype(int), len(r_trn))
        y_r_tst = np.reshape(np.round(np.reshape(r_tst[:,idx_outcome], newshape=(len(r_tst),1))).astype(int), len(r_tst))
        y_s_trn = np.reshape(np.round(np.reshape(s_trn[:,idx_outcome], newshape=(len(s_trn),1))).astype(int), len(s_trn))
        y_s_tst = np.reshape(np.round(np.reshape(s_tst[:,idx_outcome], newshape=(len(s_tst),1))).astype(int), len(s_tst))
        
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
    
    def kl_divergence(self, p, q):
        return np.sum(np.where(np.logical_and(p != 0, q != 0), p * np.log(p / q), 0))
    
    def validate_feature(self, r_feat, s_feat, var_type
                         , categorical_metric = 'euclidean'
                         , numerical_metric = 'kl'):
        
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
        
        
    def plot(self, res, analysis, file_pdf):
        
        fontsize = 6
        
        f = plt.figure()
        
        if analysis == 'feature_frequency':
        
            plt.plot([0,1],[0,1], color="gray", linestyle='--')
            plt.scatter(res['frq_r_trn'], res['frq_s_trn'], label='Train')
            plt.scatter(res['frq_r_tst'], res['frq_s_tst'], label='Test')
            plt.set_xlabel('Real feature frequency', fontsize=fontsize)
            plt.set_ylabel('Synthetic feature frequency', fontsize=fontsize)
            plt.set_xlim([0, 1])
            plt.set_ylim([0, 1])
            plt.tick_params(axis='x', labelsize=fontsize)
            plt.tick_params(axis='y', labelsize=fontsize)
            plt.legend(fontsize=fontsize)
           
        elif analysis == 'feature_effect':
            
            plt.plot([0,1],[0,1], color="gray", linestyle='--')
            plt.scatter(res['frq_r'], res['frq_s'], label='Importance')
            plt.set_xlabel('Real', fontsize=fontsize)
            plt.set_ylabel('Synthetic', fontsize=fontsize)
            plt.set_xlim([0, 1])
            plt.set_ylim([0, 1])
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
            plt.set_xlabel('False positive rate', fontsize=fontsize)
            plt.set_ylabel('True positive rate', fontsize=fontsize)

        plt.show()
        f.savefig(file_pdf, bbox_inches='tight')    
        return True
    
    def summarize(self, res, analysis, n_decimal=2):
        
        msg = ''
         
        if analysis == 'feature_frequency':
            corr = np.corrcoef(x=res['frq_r'], y=res['frq_s'])[0,1]
            msg = 'Frequency correlation: ' + str(np.round(corr, n_decimal))
        
        elif analysis == 'feature_effect':
            msg = 'Importance correlation: ' + str(np.round(corr, n_decimal))
        
        elif analysis == 'gan_train_test':
            msg = 'Realism assessment: ' + \
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
            
    def feature_frequency(self, r_trn, r_tst, s, header, missing_value):
        
        # preprocess
        pre = preprocessor(missing_value)
        m = pre.get_metadata(x = r_trn, header=header)
        d_trn = pre.get_discretized_matrix(x=r_trn, m=m, header=header,
                                           debug=False)
        d_tst = pre.get_discretized_matrix(x=r_tst, m=m, header=header,
                                           debug=False)
        d_s = pre.get_discretized_matrix(x=s, m=m, header=header,
                                           debug=False)
        
        # compare r_trn and r_tst with s
        res_trn = self.validate_univariate(d_r=d_trn, d_s=d_s, header=header)
        res_tst = self.validate_univariate(d_r=d_tst, d_s=d_s, header=header)
        
        # combine results
        return {'frq_r_trn':res_trn['frq_r'], 'frq_s_trn':res_trn['frq_s'],
                'frq_r_tst':res_tst['frq_r'], 'frq_s_tst':res_tst['frq_s']}
    
    def feature_effect(self, r_trn, r_tst, s, header, missing_value):
        
        # preprocess
        pre = preprocessor(missing_value)
        m = pre.get_metadata(x = r_trn, header=header)
        d_trn = pre.get_discretized_matrix(x=r_trn, m=m, header=header,
                                           debug=False)
        d_tst = pre.get_discretized_matrix(x=r_tst, m=m, header=header,
                                           debug=False)
        d_s = pre.get_discretized_matrix(x=s, m=m, header=header,
                                           debug=False)
        
        # compare r_trn and r_tst with s
        res_trn = self.validate_effect(d_r=d_trn, d_s=d_s, header=header)
        res_tst = self.validate_effect(d_r=d_tst, d_s=d_s, header=header)
        
        # combine results
        return {'effect_r_trn':res_trn['effect_r'], 
                'effect_s_trn':res_trn['effect_s'],
                'effect_r_tst':res_tst['effect_r'], 
                'effect_s_tst':res_tst['effect_s']}