import numpy as np
from preprocessor import preprocessor
import torch
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm

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

class realism:
    
    def __init__(self, missing_value):
        self.missing_value=missing_value
    
    def validate_univariate(self, r, s, header, discretized=False):
        
        pre=preprocessor(missing_value=self.missing_value)
        
        if discretized:
            d_r = {'x':r, 'header':header}
            d_s = {'x':s, 'header':header}
        else:
            m_r = pre.get_metadata(r, header)
            m_s = pre.get_metadata(s, header)
    
            d_r = pre.get_discretized_matrix(r, m_r, header)
            d_s = pre.get_discretized_matrix(s, m_s, header)
        
        frq_r = np.zeros(shape=d_r['x'].shape[1])
        for j in range(d_r['x'].shape[1]):
            frq_r[j] = np.mean(d_r['x'][:,j])
        
        frq_s = np.zeros(shape=d_s['x'].shape[1])
        for j in range(d_s['x'].shape[1]):
            frq_s[j] = np.mean(d_s['x'][:,j])
            
        return {'frq_r':frq_r, 'frq_s':frq_s, 
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
    
    def kl_divergence(self, p, q):
        return np.sum(np.where(p != 0, p * np.log(p / q), 0))
    
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
        
        
    
    