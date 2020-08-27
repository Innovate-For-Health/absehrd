import numpy as np
from preprocessor import preprocessor
import torch
from sklearn import metrics

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
        
    
    def validate_prediction(self, x_synth, y_synth, x_real, y_real, do_gan_train, n_epoch=5, debug=False):
        
        if (sum(y_synth) == 0 or sum(y_synth) == len(y_synth)) and do_gan_train:
            print('Error: synthetic outcome is constant')
            return None
        
        if (sum(y_real) == 0 or sum(y_real) == len(y_real)) and not do_gan_train:
            print('Error: real outcome is constant')
            return None
        
        model = mlp(input_size=x_synth.shape[1], hidden_size=256)
        
        if do_gan_train:
            x_train = torch.FloatTensor(x_synth)
            y_train = torch.FloatTensor(y_synth)
            x_test = torch.FloatTensor(x_real)
            y_test = torch.FloatTensor(y_real)
        else: 
            x_train = torch.FloatTensor(x_real)
            y_train = torch.FloatTensor(y_real)
            x_test = torch.FloatTensor(x_synth)
            y_test = torch.FloatTensor(y_synth)
        
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
       
        if do_gan_train:
            roc = metrics.roc_curve(y_true=y_real, y_score=p)
            auc = metrics.roc_auc_score(y_true=y_real, y_score=p)
        else:
            roc = metrics.roc_curve(y_true=y_synth, y_score=p)
            auc = metrics.roc_auc_score(y_true=y_synth, y_score=p)
        
        return {'mode':model, 'p':p, 'roc':roc, 'auc':auc}
    
    def gan_train(self, x_synth, y_synth, x_real, y_real, n_epoch=5, debug=False):
        return self.validate_prediction(x_synth, y_synth, x_real, y_real, 
                                   do_gan_train=True, n_epoch=n_epoch, debug=debug)
    
    def gan_test(self, x_synth, y_synth, x_real, y_real, n_epoch=5, debug=False):
        return self.validate_prediction(x_synth, y_synth, x_real, y_real, 
                                   do_gan_train=False, n_epoch=n_epoch, debug=debug)
    
    