import numpy as np
from preprocessor import preprocessor as pre
import pytorch as torch

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
    
    def validate_univariate(r, s, header):
        
        m_r = pre.get_metadata(pre, r, header)
        m_s = pre.get_metadata(pre, s, header)
        
        d_r = pre.get_discretized_matrix(pre, r, m_r, header)
        d_s = pre.get_discretized_matrix(pre, s, m_s, header)
        
        frq_r = np.zeros(shape=d_r['x'].shape[1])
        for j in range(d_r['x'].shape[1]):
            frq_r[j] = np.mean(d_r['x'][:,j])
        
        frq_s = np.zeros(shape=d_s['x'].shape[1])
        for j in range(d_s['x'].shape[1]):
            frq_s[j] = np.mean(d_s['x'][:,j])
            
        return {'frq_r':frq_r, 'frq_s':frq_s, 
                'header_r':d_r['header'], 'header_s':d_s['header']}
        
    
    def gan_train(x_synth, y_synth, x_real, y_real):
        
        model = mlp(input_size=x_synth.shape[1], hidden_size=256)
        
        
        x_train = torch.FloatTensor(x_synth)
        y_train = torch.FloatTensor(y_synth)
        x_test = torch.FloatTensor(x_real)
        y_test = torch.FloatTensor(y_real)
        
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
        
        model.eval()
        y_pred = model(x_real)
        before_train = criterion(y_pred.squeeze(), y_test)
        
        return None
    
    def gan_test():
        return None
    
    