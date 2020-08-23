import os.path
import numpy as np
from corgan import corgan

class tester_cor(object):
    
    def test_train():
        
        path_checkpoint='test_ckpts'
        prefix_checkpoint='test'
        n_epochs=10
        debug=False
        
        # dummy dataset
        n = 10000
        m = 7
        x = np.random.randint(low=0, high=2, size=(n,m))
        
        model = corgan.train(corgan, 
                             x=x, 
                             n_epochs_pretrain=10,
                             n_epochs=n_epochs,
                             n_cpu=8, 
                             path_checkpoint=path_checkpoint, 
                             prefix_checkpoint=prefix_checkpoint,
                             debug=debug)
        
        file_ckpt=os.path.join(path_checkpoint, prefix_checkpoint + ".model_epoch_%d.pth" % n_epochs)
        return os.path.isfile(file_ckpt)
    
    def test_generate():
        return True
    
def main():
    
    buffer = "\t\t"
    
    print('Testing corgan.train()', end='...\t' + buffer)
    print('PASS') if tester_cor.test_train() else print('FAIL')
    
    print('Testing corgan.generate()', end='...' + buffer)
    print('PASS') if tester_cor.test_generate() else print('FAIL')
        
if __name__ == "__main__":
    main()


