# NOTE: must set PYTHONPATH variable for pytest to recognize local modules
# export PYTHONPATH=/my/path/to/modules
# OR
# export PYTHONPATH=$(pwd)

import numpy as np
import os

# sehrd modules
from corgan import Corgan

class TestCorgan:
    
    def test_train(self):
        
        path_checkpoint='.'
        prefix_checkpoint='test'
        n_epochs=10
        cor = Corgan()
        
        # dummy dataset
        n = 1000
        m = 7
        x = np.random.randint(low=0, high=2, size=(n,m))
        
        model = cor.train(x=x, 
                             n_epochs_pretrain=10,
                             n_epochs=10,
                             batch_size=512,
                             path_checkpoint=path_checkpoint, 
                             prefix_checkpoint=prefix_checkpoint)
        
        file_ckpt=os.path.join(path_checkpoint, prefix_checkpoint + ".model_epoch_%d.pth" % n_epochs)
        res = os.path.isfile(file_ckpt)
        
        os.remove(file_ckpt)
        
        assert res

    
    def test_generate(self):
        
        path_checkpoint='.'
        prefix_checkpoint='test'
        n_epochs=10
        cor = Corgan()
        
        # dummy dataset
        n_gen = 500
        n = 1000
        m = 7
        x = np.random.randint(low=0, high=2, size=(n,m))
        
        model = cor.train(x=x, 
                             n_epochs_pretrain=10,
                             n_epochs=10,
                             path_checkpoint=path_checkpoint, 
                             prefix_checkpoint=prefix_checkpoint)
        
        x_synth = cor.generate(model = model, n_gen=n_gen)
        
        # clean up 
        file_ckpt=os.path.join(path_checkpoint, prefix_checkpoint + ".model_epoch_%d.pth" % n_epochs)
        os.remove(file_ckpt)
        
        assert len(x_synth) == n_gen
        
    def test_save_and_load(self):
        
        path_checkpoint='.'
        prefix_checkpoint='test'
        n_epochs=10
        cor = Corgan()
        
        # dummy dataset
        n_gen = 500
        n = 1000
        m = 7
        x = np.random.randint(low=0, high=2, size=(n,m))
        
        model_saved = cor.train(x=x, 
                             n_epochs_pretrain=10,
                             n_epochs=10,
                             path_checkpoint=path_checkpoint, 
                             prefix_checkpoint=prefix_checkpoint)
        
        file = 'test.pkl'
        cor.save_obj(obj=model_saved, file_name=file)
        model_loaded = cor.load_obj(file)
        x_synth = cor.generate(model = model_loaded, n_gen=n_gen)
        
        # clean up 
        file_ckpt=os.path.join(path_checkpoint, prefix_checkpoint + ".model_epoch_%d.pth" % n_epochs)
        os.remove(file_ckpt)
        os.remove(file)
        
        assert len(x_synth) == n_gen
