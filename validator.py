import pickle

class Validator:
    
    def save_obj(self, obj, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(self, file_name):
        with open(file_name, 'rb') as f:
            return pickle.load(f)
        
    def plot(self, res, analysis, file_pdf):
        return None
    
    def summarize(self, res, analysis, n_decimal=2):
        return None