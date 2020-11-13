import pickle

class Synthesizer:
    """Ancestor class for each synthetic data generator.
    """

    def save_obj(self, obj, file_name):
        """Save an object in the pickle format.

        Parameters
        ----------
        obj : object
            Object to save.
        file_name : str
            Name of the file to save the object, obj.
            
        Returns
        -------
        None.
        """

        with open(file_name, 'wb') as file_obj:
            pickle.dump(obj, file_obj, pickle.HIGHEST_PROTOCOL)

    def load_obj(self, file_name):
        """Load an object from the pickle format.

        Parameters
        ----------
        file_name : str
            Name of the file from which to load the object.
            
        Returns
        -------
        object
            Object that was stored in the file.
        """

        with open(file_name, 'rb') as file_obj:
            return pickle.load(file_obj)

    def train(self, x, n_epochs, n_cpu, *args):
        """Train a generator model.
        
        Returns
        -------
        None.
        """

        return None

    def generate(self, model, n_gen):
        """Generate synthetic samples from the trained model.
        
        Returns
        -------
        None.
        """
        return None
