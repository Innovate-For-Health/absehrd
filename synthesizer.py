import pickle

class Synthesizer:
    """
    Ancestor class for each synthetic data generator.

    Methods
    -------
    save_obj(obj, file_name)
        Save an object in the pickle format.
    load_obj(file_name)
        Load an object from the pickle format.
    train()
        Train a generator model.
    generate()
        Generate synthetic samples from the trained model.
    """

    def save_obj(self, obj, file_name):
        """Save an object in the pickle format

        Parameters
        ----------
        obj : object
            Object to save.
        file_name : str
            Name of the file to save the object, obj.
        """

        with open(file_name, 'wb') as file_obj:
            pickle.dump(obj, file_obj, pickle.HIGHEST_PROTOCOL)

    def load_obj(self, file_name):
        """Load an object from the pickle format.

        Parameters
        ----------
        file_name : str
            Name of the file to save the object.
        """

        with open(file_name, 'rb') as file_obj:
            return pickle.load(file_obj)

    def train(self):
        """Train a generator model.
        """

        return None

    def generate(self):
        """Generate synthetic samples from the trained model.
        """
        return None
