import pickle

class Validator:
    """Ancestor class for all synthetic data validators.
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

    def scale(self, arr, invert=False):
        """Scale a numeric vector between 0 and 1.

        Parameters
        ----------
        arr : array_like
            Numeric array like object to scale.
        invert : boolean
            If true, highest value is 0 and minimum value is 1;
            otherwise, highest value is 1 and minimum value is 0
        """

        scl =  (arr - min(arr)) / (max(arr) - min(arr))

        if invert:
            return 1 - scl

        return scl
    
    def plot(self, res, analysis, file_pdf):
        """Plot a result in a pdf file.

        Parameters
        ----------
        res : dictionary_like
            Result to plot.
        analysis : str
            Label for the analysis to plot.
        file_pdf : str
            Name of the pdf file to the plot.
        """
        msg = analysis + '_' + file_pdf + '_'+str(res)
        return msg

    def summarize(self, res, analysis, n_decimal=2):
        """Summarize a result in a string

        Parameters
        ----------
        res : dictionary_like
            Result to plot.
        analysis : str
            Label for the analysis to plot.
        n_decimal : int
            Number of decimal places to round the numeric parts
            of the summary string.
        """
        msg = analysis + '_' + str(n_decimal) + '_'+str(res)
        return msg
