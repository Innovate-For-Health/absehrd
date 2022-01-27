from os.path import isfile
import numpy as np
from pyarrow import feather
import pandas as pd

class Preprocessor:
    """Preprocessing functions for transforming and restoring data matrices.
    """

    def __init__(self, missing_value):
        self.missing_value=missing_value
        self.delim = '__'

    def get_file_type(self, file_name):
        """Determine the type of file.

        Parameters
        ----------
        file_name : str
            Name of the file.

        Returns
        -------
        file_type : str
            Standardized suffix of file type.

        """

        file_type = ''

        if file_name.endswith('.npy'):
            file_type = 'npy'
        elif file_name.endswith('.feather'):
            file_type = 'feather'
        elif file_name.endswith('.csv'):
            file_type = 'csv'
        elif file_name.endswith('.tsv'):
            file_type = 'tsv'
        elif file_name.endswith('.pkl'):
            file_type = 'pkl'
        else:
            file_type = None

        return file_type

    def get_default_header(self, n_col, prefix='C'):
        """Generate a header with a generic label and column number.

        Parameters
        ----------
        n_col : int
            Number of columns for header.
        prefix : str, optional
            Prefix for elements of the header. The default is 'C'.

        Returns
        -------
        header : array_like
            Array with column names.

        """

        header = np.full(shape=n_col, fill_value='',
                         dtype='<U'+str(len(str(n_col-1))+len(prefix)))
        for i in range(n_col):
            header[i] = prefix + str(i).zfill(len(str(n_col-1)))

        return header

    def read_file(self, file_name, has_header=True):
        """Read a matrix of data from file.

        Parameters
        ----------
        file_name : str
            Full path and name of the file.
        has_header : bool, optional
            True if the file has a header; false otherwise. The default is True.

        Returns
        -------
        dict
            Dictionary object containing the matrix ('x') and header ('header').

        """

        arr = None
        header = []

        if not isfile(file_name):
            return None

        if self.get_file_type(file_name) == 'npy':
            arr = np.load(file_name)
            if has_header:
                header = arr[0,:]
                arr = arr[1:len(arr),:]
        
        elif self.get_file_type(file_name) == 'pkl':
            pd_df = pd.read_pickle(file_name)
            arr = pd_df.to_numpy()
            if has_header:
                header = pd_df.columns

        elif self.get_file_type(file_name) == 'feather':
            pd_df = feather.read_feather(file_name)
            arr = pd_df.to_numpy()
            if has_header:
                header = pd_df.columns

        elif self.get_file_type(file_name) == 'csv' or self.get_file_type(file_name) == 'tsv':

            arr = []
            if has_header:

                delimiter = ','
                if self.get_file_type(file_name) == 'tsv':
                    delimiter = '\t'

                arr = np.loadtxt(file_name, dtype=str, delimiter=delimiter,
                                 skiprows=1)

                header = np.loadtxt(file_name, dtype=str, delimiter=delimiter,
                                    comments=None, skiprows=0, max_rows=1)
                header[0] = header[0].replace('# ','')
            else:
                arr = np.loadtxt(file_name, dtype=str, delimiter=delimiter)

        else:
            return None

        if len(header) == 0:
            header = self.get_default_header(arr.shape[1])

        return {'x':arr, 'header':header}

    def is_iterable(self, obj):
        """Determine if an object is iterable.

        Parameters
        ----------
        obj : object
            Object to analyze.

        Returns
        -------
        bool
            True if object is iterable; False otherwise.

        """

        try:
            iter(obj)
        except Exception:
            return False
        else:
            return True

    def is_numeric(self, arr):
        """Determine if an array is numeric.

        Parameters
        ----------
        arr : array_like
            Array of any type.

        Returns
        -------
        bool
            True if the array contains numbers; False otherwise.

        """
        
        if isinstance(arr, np.ndarray) and len(arr) == 0:
            return True

        if isinstance(arr, np.ndarray):
            ele = arr.ravel()[0]
            try:
                float(ele)
            except:
                return False
        
        elif self.is_iterable(arr):
            for ele in arr:
                try:
                    float(ele)
                except:
                    return False
        else:
            try:
                float(arr)
            except:
                return False

        return True

    def remove_na(self, arr):
        """Remove missing value elements from an array.

        Parameters
        ----------
        arr : array_like
            Array that may or may not contain missing values.

        Returns
        -------
        d : array_like
            Array with missing value elements removed.

        """

        arr_d = arr

        x_num = self.is_numeric(arr)
        m_num = self.is_numeric(self.missing_value)

        if m_num and np.isnan(self.missing_value):
            idx = np.where(np.isnan(arr))
            
        elif x_num and m_num:
            idx = np.where(arr.astype(float) == float(self.missing_value))

        else:
            idx = np.where(arr.astype(str) == str(self.missing_value))

        if len(idx) > 0:
            arr_d = np.delete(arr_d,idx)
        arr_d = np.delete(arr_d, np.where(arr_d.astype(str) == ''))

        return arr_d

    def get_minority_class(self, arr):
        """Get the least common element in the array.

        Parameters
        ----------
        arr : array_like
            Array with two unique values; may contain missing values.

        Returns
        -------
        str
            Value the occurs less frequently in the array.

        """

        arr_d = self.remove_na(arr)
        uniq_d = np.unique(arr_d)
        val_0 = 0
        val_1 = 0

        for ele in arr_d:
            if ele == uniq_d[0]:
                val_0+=1
            else:
                val_1+=1

        if val_0 <= val_1:
            return uniq_d[0]

        return uniq_d[1]

    def get_variable_type(self, arr, label=None, custom_categorical=(),
                               custom_continuous=(), custom_count=(),
                               custom_constant=(),
                               max_uniq=10, max_diff=1e-10):
        """Guess at the type of variable.

        Parameters
        ----------
        arr : array_like
            Array of values of unknown type.
        label : str, optional
            Column label of the array. The default is None.
        custom_categorical : array_like, optional
            Array of column labels that are categorical. The default is ().
        custom_continuous : array_like, optional
            Array of column labels that are continuous. The default is ().
        custom_count : array_like, optional
            Array of column labels that are count. The default is ().
        custom_constant : array_like, optional
            Array of column labels that are constant. The default is ().
        max_uniq : int, optional
            Maximum number of unique values to be considered
            categorical. The default is 10.
        max_diff : float, optional
            Maximum threshold difference between rounded value and value to 
            be considered count. The default is 1e-10.

        Returns
        -------
        str
            Type of variable.  Options are 'constant', 'categorical', 
            'count', 'continuous'.

        """

        arr_d = self.remove_na(arr)
        n_uniq = len(set(arr_d))

        if label is not None and label in custom_categorical:
            return 'categorical'

        if label is not None and label in custom_continuous:
            return 'continuous'

        if label is not None and label in custom_count:
            return 'count'

        if n_uniq == 1:
            return 'constant'

        if n_uniq == 2:
            return 'binary'

        if n_uniq > max_uniq and self.is_numeric(arr_d):

            for ele in arr_d:
                ele = float(ele)
                if abs(ele - round(ele)) > max_diff:
                    return 'continuous'
            return 'count'

        return 'categorical'

    def get_metadata(self, arr, header):
        """Gather metadata on each feature of a data matrix.

        Parameters
        ----------
        arr : array_like
            2D array for which to calculate metadata.
        header : array_like
            Array of column labels for the array.

        Returns
        -------
        meta : array_like
            Metadata matrix containing one row of information per column.

        """

        names = ['label','type', 'min', 'max', 'zero', 'one','unique','missing']
        formats = ['<U100','<U11','float64', 'float64',str(arr.dtype),str(arr.dtype), '<U1000', '?']
        meta = np.recarray(shape=arr.shape[1], names=names, formats=formats)

        for j in range(arr.shape[1]):

            arr_d = self.remove_na(arr[:,j])

            meta[j]['label'] = header[j]

            meta[j]['type'] = self.get_variable_type(arr_d, label=header[j])

            if meta[j]['type'] == 'binary':
                meta[j]['one'] = self.get_minority_class(arr_d)
                meta[j]['zero'] = set(np.unique(arr_d)).difference([meta[j]['one']]).pop()

            if meta[j]['type'] == 'constant':
                meta[j]['zero'] = arr_d[0]

            if meta[j]['type'] == 'continuous' or meta[j]['type'] == 'count':
                meta[j]['min'] = np.min(arr_d.astype(np.float))
                meta[j]['max'] = np.max(arr_d.astype(np.float))

            if meta[j]['type'] == 'categorical':
                meta[j]['unique'] = ','.join(np.unique(arr[:,j]))

            if self.is_numeric(arr):
                missing_value = self.missing_value
            else:
                missing_value = str(self.missing_value)
            if len(np.where(arr[:,j] == missing_value)[0]) > 0:
                meta[j]['missing'] = True
            else:
                meta[j]['missing'] = False

        return meta

    def get_one_hot_encoding(self, arr, label, unique_values = None, add_missing_col=False):
        """Convert a categorical variable into one-hot encoding matrix.

        Parameters
        ----------
        arr : array_like
            Array of values.
        label : str
            Column label of the feature contained in the array.
        unique_values : array_like, optional
            Custom array of unique values over which to create the encoding. The default is None.
        add_missing_col : bool, optional
            True if missing column is required; False otherwise. The default is False.

        Returns
        -------
        dict
            Contains the matrix ('x') of one hot encoded data and 
            corresponding header ('header').

        """

        if unique_values is None:
            unique_values = np.unique(arr)
        else:
            unique_values = np.array(unique_values)

        idx = np.where(unique_values == str(self.missing_value))[0]
        if add_missing_col:
            if len(idx) == 0:
                unique_values = np.append(unique_values, self.missing_value)

        n_uniq = len(unique_values)
        u_label = []
        x_hot = np.zeros(shape=(len(arr), n_uniq), dtype=int)

        for j in range(n_uniq):
            u_label = np.append(u_label, label + self.delim + str(j))

            for i, ele in enumerate(arr):
                if ele == unique_values[j]:
                    x_hot[i,j] = 1

        return {'x':x_hot, 'header':u_label}

    def scale(self, arr):
        """Scale a numeric vector to between 0 and 1

        Parameters
        ----------
        arr : array_like
            Numeric array to scale.

        Returns
        -------
        arr_s : array_like
            Scaled version of the array.

        """

        arr_d = self.remove_na(arr)
        x_min = np.min(arr_d)
        x_max = np.max(arr_d)
        arr_s = np.zeros(shape=len(arr))

        for i, ele in enumerate(arr):
            if ele != self.missing_value:
                arr_s[i] = (ele - x_min) / (x_max - x_min)
            else:
                arr_s[i] = self.missing_value

        return arr_s

    def unscale(self, arr, min_value, max_value):
        """Convert a numeric array scaled between 0 and 1 into original range.

        Parameters
        ----------
        arr : array_like
            Array of scaled numeric values.
        min_value : float
            Minimum value in the original, unscaled array.
        max_value : float
            Maximum value in the original, unscaled array.

        Returns
        -------
        arr_c : array_like
            Restored, unscaled array.

        """

        arr_c = np.zeros(len(arr))

        for i, ele in enumerate(arr):
            arr_c[i] = ele * (max_value - min_value) + min_value
        return arr_c

    def get_missing_column(self, arr):
        """Create a column to represent missing status of each feature.

        Parameters
        ----------
        arr : array_like
            Array of values which may or may not contain missing values.

        Returns
        -------
        arr_y : array_like
            Binary array of same size as input array with 1 if 
            corresponding array value is the missing_value and 0 otherwise.

        """

        arr_y = np.zeros(shape=arr.shape)
        idx = np.array([], dtype=int)

        for i, ele in enumerate(arr):
            if ele == self.missing_value or ele == str(self.missing_value):
                idx = np.append(idx, i)

        arr_y[idx] = 1

        return arr_y

    def get_na_idx(self, arr):
        """Get indices of all missing values in an array.

        Parameters
        ----------
        arr : array_like
            Array of values which may or may not contain missing values.

        Returns
        -------
        idx : array_like
            Array of integers representing the index of each missing value.

        """

        idx = np.array([])
        x_num = self.is_numeric(arr)
        m_num = self.is_numeric(self.missing_value)

        for i, ele in enumerate(arr):
            if x_num and m_num:
                if float(ele) == float(self.missing_value):
                    idx = np.append(idx, i)
            else:
                if str(ele) == str(self.missing_value):
                    idx = np.append(idx, i)

        # added
        idx = np.unique(np.append(idx, np.where(arr == '')))

        return idx

    def get_discretized_matrix(self, arr, meta, header, require_missing=True):
        """Convert a generic matrix into a representation where each element
        is scaled between 0 and 1.

        Parameters
        ----------
        arr : array_like
            2D array of values of multiple types.
        meta : array_like
            2D matrix of metadata for each column.
        header : array_like
            Array of column labels.
        require_missing : bool, optional
            If True, require a missing column for each feature; 
            False otherwise. The default is True.

        Returns
        -------
        dict
            Constains the numeric version of the matrix ('x') and 
            corresponding header ('header').

        """

        d_x = np.empty(shape=0)
        d_header = np.empty(shape=0)

        for j in range(arr.shape[1]):

            c_type = meta[j]['type']
            x_j = arr[:,j]
            s_j = []

            if require_missing:
                contains_missing = True
            else:
                contains_missing = len(self.remove_na(x_j)) < len(x_j)

            if c_type == 'constant':

                if contains_missing:
                    s_j = np.column_stack((np.zeros(shape=arr.shape[0]),
                                           self.get_missing_column(x_j)))
                    d_header = np.append(d_header,
                                         (header[j]+self.delim+header[j],
                                          header[j]+self.delim+str(self.missing_value)))
                else:
                    s_j = np.zeros(shape=arr.shape[0])
                    d_header = np.append(d_header, header[j])

            elif c_type in ['continuous', 'count']:

                if contains_missing:
                    idx = self.get_na_idx(x_j)

                    s_j_notna = self.scale(self.remove_na(x_j).astype(np.float))
                    s_j = np.random.uniform(low=0, high=1, size=len(x_j))
                    s_j[np.setdiff1d(range(len(x_j)),idx)] = s_j_notna
                    s_j = np.column_stack((s_j,
                                       self.get_missing_column(x_j)))
                    d_header = np.append(d_header,
                                         (header[j]+self.delim+header[j],
                                          header[j]+self.delim+str(self.missing_value)))
                else:
                    x_j = x_j.astype(np.float)
                    s_j = self.scale(x_j)
                    d_header = np.append(d_header, header[j])

            elif c_type == 'categorical':

                res = self.get_one_hot_encoding(arr=x_j, label=header[j],
                        unique_values=(meta[j]['unique']).split(','),
                        add_missing_col=contains_missing)
                s_j = res['x']
                d_header = np.append(d_header, res['header'])

            elif c_type == 'binary':

                s_j = np.zeros(shape=len(x_j), dtype=int)
                for i in range(len(arr)):
                    if arr[i,j] == meta[j]['one']:
                        s_j[i] = 1

                if contains_missing:
                    f_j = s_j
                    idx = np.where(x_j == str(self.missing_value))[0]
                    f_j[idx] = np.random.randint(low=0,high=2,size=len(idx))
                    s_j = np.column_stack((f_j, self.get_missing_column(x_j)))
                    d_header = np.append(d_header,
                                         (header[j]+self.delim+header[j],
                                          header[j]+self.delim+str(self.missing_value)))
                else:
                    d_header = np.append(d_header, header[j])

            else:
                print("Warning: variable type not recognized.  Appending as is.")
                s_j = x_j
                d_header = np.append(d_header, header[j])

            if len(d_x) == 0:
                d_x = s_j
            else:
                d_x = np.column_stack((d_x, s_j))

        return {'x':d_x, 'header':d_header}

    def unravel_one_hot_encoding(self, arr, unique_values):
        """Restore a binary, one-hot encoded matrix to its original
        categorical feature format.

        Parameters
        ----------
        arr : array_like
            2D array of one-hot encoded data.
        unique_values : array_like
            Unique values corresponding to each column of the array. The default is None.

        Returns
        -------
        arr_c : array_like
            1D array representing categorical feature format from one-hot encoded matrix.

        """

        arr_c = []

        for i in range(len(arr)):

            x_i = arr[i,:].astype(float)

            idx = np.where(x_i==np.max(x_i))[0]
            if len(idx) > 1:
                idx = idx[np.random.randint(low=0,high=len(idx), size=1)]
            else:
                idx = idx[0]
            arr_c = np.append(arr_c, unique_values[idx])

        return arr_c

    def restore_matrix(self, arr, meta, header):
        """Restore scaled, numeric matrix to its original format.

        Parameters
        ----------
        arr : array_like
            Discretized version of the array.
        meta : array_like
            2D matrix of metadata for each column.
        header : array_like
            Array of column labels for the discretized array.

        Returns
        -------
        dict
            Contains the original version of the matrix ('x') and 
            corresponding header ('header').

        """

        c_prime = []
        variable_names = []
        variable_values = []
        header_prime = meta['label']

        for i, ele in enumerate(header):

            splt = ele.split('__')
            variable_names = np.append(variable_names, splt[0])

            if len(splt) > 1:
                variable_values = np.append(variable_values, splt[1])
            else:
                variable_values = np.append(variable_values, '')

        for j, ele in enumerate(meta):

            c_j = []
            idx_missing = []
            idx_col = np.where(variable_names == ele['label'])[0]

            if ele['type'] != 'categorical' and len(idx_col) > 1:
                for k in np.where(variable_names == ele['label'])[0]:
                    if variable_values[k] == str(self.missing_value):
                        idx_missing = np.where(arr[:,k] == 1)[0]
                    else:
                        idx_col = k

            s_j = arr[:,idx_col]

            if ele['type'] == 'constant':
                c_j = np.full(shape=len(arr), fill_value=ele['zero'])
                c_j[idx_missing] = self.missing_value

            elif ele['type'] == 'continuous' or ele['type'] == 'count':
                min_value = float(ele["min"])
                max_value = float(ele["max"])
                c_j = self.unscale(arr=s_j.astype('float'), min_value=min_value,
                                   max_value=max_value)

                if ele['type'] == 'count':
                    c_j = np.round(c_j)

                c_j = c_j.astype(str)
                c_j[idx_missing] = self.missing_value

            elif ele['type'] == 'categorical':
                c_j = self.unravel_one_hot_encoding(arr=s_j,
                        unique_values=(ele['unique']).split(','))

            elif ele['type'] == 'binary':

                c_j = np.full(shape=len(arr), fill_value=ele['zero'], dtype='O')

                for i, val in enumerate(s_j):
                    if np.round(val) == 1:
                        c_j[i] = ele['one']

                c_j[idx_missing] = self.missing_value

            if len(c_prime) == 0:
                c_prime = c_j
            else:
                c_prime = np.column_stack((c_prime, c_j))

        return {'x':c_prime, 'header':header_prime}
