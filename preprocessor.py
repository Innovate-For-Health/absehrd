from os.path import isfile
import numpy as np
import feather

class Preprocessor:
    """Preprocessing functions for transforming and restoring data matrices.
    """

    def __init__(self, missing_value):
        self.missing_value=missing_value
        self.delim = '__'

    def get_file_type(self, file_name, debug=False):
        """Determine the type of file.

        Parameters
        ----------
        file_name : TYPE
            DESCRIPTION.
        debug : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        file_type : TYPE
            DESCRIPTION.

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
        else:
            file_type = None

        if debug:
            print(file_type)

        return file_type

    def get_default_header(self, n_col, prefix='C'):
        """Generate a header with a generic label and column number.

        Parameters
        ----------
        n : TYPE
            DESCRIPTION.
        prefix : TYPE, optional
            DESCRIPTION. The default is 'C'.

        Returns
        -------
        header : TYPE
            DESCRIPTION.

        """

        header = np.full(shape=n_col, fill_value='',
                         dtype='<U'+str(len(str(n_col-1))+len(prefix)))
        for i in range(n_col):
            header[i] = prefix + str(i).zfill(len(str(n_col-1)))

        return header

    def read_file(self, file_name, has_header=True, debug=False):
        """Read a matrix of data from file.

        Parameters
        ----------
        file_name : TYPE
            DESCRIPTION.
        has_header : TYPE, optional
            DESCRIPTION. The default is True.
        debug : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        dict
            DESCRIPTION.

        """

        arr = None
        header = []

        if not isfile(file_name):
            return None

        if self.get_file_type(file_name) == 'npy':
            if debug:
                print('Reading npy file ', file_name, '...')
            arr = np.load(file_name)
            if has_header:
                header = arr[0,:]
                arr = arr[1:len(arr),:]

        elif self.get_file_type(file_name) == 'feather':
            if debug:
                print('Reading feather file ', file_name, '...')
            pd_df = feather.read_dataframe(file_name)
            arr = pd_df.to_numpy()
            if has_header:
                header = pd_df.columns

        elif self.get_file_type(file_name) == 'csv' or self.get_file_type(file_name) == 'tsv':
            if debug:
                print('Reading ' + self.get_file_type(file_name) + ' file ', file_name, '...')

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
            if debug:
                print('Unknown file type for file', file_name)
                print("Returning None")
            return None

        if debug:
            print('  Number of samples:', arr.shape[0])
            print('  Number of features:', arr.shape[1])

        if len(header) == 0:
            header = self.get_default_header(arr.shape[1])

        return {'x':arr, 'header':header}

    def is_iterable(self, obj):
        """Determine if an object is iterable.

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.

        Returns
        -------
        bool
            DESCRIPTION.

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
        arr : TYPE
            DESCRIPTION.

        Returns
        -------
        bool
            DESCRIPTION.

        """

        if self.is_iterable(arr):
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
        x : TYPE
            DESCRIPTION.

        Returns
        -------
        d : TYPE
            DESCRIPTION.

        """

        arr_d = arr

        x_num = self.is_numeric(arr)
        m_num = self.is_numeric(self.missing_value)

        if x_num and m_num:
            idx = np.where(arr.astype(float) == float(self.missing_value))

        else:
            idx = np.where(arr.astype(str) == str(self.missing_value))

        if len(idx) > 0:
            arr_d = np.delete(arr_d,idx)

        return arr_d

    def get_minority_class(self, arr):
        """Get the least common element in the array.

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

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
                               max_uniq=10, max_diff=1e-10):
        """Guess at the type of variable.

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        label : TYPE, optional
            DESCRIPTION. The default is None.
        custom_categorical : TYPE, optional
            DESCRIPTION. The default is ().
        custom_continuous : TYPE, optional
            DESCRIPTION. The default is ().
        custom_count : TYPE, optional
            DESCRIPTION. The default is ().
        max_uniq : TYPE, optional
            DESCRIPTION. The default is 10.
        max_diff : TYPE, optional
            DESCRIPTION. The default is 1e-10.

        Returns
        -------
        str
            DESCRIPTION.

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
        x : TYPE
            DESCRIPTION.
        header : TYPE
            DESCRIPTION.

        Returns
        -------
        m : TYPE
            DESCRIPTION.

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

            if len(np.where(arr[:,j] == str(self.missing_value))[0]) > 0:
                meta[j]['missing'] = True
            else:
                meta[j]['missing'] = False

        return meta

    def get_one_hot_encoding(self, arr, label, unique_values = None, add_missing_col=False):
        """Convert a categorical variable into one-hot encoding matrix.

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        label : TYPE
            DESCRIPTION.
        unique_values : TYPE, optional
            DESCRIPTION. The default is None.
        add_missing_col : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        dict
            DESCRIPTION.

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
        x : TYPE
            DESCRIPTION.

        Returns
        -------
        s : TYPE
            DESCRIPTION.

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
        x : TYPE
            DESCRIPTION.
        min_value : TYPE
            DESCRIPTION.
        max_value : TYPE
            DESCRIPTION.

        Returns
        -------
        c : TYPE
            DESCRIPTION.

        """

        arr_c = np.zeros(len(arr))

        for i, ele in enumerate(arr):
            arr_c[i] = ele * (max_value - min_value) + min_value
        return arr_c

    def get_missing_column(self, arr):
        """Create a column to represent missing status of each feature.

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.

        Returns
        -------
        y : TYPE
            DESCRIPTION.

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
        x : TYPE
            DESCRIPTION.

        Returns
        -------
        idx : TYPE
            DESCRIPTION.

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

        return idx

    def get_discretized_matrix(self, arr, meta, header, require_missing=True, debug=False):
        """Convert a generic matrix into a representation where each element
        is scaled between 0 and 1.

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        m : TYPE
            DESCRIPTION.
        header : TYPE
            DESCRIPTION.
        require_missing : TYPE, optional
            DESCRIPTION. The default is True.
        debug : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        dict
            DESCRIPTION.

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

            if debug:
                print('Dimensions of matrix are now', d_x.shape)

        return {'x':d_x, 'header':d_header}

    def unravel_one_hot_encoding(self, arr, unique_values):
        """Restore a binary, one-hot encoded matrix to its original
        categorical feature format.

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        unique_values : TYPE
            DESCRIPTION.

        Returns
        -------
        c : TYPE
            DESCRIPTION.

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

    def restore_matrix(self, arr, met, header):
        """Restore scaled, numeric matrix to its original format.

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        m : TYPE
            DESCRIPTION.
        header : TYPE
            DESCRIPTION.

        Returns
        -------
        dict
            DESCRIPTION.

        """

        c_prime = []
        variable_names = []
        variable_values = []
        header_prime = met['label']

        for i, ele in enumerate(header):

            splt = ele.split('__')
            variable_names = np.append(variable_names, splt[0])

            if len(splt) > 1:
                variable_values = np.append(variable_values, splt[1])
            else:
                variable_values = np.append(variable_values, '')

        for j, ele in enumerate(met):

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
                    if val == 1:
                        c_j[i] = val['one']

                c_j[idx_missing] = self.missing_value

            if len(c_prime) == 0:
                c_prime = c_j
            else:
                c_prime = np.column_stack((c_prime, c_j))

        return {'x':c_prime, 'header':header_prime}
