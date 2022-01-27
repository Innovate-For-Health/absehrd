import numpy as np

file_trn = 'examples/example_train.csv'
file_tst = 'examples/example_test.csv'

n = 10000
count_min = 5
count_max = 19
constant_value = 'helloworld'
binary_A = 'A'
binary_B = 'B'
categorical_values = ['X','Y','Z']

header = np.array(['constant','binary01', 'binaryAB', 'categorical','count','continuous'])
v_constant = np.full(shape=n, fill_value=constant_value)
v_binary01 = np.random.randint(low=0, high=2, size=n)
v_binaryAB = np.concatenate((np.full(shape=n-1, fill_value=binary_A), np.array([binary_B])))
v_categorical = np.random.choice(categorical_values, size=n)
v_count = np.random.randint(low=count_min, high=count_max+1, size=n)
v_continuous = np.random.random(size=n)
x = np.column_stack((v_constant, v_binary01, v_binaryAB, v_categorical, v_count, v_continuous))

np.savetxt(fname=file_trn, fmt='%s', X=x[0:round(n/2)], delimiter=',', header=','.join(header))
np.savetxt(fname=file_tst, fmt='%s', X=x[(round(n/2)+1):(n-1)], delimiter=',', header=','.join(header))

print('Datasets written to \''+file_trn+'\' and \''+file_tst+'\'')