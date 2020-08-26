import numpy as np
from preprocessor import preprocessor as pre
from corgan import corgan
from report import report


def main():
    
    # parameters
    n = 1000
    count_min = 5
    count_max = 19
    constant_value = 'helloworld'
    binary_A = 'A'
    binary_B = 'B'
    categorical_values = ['X','Y','Z']
    n_gen = n
    outcome = 'binary01'
    file_pdf = '../plots/vignette.dummy.prediction_report.pdf'
    file_csv_real =  '../output/vignette.dummy.real.csv'
    file_csv_corgan = '../output/vignette.dummy.corgan.csv'
    
    # generate dummy dataset
    names = ['constant','binary01', 'binaryAB', 'categorical','count','continuous']
    v_constant = np.full(shape=n, fill_value=constant_value)
    v_binary01 = np.random.randint(low=0, high=2, size=n)
    v_binaryAB = np.concatenate((np.full(shape=n-1, fill_value=binary_A), np.array([binary_B])))
    v_categorical = np.random.choice(categorical_values, size=n)
    v_count = np.random.randint(low=count_min, high=count_max+1, size=n)
    v_continuous = np.random.random(size=n)
    x = np.column_stack((v_constant, v_binary01, v_binaryAB, v_categorical, v_count, v_continuous))
    
    # preprocess
    m = pre.get_metadata(pre, x=x, header=names)
    d = pre.get_discretized_matrix(pre, x, m, names)
    
    # generate synthetic data
    model = corgan.train(corgan, x=d['x'], n_cpu=15)
    s = corgan.generate(model, n_gen)
    
    # reconstruct and save synthetic data
    f = pre.restore_matrix(pre, s=s, m=m, header=d['header'])
    
    # write to file
    np.savetxt(fname=file_csv_real, fmt='%s', X=x, delimiter=',', header=','.join(names))
    np.savetxt(fname=file_csv_corgan, fmt='%s', X=f['x'], delimiter=',', header=','.join(f['header']))
    
    # report 
    report_status = report.prediction_report(report, r=d['x'], s=s, col_names=d['header'], 
                             outcome=outcome, file_pdf=file_pdf)
    
    # summary
    if report_status:
        print('Report written to', file_pdf)
    else:
        print('Error: report generation failed')
    print('Real dataset written to', file_csv_real)
    print('Synthetic dataset written to', file_csv_corgan)
    
if __name__ == "__main__":
    main()
