import numpy as np
from preprocessor import preprocessor
from corgan import corgan
from report import report


def main():
    
    # parameters
    n_gen = 1000
    outcome = 'ADMITHOS'
    file_ftr = "../data/nhamcs.raw.feather"
    file_pdf = '../plots/vignette.nhamcs.description_report.pdf'
    file_csv_real =  '../output/vignette.nhamcs.real.csv'
    file_csv_corgan = '../output/vignette.nhamcs.corgan.csv'
    missing_value = -999999
    pre = preprocessor(missing_value=missing_value)
    rep = report(missing_value=missing_value)
    
    # read nhamcs dataset
    ftr = pre.read_file(file_ftr)
    header = ftr['header']
    x = ftr['x']
    
    # preprocess
    m = pre.get_metadata(x=x, header=header)
    d = pre.get_discretized_matrix(x, m, header)
    
    # generate synthetic data
    model = corgan.train(corgan, x=d['x'], n_cpu=15, debug=True)
    s = corgan.generate(model, n_gen)
    
    # reconstruct and save synthetic data
    f = pre.restore_matrix(s=s, m=m, header=d['header'])
    
    # write to file
    np.savetxt(fname=file_csv_corgan, fmt='%s', X=f['x'], delimiter=',', header=','.join(f['header']))
    
    # report 
    idx = np.random.randint(low=0, high=len(d['x']), size=100)
    report_status = rep.description_report(r=d['x'][idx,:], s=s, col_names=d['header'], 
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
