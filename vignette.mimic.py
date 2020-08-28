import numpy as np
from preprocessor import preprocessor
from corgan import corgan
from report import report

def main():
    
    # parameters
    n_gen = 10000
    outcome = 'died_90d'
    file_label = 'mimic'
    file_csv_real = '../data/'+file_label+'.raw.csv'
    file_pdf = '../plots/vignette.'+file_label+'.description_report.pdf'
    file_csv_corgan = '../output/vignette.'+file_label+'.corgan.csv'
    missing_value = -999999
    
    pre = preprocessor(missing_value=missing_value)
    rep = report(missing_value=missing_value)
    
    # read nhamcs dataset
    ftr = pre.read_file(file_csv_real, n_header=1)
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
    report_status = rep.description_report(r=d['x'], s=s, col_names=d['header'], 
                             outcome=outcome, file_pdf=file_pdf, n_epoch=100)
    
    # summary
    if report_status:
        print('Report written to', file_pdf)
    else:
        print('Error: report generation failed')
    print('Real dataset written to', file_csv_real)
    print('Synthetic dataset written to', file_csv_corgan)
    
if __name__ == "__main__":
    main()
