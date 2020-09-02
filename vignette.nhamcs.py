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
    cor = corgan()
    
    # read nhamcs dataset
    ftr = pre.read_file(file_ftr)
    header = ftr['header']
    x = ftr['x']
    
    # preprocess
    m = pre.get_metadata(x=x, header=header)
    d = pre.get_discretized_matrix(x, m, header)
    
    # split 
    n_subset_d = round(len(d['x'])*0.5)
    r, a = random_split(d['x'], [n_subset_d, len(d['x']) - n_subset_d])
    n_subset_r = round(len(r)*0.75)
    n_subset_a = round(len(a)*0.75)
    r_trn, r_tst = random_split(r, [n_subset_r,len(r)-n_subset_r])
    a_trn, a_tst = random_split(a, [n_subset_a,len(a)-n_subset_a])
    r_trn = np.array(r_trn)
    r_tst = np.array(r_tst)
    a_trn = np.array(a_trn)
    a_tst = np.array(a_tst)

    # generate synthetic data
    model = cor.train(x=r_trn, n_cpu=15, debug=True)
    s = cor.generate(model, n_gen)
    n_subset_s = round(len(s)*0.75)
    s_trn, s_tst = random_split(s, [n_subset_s, len(s)-n_subset_s])
    s_trn = np.array(s_trn)
    s_tst = np.array(s_tst)
    
    # reconstruct and save synthetic data
    f = pre.restore_matrix(s=s, m=m, header=d['header'])
    
    # write to file
    np.savetxt(fname=file_csv_corgan, fmt='%s', X=f['x'], delimiter=',', header=','.join(f['header']))
    
    # report
    report_status = rep.description_report(r_trn=r_trn, r_tst=r_tst, s=s, col_names=d['header'], 
                             outcome=outcome, file_pdf=file_pdf, n_epoch=100, 
                             model_type='lr')
    
    # summary
    if report_status:
        print('Report written to', file_pdf)
    else:
        print('Error: report generation failed')
    print('Real dataset written to', file_csv_real)
    print('Synthetic dataset written to', file_csv_corgan)
   
if __name__ == "__main__":
    main()
