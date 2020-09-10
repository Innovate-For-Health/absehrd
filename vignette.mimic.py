import numpy as np
from torch.utils.data import random_split
from preprocessor import preprocessor
from corgan import corgan
from report import report
from privacy import privacy

def main():
    
    # parameters
    n_gen = 10000
    outcome = 'died_90d'
    file_label = 'mimic'
    file_csv_real = '../data/'+file_label+'.raw.csv'
    file_pdf = '../plots/vignette.'+file_label+'.description_report.pdf'
    file_csv_corgan = '../output/vignette.'+file_label+'.corgan.csv'
    missing_value = '-999999'
    delim = '__'
    
    pre = preprocessor(missing_value=missing_value)
    rep = report(missing_value=missing_value)
    cor = corgan()
    pri = privacy()
    
    # read nhamcs dataset
    ftr = pre.read_file(file_csv_real, has_header=True)
    header = ftr['header']
    x = ftr['x']
        
    # preprocess
    m = pre.get_metadata(x=x, header=header)
    d = pre.get_discretized_matrix(x=x, m=m, header=header, delim=delim)
    
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
    f = pre.restore_matrix(s=s, m=m, header=d['header'], delim=delim)
    
    # write to file
    np.savetxt(fname=file_csv_corgan, fmt='%s', X=f['x'], delimiter=',', header=','.join(f['header']))
    
    # report
    idx_outcome = np.unique(np.append(np.where(d['header']==outcome), np.where(d['header']==outcome+delim+outcome)))
    outcome_label = d['header'][idx_outcome][0]
    report_status = rep.description_report(r_trn=r_trn, r_tst=r_tst, s=s, col_names=d['header'], 
                             outcome=outcome_label, file_pdf=file_pdf, n_epoch=100, 
                             model_type='lr')
    
    mem_inf = pri.membership_inference(r_trn=r_trn, r_tst=r_tst, s_trn=s_trn, s_tst=s_tst,
                             a_trn=a_trn, a_tst=a_tst, model_type='svm')
    
    # summary
    if report_status:
        print('Report written to', file_pdf)
    else:
        print('Error: report generation failed')
    print('Real dataset written to', file_csv_real)
    print('Synthetic dataset written to', file_csv_corgan)
    print('AUC for authentic-synthetic: ', mem_inf['auc_as'])
    print('AUC for real train-test: ', mem_inf['auc_rr'])
    
if __name__ == "__main__":
    main()
