import numpy as np
from preprocessor import preprocessor
from corgan import corgan
from report import report
from privacy import privacy
import pickle
from os.path import isfile

def save_obj(obj, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def main():
    
    # files
    file_label = 'mimic'
    file_version = '2'
    file_csv_real = '../data/'+file_label+'_raw_v'+file_version+'.csv'
    file_desc_pdf = '../plots/vignette.'+file_label+'.description_report.pdf'
    file_pred_pdf = '../plots/vignette.'+file_label+'.prediction_report.pdf'
    file_real_trn =  '../output/'+file_label+'_real_train_v'+file_version+'.csv'
    file_real_tst =  '../output/'+file_label+'_real_test_v'+file_version+'.csv'
    file_synth = '../output/'+file_label+'_synth_v'+file_version+'.csv'
    file_model = '../output/'+file_label+'.corgan.pkl'
    
    # parameters
    #outcome = 'died_90d'
    outcome = 'died_365d'
    missing_value = '-999999'
    delim = '__'
    trn_frac = 0.75
    n_epoch = 100
    n_epoch_pre = 100
    use_saved_model = False
    
    pre = preprocessor(missing_value=missing_value)
    rep = report(missing_value=missing_value)
    cor = corgan()
    pri = privacy()
    
    # read mimic dataset
    ftr = pre.read_file(file_csv_real, has_header=True)
    header = ftr['header']
    x = ftr['x']
        
    # preprocess
    m = pre.get_metadata(x=x, header=header)
    d = pre.get_discretized_matrix(x=x, m=m, header=header, delim=delim)
    
    # split 
    if isfile(file_real_trn) and use_saved_model:
        r_trn = np.loadtxt(file_real_trn, dtype=str, delimiter=',')
        r_tst = np.loadtxt(file_real_tst, dtype=str, delimiter=',')
    else:
        r = d['x']
        n_subset_r = round(len(r) * trn_frac)
        idx_trn = np.random.choice(len(r), n_subset_r, replace=False)
        idx_tst = np.setdiff1d(range(len(r)), idx_trn)
        r_trn = r[idx_trn,:]
        r_tst = r[idx_tst,:]

    # train generator
    if isfile(file_model) and use_saved_model:
        model = load_obj(file_model)
    else:
        model = cor.train(x=r_trn, n_cpu=16, debug=True, n_epochs=n_epoch, 
                          n_epochs_pretrain=n_epoch_pre)
        save_obj(model, file_model)
    
    # generate synthetic data
    s = cor.generate(model, n_gen=len(r_trn)+len(r_tst))
    n_subset_s = round(len(s) * trn_frac)
    idx_trn = np.random.choice(len(s), n_subset_s, replace=False)
    idx_tst = np.setdiff1d(range(len(s)), idx_trn)
    s_trn = s[idx_trn,:]
    s_tst = s[idx_tst,:]
    
    # reconstruct and save synthetic data
    f = pre.restore_matrix(s=s, m=m, header=d['header'], delim=delim)
    x_r_trn = pre.restore_matrix(s=r_trn, m=m, header=d['header'])
    x_r_tst = pre.restore_matrix(s=r_tst, m=m, header=d['header'])

    # write to file
    np.savetxt(fname=file_real_trn, fmt='%s', X=x_r_trn['x'], delimiter=',', header=','.join(x_r_trn['header']))
    np.savetxt(fname=file_real_tst, fmt='%s', X=x_r_tst['x'], delimiter=',', header=','.join(x_r_tst['header']))
    np.savetxt(fname=file_synth, fmt='%s', X=f['x'], delimiter=',', header=','.join(f['header']))
    
    # report
    idx_outcome = np.unique(np.append(np.where(d['header']==outcome), np.where(d['header']==outcome+delim+outcome)))
    outcome_label = d['header'][idx_outcome][0]
    report_status_desc = rep.description_report(r_trn=r_trn, r_tst=r_tst, 
                             s_trn=s_trn, s_tst=s_tst, col_names=d['header'], 
                             outcome=outcome_label, file_pdf=file_desc_pdf, 
                             n_epoch=100, model_type='lr')
    report_status_pred = rep.prediction_report(r_trn=r_trn, r_tst=r_tst, 
                             s_trn=s_trn, s_tst=s_tst, col_names=d['header'], 
                             outcome=outcome_label, file_pdf=file_pred_pdf, 
                             n_epoch=100, model_type='lr')
        
    # summary
    if report_status_desc:
        print('Description report written to', file_desc_pdf)
    else:
        print('Error: description report generation failed')
    if report_status_pred:
        print('Description report written to', file_pred_pdf)
    else:
        print('Error: prediction report generation failed')
    
if __name__ == "__main__":
    main()
