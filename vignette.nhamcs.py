import numpy as np
from torch.utils.data import random_split
from preprocessor import preprocessor
from corgan import corgan
from report import report
from privacy import privacy
from os.path import isfile
import pickle

def save_obj(obj, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def main():
    
    # files
    outcome = 'ADMITHOS'
    file_ftr = "../data/nhamcs.raw.feather"
    file_pdf = '../plots/vignette.nhamcs.description_report.pdf'
    file_real_trn =  '../output/nhamcs_real_train.csv'
    file_real_tst =  '../output/nhamcs_real_test.csv'
    file_aux_trn = '../output/nhamcs_aux_train.csv'
    file_aux_tst = '../output/nhamcs_aux_test.csv'
    file_synth = '../output/nhamcs_synth.csv'
    file_model = '../output/nhamcs.corgan.pkl'
    
    # parameters
    missing_value = '-999999'
    delim = '__'
    use_saved_model = False
    n_epoch = 100
    n_epoch_pretrain = 150
    
    pre = preprocessor(missing_value=missing_value)
    rep = report()
    cor = corgan()
    pri = privacy()
    
    # read nhamcs dataset
    ftr = pre.read_file(file_ftr)
    header = ftr['header']
    x = ftr['x']
    
    # preprocess
    m = pre.get_metadata(x=x, header=header)
    d = pre.get_discretized_matrix(x, m, header, delim=delim)
    
    # split 
    if isfile(file_real_trn) and use_saved_model:
        r_trn = load_obj(file_real_trn)
        r_tst = load_obj(file_real_tst)
        a_trn = load_obj(file_aux_trn)
        a_tst = load_obj(file_aux_tst)
    else:
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

    # train generator
    if isfile(file_model) and use_saved_model:
        model = load_obj(file_model)
    else:
        model = cor.train(x=r_trn, n_cpu=15, debug=True,
                n_epochs=n_epoch, n_epochs_pretrain=n_epoch_pretrain)
        save_obj(model, file_model)
    
    # generate synthetic samples
    s = cor.generate(model, n_gen=len(r_tst))
    n_subset_s = round(len(s)*0.75)
    s_trn, s_tst = random_split(s, [n_subset_s, len(s)-n_subset_s])
    s_trn = np.array(s_trn)
    s_tst = np.array(s_tst)
    
    # reconstruct and save synthetic data
    f = pre.restore_matrix(s=s, m=m, header=d['header'])
    x_r_trn = pre.restore_matrix(s=r_trn, m=m, header=d['header'])
    x_r_tst = pre.restore_matrix(s=r_tst, m=m, header=d['header'])
    x_a_trn = pre.restore_matrix(s=a_trn, m=m, header=d['header'])
    x_a_tst = pre.restore_matrix(s=a_tst, m=m, header=d['header'])
    
    # write to file
    np.savetxt(fname=file_real_trn, fmt='%s', X=x_r_trn['x'], delimiter=',', header=','.join(x_r_trn['header']))
    np.savetxt(fname=file_real_tst, fmt='%s', X=x_r_tst['x'], delimiter=',', header=','.join(x_r_tst['header']))
    np.savetxt(fname=file_aux_trn, fmt='%s', X=x_a_trn['x'], delimiter=',', header=','.join(x_a_trn['header']))
    np.savetxt(fname=file_aux_tst, fmt='%s', X=x_a_tst['x'], delimiter=',', header=','.join(x_a_tst['header']))
    np.savetxt(fname=file_synth, fmt='%s', X=f['x'], delimiter=',', header=','.join(f['header']))
    
    # report
    idx_outcome = np.unique(np.append(np.where(d['header']==outcome), np.where(d['header']==outcome+delim+outcome)))
    outcome_label = d['header'][idx_outcome][0]
    report_status = rep.description_report(r_trn=r_trn, r_tst=r_tst, s_trn=s_trn, 
                             s_tst=s_tst, col_names=d['header'], 
                             outcome=outcome_label,  file_pdf=file_pdf, 
                             n_epoch=100, model_type='lr', penalty='l1')
    
    # membership inference
    mem_inf = pri.membership_inference(r_trn=r_trn, r_tst=r_tst, s_trn=s_trn, s_tst=s_tst,
                         a_trn=a_trn, a_tst=a_tst, model_type='svm')
    
    # summary
    if report_status:
        print('Report written to', file_pdf)
    else:
        print('Error: report generation failed')
    print('Synthetic dataset written to', file_synth)
    print('AUC for authentic-synthetic: ', mem_inf['auc_as'])
    print('AUC for real train-test: ', mem_inf['auc_rr'])
   
if __name__ == "__main__":
    main()
