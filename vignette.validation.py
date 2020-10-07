import numpy as np
from preprocessor import preprocessor
from corgan import corgan
from realism import realism
from os.path import isfile
import pickle

def save_obj(obj, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)
    
def generate_unique_strings(n):
    
    a = 97
    z = a + 25
    strs = np.full(shape=n,fill_value='aaa')
    index = 0
    
    for i in range(a,z):
        for j in range(a,z):
            for k in range(a,z):
                
                if index >= n:
                    break
                strs[index] = chr(i)+chr(j)+chr(k)
                index += 1
                
    return(strs)

def main():
    
    # files
    file_real = '../output/demo_raw.csv'
    file_model = '../output/demo_corgan.pkl'
    
    # parameters
    n = 10000
    n_gen = n
    missing_value = '-99999'
    delim = '__'
    use_saved_model = False
    
    # sweeps
    binary_prob = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    categorical_k = [3,10,30,100,300,1000]
    count_lambda = [0.3, 1, 3, 10, 30, 100]
    
    # sehrd objects
    pre = preprocessor(missing_value=missing_value)
    cor = corgan()
    rea = realism(missing_value=missing_value)
    
    if isfile(file_real) and use_saved_model:
        x = np.loadtxt(file_real, dtype=str, delimiter=',')
    
    else:
        
        # constant variable
        names = ['constant']
        x = np.zeros(shape=(n,1))
        
        # binary 
        for i in range(len(binary_prob)):
            names = np.append(names, 'binary_'+str(binary_prob[i]))
            x = np.hstack((x, np.random.binomial(n=1, p=binary_prob[i], size=(n,1))))
            
        # categorical
        for i in range(len(categorical_k)):
            
            uniq_vals = generate_unique_strings(n=categorical_k[i])
            
            # uniform 
            names = np.append(names, 'categorical_uniform_'+str(categorical_k[i]))
            x = np.hstack((x, np.random.choice(uniq_vals, size=(n,1))))
            
            # exponential
            names = np.append(names, 'categorical_exponential_'+str(categorical_k[i]))
            p = np.random.exponential(scale=1.0, size=categorical_k[i])
            p = p / sum(p)
            x = np.hstack((x, np.random.choice(uniq_vals, size=(n,1), p=p)))
            
        # count
        for i in range(len(count_lambda)):
            names = np.append(names, 'count_'+str(count_lambda[i]))
            x = np.hstack((x, np.random.poisson(lam=count_lambda[i], size=(n,1))))
            
        # continuous, uniform
        names = np.append(names, 'continuous_uniform')
        x = np.hstack((x, np.random.uniform(low=0.0, high=1.0, size=(n,1))))
       
        # continuous, exponential
        names = np.append(names, 'continuous_exponential')
        x = np.hstack((x, np.random.normal(loc=0, scale=1, size=(n,1))))
        
        # continuous, normal
        names = np.append(names, 'continuous_normal')
        x = np.hstack((x, np.random.normal(loc=0, scale=1, size=(n,1))))
        
        # save real demo data
        np.savetxt(fname=file_real, fmt='%s', X=x, delimiter=',', header=','.join(names))
    
    # preprocess
    m = pre.get_metadata(x=x, header=names)
    d = pre.get_discretized_matrix(x, m, names, delim=delim)

    # generate synthetic data
    if isfile(file_model) and use_saved_model:
        model = load_obj(file_model)
    else:
        model = cor.train(x=d['x'], n_cpu=15, debug=True)
        save_obj(model, file_model)
    s = cor.generate(model, n_gen)
    
    # reconstruct and save synthetic data
    f = pre.restore_matrix(s=s, m=m, header=d['header'], delim=delim)
    
    # distance between all features
    dist_feat = np.zeros(len(m))
    for j in range(len(m)):
        dist_feat[j] = rea.validate_feature(r_feat=d['x'][:,j], s_feat=f['x'][:,j], var_type=m[j]['type'])
        
    print(dist_feat)
        
    
if __name__ == "__main__":
    main()
