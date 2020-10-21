import argparse
import sys
import multiprocessing
import numpy as np
from os.path import isfile

# sehrd packages
from preprocessor import preprocessor
from corgan import corgan
from realism import realism
from privacy import privacy

# cli ------------------------

# valid values (default is always first)
choices_train_type = ['corgan']
choices_report_type = ['feature_frequency','feature_effect',
                    'gan_train_test','nearest_neighbors',
                    'membership_inference']

# parsers
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help='task to perform', 
                                   dest='task',
                                   required=True)
parser_t = subparsers.add_parser('train', 
                    help='train a synthetic data generator')
parser_g = subparsers.add_parser('generate', 
                    help='generate synthetic dataset')
parser_r = subparsers.add_parser('validate', 
                    help='validate synthetic dataset')

# subparser: train
parser_t.add_argument('--file_data', 
                    help='path to file containing real data', 
                    type=str,
                    required=True)
parser_t.add_argument('--outprefix_train', 
                    help='file prefix to model pkl dump', 
                    type=str, 
                    required=True)
parser_t.add_argument('--train_type', 
                    help='type of generative model to train', 
                    choices=choices_train_type,
                    default=choices_train_type[0],
                    required=False)
parser_t.add_argument('--n_epoch_pre', 
                    help='number of pre-training epochs (>0)', 
                    type=int,
                    default=100,
                    required=False)
parser_t.add_argument('--n_epoch', 
                    help='number of training epochs (>0)', 
                    type=int,
                    default=100,
                    required=False)
parser_t.add_argument('--missing_value',
                    help='representation of missing value',
                    type=str,
                    default='-999999',
                    required=False)
parser_t.add_argument('--frac_train', 
                    help='fraction of real data to train generative model', 
                    type=float,
                    default=0.75,
                    required=False)
parser_t.add_argument('-v', '--verbose', 
                    help='print verbose output', 
                    action='store_true', 
                    required=False)
parser_t.add_argument('--n_cpu_train', 
                    help='number of CPUs to use (>0)', 
                    type=int, 
                    default=1,
                    required=False)

# subparser: generate
parser_g.add_argument('--file_model', 
                    help='path to file containing trained generator', 
                    type=str,
                    required=True)
parser_g.add_argument('-o', '--outprefix_generate', 
                    help='file prefix to synthetic data CSV', 
                    type=str, 
                    required=True)
parser_g.add_argument('--generate_size', 
                    help='number of synthetic samples to generate (>0)', 
                    type=int,
                    default=1000,
                    required=False)
parser_g.add_argument('-v', '--verbose', 
                    help='print verbose output', 
                    action='store_true', 
                    required=False)
parser_g.add_argument('--n_cpu_generate', 
                    help='number of CPUs to use (>0)', 
                    type=int, 
                    default=1,
                    required=False)

# subparser: report
parser_r.add_argument('--outprefix_report', 
                    metavar='OUTPREFIX',
                    help='file prefix for validation results', 
                    type=str, 
                    required=True)
parser_r.add_argument('--feature_frequency', 
                    help='compare real and synthetic feature frequency', 
                    action='store_true',
                    required=False)
parser_r.add_argument('--effect_size', 
                    metavar='OUTCOME',
                    help='compare real and synthetic feature effect sizes', 
                    type=str,
                    default='outcome',
                    required=False)
parser_r.add_argument('--gan_train_test', 
                    help='compare real and synthetic feature effect sizes', 
                    type=str,
                    metavar='OUTCOME',
                    default='outcome',
                    required=False)
parser_r.add_argument('--nearest_neighbors', 
                    help='compare nearest neighbor distances', 
                    action='store_true',
                    required=False)
parser_r.add_argument('--membership_inference', 
                    help='calculate risk of membership inference', 
                    action='store_true',
                    required=False)
parser_r.add_argument('--plot', 
                    help='plot applicable results', 
                    action='store_true',
                    required=False)

# check user input -------------

# print help and exit if no arguments specified
if len(sys.argv)==1:
    parser.print_help()
    sys.exit(0)
    
# get command line arguments
args = parser.parse_args()
print(args)


# check arguments for 'train' task        
if args.task == 'train':
    
    if args.n_epoch_pre <= 0:
        parser.print_usage()
        print('sehrd.py: error: argument --n_epoch_pre: invalid choice: \'' +
              str(args.n_epoch_pre) + '\' (choose integer greater than 0)')
        sys.exit(0)
    
    if args.n_epoch <= 0:
        parser.print_usage()
        print('sehrd.py: error: argument --n_epoch: invalid choice: \'' +
              str(args.n_epoch) + '\' (choose integer greater than 0)')
        sys.exit(0)
    
    max_cpu = multiprocessing.cpu_count()
    if args.n_cpu_train < 1 or args.n_cpu_train > max_cpu:
        parser.print_usage()
        print('sehrd.py: error: argument --n_cpu_train: invalid choice: \'' +
              str(args.n_cpu_train) + '\' (choose integer in range [1,' + 
              str(max_cpu) + '])')
        sys.exit(0)
    
    if args.frac_train < 0 or args.frac_train > 1:
        parser.print_usage()
        print('sehrd.py: error: argument --frac_train: invalid choice: \'' +
              str(args.n_cpu) + '\' (choose float in range [0,1])')
        sys.exit(0)
    
    if not isfile(args.file_data):
        parser.print_usage()
        print('sehrd.py: error: argument --file_data: file does not exist: \'' +
              str(args.file_data) + '\' (check path to file)')
        sys.exit(0)
        
# check arguments for 'generate' task
if args.task == 'generate':
    
    if args.generate_size <= 0:
        parser.print_usage()
        print('sehrd.py: error: argument --generate_size: invalid choice: \'' +
              str(args.generate_size) + '\' (choose integer greater than 0)')
        sys.exit(0)
        
    if not isfile(args.file_model):
        parser.print_usage()
        print('sehrd.py: error: argument --file_model: file does not exist: \'' +
              str(args.file_model) + '\' (check path to file)')
        sys.exit(0)
        
    max_cpu = multiprocessing.cpu_count()
    if args.n_cpu_generate < 1 or args.n_cpu_generate > max_cpu:
        parser.print_usage()
        print('sehrd.py: error: argument --n_cpu_generate: invalid choice: \'' +
              str(args.n_cpu_generate) + '\' (choose integer in range [1,' + 
              str(max_cpu) + '])')
        sys.exit(0)
            
# scenarios --------------

if args.task == 'train':
    
    # instantiate
    pre = preprocessor(missing_value=args.missing_value)
    cor = corgan()
    outfile = args.outprefix_train + '.pkl'
        
    # load real data
    ftr = pre.read_file(args.file, has_header=True)
    header = ftr['header']
    x = ftr['x']
        
    # preprocess
    m = pre.get_metadata(x=x, header=header)
    d = pre.get_discretized_matrix(x=x, m=m, header=header)
    
    # split data
    r = d['x']
    n_subset_r = round(len(r) * args.frac_train)
    idx_trn = np.random.choice(len(r), n_subset_r, replace=False)
    idx_tst = np.setdiff1d(range(len(r)), idx_trn)
    r_trn = r[idx_trn,:]
    r_tst = r[idx_tst,:]
    
    # train and save model
    model = cor.train(x=r_trn, n_cpu=args.n_cpu_train, debug=False, 
                      n_epochs=args.n_epoch, 
                      n_epochs_pretrain=args.n_epoch_pre)
    model['m'] = m
    model['header'] = d['header']
    save_obj(model, outfile)
    
elif args.task == 'generate':
    
    cor = corgan()
    outfile = args.outprefix_generate + '.csv'
    
    model = load_obj(args.file_model)
    s = cor.generate(model, n_gen=args.generate_size)
    
    f = pre.restore_matrix(s=s, m=model['m'], header=model['header'])
    np.savetxt(fname=outfile, fmt='%s', X=f['x'], delimiter=',', 
               header=','.join(f['header']))
    
elif args.task == 'validate':

    rea = realism()
    pri = privacy()
    outfile = args.outprefix_report + '.pkl'
    outplot = args.outprefix_report + '.pdf'
     
        
        


