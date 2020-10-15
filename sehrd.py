import argparse
import sys
import multiprocessing
from os.path import isfile

# sehrd packages
from preprocessor import preprocessor
from corgan import corgan
from report import report

# cli ------------------------

# valid values (default is always first)
choices_train_type = ['corgan']
choices_report_type = ['prediction','description']

# command line interface setup
parser = argparse.ArgumentParser()

# file paths and prefixes
parser.add_argument('-i', '--infile_prefix', help='prefix to input files', type=str)
parser.add_argument('-o', '--outfile_prefix', help='prefix to output files', type=str)

# flags
parser.add_argument('-t', '--train', help='train a generative model', action='store_true')
parser.add_argument('-g', '--generate', help='generate synthetic samples', action='store_true')
parser.add_argument('-r', '--report', help='generate a synthetic data report', action='store_true')
parser.add_argument('-v', '--verbose', help='print verbose output', action='store_true')

# scenario options
parser.add_argument('--train_type', 
                    help='type of generative model to train', 
                    choices=choices_train_type,
                    default=choices_train_type[0])
parser.add_argument('--generate_size', 
                    help='number of synthetic samples to generate (>0)', 
                    type=int,
                    default=1000)
parser.add_argument('--report_type', 
                    help='type of synthetic data report to generate', 
                    choices=choices_report_type,
                    default=choices_report_type[0])

# other options
parser.add_argument('--n_cpu', 
                    help='number of CPUs to use (>0)', 
                    type=int,
                    default=1)
parser.add_argument('--n_epoch_pre', 
                    help='number of pre-training epochs (>0)', 
                    type=int,
                    default=100)
parser.add_argument('--n_epoch', 
                    help='number of training epochs (>0)', 
                    type=int,
                    default=100)
parser.add_argument('--missing_value',
                    help='representation of missing value',
                    type=str,
                    default='-999999')
parser.add_argument('--outcome',
                    help='column name of feature to be used as prediction target',
                    type=str,
                    default='outcome')
parser.add_argument('--frac_train', 
                    help='fraction of real data to train generative model', 
                    type=float,
                    default=0.75)

# check user input -------------

# print help and exit if no arguments specified
if len(sys.argv)==1:
    parser.print_help()
    sys.exit(0)
    
# get command lines
args = parser.parse_args()

# check user input
if args.generate_size <= 0:
    parser.print_usage()
    print('sehrd.py: error: argument --generate_size: invalid choice: \'' +
          str(args.generate_size) + '\' (choose integer greater than 0)')
    sys.exit(0)
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
if args.n_cpu < 1 or args.n_cpu > max_cpu:
    parser.print_usage()
    print('sehrd.py: error: argument --n_cpu: invalid choice: \'' +
          str(args.n_cpu) + '\' (choose integer in range [1,' + 
          str(max_cpu) + '])')
    sys.exit(0)
if args.frac_train < 0 or args.frac_train > 1:
    parser.print_usage()
    print('sehrd.py: error: argument --frac_train: invalid choice: \'' +
          str(args.n_cpu) + '\' (choose float in range [0,1])')
    sys.exit(0)
    
# scenarios --------------

# initilize
pre = preprocessor(missing_value=args.missing_value)
rep = report(missing_value=args.missing_value)
cor = corgan()

# loading real data, if applicable
if args.generate and not (args.train or args.report):
    if(args.verbose):
        print('Skipping real data loading')
else:
    
    # check that args.file_real exists
    if not isfile(args.file_real):
        
    
    # load real data
    ftr = pre.read_file(args.file_real, has_header=True)
    header = ftr['header']
    x = ftr['x']
        
    # preprocess
    m = pre.get_metadata(x=x, header=header)
    d = pre.get_discretized_matrix(x=x, m=m, header=header)
    
    # split data
    r = d['x']
    n_subset_r = round(len(r) * trn_frac)
    idx_trn = np.random.choice(len(r), n_subset_r, replace=False)
    idx_tst = np.setdiff1d(range(len(r)), idx_trn)
    r_trn = r[idx_trn,:]
    r_tst = r[idx_tst,:]

    
if args.train and args.generate and args.report: 
    
    if args.verbose:
        print('Scenario 1: train, generate, report')
        
    
    
    # train
    
    # generate
    
    # report

elif args.train and args.generate:
    
    if args.verbose:
        print('Scenario 2: train, generate')
        
elif args.generate and args.report: 
    
    if args.verbose:
        print('Scenario 3: generate, report')
        
elif args.train and args.report: 
    
    if args.verbose:
        print('Scenario 4: train, report')
        
elif args.train: 
    
    if args.verbose:
        print('Scenario 5: train')
        
elif args.generate: 
    
    if args.verbose:
        print('Scenario 6: generate')

elif args.report: 
    
    if args.verbose:
        print('Scenario 7: report')
        
