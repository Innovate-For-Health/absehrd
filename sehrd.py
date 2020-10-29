import argparse
import sys
import multiprocessing
import numpy as np
from os.path import isfile
import datetime as dt

# sehrd packages
from preprocessor import preprocessor
from corgan import corgan
from realism import realism
from privacy import privacy

# cli ------------------------

# valid values (default is always first)
choices_train_type = ['corgan']
choices_analysis_realism = ['feature_frequency',
                    'feature_effect',
                    'gan_train_test']
choices_analysis_privacy = ['nearest_neighbors',
                    'membership_inference']
choices_output = ['summary', 'file', 'plot', 'all']

# parsers
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help='task to perform', 
                                   dest='task',
                                   required=True)
parser_t = subparsers.add_parser('train', 
                    help='train a synthetic data generator')
parser_g = subparsers.add_parser('generate', 
                    help='generate synthetic dataset')
parser_r = subparsers.add_parser('realism', 
                    help='assess realism of synthetic dataset')
parser_p = subparsers.add_parser('privacy', 
                    help='assess privacy risk of synthetic dataset')

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

# subparser: realism
parser_r.add_argument('--outprefix_realism', 
                    metavar='OUTPREFIX',
                    help='file prefix for realism assessments', 
                    type=str, 
                    required=True)
parser_r.add_argument('--file_realism_real_train', 
                    help='file containing real data used to train the synthetic data generator', 
                    type=str,
                    required=True)
parser_r.add_argument('--file_realism_real_test', 
                    help='file containing real data not used to train the synthetic data generator', 
                    type=str,
                    required=True)
parser_r.add_argument('--file_realism_synth', 
                    help='file containing synthetic data', 
                    type=str,
                    required=True)
parser_r.add_argument('--outcome',
                      help='outcome for realism metrics',
                      type=str,
                      required=False)
parser_r.add_argument('--missing_value_realism',
                    help='representation of missing value',
                    type=str,
                    default='-999999',
                    required=False)
parser_r.add_argument('--analysis_realism',
                    help='type of realism validation analysis',
                    choices=choices_realism_privacy,
                    default=choices_realism_privacy[0],
                    required=False)
parser_r.add_argument('--output_realism', 
                    help='type of output for realism analysis', 
                    choices=choices_output,
                    default=choices_output[0],
                    required=False)

# subparser: privacy
parser_p.add_argument('--outprefix_privacy', 
                    metavar='OUTPREFIX',
                    help='file prefix for realism assessments', 
                    type=str, 
                    required=True)
parser_p.add_argument('--file_privacy_real_train', 
                    help='file containing real data used to train the synthetic data generator', 
                    type=str,
                    required=True)
parser_p.add_argument('--file_privacy_real_test', 
                    help='file containing real data not used to train the synthetic data generator', 
                    type=str,
                    required=True)
parser_p.add_argument('--file_privacy_synth', 
                    help='file containing synthetic data', 
                    type=str,
                    required=True)
parser_p.add_argument('--analysis_privacy',
                    help='type of privacy validation analysis',
                    choices=choices_analysis_privacy,
                    default=choices_analysis_privacy[0],
                    required=False)
parser_p.add_argument('--output_privacy', 
                    help='type of output for privacy analysis', 
                    choices=choices_output,
                    default=choices_output[0],
                    required=False)

# check user input -------------

# print help and exit if no arguments specified
if len(sys.argv)==1:
    parser.print_help()
    sys.exit(0)
    
# get command line arguments
args = parser.parse_args()

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
  
# initial message -----------------

# TODO: porint initial message to user
#   - start time
#   - command 
tic = dt.datetime.now()
print()
print('Command ')
print(args)
print('Started at ' + str(tic.replace(microsecond=0)))


          
# tasks --------------

outfile = 'none'

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
    cor.save_obj(model, outfile)
    
elif args.task == 'generate':
    
    cor = corgan()
    outfile = args.outprefix_generate + '.csv'
    
    model = cor.load_obj(args.file_model)
    s = cor.generate(model, n_gen=args.generate_size)
    
    f = pre.restore_matrix(s=s, m=model['m'], header=model['header'])
    np.savetxt(fname=outfile, fmt='%s', X=f['x'], delimiter=',', 
               header=','.join(f['header']))
    
elif args.task == 'realism':

    rea = realism()
    pre = preprocessor(missing_value=args.missing_value_realism)
    r_trn = pre.read_file(args.file_realism_train)
    r_tst = pre.read_file(args.file_realism_test)
    s = pre.read_file(args.file_realism_synth)
        
    # analysis
    if args.analysis_realism == 'feature_frequency':
        res = rea.feature_frequency(r_trn=r_trn['x'], r_tst=r_tst['x'], s=s['x'], 
                                    header=r_trn['header'],
                                    missing_value=args.missing_value_realism)
        
    elif args.analysis_realism == 'feature_effect':
        res = rea.feature_effect(r_trn=r_trn['x'], r_tst=r_tst['x'], s=s['x'], 
                                    header=r_trn['header'],
                                    missing_value=args.missing_value_realism)
        
    elif args.analysis_realism == 'gan_train_test':
        res = rea.gan_train_test(r_trn, r_tst, s, args.outcome)
        
    else:
        print('Error: do not recognize analysis_realism option ' 
              + args.analysis_realism)
        sys.exit(0)
        
    # output
    if args.output_realism == 'file':
        outfile = args.outprefix_realism + '_' + args.analysis_realism + '.pkl'
        rea.save_obj(res, file_name=outfile)
        
    elif args.output_realism == 'plot':
        outfile = args.outprefix_realism + '_' + args.analysis_realism+ '.pdf'
        rea.plot(res, analysis=args.output_realism, file_pdf=outfile)
    
    elif args.output_realism == 'summary':
        msg = rea.summarize(res, analysis=args.output_realism)
        print(msg)
        
    else:
        print('Error: do not recognize output_realism option ' 
              + args.output_realism)
        sys.exit(0)
        
elif args.task == 'privacy':

    pri = privacy()

    # analysis
    if args.analysis_privacy == 'nearest_neighbors':
        res = None
    elif args.analysis_privacy == 'membership_inference':
        res = None
    else:
        print('Error: do not recognize analysis_privacy option ' 
              + args.analysis_privacy)
        sys.exit(0)
    
    # output
    if args.output_privacy == 'file':
        outfile = args.outprefix_privacy + '_' + args.analysis_privacy + '.pkl'
        rea.save_obj(res, file_name=outfile)
        
    elif args.output_privacy == 'plot':
        # TODO: write plot_privacy function
        outfile = args.outprefix_privacy + '_' + args.analysis_privacy+ '.pdf'
        pri.plot(res, analysis=args.output_privacy, file_pdf=outfile)
    
    elif args.output_privacy == 'summary':
        #TODO: write summarize_privacy function
        msg = pri.summarize(res, analysis=args.output_privacy)
        print(msg)
        
    else:
        print('Error: do not recognize output_privacy option ' 
              + args.output_privacy)
        sys.exit(0)
        
# final message -----------------

toc = dt.datetime.now()
print()
print('Command ')
print(args)
print('Completed at ' + str(toc.replace(microsecond=0)))
print('Runtime was ' + str(toc - tic))
print('Output file: ' + outfile)


