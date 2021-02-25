import argparse
import sys
import multiprocessing
import numpy as np
from os.path import isfile
import datetime as dt

# sehrd packages
from preprocessor import Preprocessor
from corgan import Corgan
#from ppgan import Ppgan
from realism import Realism
from privacy import Privacy

# cli ------------------------

# valid values (default is always first)
choices_train_type = ['corgan', 'ppgan']
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
parser_t.add_argument('--verbose',
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
parser_g.add_argument('--outprefix_generate',
                    help='file prefix to synthetic data CSV',
                    type=str,
                    required=True)
parser_g.add_argument('--generate_size',
                    help='number of synthetic samples to generate (>0)',
                    type=int,
                    default=1000,
                    required=False)
parser_g.add_argument('--n_cpu_generate',
                    help='number of CPUs to use (>0)',
                    type=int,
                    default=1,
                    required=False)
parser_g.add_argument('--missing_value_generate',
                    help='representation of missing value',
                    type=str,
                    default='-999999',
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
                    choices=choices_analysis_realism,
                    default=choices_analysis_realism[0],
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
parser_p.add_argument('--missing_value_privacy',
                    help='representation of missing value',
                    type=str,
                    default='-999999',
                    required=False)
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
parser_p.add_argument('--sample_privacy',
                    help='number of samples on which to conduct privacy analysis',
                    type=int,
                    default=10000,
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
              str(args.n_cpu_train) + '\' (choose float in range [0,1])')
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

tic = dt.datetime.now()
print('\n------------------------------')
print('\nStarted: ' + str(tic.replace(microsecond=0)))
print()
print('Command: ')
print(args)

# tasks --------------

outfile = 'none'

if args.task == 'train':

    # instantiate
    pre = Preprocessor(missing_value=args.missing_value)
    outfile = args.outprefix_train + '.pkl'
    
    if args.verbose:
        debug = True
    else:
        debug = False

    # load real data
    ftr = pre.read_file(args.file_data, has_header=True)
    header = ftr['header']
    arr = ftr['x']

    # preprocess
    meta = pre.get_metadata(arr=arr, header=header)
    obj_d = pre.get_discretized_matrix(arr=arr, meta=meta, header=header)

    # split data
    r_all = obj_d['x']
    n_subset_r = round(len(r_all) * args.frac_train)
    idx_trn = np.random.choice(len(r_all), n_subset_r, replace=False)
    idx_tst = np.setdiff1d(range(len(r_all)), idx_trn)
    r_trn = r_all[idx_trn,:]
    r_tst = r_all[idx_tst,:]

    # train and save model
    if args.train_type == 'corgan':
        syn = Corgan(debug=debug, n_cpu=args.n_cpu_train)
    elif args.train_type == 'ppgan':
        syn = Ppgan(debug=debug, n_cpu=args.n_cpu_train)
    model = syn.train(x=r_trn, n_epochs=args.n_epoch)
    model['m'] = meta
    model['header'] = obj_d['header']
    syn.save_obj(model, outfile)

elif args.task == 'generate':

    pre = Preprocessor(missing_value=args.missing_value_generate)
    outfile = args.outprefix_generate + '.csv'

    syn = Corgan()
    model = syn.load_obj(args.file_model)
    if model['parameter_dict']['model'] == 'corgan':
        syn = Corgan()
    elif model['parameter_dict']['model'] == 'ppgan':
        syn = Ppgan(debug=False, n_cpu=1)
    
    s = syn.generate(model, n_gen=args.generate_size)

    f = pre.restore_matrix(arr=s, meta=model['m'], header=model['header'])
    np.savetxt(fname=outfile, fmt='%s', X=f['x'], delimiter=',',
               header=','.join(f['header']), comments = '')

elif args.task == 'realism':

    rea = Realism()
    pre = Preprocessor(missing_value=args.missing_value_realism)
    r_trn = pre.read_file(args.file_realism_real_train)
    r_tst = pre.read_file(args.file_realism_real_test)
    s = pre.read_file(args.file_realism_synth)

    # analysis
    if args.analysis_realism == 'feature_frequency':
        res = rea.feature_frequency(mat_f_r_trn=r_trn['x'],
                                    mat_f_r_tst=r_tst['x'],
                                    mat_f_s=s['x'],
                                    header=r_trn['header'],
                                    missing_value=args.missing_value_realism)

    elif args.analysis_realism == 'feature_effect':
        res = rea.feature_effect(mat_f_r_trn=r_trn['x'],
                                 mat_f_r_tst=r_tst['x'],
                                 mat_f_s=s['x'],
                                 header=r_trn['header'],
                                 outcome=args.outcome,
                                 missing_value=args.missing_value_realism)

    elif args.analysis_realism == 'gan_train_test':
        res = rea.gan_train_test(mat_f_r_trn=r_trn['x'],
                                 mat_f_r_tst=r_tst['x'],
                                 mat_f_s=s['x'],
                                 header=r_trn['header'],
                                 outcome=args.outcome,
                                 n_epoch=100,
                                 model_type='lr',
                                 missing_value=args.missing_value_realism)

    else:
        print('Error: do not recognize analysis_realism option '
              + args.analysis_realism)
        sys.exit(0)

    # output
    if args.output_realism == 'file':
        outfile = args.outprefix_realism + '_' + args.analysis_realism + '.pkl'
        res['analysis'] = args.analysis_realism
        rea.save_obj(res, file_name=outfile)

    elif args.output_realism == 'plot':
        outfile = args.outprefix_realism + '_' + args.analysis_realism+ '.pdf'
        res['analysis'] = args.analysis_realism
        rea.plot(res, file_pdf=outfile)

    elif args.output_realism == 'summary':
        msg = rea.summarize(res)
        print(msg)

    elif args.output_realism == 'all':
        res['analysis'] = args.analysis_realism
        outfile = args.outprefix_realism + '_' + args.analysis_realism + '.pkl'
        rea.save_obj(res, file_name=outfile)

        outfile = args.outprefix_realism + '_' + args.analysis_realism+ '.pdf'
        rea.plot(res, file_pdf=outfile)

        msg = rea.summarize(res)
        print(msg)

    else:
        print('Error: do not recognize output_realism option '
              + args.output_realism)
        sys.exit(0)

elif args.task == 'privacy':

    pri = Privacy()
    pre = Preprocessor(missing_value=args.missing_value_privacy)
    r_trn = pre.read_file(args.file_privacy_real_train)
    r_tst = pre.read_file(args.file_privacy_real_test)
    s = pre.read_file(args.file_privacy_synth)
    
    # subsample 
    if args.sample_privacy < len(s['x']):
        idx = np.random.choice(range(len(s['x'])), args.sample_privacy, replace=False)
        s['x'] = s['x'][idx,:]
    if args.sample_privacy < len(r_trn['x']):
        idx = np.random.choice(range(len(r_trn['x'])), args.sample_privacy, replace=False)
        r_trn['x'] = r_trn['x'][idx,:]
    if args.sample_privacy < len(r_tst['x']):
        idx = np.random.choice(range(len(r_tst['x'])), args.sample_privacy, replace=False)
        r_tst['x'] = r_tst['x'][idx,:]

    # analysis
    if args.analysis_privacy == 'nearest_neighbors':
        res = pri.assess_memorization(mat_f_r=r_trn['x'],
                                    mat_f_s=s['x'],
                                    missing_value=args.missing_value_privacy,
                                    header=r_trn['header'])
    elif args.analysis_privacy == 'membership_inference':
        res = pri.membership_inference(mat_f_r_trn=r_trn['x'],
                                    mat_f_r_tst=r_tst['x'],
                                    mat_f_s=s['x'],
                                    header=r_trn['header'],
                                    missing_value=args.missing_value_privacy)
    else:
        print('Error: do not recognize analysis_privacy option '
              + args.analysis_privacy)
        sys.exit(0)

    # output
    if args.output_privacy == 'file':
        outfile = args.outprefix_privacy + '_' + args.analysis_privacy + '.pkl'
        res['analysis'] = args.analysis_privacy
        pri.save_obj(res, file_name=outfile)

    elif args.output_privacy == 'plot':
        # TODO: write plot_privacy function
        outfile = args.outprefix_privacy + '_' + args.analysis_privacy+ '.pdf'
        res['analysis'] = args.analysis_privacy
        pri.plot(res, file_pdf=outfile)

    elif args.output_privacy == 'summary':
        #TODO: write summarize_privacy function
        res['analysis'] = args.analysis_privacy
        msg = pri.summarize(res)
        print(msg)

    elif args.output_privacy == 'all':
        res['analysis'] = args.analysis_privacy
        outfile = args.outprefix_privacy + '_' + args.analysis_privacy + '.pkl'
        pri.save_obj(res, file_name=outfile)

        outfile = args.outprefix_privacy + '_' + args.analysis_privacy+ '.pdf'
        pri.plot(res, file_pdf=outfile)

        msg = pri.summarize(res)
        print(msg)

    else:
        print('Error: do not recognize output_privacy option '
              + args.output_privacy)
        sys.exit(0)

# final message -----------------

toc = dt.datetime.now()
print()
print('Output file: ' + outfile)
print('Runtime: ' + str(toc - tic))
print('\nCompleted: ' + str(toc.replace(microsecond=0)))
print('------------------------------\n')
