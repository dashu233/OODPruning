# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid
import copy

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

from utils import check_sparsity,speed_up_inference_for_channel
from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from domainbed.kde import opt_kde
from domainbed.feature_checker import feature_extractor_for_pipline

import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

def torch_to_numpy(d):
    return {
        key: d[key].cpu().numpy()
        for key in d.keys()
        if d[key] is not None
    }

def to_str(lis):
    s = ""
    for w in lis:
        s = s + str(w).ljust(10," ") + ", "
    return s

def calculate_variation(algorithm,step,
                        eval_loader_names, eval_loaders,dataset,args,whether_write=False):
    # Return:
    # a list of shape [9][4]
    # each line contains: train variation, test variation, train_info, remain_feature
    # recommend: use [6][0](train) or [6][1](test) as a criterion of variation
    if args.output_dir[-1] == '/':
        marker = args.output_dir + "extracted_{}".format(step)
    else:
        marker = args.output_dir + "/" + "extracted_{}".format(step)

    datas = feature_extractor_for_pipline(algorithm, zip(
        eval_loader_names, eval_loaders), device, dataset.num_classes, marker)
    env_list = ['env{}'.format(i) for i in range(len(dataset))]
    train_env = copy.deepcopy(env_list)
    for ev in args.test_envs:
        train_env.remove('env{}'.format(ev))
    if args.dataset == 'ColoredMNIST':
        feature_num = 128
    else:
        feature_num = 512 if hparams['resnet18'] else 2048
    opt_for_pipline = opt_kde(env_list, train_env, dataset.num_classes,
                              feature_num, datas, sample_size=10000, device=device)
    compute_result = torch_to_numpy(
        opt_for_pipline.forward(cal_info=True))
    compute_result['eig_value'] = opt_for_pipline.eig_val()

    if whether_write:
        mmstr = '_mean'
        new_for_save = np.array(compute_result)
        np.save(marker + "before_new_L1_" + mmstr + "_save.npy", new_for_save)
        del new_for_save

    train_distance = compute_result['train_results'].max(axis=0)
    test_distance = compute_result['test_results'].max(axis=0)
    info = compute_result['train_info'][0]
    print("———————— before info filter ————————")
    print("train_dis:", train_distance)
    print("test_dis:", test_distance)
    print("info:", info)
    line = ''
    threshold_list = [0.05*i for i in range(9)]
    res_list = []
    for thr in threshold_list:
        select_index = [i for i in range(len(info)) if info[i] >= thr]
        # print(select_index)
        if len(select_index) == 0:
            train_mean = float('nan')
            test_mean = float('nan')
            info_mean = float('nan')
            line += to_str([train_mean, test_mean, info_mean, 0])
            res_list.append([train_mean, test_mean, info_mean, 0])
        else:
            train_mean = train_distance[select_index].mean()
            test_mean = test_distance[select_index].mean()
            info_mean = info[select_index].mean()
            line += to_str([train_mean, test_mean, info_mean, len(select_index)])
            res_list.append([train_mean, test_mean, info_mean, len(select_index)])
    line += '\n'
    del compute_result

    if whether_write and args.output_result_file is not None:
        with open(args.output_dir + '/' + 'before_' + args.output_result_file, 'a+') as f:
            f.write(line)
    return res_list[6]
    # train_variation, test_variation, //train_info, feature_num


    # val_acc = avg(train_out_acc)

    # model val_acc > max_val_acc - 0.1
    # A=std(val), B=std(variation)
    # val - A/B variation





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--task', type=str, default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')

    # add for prune model
    parser.add_argument('--pruning_method',type=str,default=None,help="Pruning method for model. default is not to prune")
    parser.add_argument('--init_model',type=str,default=None,
        help="state dict saved by save_checkpoint, used as init model state dict")
    parser.add_argument('--prune_gamma',type=float,default=0.2,
                        help="pruned percent for each prune step")
    parser.add_argument('--subdir',type=str,default=None)
    parser.add_argument('--debug',action='store_true')


    args = parser.parse_args()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None
    if args.init_model:
        alg_dict = torch.load(args.init_model)
        assert 'model_dict' in alg_dict,'not correct model state dict'
        algorithm_dict = alg_dict['model_dict']

    os.makedirs(args.output_dir, exist_ok=True)
    if args.subdir:
        args.output_dir = os.path.join(args.output_dir,args.subdir)
        os.makedirs(args.output_dir, exist_ok=True)

    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs, hparams)
    else:
        raise NotImplementedError

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.
    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):
        uda = []

        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))

        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                int(len(in_)*args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]

    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(uda_splits)
        if i in args.test_envs]

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)]
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
        for i in range(len(uda_splits))]

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams)

    algorithm.cpu()
    if algorithm_dict is not None:
        #print('arch_dict:',algorithm.state_dict().keys())
        #print('save_dict:',algorithm_dict.keys())
        algorithm.load_state_dict(algorithm_dict)
    del algorithm_dict
    algorithm.to(device)

    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env)/hparams['batch_size'] for env,_ in in_splits])

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.cpu().state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))
        algorithm.to(device)


    last_results_keys = None

    prune_step_list = [_*n_steps//10+1 for _ in range(1,10)]
    if args.pruning_method == 'LTH':
        for name, module in algorithm.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                prune.l1_unstructured(module, name='weight', amount=0)
            elif isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=0)
        forg_dict = algorithm.state_dict()
        org_dict = {}
        for key in forg_dict:
            if 'orig' in key:
                org_dict[key] = copy.deepcopy(forg_dict[key])
        del forg_dict



    evals = zip(eval_loader_names, eval_loaders, eval_weights)
    start_time = time.time()
    with torch.no_grad():
        for name, loader, weights in evals:
            acc = misc.accuracy(algorithm, loader, weights, device)
            print(name, acc)
    end_time = time.time()
    print('pretrained time:', -start_time + end_time)


    if args.debug:
        for name, module in algorithm.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                prune.ln_structured(module, name='weight', amount=0.4, dim=0, n=2)
                speed_up_inference_for_channel(module)
            elif isinstance(module, torch.nn.Linear):
                prune.ln_structured(module, name='weight', amount=0.4, dim=0, n=2)
                speed_up_inference_for_channel(module)

        evals = zip(eval_loader_names, eval_loaders, eval_weights)
        start_time = time.time()
        with torch.no_grad():
            for name, loader, weights in evals:
                acc = misc.accuracy(algorithm, loader, weights, device)
                print(name, acc)
        end_time = time.time()
        print('speed up time:', -start_time + end_time)


    for step in range(start_step, n_steps):

        step_start_time = time.time()
        if args.pruning_method == 'IMP':
            if step in prune_step_list:
                for name, module in algorithm.named_modules():
                    if isinstance(module, torch.nn.Conv2d):
                        prune.l1_unstructured(module, name='weight', amount=args.prune_gamma)
                    elif isinstance(module, torch.nn.Linear):
                        prune.l1_unstructured(module, name='weight', amount=args.prune_gamma)
        if args.pruning_method == 'SIMP':
            if step in prune_step_list:
                for name, module in algorithm.named_modules():
                    if isinstance(module, torch.nn.Conv2d):
                        prune.ln_structured(module, name='weight', amount=args.prune_gamma,dim=0,n=2)
                    elif isinstance(module, torch.nn.Linear):
                        prune.ln_structured(module, name='weight', amount=args.prune_gamma,dim=0,n=2)

        if args.pruning_method == 'LTH':
            if step in prune_step_list:
                for name, module in algorithm.named_modules():
                    if isinstance(module, torch.nn.Conv2d):
                        prune.l1_unstructured(module, name='weight', amount=args.prune_gamma)
                    elif isinstance(module, torch.nn.Linear):
                        prune.l1_unstructured(module, name='weight', amount=args.prune_gamma)

                tmp_dict = algorithm.state_dict()
                tmp_dict.update(org_dict)

        minibatches_device = [(x.to(device), y.to(device))
            for x,y in next(train_minibatches_iterator)]
        if args.task == "domain_adaptation":
            uda_device = [x.to(device)
                for x,_ in next(uda_minibatches_iterator)]
        else:
            uda_device = None
        step_vals = algorithm.update(minibatches_device, uda_device)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)
            variation = calculate_variation(algorithm,step,eval_loader_names,eval_loaders,
                                dataset,args)

            results['train_variation'] = float(variation[0])
            results['test_variation'] = float(variation[1])

            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            for name, loader, weights in evals:
                acc = misc.accuracy(algorithm, loader, weights, device)
                results[name+'_acc'] = acc
            sparse = check_sparsity(algorithm.network)
            results['sparse'] = sparse
            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                colwidth=12)

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}.pkl')

    evals = zip(eval_loader_names, eval_loaders, eval_weights)
    start_time = time.time()
    for name, loader, weights in evals:
        acc = misc.accuracy(algorithm, loader, weights, device)
        print(name, acc)
    end_time = time.time()
    print('pruned time:',-start_time+end_time)
    for name, module in algorithm.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.remove(module,'weight')
            #prune.l1_unstructured(module, name='weight', amount=0)
        elif isinstance(module, torch.nn.Linear):
            prune.remove(module,'weight')

    evals = zip(eval_loader_names, eval_loaders, eval_weights)
    start_time = time.time()
    for name, loader, weights in evals:
        acc = misc.accuracy(algorithm, loader, weights, device)
        print(name, acc)
    end_time = time.time()
    print('removed time:', -start_time + end_time)

    save_checkpoint('model.pkl')

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')

