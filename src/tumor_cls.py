#!/usr/bin/env python3

import argparse
import ast
import time
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchio as tio
from sklearn.model_selection import ParameterGrid, RepeatedStratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

import models
import push
from interpret import attr_methods, attribute
from utils import (FocalLoss, accuracy, balanced_accuracy, cross_entropy, get_hashes, get_m_indexes,
                   get_transform_aug, iad, lc, load_data, load_subjs_batch, makedir, output_results,
                   preload, preprocess, print_param, print_results, process_iad, save_cvs)


def train(net, data_loader, optimizer, grid=None, **kwargs):
    assert use_amp is not None and scaler is not None and criterion is not None
    if use_da:
        torch.backends.cudnn.benchmark = False
        # torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    net.train()
    if 'MProtoNet' in model_name:
        return train_mppnet(net, data_loader, optimizer, coefs=grid['coefs'], **kwargs)
    for b, subjs_batch in enumerate(data_loader):
        data, target, _ = load_subjs_batch(subjs_batch)
        data = data.to(device, non_blocking=True)
        target = target.argmax(1).to(device, non_blocking=True)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            output = net(data)
            loss = criterion(output, target)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


def train_mppnet(net, data_loader, optimizer, use_l1_mask=True, coefs=None, stage=None, log=print):
    if stage in ['warm_up', 'joint']:
        log()
    log(f"\t{stage}", end='', flush=True)
    start = time.time()
    n_examples, n_correct, n_batches = 0, 0, 0
    total_loss, total_ce, total_clst, total_sep, total_avg_sep, total_l1 = 0, 0, 0, 0, 0, 0
    total_ortho, total_ss, total_map, total_ce2 = 0, 0, 0, 0
    for b, subjs_batch in enumerate(data_loader):
        data, target, _ = load_subjs_batch(subjs_batch)
        data = data.to(device, non_blocking=True)
        target = target.argmax(1).to(device, non_blocking=True)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            if getattr(net, 'module.p_mode' if is_parallel else 'p_mode') >= 2:
                output, min_distances, x, p_map = net(data)
            else:
                output, min_distances = net(data)
            # Calculate cross-entropy loss
            cross_entropy = criterion(output, target)
            loss_ce = coefs['ce'] * cross_entropy
            total_ce += loss_ce.item()
            if stage in ['warm_up', 'joint']:
                # Calculate cluster loss
                max_dist = torch.prod(torch.tensor(net.prototype_shape[1:])).to(device)
                target_weight = class_weight.to(device)[target]
                target_weight = target_weight / target_weight.sum()
                # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
                prototypes_correct = net.prototype_class_identity[:, target].mT
                inv_distances_correct = ((max_dist - min_distances) * prototypes_correct).amax(1)
                cluster = ((max_dist - inv_distances_correct) * target_weight).sum()
                loss_clst = coefs['clst'] * cluster
                total_clst += loss_clst.item()
                # Calculate separation loss
                prototypes_wrong = 1 - prototypes_correct
                inv_distances_wrong = ((max_dist - min_distances) * prototypes_wrong).amax(1)
                separation = ((max_dist - inv_distances_wrong) * target_weight).sum()
                loss_sep = coefs['sep'] * separation
                total_sep += loss_sep.item()
                # Calculate average separation
                avg_separation = (min_distances * prototypes_wrong).sum(1) / prototypes_wrong.sum(1)
                avg_separation = (avg_separation * target_weight).sum()
                total_avg_sep += avg_separation.item()
                # Calculate orthogonality loss
                p = net.prototype_vectors.reshape(net.num_classes, net.num_prototypes_per_class, -1)
                i_p = torch.eye(p.shape[1]).to(device)
                orthogonality = torch.relu(torch.linalg.norm(p @ p.mT - i_p, dim=(1, 2))).sum()
                loss_ortho = coefs['ortho'] * orthogonality
                total_ortho += loss_ortho.item()
                # Calculate subspace-separation loss
                subspace_separation = F.pdist((p.mT @ p).flatten(1)).sum() * 2 ** (-0.5)
                loss_ss = coefs['ss'] * subspace_separation
                total_ss += loss_ss.item()
                # Calculate mapping loss
                if getattr(net, 'module.p_mode' if is_parallel else 'p_mode') >= 2:
                    ri = torch.randint(2, (1,)).item()
                    f_affine = lambda t: F.interpolate(
                        t, scale_factor=(0.75, 0.875)[ri],
                        mode='trilinear' if x.ndim == 5 else 'bilinear', align_corners=True
                    )
                    f_l1 = lambda t: t.abs().mean()
                    mapping = f_l1(f_affine(p_map) - net.get_p_map(f_affine(x))) + f_l1(p_map)
                    loss_map = coefs['map'] * mapping
                    total_map += loss_map.item()
                # Calculate cross-entropy 2nd loss
                if getattr(net, 'module.p_mode' if is_parallel else 'p_mode') >= 4:
                    p_x = net.lse_pooling(net.p_map[:-3](x).flatten(2))
                    output2 = net.last_layer(p_x @ net.p_map[-3].weight.flatten(1).mT)
                    cross_entropy2 = criterion(output2, target)
                    loss_ce2 = coefs['ce2'] * cross_entropy2
                    total_ce2 += loss_ce2.item()
                # Calculate total loss
                loss = loss_ce + loss_clst + loss_sep + loss_ortho + loss_ss
                if getattr(net, 'module.p_mode' if is_parallel else 'p_mode') >= 2:
                    loss = loss + loss_map
                if getattr(net, 'module.p_mode' if is_parallel else 'p_mode') >= 4:
                    loss = loss + loss_ce2
            else:
                # Calculate L1 loss
                if use_l1_mask:
                    l1_mask = 1 - net.prototype_class_identity.mT
                    l1 = torch.linalg.vector_norm(net.last_layer.weight * l1_mask, ord=1)
                else:
                    l1 = torch.linalg.vector_norm(net.last_layer.weight, ord=1)
                loss_l1 = coefs['l1'] * l1
                total_l1 += loss_l1.item()
                # Calculate total loss
                loss = loss_ce + loss_l1
            total_loss += loss.item()
            # Evaluation statistics
            n_examples += target.shape[0]
            n_correct += (output.data.argmax(1) == target).sum().item()
            n_batches += 1
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # print(f"Before del: {torch.cuda.memory_allocated(device) >> 20} MB (allocated)", end='',
        #       flush=True)
        # print(f" {torch.cuda.memory_reserved(device) >> 20} MB (reserved)", end='', flush=True)
        # del data, target, output, min_distances
        # if getattr(net, 'module.p_mode' if is_parallel else 'p_mode') >= 2:
        #     del x, p_map
        # print(f"  After del: {torch.cuda.memory_allocated(device) >> 20} MB (allocated)", end='',
        #       flush=True)
        # print(f" {torch.cuda.memory_reserved(device) >> 20} MB (reserved)")
    with torch.no_grad():
        p_avg_pdist = F.pdist(net.prototype_vectors.flatten(1)).mean().item()
    end = time.time()
    log(f"\ttime: {end - start:.2f}s,"
        f" acc: {n_correct / n_examples:.4f},"
        f" loss: {total_loss / n_batches:.4f},"
        f" ce: {total_ce / n_batches:.4f},"
        f" clst: {total_clst / n_batches:.4f},"
        f" sep: {total_sep / n_batches:.4f},"
        f" avg_sep: {total_avg_sep / n_batches:.4f},"
        f" l1: {total_l1 / n_batches:.4f},"
        f" ortho: {total_ortho / n_batches:.4f},"
        f" ss: {total_ss / n_batches:.4f},"
        f" map: {total_map / n_batches:.4f},"
        f" ce2: {total_ce2 / n_batches:.4f},"
        f" p_avg_pdist: {p_avg_pdist:.4f}")


def test(net, data_loader, grid=None):
    if not use_da:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    net.eval()
    f_x, m_f_xes, lcs, iads = [], {}, {}, {}
    if grid and grid.get('attrs'):
        methods = grid['attrs']
    elif 'MProtoNet' in model_name:
        methods = 'MG'
    else:
        methods = 'G'
    with torch.no_grad():
        for b, subjs_batch in enumerate(data_loader):
            data, target, seg_map = load_subjs_batch(subjs_batch)
            data = data.to(device, non_blocking=True)
            target = target.argmax(1).to(device, non_blocking=True)
            seg_map = seg_map.to(device, non_blocking=True)
            if grid and grid.get('trans'):
                data1 = data.permute(0, 4, 1, 2, 3).reshape((-1,) + data.shape[1:-1])
                data1 = net.trans(data1).reshape(data.shape[0], data.shape[4], 3, data.shape[2:-1])
                data1 = data1.permute(0, 1, 3, 4, 2).detach().cpu().numpy()
                print(data1.min(), data1.max(), data1.mean(), data1.std())
                data1 = (data1 - data1.min()) / (data1.max() - data1.min())
                trans_dir = f'../results/saved_imgs/{args.model_name}' \
                            f'{"f" if grid.get("fixed") else ""}_trans/'
                makedir(trans_dir)
                matplotlib.use('agg')
                for d in range(data1.shape[0]):
                    plt.figure(figsize=(6, 4))
                    plt.axis('off')
                    plt.imshow(data1[d, data1.shape[1] // 2, :, :, :])
                    plt.savefig(f'{trans_dir}i{i}_b{b:02d}_{d:02d}.png', bbox_inches='tight',
                                pad_inches=0)
                    plt.close()
            f_x.append(F.softmax(net(data), dim=1).cpu().numpy())
            print("Missing Modalities:", end='', flush=True)
            m_indexes = get_m_indexes()
            tic = time.time()
            for remaining, m_index in m_indexes.items():
                if m_f_xes.get(remaining) is None:
                    m_f_xes[remaining] = []
                data_missing = data.clone()
                r_index = list(set(range(data.shape[1])) - set(m_index))
                data_missing[:, m_index] = data[:, r_index].mean(1, keepdim=True)
                m_f_xes[remaining].append(F.softmax(net(data_missing), dim=1).cpu().numpy())
            toc = time.time()
            print(f" {toc - tic:.6f}s,", end='', flush=True)
            for method_i in methods:
                method = attr_methods[method_i]
                print(f" {method}:", end='', flush=True)
                if not lcs.get(method):
                    lcs[method] = {f'({a}, Th=0.5) {m}': []
                                   for a in ['WT', 'TC'] for m in ['AP', 'DSC', 'IoU']}
                if not iads.get(method):
                    iads[method] = {m: [] for m in ['IA', 'ID', 'IAD']}
                tic = time.time()
                attr = attribute(net, data, target, device, method)
                lcs[method]['(WT, Th=0.5) AP'].append(
                    lc(attr, seg_map, annos=[1, 2, 4], threshold=0.5, metric='AP'))
                lcs[method]['(WT, Th=0.5) DSC'].append(
                    lc(attr, seg_map, annos=[1, 2, 4], threshold=0.5, metric='DSC'))
                lcs[method]['(WT, Th=0.5) IoU'].append(
                    lc(attr, seg_map, annos=[1, 2, 4], threshold=0.5, metric='IoU'))
                lcs[method]['(TC, Th=0.5) AP'].append(
                    lc(attr, seg_map, annos=[1, 4], threshold=0.5, metric='AP'))
                lcs[method]['(TC, Th=0.5) DSC'].append(
                    lc(attr, seg_map, annos=[1, 4], threshold=0.5, metric='DSC'))
                lcs[method]['(TC, Th=0.5) IoU'].append(
                    lc(attr, seg_map, annos=[1, 4], threshold=0.5, metric='IoU'))
                iads[method]['IA'].append(
                    iad(net, data, attr, n_intervals=50, quantile=True, addition=True))
                iads[method]['ID'].append(
                    iad(net, data, attr, n_intervals=50, quantile=True, addition=False))
                toc = time.time()
                print(f" {toc - tic:.6f}s,", end='', flush=True)
            print(" Finished.")
    for remaining, m_f_x in m_f_xes.items():
        m_f_xes[remaining] = np.vstack(m_f_x)
    for method, lcs_ in lcs.items():
        for metric, lcs__ in lcs_.items():
            lcs[method][metric] = np.vstack(lcs__)
    for method, iads_ in iads.items():
        for metric, iads__ in iads_.items():
            if metric == 'IAD':
                continue
            iads[method][metric] = np.concatenate(iads__, axis=1)
    return np.vstack(f_x), m_f_xes, lcs, iads


def parse_arguments():
    # Define command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-name', type=str, required=True, help="name of the model")
    parser.add_argument('-d', '--data-path', type=str,
                        default='../data/BraTS_2020/MICCAI_BraTS2020_TrainingData',
                        help="path to the data files")
    parser.add_argument('-n', '--max-n-epoch', type=int, default=200,
                        help="maximum number of epochs to train on")
    parser.add_argument('-p', '--param-grid', type=str, default=None,
                        help="grid of hyper-parameters")
    parser.add_argument('-s', '--seed', type=int, default=0, help="random seed")
    parser.add_argument('-v', '--v-mod', type=int, choices={0, 1}, default=0, help="verbose mode")
    parser.add_argument('--bc-opt', type=str,
                        choices={'Off', 'BCE', 'B2CE', 'CBCE', 'FL', 'BFL', 'B2FL', 'CBFL'},
                        default='BFL', help="balanced classification option")
    parser.add_argument('--op-opt', type=str, choices={'Adam', 'AdamW'}, default='AdamW',
                        help="optimizer option")
    parser.add_argument('--lr-opt', type=str,
                        choices={'Off', 'StepLR', 'CosALR', 'WUStepLR', 'WUCosALR'},
                        default='WUCosALR', help="learning rate scheduler option")
    parser.add_argument('--lr-n', type=int, default=0, help="learning rate scheduler number")
    parser.add_argument('--wu-n', type=int, default=0, help="number of warm-up epochs")
    parser.add_argument('--early-stop', type=int, choices={0, 1}, default=0,
                        help="whether to enable early stopping")
    parser.add_argument('--n-workers', type=int, default=8, help="number of workers in data loader")
    parser.add_argument('--n-threads', type=int, default=4, help="number of CPU threads")
    parser.add_argument('--preloaded', type=int, choices={0, 1, 2}, default=1,
                        help="whether to preprocess (1) and preload (2) the dataset")
    parser.add_argument('--augmented', type=int, choices={0, 1}, default=1,
                        help="whether to perform data augmentation during training")
    parser.add_argument('--aug-seq', type=str, default=None, help="data augmentation sequence")
    parser.add_argument('--use-cuda', type=int, choices={0, 1}, default=1,
                        help="whether to use CUDA if available")
    parser.add_argument('--use-amp', type=int, choices={0, 1}, default=1,
                        help="whether to use automatic mixed precision")
    parser.add_argument('--use-da', type=int, choices={0, 1}, default=0,
                        help="whether to use deterministic algorithms.")
    parser.add_argument('--gpus', type=str, default='0', help="indexes of GPUs")
    parser.add_argument('--load-model', type=str, default=None,
                        help="whether to load the model files")
    parser.add_argument('--save-model', type=int, choices={0, 1}, default=0,
                        help="whether to save the best model")
    return parser.parse_args()


if __name__ == '__main__':
    tic = time.time()
    # Parse command-line arguments
    args = parse_arguments()
    model_name, data_path = args.model_name, args.data_path
    max_n_epoch, param_grid, seed, v_mod = args.max_n_epoch, args.param_grid, args.seed, args.v_mod
    bc_opt, op_opt, lr_opt, lr_n, wu_n = args.bc_opt, args.op_opt, args.lr_opt, args.lr_n, args.wu_n
    early_stop, n_workers, n_threads = args.early_stop, args.n_workers, args.n_threads
    preloaded, augmented, aug_seq = args.preloaded, args.augmented, args.aug_seq
    use_cuda, use_amp, use_da, gpus = args.use_cuda, args.use_amp, args.use_da, args.gpus
    load_model, save_model = args.load_model, args.save_model
    if param_grid is not None:
        param_grid = ast.literal_eval(param_grid)
        for k, v in param_grid.items():
            if not isinstance(v, list):
                param_grid[k] = [v]
    else:
        param_grid = {'batch_size': [32], 'lr': [1e-3], 'wd': [1e-2], 'features': ['resnet152_ri'],
                      'n_layers': [6]}
    transform = [tio.ToCanonical(), tio.CropOrPad(target_shape=(192, 192, 144))]
    transform += [tio.Resample(target=(1.5, 1.5, 1.5))]
    transform += [tio.ZNormalization()]
    if augmented:
        if aug_seq is not None:
            transform_aug = get_transform_aug(aug_seq=aug_seq)
        else:
            transform_aug = get_transform_aug()
    else:
        transform_aug = []
    transform_train = tio.Compose(transform + transform_aug)
    transform = tio.Compose(transform)
    if '_pm' in model_name:
        p_mode = int(model_name[model_name.find('_pm') + 3])
        model_name = model_name.replace(f'_pm{p_mode}', '')
    gpu_ids = ast.literal_eval(f'[{gpus}]')
    device = torch.device(
        'cuda:' + str(gpu_ids[0]) if use_cuda and torch.cuda.is_available() else 'cpu')
    use_amp = use_amp == 1 and use_cuda == 1 and torch.cuda.is_available()
    if use_da:
        torch.backends.cudnn.benchmark = False
        # torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    torch.set_num_threads(n_threads)
    opts_hash = get_hashes(args)
    warnings.filterwarnings('ignore', message="The epoch parameter in `scheduler.step\(\)` was not")
    # Load data
    x, y = load_data(data_path)
    if preloaded:
        np.random.seed(seed)
        torch.manual_seed(seed)
        dataset = tio.SubjectsDataset(list(x), transform=transform)
        if preloaded > 1:
            data_loader = DataLoader(dataset, batch_size=(n_workers + 1) // 2,
                                     num_workers=n_workers)
            x, y, seg = preload(data_loader)
            n_workers = 0
        else:
            data_loader = DataLoader(dataset, num_workers=n_workers)
            x = preprocess(data_loader)
            transform_train = tio.Compose(transform_aug) if augmented else None
            transform = None
        del dataset, data_loader
        toc = time.time()
        print(f"Elapsed time is {toc - tic:.6f} seconds.")
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=seed)
    cv_train = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=seed)
    # 5-fold CV
    cv_fold = cv.get_n_splits()
    cv_train_fold = cv_train.get_n_splits()
    splits = np.zeros(y.shape[0], dtype=int)
    param_grids = ParameterGrid(param_grid)
    f_x, m_f_xes, lcs, n_prototypes, iads = np.zeros(y.shape), {}, {}, None, {}
    for i, (I_train, I_test) in enumerate(cv.split(x, y.argmax(1))):
        print(f">>>>>>>> CV = {i + 1}:")
        splits[I_test] = i
        if len(param_grids) > 1 or early_stop:
            # TODO: 5-fold inner CV
            pass
        else:
            best_grid = param_grids[0]
        if early_stop:
            # TODO: 5-fold inner CV
            best_n_epoch = max_n_epoch
        else:
            best_n_epoch = max_n_epoch
        # Training and test
        np.random.seed(seed)
        torch.manual_seed(seed)
        if preloaded > 1:
            dataset_train = TensorDataset(x[I_train], y[I_train], seg[I_train])
            dataset_test = TensorDataset(x[I_test], y[I_test], seg[I_test])
            in_size = (4,) + x.shape[2:]
            out_size = y.shape[1]
        else:
            dataset_train = tio.SubjectsDataset(list(x[I_train]), transform=transform_train)
            if 'MProtoNet' in model_name:
                dataset_push = tio.SubjectsDataset(list(x[I_train]), transform=transform)
            dataset_test = tio.SubjectsDataset(list(x[I_test]), transform=transform)
            in_size = (4,) + dataset_train[0]['t1']['data'].shape[1:]
            out_size = dataset_train[0]['label'].shape[0]
        loader_train = DataLoader(dataset_train, batch_size=best_grid['batch_size'], shuffle=True,
                                  num_workers=n_workers, pin_memory=True, drop_last=True)
        if v_mod > 0:
            loader_train_ = DataLoader(dataset_train, batch_size=best_grid['batch_size'],
                                       num_workers=n_workers, pin_memory=True)
        loader_test = DataLoader(dataset_test, batch_size=(best_grid['batch_size'] + 1) // 2,
                                 num_workers=n_workers, pin_memory=True)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        kwargs = {'in_size': in_size, 'out_size': out_size}
        for arg in ['kernel_size', 'stride', 'features', 'n_layers', 'fixed']:
            if best_grid.get(arg):
                kwargs[arg] = best_grid[arg]
        if 'MProtoNet' in model_name:
            if '_pm' in args.model_name:
                kwargs['p_mode'] = p_mode
            for arg in ['prototype_shape', 'f_dist', 'topk_p']:
                if best_grid.get(arg):
                    kwargs[arg] = best_grid[arg]
        net = getattr(models, model_name)(**kwargs).to(device)
        if use_cuda and torch.cuda.is_available() and len(gpu_ids) > 1:
            net = nn.DataParallel(net, device_ids=gpu_ids)
            is_parallel = True
        else:
            is_parallel = False
        if load_model is not None:
            if load_model.startswith(args.model_name):
                model_name_i = f'{load_model}_cv{i}'
                model_path_i = f'../results/models/{model_name_i}.pt'
            else:
                model_name_i = f'{load_model[load_model.find(args.model_name):]}_cv{i}'
                model_path_i = f'{load_model}_cv{i}.pt'
        else:
            model_name_i = f'{args.model_name}_{opts_hash}_cv{i}'
        print(f"Model: {model_name_i}\n{str(net)}")
        print_param(net, show_each=False)
        print(f"Hyper-parameters = {param_grid}")
        print(f"Best Hyper-parameters = {best_grid}")
        print(f"Best Number of Epoch = {best_n_epoch}")
        if 'MProtoNet' in model_name:
            if preloaded > 1:
                loader_push = DataLoader(dataset_train, batch_size=best_grid['batch_size'],
                                         num_workers=n_workers, pin_memory=True)
            else:
                loader_push = DataLoader(dataset_push, batch_size=best_grid['batch_size'],
                                         num_workers=n_workers, pin_memory=True)
            img_dir = f'../results/saved_imgs/{model_name_i}/'
            makedir(img_dir)
            prototype_img_filename_prefix = 'prototype-img'
            prototype_self_act_filename_prefix = 'prototype-self-act'
            proto_bound_boxes_filename_prefix = 'bb'
            if net.features_3d:
                params = [
                    {'params': net.features.parameters(), 'lr': best_grid['lr'],
                     'weight_decay': best_grid['wd']},
                    {'params': net.add_ons.parameters(), 'lr': best_grid['lr'],
                     'weight_decay': best_grid['wd']},
                    {'params': net.prototype_vectors, 'lr': best_grid['lr'], 'weight_decay': 0},
                ]
            else:
                params = [
                    {'params': net.trans.parameters(), 'lr': best_grid['lr'] * 0.1,
                     'weight_decay': best_grid['wd']},
                    {'params': net.features.parameters(), 'lr': best_grid['lr'] * 0.1,
                     'weight_decay': best_grid['wd']},
                    {'params': net.add_ons.parameters(), 'lr': best_grid['lr'],
                     'weight_decay': best_grid['wd']},
                    {'params': net.prototype_vectors, 'lr': best_grid['lr'], 'weight_decay': 0},
                ]
                params_warm_up = [
                    {'params': net.trans.parameters(), 'lr': best_grid['lr'] * 0.1,
                     'weight_decay': best_grid['wd']},
                    {'params': net.add_ons.parameters(), 'lr': best_grid['lr'],
                     'weight_decay': best_grid['wd']},
                    {'params': net.prototype_vectors, 'lr': best_grid['lr'], 'weight_decay': 0},
                ]
            if getattr(net, 'module.p_mode' if is_parallel else 'p_mode') >= 2:
                params += [
                    {'params': net.p_map.parameters(), 'lr': best_grid['lr'],
                     'weight_decay': best_grid['wd']},
                ]
                if not net.features_3d:
                    params_warm_up += [
                        {'params': net.p_map.parameters(), 'lr': best_grid['lr'],
                         'weight_decay': best_grid['wd']},
                    ]
            params_last_layer = [
                {'params': net.last_layer.parameters(), 'lr': best_grid['lr'],
                 'weight_decay': 0},
            ]
            if op_opt == 'Adam':
                optimizer = optim.Adam(params)
                if not net.features_3d:
                    optimizer_warm_up = optim.Adam(params_warm_up)
                optimizer_last_layer = optim.Adam(params_last_layer)
            elif op_opt == 'AdamW':
                optimizer = optim.AdamW(params)
                if not net.features_3d:
                    optimizer_warm_up = optim.AdamW(params_warm_up)
                optimizer_last_layer = optim.AdamW(params_last_layer)
            else:
                optimizer = optim.SGD(params, momentum=0.9)
                if not net.features_3d:
                    optimizer_warm_up = optim.SGD(params_warm_up, momentum=0.9)
                optimizer_last_layer = optim.SGD(params_last_layer, momentum=0.9)
        else:
            if op_opt == 'Adam':
                optimizer = optim.Adam(net.parameters(), lr=best_grid['lr'],
                                       weight_decay=best_grid['wd'])
            elif op_opt == 'AdamW':
                optimizer = optim.AdamW(net.parameters(), lr=best_grid['lr'],
                                        weight_decay=best_grid['wd'])
            else:
                optimizer = optim.SGD(net.parameters(), lr=best_grid['lr'], momentum=0.9,
                                      weight_decay=best_grid['wd'])
        if ('WU' in lr_opt or 'MProtoNet' in model_name or '_pt' in model_name) and wu_n <= 0:
            wu_n = best_n_epoch // 5
        wu_e = wu_n // 2
        if 'StepLR' in lr_opt:
            if lr_n <= 0:
                lr_n = best_n_epoch // 10
            scheduler = optim.lr_scheduler.StepLR(optimizer, lr_n)
        elif 'CosALR' in lr_opt:
            if lr_n <= 0:
                lr_n = best_n_epoch - wu_n if 'WU' in lr_opt else best_n_epoch - wu_e
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, lr_n)
        if 'WU' in lr_opt:
            if 'MProtoNet' in model_name and not net.features_3d:
                scheduler_warm_up = optim.lr_scheduler.LambdaLR(optimizer_warm_up,
                                                                lambda e: (e + 1) / wu_n)
                scheduler0 = optim.lr_scheduler.LambdaLR(optimizer,
                                                         lambda e: (e + 1) / wu_n + wu_e / wu_n)
                scheduler = optim.lr_scheduler.SequentialLR(optimizer, [scheduler0, scheduler],
                                                            [wu_n - wu_e])
            else:
                scheduler0 = optim.lr_scheduler.LambdaLR(optimizer, lambda e: (e + 1) / wu_n)
                scheduler = optim.lr_scheduler.SequentialLR(optimizer, [scheduler0, scheduler],
                                                            [wu_n])
        if bc_opt in ['BCE', 'BFL']:
            class_weight = torch.FloatTensor(1 / y[I_train].sum(0))
        elif bc_opt in ['B2CE', 'B2FL']:
            class_weight = torch.FloatTensor(1 / y[I_train].sum(0) ** 0.5)
        elif bc_opt in ['CBCE', 'CBFL']:
            beta = 1 - 1 / y[I_train].sum()
            class_weight = torch.FloatTensor((1 - beta) / (1 - beta ** y[I_train].sum(0)))
        else:
            class_weight = torch.ones(y[I_train].shape[1])
        if 'FL' in bc_opt:
            criterion = FocalLoss(weight=class_weight).to(device)
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weight).to(device)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        if load_model is not None:
            if is_parallel:
                net.module.load_state_dict(torch.load(model_path_i, map_location=device))
            else:
                net.load_state_dict(torch.load(model_path_i, map_location=device))
        else:
            print("Epoch =", end='', flush=True)
            if v_mod > 0:
                train_acc, test_acc = np.zeros(best_n_epoch), np.zeros(best_n_epoch)
                train_bac, test_bac = np.zeros(best_n_epoch), np.zeros(best_n_epoch)
                train_ce, test_ce = np.zeros(best_n_epoch), np.zeros(best_n_epoch)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            for e in range(best_n_epoch):
                print(f" {e + 1}", end='', flush=True)
                if '_pt' in model_name and not best_grid.get('fixed'):
                    if lr_opt != 'Off' and ('WU' in lr_opt or e >= wu_e):
                        print(f"(lr={scheduler.get_last_lr()[0]:g})", end='', flush=True)
                    if e < wu_e:
                        for p in net.features.parameters():
                            p.requires_grad = False
                    elif e == wu_e:
                        for p in net.features.parameters():
                            p.requires_grad = True
                    train(net, loader_train, optimizer)
                    if lr_opt != 'Off' and ('WU' in lr_opt or e >= wu_e):
                        scheduler.step()
                elif 'MProtoNet' in model_name and not net.features_3d and e < wu_e:
                    if 'WU' in lr_opt:
                        print(f"(lr={scheduler_warm_up.get_last_lr()[0]:g})", end='', flush=True)
                    train(net, loader_train, optimizer_warm_up, grid=best_grid, stage='warm_up')
                    if 'WU' in lr_opt:
                        scheduler_warm_up.step()
                else:
                    if lr_opt != 'Off':
                        print(f"(lr={scheduler.get_last_lr()[0]:g})", end='', flush=True)
                    train(net, loader_train, optimizer, grid=best_grid, stage='joint')
                    if lr_opt != 'Off':
                        scheduler.step()
                if 'MProtoNet' in model_name and e + 1 >= 10 and e + 1 in [i for i in
                                                                           range(best_n_epoch + 1)
                                                                           if i % 10 == 0]:
                    if not use_da:
                        torch.backends.cudnn.benchmark = False
                        torch.backends.cudnn.deterministic = True
                    push.push_prototypes(
                        loader_push,
                        net,
                        root_dir_for_saving_prototypes=None if e + 1 < best_n_epoch else img_dir,
                        prototype_img_filename_prefix=prototype_img_filename_prefix,
                        prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                        proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix
                    )
                    for j in range(10):
                        train(net, loader_train, optimizer_last_layer, grid=best_grid,
                              stage=f'last_{j}')
                if v_mod > 0:
                    f_x_train, _ = test(net, loader_train_)
                    train_acc[e] = accuracy(f_x_train, y[I_train])
                    train_bac[e] = balanced_accuracy(f_x_train, y[I_train])
                    train_ce[e] = cross_entropy(f_x_train, y[I_train])
                    f_x_test, _ = test(net, loader_test)
                    test_acc[e] = accuracy(f_x_test, y[I_test])
                    test_bac[e] = balanced_accuracy(f_x_test, y[I_test])
                    test_ce[e] = cross_entropy(f_x_test, y[I_test])
            print()
        # TODO: Pruning
        del dataset_train, loader_train
        if 'MProtoNet' in model_name:
            del loader_push
        f_x[I_test], m_f_xes_test, lcs_test, iads_test = test(net, loader_test, grid=best_grid)
        del dataset_test, loader_test
        for remaining, m_f_x in m_f_xes_test.items():
            if m_f_xes.get(remaining) is None:
                m_f_xes[remaining] = np.zeros(y.shape)
            m_f_xes[remaining][I_test] = m_f_x
        for method, lcs_ in lcs_test.items():
            if not lcs.get(method):
                lcs[method] = {f'({a}, Th=0.5) {m}': np.zeros((cv_fold, 4))
                               for a in ['WT', 'TC'] for m in ['AP', 'DSC', 'IoU']}
            for metric, lcs__ in lcs_.items():
                lcs[method][metric][i] = lcs__.mean(0)
        if 'MProtoNet' in model_name:
            if n_prototypes is None:
                n_prototypes = np.zeros((cv_fold, out_size))
            n_prototypes[i] = net.prototype_class_identity.sum(0).cpu().numpy()
            n_prototype = n_prototypes[i:i + 1]
        else:
            n_prototype = None
        process_iad(iads_test, y[I_test], model_name=model_name_i)
        for method, iads_ in iads_test.items():
            if not iads.get(method):
                iads[method] = {m: np.zeros((cv_fold, 2)) for m in ['IA', 'ID', 'IAD']}
            for metric, iads__ in iads_.items():
                iads[method][metric][i] = iads__
        if v_mod > 0:
            print("Training Accuracy:")
            print(np.array2string(train_acc, max_line_width=88,
                                  formatter={'float_kind': lambda x: "%7.4f" % x}))
            print("Training Balanced Accuracy:")
            print(np.array2string(train_bac, max_line_width=88,
                                  formatter={'float_kind': lambda x: "%7.4f" % x}))
            print("Training Cross Entropy:")
            print(np.array2string(train_ce, max_line_width=88,
                                  formatter={'float_kind': lambda x: "%7.4f" % x}))
            print("Test Accuracy:")
            print(np.array2string(test_acc, max_line_width=88,
                                  formatter={'float_kind': lambda x: "%7.4f" % x}))
            print("Test Balanced Accuracy:")
            print(np.array2string(test_bac, max_line_width=88,
                                  formatter={'float_kind': lambda x: "%7.4f" % x}))
            print("Test Cross Entropy:")
            print(np.array2string(test_ce, max_line_width=88,
                                  formatter={'float_kind': lambda x: "%7.4f" % x}))
        print_results("Test", f_x[I_test], y[I_test], m_f_xes_test, lcs_test, n_prototype,
                      iads_test)
        if save_model and load_model is None:
            model_dir = '../results/models/'
            makedir(model_dir)
            if is_parallel:
                torch.save(net.module.state_dict(), f'{model_dir}{model_name_i}.pt')
            else:
                torch.save(net.state_dict(), f'{model_dir}{model_name_i}.pt')
        toc = time.time()
        print(f"Elapsed time is {toc - tic:.6f} seconds.")
        print()
    print(f">>>>>>>> {cv_fold}-fold CV Results:")
    print_results("Test", f_x, y, m_f_xes, lcs, n_prototypes, iads, splits)
    output_results("BraTS_2020", args, best_grid['batch_size'], f_x, y, m_f_xes, lcs, n_prototypes,
                   iads, splits)
    cv_dir = '../results/cvs/'
    makedir(cv_dir)
    save_cvs(cv_dir, args, f_x, y, lcs, iads, splits)
    print("Finished.")
    toc = time.time()
    print(f"Elapsed time is {toc - tic:.6f} seconds.")
    print()
