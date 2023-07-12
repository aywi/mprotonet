#!/usr/bin/env python3

import argparse
import ast
import time
import warnings

import captum.attr as ctattr
import numpy as np
import torch
import torch.nn as nn
import torchio as tio
from sklearn.model_selection import ParameterGrid, RepeatedStratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

import models
from utils import load_data, preload, preprocess, print_param

attr_methods: dict = {
    'M': 'MProtoNet',
    'D': 'Deconvolution',
    'G': 'GradCAM',
    'U': 'Guided GradCAM',
    'O': 'Occlusion',
}


def upsample(attr, data):
    if attr.ndim == 5:
        attr = ctattr.LayerAttribution.interpolate(attr, data.shape[2:],
                                                   interpolate_mode='trilinear')
    else:
        attr = ctattr.LayerAttribution.interpolate(attr, data.shape[2:4],
                                                   interpolate_mode='bilinear')
        attr = attr.reshape((data.shape[0],
                             data.shape[4], 1) + data.shape[2:4]).permute(0, 2, 3, 4, 1)
    if attr.shape[0] // data.shape[0] == data.shape[1] // attr.shape[1]:
        return attr.reshape_as(data)
    else:
        return attr.expand_as(data).clone()


def attribute(net, data, target, device, method, show_progress=False):
    warnings.filterwarnings('ignore', message="Input Tensor \d+ did not already require gradients,")
    warnings.filterwarnings('ignore', message="Setting backward hooks on ReLU activations.The hook")
    warnings.filterwarnings('ignore', message="Setting forward, backward hooks and attributes on n")
    if 'GradCAM' in method:
        conv_name = [n for n, m in net.named_modules() if isinstance(m, (nn.Conv2d, nn.Conv3d))][-1]
    if method == 'MProtoNet':
        if isinstance(net, nn.DataParallel):
            net = net.module
        if net.p_mode >= 2:
            _, _, attr = net.push_forward(data)
        else:
            _, attr = net.push_forward(data)
            attr = net.distance_2_similarity(attr)
        prototype_filters = net.prototype_class_identity[:, target].mT
        attr = (attr * prototype_filters[(...,) + (None,) * (attr.ndim - 2)]).mean(1, keepdim=True)
        return upsample(attr, data)
    elif method == 'Deconvolution':
        deconv = ctattr.Deconvolution(net)
        return deconv.attribute(data, target=target)
    elif method == 'GradCAM':
        gc = ctattr.LayerGradCam(net, net.get_submodule(conv_name))
        attr = gc.attribute(data, target=target, relu_attributions=True)
        return upsample(attr, data)
    elif method == 'Guided GradCAM':
        gc = ctattr.LayerGradCam(net, net.get_submodule(conv_name))
        attr = gc.attribute(data, target=target, relu_attributions=True)
        guided_bp = ctattr.GuidedBackprop(net)
        return guided_bp.attribute(data, target=target) * upsample(attr, data)
    elif method == 'Occlusion':
        occlusion = ctattr.Occlusion(net)
        sliding_window = (1,) + (11,) * len(data.shape[2:])
        strides = (1,) + (5,) * len(data.shape[2:])
        return occlusion.attribute(data, sliding_window, strides=strides, target=target,
                                   perturbations_per_eval=1, show_progress=show_progress)


def demo():
    # TODO: Demo
    pass


def parse_arguments():
    # Define command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-name', type=str, required=True, help="name of the model")
    parser.add_argument('-d', '--data-path', type=str,
                        default='../data/BraTS_2020/MICCAI_BraTS2020_TrainingData',
                        help="path to the data files")
    parser.add_argument('-p', '--param-grid', type=str, default=None,
                        help="grid of hyper-parameters")
    parser.add_argument('-s', '--seed', type=int, default=0, help="random seed")
    parser.add_argument('--n-workers', type=int, default=8, help="number of workers in data loader")
    parser.add_argument('--n-threads', type=int, default=4, help="number of CPU threads")
    parser.add_argument('--preloaded', type=int, choices={0, 1, 2}, default=2,
                        help="whether to preprocess (1) and preload (2) the dataset")
    parser.add_argument('--use-cuda', type=int, choices={0, 1}, default=1,
                        help="whether to use CUDA if available")
    parser.add_argument('--use-da', type=int, choices={0, 1}, default=0,
                        help="whether to use deterministic algorithms.")
    parser.add_argument('--gpus', type=str, default='0', help="indexes of GPUs")
    parser.add_argument('--load-model', type=str, default=None,
                        help="whether to load the model files")
    return parser.parse_args()


if __name__ == '__main__':
    tic = time.time()
    # Parse command-line arguments
    args = parse_arguments()
    model_name, data_path, param_grid = args.model_name, args.data_path, args.param_grid
    seed, n_workers, preloaded = args.seed, args.n_workers, args.preloaded
    n_threads, use_cuda, use_da, gpus = args.n_threads, args.use_cuda, args.use_da, args.gpus
    load_model = args.load_model
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
    transform = tio.Compose(transform)
    if '_pm' in model_name:
        p_mode = int(model_name[model_name.find('_pm') + 3])
        model_name = model_name.replace(f'_pm{p_mode}', '')
    gpu_ids = ast.literal_eval(f'[{gpus}]')
    device = torch.device(
        'cuda:' + str(gpu_ids[0]) if use_cuda and torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = False
    if use_da:
        # torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.deterministic = True
    torch.set_num_threads(n_threads)
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
            transform = None
        del dataset, data_loader
        toc = time.time()
        print(f"Elapsed time is {toc - tic:.6f} seconds.")
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=seed)
    # 5-fold CV
    cv_fold = cv.get_n_splits()
    param_grids = ParameterGrid(param_grid)
    for i, (I_train, I_test) in enumerate(cv.split(x, y.argmax(1))):
        print(f">>>>>>>> CV = {i + 1}:")
        best_grid = param_grids[0]
        np.random.seed(seed)
        torch.manual_seed(seed)
        if preloaded > 1:
            dataset_train = TensorDataset(x[I_train], y[I_train], seg[I_train])
            dataset_test = TensorDataset(x[I_test], y[I_test], seg[I_test])
            in_size = (4,) + x.shape[2:]
            out_size = y.shape[1]
        else:
            dataset_train = tio.SubjectsDataset(list(x[I_train]), transform=transform)
            dataset_test = tio.SubjectsDataset(list(x[I_test]), transform=transform)
            in_size = (4,) + dataset_train[0]['t1']['data'].shape[1:]
            out_size = dataset_train[0]['label'].shape[0]
        loader_train = DataLoader(dataset_train, batch_size=(best_grid['batch_size'] + 1) // 2,
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
        if load_model.startswith(args.model_name):
            model_name_i = f'{load_model}_cv{i}'
            model_path_i = f'../results/models/{model_name_i}.pt'
        else:
            model_name_i = f'{load_model[load_model.find(args.model_name):]}_cv{i}'
            model_path_i = f'{load_model}_cv{i}.pt'
        print(f"Model: {model_name_i}\n{str(net)}")
        print_param(net, show_each=False)
        print(f"Hyper-parameters = {param_grid}")
        print(f"Best Hyper-parameters = {best_grid}")
        if is_parallel:
            net.module.load_state_dict(torch.load(model_path_i, map_location=device))
        else:
            net.load_state_dict(torch.load(model_path_i, map_location=device))
        # f_x_test, lcs_test = test(net, loader_test)
        demo()
