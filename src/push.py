#!/usr/bin/env python3

import glob
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from receptive_field import compute_rf_prototype
from utils import find_high_activation_crop, load_subjs_batch, makedir


# push each prototype to the nearest patch in the training set
def push_prototypes(data_loader,  # pytorch dataloader (must be unnormalized in [0,1])
                    ppnet,  # pytorch network with prototype_vectors
                    class_specific=True,
                    preprocess_input_function=None,  # normalize if needed
                    prototype_layer_stride=1,
                    root_dir_for_saving_prototypes=None,  # if not None, prototypes will be saved here
                    epoch_number=None,  # if not provided, prototypes saved previously will be overwritten
                    prototype_img_filename_prefix=None,
                    prototype_self_act_filename_prefix=None,
                    proto_bound_boxes_filename_prefix=None,
                    save_prototype_class_identity=True,  # which class the prototype image comes from
                    log=print,
                    prototype_activation_function_in_numpy=None):
    ppnet.eval()
    log("\tpush", end='', flush=True)

    start = time.time()
    prototype_shape = ppnet.prototype_shape
    n_prototypes = ppnet.num_prototypes
    # saves the closest distance seen so far
    global_min_proto_dist = np.full(n_prototypes, np.inf)
    # saves the patch representation that gives the current smallest distance
    global_min_fmap_patches = np.zeros(prototype_shape)

    '''
    proto_rf_boxes and proto_bound_boxes column:
    0: image index in the entire dataset
    1: height start index
    2: height end index
    3: width start index
    4: width end index
    5: (optional) class identity
    '''
    if save_prototype_class_identity:
        proto_rf_boxes = np.full(shape=[n_prototypes, 6],
                                    fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 6],
                                            fill_value=-1)
    else:
        proto_rf_boxes = np.full(shape=[n_prototypes, 5],
                                    fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 5],
                                            fill_value=-1)

    if root_dir_for_saving_prototypes != None:
        if epoch_number != None:
            proto_epoch_dir = os.path.join(root_dir_for_saving_prototypes,
                                           'epoch-'+str(epoch_number))
            makedir(proto_epoch_dir)
        else:
            proto_epoch_dir = root_dir_for_saving_prototypes
    else:
        proto_epoch_dir = None

    search_batch_size = data_loader.batch_size

    num_classes = ppnet.num_classes

    for b, subjs_batch in enumerate(data_loader):
        '''
        start_index_of_search keeps track of the index of the image
        assigned to serve as prototype
        '''
        data, target, _ = load_subjs_batch(subjs_batch)
        start_index_of_search_batch = b * search_batch_size

        update_prototypes_on_batch(data,
                                   start_index_of_search_batch,
                                   ppnet,
                                   global_min_proto_dist,
                                   global_min_fmap_patches,
                                   proto_rf_boxes,
                                   proto_bound_boxes,
                                   class_specific=class_specific,
                                   search_y=target.argmax(1) if target.ndim > 1 else target,
                                   num_classes=num_classes,
                                   preprocess_input_function=preprocess_input_function,
                                   prototype_layer_stride=prototype_layer_stride,
                                   dir_for_saving_prototypes=proto_epoch_dir,
                                   prototype_img_filename_prefix=prototype_img_filename_prefix,
                                   prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                                   prototype_activation_function_in_numpy=prototype_activation_function_in_numpy)

    if proto_epoch_dir != None and proto_bound_boxes_filename_prefix != None:
        np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + '-receptive_field' + str(epoch_number) + '.npy'),
                proto_rf_boxes)
        np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + str(epoch_number) + '.npy'),
                proto_bound_boxes)

    log("\tExecuting push...", end='', flush=True)
    prototype_update = global_min_fmap_patches.reshape(prototype_shape)
    ppnet.prototype_vectors.data.copy_(torch.tensor(prototype_update).to(ppnet.device))
    end = time.time()
    log(f"\tpush time: {end - start:.6f}s")

# update each prototype for current search batch
def update_prototypes_on_batch(search_batch_input,
                               start_index_of_search_batch,
                               ppnet,
                               global_min_proto_dist, # this will be updated
                               global_min_fmap_patches, # this will be updated
                               proto_rf_boxes, # this will be updated
                               proto_bound_boxes, # this will be updated
                               class_specific=True,
                               search_y=None, # required if class_specific == True
                               num_classes=None, # required if class_specific == True
                               preprocess_input_function=None,
                               prototype_layer_stride=1,
                               dir_for_saving_prototypes=None,
                               prototype_img_filename_prefix=None,
                               prototype_self_act_filename_prefix=None,
                               prototype_activation_function_in_numpy=None):
    ppnet.eval()

    if preprocess_input_function is not None:
        # print('preprocessing input for pushing ...')
        # search_batch = copy.deepcopy(search_batch_input)
        search_batch = preprocess_input_function(search_batch_input)
    else:
        search_batch = search_batch_input

    with torch.no_grad():
        search_batch = search_batch.to(ppnet.device, non_blocking=True)
        # this computation currently is not parallelized
        if ppnet.p_mode >= 2:
            protoL_input_torch, proto_dist_torch, p_map = ppnet.push_forward(search_batch)
            p_map_ = np.copy(p_map.detach().cpu().numpy())
            del p_map
        else:
            protoL_input_torch, proto_dist_torch = ppnet.push_forward(search_batch)
        protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
        proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())
        del search_batch, protoL_input_torch, proto_dist_torch

    if ppnet.c_mode >= 1:
        search_y = search_y[:, None].repeat(1, search_batch_input.shape[1]).reshape(-1)
        c_index = torch.arange(search_batch_input.shape[1]).repeat(search_batch_input.shape[0])
        search_batch_input = search_batch_input.reshape((-1, 1) + search_batch_input.shape[2:])
    if class_specific:
        class_to_img_index_dict = {key: [] for key in range(num_classes)}
        # img_y is the image's integer label
        for img_index, img_y in enumerate(search_y):
            img_label = img_y.item()
            class_to_img_index_dict[img_label].append(img_index)

    prototype_shape = ppnet.prototype_shape
    if len(prototype_shape) >= 5:
        is_3D = True
    else:
        is_3D = False
    n_prototypes = prototype_shape[0]
    proto_h = prototype_shape[2]
    proto_w = prototype_shape[3]
    if is_3D:
        proto_a = prototype_shape[4]
    max_dist = np.prod(prototype_shape[1:])

    for j in range(n_prototypes):
        #if n_prototypes_per_class != None:
        if class_specific:
            # target_class is the class of the class_specific prototype
            target_class = torch.argmax(ppnet.prototype_class_identity[j]).item()
            # if there is not images of the target_class from this batch
            # we go on to the next prototype
            if len(class_to_img_index_dict[target_class]) == 0:
                continue
            proto_dist_j = proto_dist_[class_to_img_index_dict[target_class]][:, j]
            if ppnet.c_mode >= 2:
                c_index_j = c_index[class_to_img_index_dict[target_class]]
                c_prototype_j = ppnet.c_prototype_identity[j].argmax().item()
                proto_dist_j = proto_dist_j[c_index_j == c_prototype_j]
        else:
            # if it is not class specific, then we will search through
            # every example
            proto_dist_j = proto_dist_[:, j]

        batch_min_proto_dist_j = np.amin(proto_dist_j)
        if batch_min_proto_dist_j < global_min_proto_dist[j]:
            batch_argmin_proto_dist_j = list(np.unravel_index(np.argmin(proto_dist_j, axis=None),
                                                              proto_dist_j.shape))
            if class_specific:
                '''
                change the argmin index from the index among
                images of the target class to the index in the entire search
                batch
                '''
                if ppnet.c_mode >= 2:
                    batch_argmin_proto_dist_j[0] = np.array(class_to_img_index_dict[target_class])[c_index_j == c_prototype_j][batch_argmin_proto_dist_j[0]]
                else:
                    batch_argmin_proto_dist_j[0] = class_to_img_index_dict[target_class][batch_argmin_proto_dist_j[0]]

            # retrieve the corresponding feature map patch
            img_index_in_batch = batch_argmin_proto_dist_j[0]
            fmap_height_start_index = batch_argmin_proto_dist_j[1] * prototype_layer_stride
            fmap_height_end_index = fmap_height_start_index + proto_h
            fmap_width_start_index = batch_argmin_proto_dist_j[2] * prototype_layer_stride
            fmap_width_end_index = fmap_width_start_index + proto_w
            if is_3D:
                fmap_axial_start_index = batch_argmin_proto_dist_j[3] * prototype_layer_stride
                fmap_axial_end_index = fmap_axial_start_index + proto_a
                if ppnet.p_mode >= 2:
                    batch_min_fmap_patch_j = protoL_input_[img_index_in_batch, j, :,
                                             fmap_height_start_index:fmap_height_end_index,
                                             fmap_width_start_index:fmap_width_end_index,
                                             fmap_axial_start_index:fmap_axial_end_index]
                else:
                    batch_min_fmap_patch_j = protoL_input_[img_index_in_batch, :,
                                             fmap_height_start_index:fmap_height_end_index,
                                             fmap_width_start_index:fmap_width_end_index,
                                             fmap_axial_start_index:fmap_axial_end_index]
            else:
                batch_min_fmap_patch_j = protoL_input_[img_index_in_batch, :,
                                         fmap_height_start_index:fmap_height_end_index,
                                         fmap_width_start_index:fmap_width_end_index]

            global_min_proto_dist[j] = batch_min_proto_dist_j
            if is_3D and global_min_fmap_patches[j].ndim < batch_min_fmap_patch_j.ndim:
                global_min_fmap_patches[j] = batch_min_fmap_patch_j.squeeze(-1)
            else:
                global_min_fmap_patches[j] = batch_min_fmap_patch_j

            # get the receptive field boundary of the image patch
            # that generates the representation
            protoL_rf_info = ppnet.proto_layer_rf_info
            rf_prototype_j = compute_rf_prototype(search_batch_input.size(2), batch_argmin_proto_dist_j, protoL_rf_info)

            # get the whole image
            if is_3D:
                ratio = search_batch_input.shape[4] // proto_dist_.shape[-1]
                axial_start = fmap_axial_start_index * ratio
                axial_end = fmap_axial_end_index * ratio
                axial_median = (axial_start + axial_end) // 2
                original_img_j = search_batch_input[rf_prototype_j[0], :, :, :, axial_median]
            else:
                original_img_j = search_batch_input[rf_prototype_j[0]]
            original_img_j = original_img_j.numpy()
            original_img_j = np.transpose(original_img_j, (1, 2, 0))
            original_img_size = original_img_j.shape[0]

            # crop out the receptive field
            rf_img_j = original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                                      rf_prototype_j[3]:rf_prototype_j[4], :]

            # save the prototype receptive field information
            if ppnet.c_mode >= 1:
                pass
            else:
                proto_rf_boxes[j, 0] = rf_prototype_j[0] + start_index_of_search_batch
                proto_rf_boxes[j, 1] = rf_prototype_j[1]
                proto_rf_boxes[j, 2] = rf_prototype_j[2]
                proto_rf_boxes[j, 3] = rf_prototype_j[3]
                proto_rf_boxes[j, 4] = rf_prototype_j[4]
                if proto_rf_boxes.shape[1] == 6 and search_y is not None:
                    proto_rf_boxes[j, 5] = search_y[rf_prototype_j[0]].item()

            # find the highly activated region of the original image
            if is_3D:
                if ppnet.p_mode >= 2:
                    ratio = p_map_.shape[4] // proto_dist_.shape[-1]
                    axial_start = fmap_axial_start_index * ratio
                    axial_end = fmap_axial_end_index * ratio
                    axial_median = (axial_start + axial_end) // 2
                    proto_dist_img_j = p_map_[img_index_in_batch, j, :, :, axial_median]
                else:
                    axial_median = (fmap_axial_start_index + fmap_axial_end_index) // 2
                    proto_dist_img_j = proto_dist_[img_index_in_batch, j, :, :, axial_median]
            else:
                proto_dist_img_j = proto_dist_[img_index_in_batch, j, :, :]
            if ppnet.p_mode >= 2:
                proto_act_img_j = proto_dist_img_j
            elif ppnet.f_dist == 'cos':
                proto_act_img_j = np.maximum(1 - proto_dist_img_j, 0)
            elif ppnet.prototype_activation_function == 'log':
                proto_act_img_j = np.log((proto_dist_img_j + 1) / (proto_dist_img_j + ppnet.epsilon))
            elif ppnet.prototype_activation_function == 'linear':
                proto_act_img_j = max_dist - proto_dist_img_j
            else:
                proto_act_img_j = prototype_activation_function_in_numpy(proto_dist_img_j)
            upsampled_act_img_j = cv2.resize(proto_act_img_j, dsize=(original_img_size, original_img_size),
                                             interpolation=cv2.INTER_CUBIC)
            proto_bound_j = find_high_activation_crop(upsampled_act_img_j)
            # crop out the image patch with high activation as prototype image
            proto_img_j = original_img_j[proto_bound_j[0]:proto_bound_j[1],
                                         proto_bound_j[2]:proto_bound_j[3], :]

            # save the prototype boundary (rectangular boundary of highly activated region)
            if ppnet.c_mode >= 1:
                pass
            else:
                proto_bound_boxes[j, 0] = proto_rf_boxes[j, 0]
                proto_bound_boxes[j, 1] = proto_bound_j[0]
                proto_bound_boxes[j, 2] = proto_bound_j[1]
                proto_bound_boxes[j, 3] = proto_bound_j[2]
                proto_bound_boxes[j, 4] = proto_bound_j[3]
                if proto_bound_boxes.shape[1] == 6 and search_y is not None:
                    proto_bound_boxes[j, 5] = search_y[rf_prototype_j[0]].item()

            if dir_for_saving_prototypes is not None:
                # if prototype_self_act_filename_prefix is not None:
                #     # save the numpy array of the prototype self activation
                #     np.save(os.path.join(dir_for_saving_prototypes,
                #                          prototype_self_act_filename_prefix + str(j) + '.npy'),
                #             proto_act_img_j)
                if prototype_img_filename_prefix is not None:
                    # save the whole image containing the prototype as png
                    n_d = len(f'{n_prototypes}')
                    modalities = ['t1', 't1ce', 't2', 'flair']
                    if ppnet.c_mode >= 1:
                        c_index_j = c_index[img_index_in_batch]
                        modalities = modalities[c_index_j:c_index_j + 1]
                        if ppnet.c_mode == 1:
                            for img in glob.glob(os.path.join(dir_for_saving_prototypes,
                                                              f'{prototype_img_filename_prefix}'
                                                              f'*{j + 1:0{n_d}d}-*.png')):
                                os.remove(img)
                    original_img_j_ = original_img_j - np.amin(original_img_j)
                    original_img_j_ = original_img_j_ / np.amax(original_img_j_).clip(ppnet.epsilon)
                    # for m in range(len(modalities)):
                    #     plt.imsave(os.path.join(dir_for_saving_prototypes,
                    #                             f'{prototype_img_filename_prefix}'
                    #                             f'-original{j + 1:0{n_d}d}-{modalities[m]}.png'),
                    #                original_img_j_[:, :, m].T, vmin=0., vmax=1., origin='lower')
                    # overlay (upsampled) self activation on original image and save the result
                    rescaled_act_img_j = upsampled_act_img_j - np.amin(upsampled_act_img_j)
                    rescaled_act_img_j = (rescaled_act_img_j
                                          / np.amax(rescaled_act_img_j).clip(ppnet.epsilon))
                    heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_act_img_j),
                                                cv2.COLORMAP_JET)
                    heatmap = np.float32(heatmap) / 255
                    heatmap = heatmap[..., ::-1]
                    for m in range(len(modalities)):
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                f'{prototype_img_filename_prefix}'
                                                f'-original_with_self_act{j + 1:0{n_d}d}'
                                                f'-{modalities[m]}.png'),
                                   (0.5 * original_img_j_[:, :, m:m + 1]
                                    + 0.3 * heatmap).transpose((1, 0, 2)),
                                   vmin=0., vmax=1., origin='lower')
                    # # if different from the original (whole) image, save the prototype receptive field as png
                    # if rf_img_j.shape[0] != original_img_size or rf_img_j.shape[1] != original_img_size:
                    #     rf_img_j_ = original_img_j_[rf_prototype_j[1]:rf_prototype_j[2],
                    #                                 rf_prototype_j[3]:rf_prototype_j[4]]
                    #     for m in range(len(modalities)):
                    #         plt.imsave(os.path.join(dir_for_saving_prototypes,
                    #                                 f'{prototype_img_filename_prefix}'
                    #                                 f'-receptive_field{j + 1:0{n_d}d}'
                    #                                 f'-{modalities[m]}.png'),
                    #                    rf_img_j_[:, :, m].T, vmin=0., vmax=1., origin='lower')
                    #     rf_heatmap = heatmap[rf_prototype_j[1]:rf_prototype_j[2],
                    #                          rf_prototype_j[3]:rf_prototype_j[4]]
                    #     for m in range(len(modalities)):
                    #         plt.imsave(os.path.join(dir_for_saving_prototypes,
                    #                                 f'{prototype_img_filename_prefix}'
                    #                                 f'-receptive_field_with_self_act{j + 1:0{n_d}d}'
                    #                                 f'-{modalities[m]}.png'),
                    #                    (0.5 * rf_img_j_[:, :, m:m + 1]
                    #                     + 0.3 * rf_heatmap).transpose((1, 0, 2)),
                    #                    vmin=0., vmax=1., origin='lower')
                    # save the prototype image (highly activated region of the whole image)
                    proto_img_j_ = original_img_j_[proto_bound_j[0]:proto_bound_j[1],
                                                   proto_bound_j[2]:proto_bound_j[3]]
                    for m in range(len(modalities)):
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                f'{prototype_img_filename_prefix}'
                                                f'{j + 1:0{n_d}d}-{modalities[m]}.png'),
                                   proto_img_j_[:, :, m].T, vmin=0., vmax=1., origin='lower')
