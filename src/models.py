#!/usr/bin/env python3

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as vision_models

from receptive_field import compute_proto_layer_rf_info_v2
from utils import replace_module


def build_resnet_features(net):
    return nn.Sequential(OrderedDict([
        ('conv1', net.conv1),
        ('bn1', net.bn1),
        ('relu', net.relu),
        ('maxpool', net.maxpool),
        ('layer1', net.layer1),
        ('layer2', net.layer2),
        ('layer3', net.layer3),
        ('layer4', net.layer4)
    ]))


def features_imagenet1k(features):
    if features == 'resnet18':
        return build_resnet_features(vision_models.resnet18(weights='IMAGENET1K_V1'))
    elif features == 'resnet18_ri':
        return build_resnet_features(vision_models.resnet18())
    elif features == 'resnet34':
        return build_resnet_features(vision_models.resnet34(weights='IMAGENET1K_V1'))
    elif features == 'resnet34_ri':
        return build_resnet_features(vision_models.resnet34())
    elif features == 'resnet50':
        return build_resnet_features(vision_models.resnet50(weights='IMAGENET1K_V2'))
    elif features == 'resnet50_ri':
        return build_resnet_features(vision_models.resnet50())
    elif features == 'resnet101':
        return build_resnet_features(vision_models.resnet101(weights='IMAGENET1K_V2'))
    elif features == 'resnet101_ri':
        return build_resnet_features(vision_models.resnet101())
    elif features == 'resnet152':
        return build_resnet_features(vision_models.resnet152(weights='IMAGENET1K_V2'))
    elif features == 'resnet152_ri':
        return build_resnet_features(vision_models.resnet152())


def init_conv(net):
    for m in net.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def init_resnet3d_features(net, in_channels=3):
    replace_module(net, [nn.Conv2d, {'in_channels': 3, 'out_channels': 64}], nn.Conv3d, in_channels,
                   64, 7, stride=2, padding=3, bias=False)
    for ic, oc, ks, st, pd in [(64, 64, 1, 1, 0), (64, 64, 3, 1, 1), (64, 128, 1, 2, 0),
                               (64, 128, 3, 2, 1), (64, 256, 1, 1, 0), (128, 128, 3, 1, 1),
                               (128, 128, 3, 2, 1), (128, 256, 1, 2, 0), (128, 256, 3, 2, 1),
                               (128, 512, 1, 1, 0), (256, 64, 1, 1, 0), (256, 128, 1, 1, 0),
                               (256, 256, 3, 1, 1), (256, 256, 3, 2, 1), (256, 512, 1, 2, 0),
                               (256, 512, 3, 2, 1), (256, 1024, 1, 1, 0), (512, 128, 1, 1, 0),
                               (512, 256, 1, 1, 0), (512, 512, 3, 1, 1), (512, 512, 3, 2, 1),
                               (512, 1024, 1, 2, 0), (512, 2048, 1, 1, 0), (1024, 256, 1, 1, 0),
                               (1024, 512, 1, 1, 0), (1024, 2048, 1, 2, 0), (2048, 512, 1, 1, 0)]:
        replace_module(net,
                       [nn.Conv2d, {'in_channels': ic, 'out_channels': oc, 'kernel_size': (ks, ks),
                                    'stride': (st, st), 'padding': (pd, pd)}],
                       nn.Conv3d, ic, oc, ks, stride=st, padding=pd, bias=False)
    for nf in [64, 128, 256, 512, 1024, 2048]:
        replace_module(net, [nn.BatchNorm2d, {'num_features': nf}], nn.BatchNorm3d, nf)
    replace_module(net, nn.MaxPool2d, nn.MaxPool3d, 3, stride=2, padding=1)
    init_conv(net)


def conv_info(net):
    kernel_sizes, strides, paddings = [], [], []
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            if m.kernel_size[0] == 1 and m.stride[0] == 2:
                continue
            kernel_sizes += [m.kernel_size[0]]
            strides += [m.stride[0]]
            paddings += [m.padding[0]]
        elif isinstance(m, (nn.MaxPool2d, nn.MaxPool3d)):
            kernel_sizes += [m.kernel_size if isinstance(m.kernel_size, int) else m.kernel_size[0]]
            strides += [m.stride if isinstance(m.stride, int) else m.stride[0]]
            paddings += [m.padding if isinstance(m.padding, int) else m.padding[0]]
    return kernel_sizes, strides, paddings


# 3-dimensional convolutional neural network with predefined architectures
class CNN(nn.Module):
    def __init__(self, in_size=(4, 240, 240, 155), out_size=2, features='resnet50_ri', n_layers=7,
                 **kwargs):
        super(CNN, self).__init__()
        self.features = features_imagenet1k(features)[:n_layers]
        self.features_name = features + f'[:{n_layers}]' if n_layers else features
        add_ons_channels = [m for m in self.features.modules()
                            if isinstance(m, nn.BatchNorm2d)][-1].num_features
        self.add_ons = nn.Sequential(
            nn.Conv3d(add_ons_channels, 128, 1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, 1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(1)
        )
        self.fc = nn.Linear(128, out_size, bias=False)
        if self.features_name.startswith('resnet'):
            init_resnet3d_features(self.features, in_channels=in_size[0])
        init_conv(self.add_ons)

    def forward(self, x, missing=None):
        x = self.features(x)
        x = self.add_ons(x)
        x = self.fc(x)
        return x


def CNN3D(**kwargs):
    return CNN(**kwargs)


# Medical Prototype Network (MProtoNet)
class MProtoNet(nn.Module):
    def __init__(self, in_size=(4, 240, 240, 155), out_size=2, features='resnet50_ri', n_layers=7,
                 prototype_shape=(20, 128, 1, 1, 1), init_weights=True, f_dist='l2',
                 prototype_activation_function='log', p_mode=0, topk_p=1, **kwargs):
        super(MProtoNet, self).__init__()
        self.in_size = in_size
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.num_classes = out_size
        self.epsilon = 1e-4

        self.f_dist = f_dist
        # prototype_activation_function could be 'log', 'linear', or a generic function that
        # converts distance to similarity score
        self.prototype_activation_function = prototype_activation_function

        assert self.num_prototypes % self.num_classes == 0
        # A onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = nn.Parameter(
            torch.zeros(self.num_prototypes, self.num_classes), requires_grad=False)
        self.num_prototypes_per_class = self.num_prototypes // self.num_classes
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // self.num_prototypes_per_class] = 1

        while len(self.prototype_shape) < 5:
            self.prototype_shape += (1,)
        self.features = features_imagenet1k(features)[:n_layers]
        self.features_name = features + f'[:{n_layers}]' if n_layers else features
        add_ons_channels = [m for m in self.features.modules()
                            if isinstance(m, nn.BatchNorm2d)][-1].num_features
        self.add_ons = nn.Sequential(
            nn.Conv3d(add_ons_channels, self.prototype_shape[1], 1, bias=False),
            nn.BatchNorm3d(self.prototype_shape[1]),
            nn.ReLU(inplace=True),
            nn.Conv3d(self.prototype_shape[1], self.prototype_shape[1], 1, bias=False),
            nn.BatchNorm3d(self.prototype_shape[1]),
            nn.Sigmoid()
        )

        layer_filter_sizes, layer_strides, layer_paddings = conv_info(self.features)
        self.proto_layer_rf_info = compute_proto_layer_rf_info_v2(
            img_size=self.in_size[1], layer_filter_sizes=layer_filter_sizes,
            layer_strides=layer_strides, layer_paddings=layer_paddings,
            prototype_kernel_size=self.prototype_shape[2]
        )

        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape))

        # Do not make this just a tensor, since it will not be automatically moved to GPU(s)
        self.ones = nn.Parameter(torch.ones(self.prototype_shape), requires_grad=False)

        self.p_mode = p_mode
        if self.p_mode >= 1:
            p_size_w = compute_proto_layer_rf_info_v2(
                img_size=self.in_size[2], layer_filter_sizes=layer_filter_sizes,
                layer_strides=layer_strides, layer_paddings=layer_paddings,
                prototype_kernel_size=self.prototype_shape[3]
            )[0]
            p_size_a = compute_proto_layer_rf_info_v2(
                img_size=self.in_size[3], layer_filter_sizes=layer_filter_sizes,
                layer_strides=layer_strides, layer_paddings=layer_paddings,
                prototype_kernel_size=self.prototype_shape[4]
            )[0]
            self.p_size = (self.proto_layer_rf_info[0], p_size_w, p_size_a)
            self.topk_p = int(self.p_size[0] * self.p_size[1] * self.p_size[2] * topk_p / 100)
            assert self.topk_p >= 1
        if self.p_mode >= 2:
            self.p_map = nn.Sequential(
                nn.Conv3d(add_ons_channels, self.prototype_shape[1], 1, bias=False),
                nn.BatchNorm3d(self.prototype_shape[1]),
                nn.ReLU(inplace=True),
                nn.Conv3d(self.prototype_shape[1], self.prototype_shape[0], 1, bias=False),
                nn.BatchNorm3d(self.prototype_shape[0]),
                nn.Sigmoid()
            )

        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes, bias=False)

        if init_weights:
            self._initialize_weights()

    # https://github.com/pytorch/pytorch/issues/7460
    @property
    def device(self):
        device, = set([p.device for p in self.parameters()] + [b.device for b in self.buffers()])
        return device

    def scale(self, x, dim):
        x = x - x.amin(dim, keepdim=True)
        return x / x.amax(dim, keepdim=True).clamp_min(self.epsilon)

    def sigmoid(self, x, omega=10, sigma=0.5):
        return torch.sigmoid(omega * (x - sigma))

    def get_p_map(self, x):
        if self.p_mode in [3, 5]:
            p_map = F.relu(self.p_map[:-1](x))
            p_map = self.scale(p_map, tuple(range(1, p_map.ndim)))
            return self.sigmoid(p_map)
        else:
            return self.p_map(x)

    def lse_pooling(self, x, r=10, dim=-1):
        return (torch.logsumexp(r * x, dim=dim) - torch.log(torch.tensor(x.shape[dim]))) / r

    def conv_features(self, x):
        x = self.features(x)
        f_x = self.add_ons(x)
        if self.p_mode >= 2:
            p_map = self.get_p_map(x)
            return f_x, p_map, x
        else:
            return f_x

    def l2_convolution_2D(self, x):
        x_2 = F.conv2d(x ** 2, self.ones)
        xp = F.conv2d(x, self.prototype_vectors)
        p_2 = (self.prototype_vectors ** 2).sum((1, 2, 3)).reshape(-1, 1, 1)
        return F.relu(x_2 - 2 * xp + p_2)

    def l2_convolution_3D(self, x):
        if x.shape[1:] == self.prototype_shape:
            x_2 = (x ** 2).sum(2)
            xp = (x * self.prototype_vectors).sum(2)
        else:
            x_2 = F.conv3d(x ** 2, self.ones)
            xp = F.conv3d(x, self.prototype_vectors)
        p_2 = (self.prototype_vectors ** 2).sum((1, 2, 3, 4)).reshape(-1, 1, 1, 1)
        return F.relu(x_2 - 2 * xp + p_2)

    def cosine_convolution_2D(self, x):
        assert x.min() >= 0, f"{x.min():.16g} >= 0"
        x_unit = F.normalize(x, p=2, dim=1)
        prototype_vectors_unit = F.normalize(self.prototype_vectors, p=2, dim=1)
        return F.relu(1 - F.conv2d(input=x_unit, weight=prototype_vectors_unit))

    def cosine_convolution_3D(self, x):
        assert x.min() >= 0, f"{x.min():.16g} >= 0"
        prototype_vectors_unit = F.normalize(self.prototype_vectors, p=2, dim=1)
        if x.shape[1:] == self.prototype_shape:
            x_unit = F.normalize(x, p=2, dim=2)
            return F.relu(1 - (x_unit * prototype_vectors_unit).sum(2))
        else:
            x_unit = F.normalize(x, p=2, dim=1)
            return F.relu(1 - F.conv3d(input=x_unit, weight=prototype_vectors_unit))

    def prototype_distances(self, x, p_map=None):
        if self.p_mode >= 2:
            p_size = x.flatten(2).shape[2]
            p_x = (torch.einsum('bphwa,bchwa->bpc', p_map, x) / p_size)[(...,) + (None,) * 3]
            if self.f_dist == 'l2':
                return self.l2_convolution_3D(p_x), p_map, p_x
            elif self.f_dist == 'cos':
                return self.cosine_convolution_3D(p_x), p_map, p_x
        else:
            if self.f_dist == 'l2':
                return self.l2_convolution_3D(x)
            elif self.f_dist == 'cos':
                return self.cosine_convolution_3D(x)

    def distance_2_similarity(self, distances):
        if self.f_dist == 'cos':
            return F.relu(1 - distances)
        elif self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        else:
            return self.prototype_activation_function(distances)

    def forward(self, x, missing=None):
        if self.p_mode >= 2:
            f_x, p_map, x = self.conv_features(x)
            distances, p_map, _ = self.prototype_distances(f_x, p_map)
        else:
            f_x = self.conv_features(x)
            distances = self.prototype_distances(f_x)
        distances = distances.flatten(2)
        if self.p_mode == 1:
            min_distances = distances.topk(self.topk_p, dim=2, largest=False)[0].mean(2)
        elif self.p_mode >= 2:
            min_distances = distances.flatten(1)
        else:
            min_distances = distances.amin(2)
        prototype_activations = self.distance_2_similarity(min_distances)
        logits = self.last_layer(prototype_activations)
        if self.p_mode >= 2 and self.training:
            return logits, min_distances, x, p_map
        elif self.training:
            return logits, min_distances
        else:
            return logits

    def push_forward(self, x):
        if self.p_mode >= 2:
            f_x, p_map, _ = self.conv_features(x)
            distances, p_map, p_x = self.prototype_distances(f_x, p_map)
            return p_x, distances, p_map
        else:
            f_x = self.conv_features(x)
            distances = self.prototype_distances(f_x)
            return f_x, distances

    def prune_prototypes(self, prototypes_to_prune):
        # TODO: Pruning
        pass

    def __repr__(self):
        # MProtoNet(self, features, img_size, prototype_shape, proto_layer_rf_info, num_classes,
        # epsilon, p_size, topk_p):
        return (
            f"MProtoNet(\n"
            f"\tfeatures: {self.features_name},\n"
            f"\timg_size: {self.in_size[1:]},\n"
            f"\tprototype_shape: {self.prototype_shape},\n"
            f"\tproto_layer_rf_info: {self.proto_layer_rf_info},\n"
            f"\tnum_classes: {self.num_classes},\n"
            f"\tepsilon: {self.epsilon},\n"
            f"\tp_size: {self.p_size if self.p_mode >= 1 else None},\n"
            f"\ttopk_p: {self.topk_p if self.p_mode >= 1 else None}\n"
            f")"
        )

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        positive_one_weights_locations = self.prototype_class_identity.mT
        negative_one_weights_locations = 1 - positive_one_weights_locations
        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations
        )

    def _initialize_weights(self):
        if self.features_name.startswith('resnet'):
            init_resnet3d_features(self.features, in_channels=self.in_size[0])
        init_conv(self.add_ons)
        if self.p_mode >= 2:
            init_conv(self.p_map)
        self.set_last_layer_incorrect_connection(-0.5)


def MProtoNet3D(**kwargs):
    return MProtoNet(**kwargs)
