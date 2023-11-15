# MProtoNet

This repository contains the official implementation of MProtoNet from the paper "[MProtoNet: A Case-Based Interpretable Model for Brain Tumor Classification with 3D Multi-parametric Magnetic Resonance Imaging](https://openreview.net/forum?id=6Wbj3QCo4U4 "https://openreview.net/forum?id=6Wbj3QCo4U4")" (accepted at [MIDL 2023](https://2023.midl.io/papers/p218 "https://2023.midl.io/papers/p218")) by Yuanyuan Wei, Roger Tam, and Xiaoying Tang.

Talk: <https://www.youtube.com/watch?v=DtfXwrliVQg>

![Architecture](./images/architecture.png "The overall architecture of MProtoNet")

- [Directory Structure](#directory-structure)
- [Dataset](#dataset)
- [Environment Configuration](#environment-configuration)
- [Experiments](#experiments)
- [Results](#results)
- [Acknowledgment](#acknowledgment)
- [Citation](#citation)
- [To-do List](#to-do-list)

## Directory Structure

- `data/`: Dataset
- `env_config/`: Environment configuration
- `results/`: Results
- `src/`: Source code
  - `tumor_cls.py`: Major code for running the experiments

## Dataset

- [MICCAI Brain Tumor Segmentation (BraTS) Challenge 2020](https://www.med.upenn.edu/cbica/brats2020/ "https://www.med.upenn.edu/cbica/brats2020/") (BraTS_2020)

Please follow the instructions on the website above to request the BraTS_2020 dataset (it may take 1~4 days) and download `MICCAI_BraTS2020_TrainingData.zip` to the `data/` folder. Then unzip the dataset:

```sh
unzip ./data/MICCAI_BraTS2020_TrainingData.zip -d ./data/BraTS_2020
```

## Environment Configuration

Prerequisites: Python (>=3.8), NumPy, SciPy, scikit-learn, Matplotlib, Jupyter, PyTorch, Captum, OpenCV, TorchIO.

If you are using Miniconda/Anaconda, you can run following commands (Linux & Windows) to create a new conda environment called `mprotonet` with all the required packages (they can take up about 10 GB of disk space).

```sh
conda env create -f ./env_config/environment.yml -n mprotonet -y
# Or if you want to install the latest versions of the packages (with Python 3.11 and CUDA 11):
# conda env create -f ./env_config/environment_latest.yml -n mprotonet -y
conda activate mprotonet
conda clean -a -y
conda env list
```

## Experiments

The following experiments can consume about 12~18 GB of GPU memory.

```sh
cd ./src
# CNN (with GradCAM)
python ./tumor_cls.py -m CNN3D -n 100 -p "{'batch_size': [32], 'lr': [0.001], 'wd': [0.01], 'features': ['resnet152_ri'], 'n_layers': [6]}" --save-model 1
# ProtoPNet
python ./tumor_cls.py -m MProtoNet3D_pm1 -n 100 -p "{'batch_size': [32], 'lr': [0.001], 'wd': [0.01], 'features': ['resnet152_ri'], 'n_layers': [6], 'prototype_shape': [(30, 128, 1, 1, 1)], 'f_dist': ['cos'], 'topk_p': [1], 'coefs': [{'cls': 1, 'clst': 0.8, 'sep': -0.08, 'L1': 0.01, 'map': 0, 'OC': 0}]}" --save-model 1
# XProtoNet
python ./tumor_cls.py -m MProtoNet3D_pm2 -n 100 -p "{'batch_size': [32], 'lr': [0.001], 'wd': [0.01], 'features': ['resnet152_ri'], 'n_layers': [6], 'prototype_shape': [(30, 128, 1, 1, 1)], 'f_dist': ['cos'], 'topk_p': [1], 'coefs': [{'cls': 1, 'clst': 0.8, 'sep': -0.08, 'L1': 0.01, 'map': 0.5, 'OC': 0}]}" --save-model 1
# MProtoNet A
python ./tumor_cls.py -m MProtoNet3D_pm3 -n 100 -p "{'batch_size': [32], 'lr': [0.001], 'wd': [0.01], 'features': ['resnet152_ri'], 'n_layers': [6], 'prototype_shape': [(30, 128, 1, 1, 1)], 'f_dist': ['cos'], 'topk_p': [1], 'coefs': [{'cls': 1, 'clst': 0.8, 'sep': -0.08, 'L1': 0.01, 'map': 0.5, 'OC': 0}]}" --save-model 1
# MProtoNet B
python ./tumor_cls.py -m MProtoNet3D_pm4 -n 100 -p "{'batch_size': [32], 'lr': [0.001], 'wd': [0.01], 'features': ['resnet152_ri'], 'n_layers': [6], 'prototype_shape': [(30, 128, 1, 1, 1)], 'f_dist': ['cos'], 'topk_p': [1], 'coefs': [{'cls': 1, 'clst': 0.8, 'sep': -0.08, 'L1': 0.01, 'map': 0.5, 'OC': 0.05}]}" --save-model 1
# MProtoNet C
python ./tumor_cls.py -m MProtoNet3D_pm5 -n 100 -p "{'batch_size': [32], 'lr': [0.001], 'wd': [0.01], 'features': ['resnet152_ri'], 'n_layers': [6], 'prototype_shape': [(30, 128, 1, 1, 1)], 'f_dist': ['cos'], 'topk_p': [1], 'coefs': [{'cls': 1, 'clst': 0.8, 'sep': -0.08, 'L1': 0.01, 'map': 0.5, 'OC': 0.05}]}" --save-model 1
```

## Results

## Acknowledgment

This repository contains modified source code from [cfchen-duke/ProtoPNet](https://github.com/cfchen-duke/ProtoPNet "https://github.com/cfchen-duke/ProtoPNet") ([MIT License](https://github.com/cfchen-duke/ProtoPNet/blob/81bf2b70cb60e4f36e25e8be386eb616b7459321/LICENSE "https://github.com/cfchen-duke/ProtoPNet/blob/81bf2b70cb60e4f36e25e8be386eb616b7459321/LICENSE")) by Chaofan Chen, Oscar Li, Chaofan Tao, Alina Jade Barnett, and Cynthia Rudin.

## Citation

Yuanyuan Wei, Roger Tam, and Xiaoying Tang. MProtoNet: a case-based interpretable model for brain tumor classification with 3D multi-parametric magnetic resonance imaging. In *Medical Imaging with Deep Learning*, Nashville, United States, July 2023.

```bibtex
@inproceedings{wei2023mprotonet,
  title = {{{MProtoNet}}: A Case-Based Interpretable Model for Brain Tumor Classification with {{3D}} Multi-Parametric Magnetic Resonance Imaging},
  shorttitle = {{{MProtoNet}}},
  booktitle = {Medical {{Imaging}} with {{Deep Learning}}},
  author = {Wei, Yuanyuan and Tam, Roger and Tang, Xiaoying},
  year = {2023},
  month = jul,
  address = {{Nashville, United States}},
  url = {https://openreview.net/forum?id=6Wbj3QCo4U4}
}
```

## To-do List

- [x] Dataset
- [x] Environment configuration
- [x] Upload source code
- [x] Experiments
- [ ] Results
