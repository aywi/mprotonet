conda create -n mprotonet -y python=3.10
. $(conda info --base)/etc/profile.d/conda.sh
conda activate mprotonet
conda install -y pip numpy scipy matplotlib ipython jupyter
conda install -c pytorch -c nvidia -y pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6
conda install -c pytorch -y captum
pip install -U torchio
conda clean -a -y
conda info
