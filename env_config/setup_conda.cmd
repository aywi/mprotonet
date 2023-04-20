@echo off
call conda create -n mprotonet -y python=3.10
call conda activate mprotonet
call conda install -y pip numpy scipy matplotlib ipython jupyter
call conda install -c pytorch -c nvidia -y pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6
call conda install -c pytorch -y captum
pip install -U torchio
call conda clean -a -y
call conda info
pause
