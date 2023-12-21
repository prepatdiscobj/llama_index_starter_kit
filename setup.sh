#!bin/bash

#Optional
#Download the pytorch and torchvision documentation
#wget https://github.com/unknownue/PyTorch.docs/releases/download/v2.1.0/torch.zip
#wget https://github.com/unknownue/PyTorch.docs/releases/download/v2.1.0/torchvision.zip

# mkdir torch_documentation
# mkdir -p torch_documentation/torch
# mkdir -p torch_documentation/torchvision

# unzip torch.zip -d torch_documentation/torch
# unzip torchvision.zip -d torch_documentation/torchvision

# Download and install the miniconda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

~/miniconda3/bin/conda init bash
# Create virtual environement

conda create env -n beautifulmind python=3.10
conda activate beautifulmind

#Install requirements
pip install -r requirements.txt

