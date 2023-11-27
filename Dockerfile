FROM nvidia/cuda:11.0.3-devel-ubuntu20.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update && apt-get install --yes gcc wget vim git build-essential \
    && apt-get upgrade -y libstdc++6

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-py38_23.5.2-0-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash /Miniconda3-py38_23.5.2-0-Linux-x86_64.sh -b \
    && rm -f /Miniconda3-py38_23.5.2-0-Linux-x86_64.sh

RUN pip3 install opencv-python meerkat-ml scikit-learn==1.3.1  numpy==1.18.0 tqdm==4.49.0 ipywidgets seaborn pydantic==1.10.8

# RUN conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
RUN conda install pandas matplotlib jupyter notebook 
# RUN conda install wandb gxx gcc --channel conda-forge

RUN pip3 install "domino[all] @ git+https://github.com/HazyResearch/domino@main"
