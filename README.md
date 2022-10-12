<div align="center">

# Pooling Strategies for Simplicial Convolutional Networks

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
 <!---
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)
-->
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://arxiv.org/abs/2210.05490)
    
</div>

## Description

This repository contains the code for the paper [Pooling Strategies for Simplicial Convolutional Networks](https://arxiv.org/abs/2210.05490).

_The goal of this paper is to introduce pooling strategies for simplicial convolutional neural networks. Inspired by graph pooling methods, we introduce a general formulation for a simplicial pooling layer that performs: i) local aggregation of simplicial signals; ii) principled selection of sampling sets; iii) downsampling and simplicial topology adaptation. The general layer is then customized to design four different pooling strategies (i.e., max, top-k, self-attention, and separated top-k) grounded in the theory of topological signal processing. Also, we leverage the proposed layers in a hierarchical architecture that reduce complexity while representing data at different resolutions. Numerical results on real data benchmarks (i.e., flow and graph classification) illustrate the advantage of the proposed methods with respect to the state of the art._

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/domenicocinque/spm
cd spm

# [OPTIONAL] create conda environment
conda create -n spm python=3.9
conda activate spm

# install pytorch according to instructions
# https://pytorch.org/get-started/
# install pytorch geometric according to instructions
# https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

# install requirements
pip install -r requirements.txt
```

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 datamodule.batch_size=64
```
