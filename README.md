<div align="center">

# Pooling Layers For Simplicial Convolutional Neural Networks

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
 <!---
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)
-->
    
</div>

## Description

This repository contains the code for the paper Pooling Layers For Simplicial Convolutional Neural Networks.

_The goal of this paper is to introduce pooling layers for simplicial convolutional neural networks (SCNs). Inspired by well-known graph pooling layers, we propose a general formulation for a simplicial pooling layer, and we tailor it to design four different layers, grounded in the theory of topological signal processing and based on local aggregation of simplicial signals, principled selection of sampling sets and consequent downsampling and topology adaptation. We leverage the proposed layers in a hierarchical architecture to enhance the possibilities of reducing complexity and representing data at different resolutions. We numerically show that the proposed strategies generally improve learning performance with respect to GCNs-based methods and SCNs without pooling layers._

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/domenicocinque/spm
cd sap

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
