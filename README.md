# HybridHydro | Hybrid PB-ML models

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7596997.svg)](https://doi.org/10.5281/zenodo.7596997)

[Lu Li](https://www.researchgate.net/profile/Lu-Li-69?ev=hdr_xprf)

### Introduction

To adequately incorporate physical information to improve pure DL models, we developed a hybrid model based on an attention mechanism and condition hybrid schemes. We further proposed an ensemble model by averaging outputs of various hybrid models using simple average, condition, and attention methods. We applied two proposed models to improve ConvLSTM by GFS forecasts and compared with three widely used hybrid methods based on in-situ and gridded data evaluation. The results showed that the proposed ensemble hybrid model achieves the best general performance among all hybrid models from 1 to 16 days forecasting, and is amenable to different soil conditions. It is highlighted that the ensemble model improves at least 65% of R compared to ConvLSTM for 16-day forecasting, and outperforms it over 79.5% in-situ stations. Moreover, our proposed attention-based hybrid model, which detects 60.6% and 56.8% drought events separately for 1-week and 2-week forecasts, achieves the best drought events predictability over arid, temperate, cold and polar regions. Our findings emphasized that the proposed hybrid models could address the problem of pure DL models on long-term and extreme forecasting and could break the performance ceiling constrained by training datasets.

### How to use

**1. Clone our repo**

```shell
git clone git@github.com:leelew/HybridHydro.git
```

**2. Download dataset**

The dataset used in the paper could be download from [zenodo](https://doi.org/10.5281/zenodo.7596997). You could test the repo with downloading each sub-tasks rather than the whole data. Please make an `input` dir and move download data into this fold.

**3. Wandb init**

The code is monitored by Weight & Biases ([Wandb](https://wandb.ai/)). Please register and wandb init in your own server (choose "Create New" when first set up wandb project).

**4. (Optional) Search for best parameters**

We use [Wandb Sweeps](https://docs.wandb.ai/guides/sweeps) to automate hyperparameter search with Bayesian methods (All logs is shown in [logs](https://wandb.ai/lilu)). If you want to explore the best parameters of HybridHydro, you could train models with our sweep configuration (`sweep.yml`)

(1) Initialize sweep and get the Sweep ID:

```shell
wandb sweep sweep.yml
```

(2) Start sweep agent

```shell
wandb agent sweep_id
```

**5. (Optional) Change configuration**

Change `configs.py` if you want to train HybridHydro model with your own configurations. Otherwise, it will train by default parameters.

**6. Run**

Change your work path (e.g., forecast, saved models), job numbers and model name in `run.sh`, and perform:

```shell
bash run.sh
```

**7. Postprocess**

The forecast of models is shown in forecast path defined in `run.sh`. If you have trained all 24 sub-tasks models, change the model name in `postprocess.sh` and perform:

```shell
bash postprocess.sh
```

### Edition

HybridHydro has five edition, shown in different branchs.

[V1](): train local model for each patch (112 x 112)

[V2](https://github.com/leelew/HybridHydro/tree/V2): train global model for all patches (112 x 112)

[V3](https://github.com/leelew/HybridHydro/tree/V3): train global model for all patches (28 x 28)

[V4](https://github.com/leelew/HybridHydro/tree/V4): train local model for each patch based on transfer learning V2 (112 x 112)

[V5](https://github.com/leelew/HybridHydro/tree/V5): train local model for each patch based on transfer learning V3 (28 x 28)

### Notation

1. Add ancillary data (DEM, land cover) would slightly decrease the performance. 

### Citation

In case you use HybridHydro in your research or work, please cite:

```bibtex
@article{Lu Li,
    author = {Lu Li, Yongjiu Dai et al.},
    title = {Enhancing Deep Learning Soil Moisture Forecasting Models by Integrating Physics-based Models},
    journal = {Advances in Atompsheric Sciences},
    year = {2024},
    DOI = {https://doi.org/10.1007/s00376-023-3181-8}
}
```

### [License](https://github.com/leelew/HybridHydro/LICENSE)

Copyright (c) 2022, Lu Li
