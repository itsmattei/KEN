# KEN pruning algorithm 🕶️
KEN (Kernel density Estimator for Neural Network compression): a straightforward, universal and unstructured pruning algorithm based on Kernel Density Estimation (KDE) for transformer compression.

![](https://github.com/itsmattei/KEN/blob/main/files/KEN_workflow.gif)

This repository contains all the code to replicate the experiments shown in [_KEN: a Universal and Simple Non-Parametric Pruning Algorithm for Large Language Models_](https://pages.github.com/)

Based on the different KEN application, this repository includes the following packages:
```bash
KEN
├── easy_train                              <-- an useful package used for train in a very fast faschion your LLM
    ├── easy_train.py
    └── easy_train_large_models.py         
├── model_compression                       <-- for the download of the compressed model and its support dictionary
└──compress_file.py
├── pretrained_model_injection              <-- KEN applied the selected fine-tuned params in a pre-trained model
    ├── inject_all_layers.py
    └── inject_attention_layers.py
└── trained_model_injection                 <-- KEN substitute the not selected parameters with its pre-trained values
    ├── inject_all_layers.py
    └── inject_attention_layers.py

KENviz                                      <-- Visualization tool
└── KEN_viz.py

```

## Usage
