# KEN pruning algorithm 🕶️
KEN (Kernel density Estimator for Neural Network compression): a straightforward, universal and unstructured pruning algorithm based on Kernel Density Estimation (KDE) for transformer compression.

![](https://github.com/itsmattei/KEN/blob/main/files/KEN_workflow.gif)

This repository contains all the code to replicate the experiments shown in [_KEN: a Universal and Simple Non-Parametric Pruning Algorithm for Large Language Models_](https://pages.github.com/)

Based on the different KEN application, this repository includes the following packages:
```bash
KEN
├── setup                              <-- a useful package to train your LLM very quickly
    ├── easy_train.py
    └── easy_train_large_models.py         
├── model_compression                       <-- for downloading the compressed model and its supporting dictionary
└──compress_file.py
├── pretrained_model_injection              <-- KEN injects the selected fine-tuned params in a pre-trained model
    ├── inject_all_layers.py
    └── inject_attention_layers.py
└── trained_model_injection                 <-- KEN replaces unselected parameters with its pre-set values
    ├── inject_all_layers.py
    └── inject_attention_layers.py

KENviz                                      <-- Visualization tool
└── KEN_viz.py
```

## Usage
To use KEN, you can simply follow these steps:

1. Clone the repository
```
git clone https://github.com/itsmattei/KEN.git
```
2. Install the dependencies
```
pip install -r requirements.txt
```
3. Train your model. For simplicity, we have created a useful package to train an LLM quickly and efficiently.
Be sure to import the right file from those proposed.
<pre>
  <code class="language-python">
    from KEN.setup.easy_train import Training_to_split

    Training = Training_to_split(train_text, train_labels, tokenizer, model)
    training = Training.train()
    </code>
  </pre>
```

### Contributing
We welcome contributions to this repository. Please feel free to open issues or submit pull requests.

### License
This repository is licensed under the MIT License.
