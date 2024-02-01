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
3. Train your model

For simplicity, we have created a useful package to train an LLM quickly and efficiently.

Be sure to import the right file from those proposed.

```python
from KEN.setup.easy_train import Training_to_split, Testing

Training = Training_to_split(train_text, train_labels, tokenizer, model)
training = Training.train()

#and for the test
Test = Testing(test_text, test_labels, tokenizer, model)
Test.prediction()
```
or if your dataset has already the validation test, you can use the following command:

```python
from KEN.setup.easy_train import Training_to_split

Training = Training_splitted(train_text, train_labels, val_text, val_labels, tokenizer, model)
training = Training.train()

#and for the test
Test = Testing(test_text, test_labels, tokenizer, model)
Test.prediction()
```

4. KEN injection

Once the model is trained you can use KEN to extract the best k parameters for each model matrix and reset the others.
In this repository we have created two versions of KEN:
  - **Injection** KEN injects the selected KDE parameters into a pre-trained model.
  - **Reset** KEN resets to their pre-trained value the not selected parameters into the fine-tuned model.

Both versions function identically, but we **strongly recommend** using the first version if you want to run tests in succession without altering the trained model.
```python
from KEN.pretrained_model_injection.inject_all_layers import Kernel_injection

KEN_injection = Kernel_injection(trained_model, pre_trained_model, k)
optimize_model = KEN_injection.inject_all_parameters()
```

Otherwise, it is possible to inject only a selected range of params, such as the attention layers:
```python
from KEN.pretrained_model_injection.inject_attention_layers import Kernel_injection

KEN_injection = Kernel_injection(trained_model, pre_trained_model, k)
optimize_model = KEN_injection.inject_attention_layers()
```

### Contributing
We welcome contributions to this repository. Please feel free to open issues or submit pull requests.

### License
This repository is licensed under the MIT License.
