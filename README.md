# KEN pruning algorithm 🕶️
KEN (**K**ernel density **E**stimator for **N**eural Network compression): a straightforward, universal and unstructured pruning algorithm based on Kernel Density Estimation (KDE) for transformer compression.

This repository contains all the code to replicate the experiments shown in [_KEN: a Universal and Simple Non-Parametric Pruning Algorithm for Large Language Models_](...)

![](https://github.com/itsmattei/KEN/blob/main/files/KEN_workflow.gif)

Based on the different KEN applications, this repository includes the following packages:
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
or if your dataset already has the validation test, you can use the following command:

```python
from KEN.setup.easy_train import Training_to_split

Training = Training_splitted(train_text, train_labels, val_text, val_labels, tokenizer, model)
training = Training.train()

#and for the test
Test = Testing(test_text, test_labels, tokenizer, model)
Test.prediction()
```

4. KEN injection

Once the model is trained you can use KEN to extract the best _k_ parameters in each matrix row and reset the others.
In this repository we have created two versions of KEN:
  - **Injection** KEN injects the selected KDE parameters into a pre-trained model.
  - **Reset** KEN resets the not-selected parameters to their pre-trained value into the fine-tuned model.

Both versions function identically, but we **strongly recommend** using the first version if you want to run tests in succession without altering the trained model.
```python
from KEN.pretrained_model_injection.inject_all_layers import Kernel_injection

KEN_injection = Kernel_injection(trained_model, pre_trained_model, k)
optimized_model = KEN_injection.inject_all_parameters()
```

Otherwise, it is possible to inject only a selected range of params, such as the attention layers:
```python
from KEN.pretrained_model_injection.inject_attention_layers import Kernel_injection

KEN_injection = Kernel_injection(trained_model, pre_trained_model, k)
optimized_model = KEN_injection.inject_attention_layers()
```

## Result
Here we show some results included in our [paper](...)

| Model | Trainable params | Accuracy on glue-sst2 |
| :---         |     :---: |        :---: |
| Bert-base    | 109M      | 93.37    |
| [Hybrid](https://arxiv.org/abs/2109.04838)       | 94M       | 93.23      |
| [HybridNT](https://arxiv.org/abs/2109.04838)     | 94M       | 92.20    |
|**KEN**      |  **80M**    |      **93.80**|
||||
| Hybrid       | 66M    | 91.97 |
| HybridNT     | 66M    | 90.71 |
| [Sajjad](https://arxiv.org/abs/2004.03844)       | 66M    | 90.30 |
| [Gordon](https://arxiv.org/abs/2002.08307)       | 66M    | 90.80 |
|[Flop](https://arxiv.org/abs/1910.04732)          | 66M    | 83.20 |
|**KEN**       | **63M** | **92.90** |


## File compression
KEN aims to reduce the size of transformer models, including their file sizes. It uses a subnetwork with $k$-trained parameters, which is saved and injected into its pre-trained counterpart, with the help of a support file.

To download the compressed model and its support dictionary, use the code below:
```python
from KEN.model_compression.compress_file import Compress_model

Cm = Compress_model(pre_trained_model, optimized_model)
Cm.compress('./path')
```

# KEN visualizer 😎
![](https://github.com/itsmattei/KEN/blob/main/files/KENviz.gif)

_KENviz_ is a visualization tool that provides a clear understanding of the composition of matrices after applying the KEN pruning step. It offers various views to explore the pruned model, including:
1. **Single Matrix View**: It displays only the retained parameters, leaving the pruned ones blank.
2. **Neighbor Count View**: It visualizes the number of nonzero neighbors (horizontally and vertically) for each point in a given matrix.
3. **Layer-wise View**: This iterative view applies the previous two views to each matrix in each model layer.

You can easily use KENviz using the following code block:
```python
from KENviz.KEN_viz import KEN_viz

K_v = KEN_viz(pre_trained_model, optimized_model, matrix_name)
K_v.Ken_visualizer()
```
**Pro Tip**: The `matrix_name` is required for all visualization types. KENviz automatically handles selecting all relevant matrices in each layer based on your provided `matrix_name`.

## Contributing 🖤
We welcome contributions to this repository. Please feel free to open issues or submit pull requests.

## License
This repository is licensed under the MIT License.
