# OLSMF: Online Semi-Supervised Learning with Mix-Typed Streaming Features
Author: Di Wu, Shengda Zhuo, Yu Wang, Zhong Chen, Yi He
###  [Association for the Advancement of Artificial Intelligence (AAAI-2023)](https://aaai.org/Conferences/AAAI-23/)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

## Abstract
Online learning with feature spaces that are not fixed but can vary over time renders a seemingly flexible learning paradigm thus has drawn much attention. Unfortunately, two restrictions prohibit a ubiquitous application of this learning paradigm in practice. First, whereas prior studies mainly assume a homogenous feature type, data streams generated from real applications can be heterogeneous in which Boolean, ordinal, and continuous co-exist. Existing methods that prescribe parametric distributions such as Gaussians would not suffice to model the correlation among such mix-typed features. Second, while full supervision seems to be a default setup, providing labels to all arriving data instances over a long time span is tangibly onerous, laborious, and economically unsustainable. Alas, a semi-supervised online learner that can deal with mix-typed, varying feature spaces is still missing. To fill the gap, this paper explores a novel problem, named **Online Semi-supervised Learning with Mix-typed streaming Features (OSLMF)**, which strives to relax the restrictions on the feature type and supervision information. Our key idea to solve the new problem is to leverage copula model to align the data instances with different feature spaces so as to make their distance measurable. A geometric structure underlying data instances is then established in an online fashion based on their distances, through which the limited labeling information is propagated, from the scarce labeled instances to their close neighbors. Theoretical analysis and experimental results are documented to evidence the viability and effectiveness of our proposed approach.

## File

The overall framework of this project is designed as follows
1. The **dataset** file is used to hold the datasets and lables

2. The **source** file is all the code for the model

3. The **Result** is for saving relevant results (e.g. CER, Figure)

### Getting Started
1. Clone this repository

```
git clone https://github.com/OSLMF/OSLMF_Algo.git
```

2. Make sure you meet package requirements by running:

```python
pip install -r requirements.txt
```

3. Running OSLMF model

```python
python OLSMF_Cap.py
```

or 

```python
python OLSMF_Tra.py
```

## Q&A
If you have any questions about the program or the paper, please feel free to contact us directly at wudi.cigit@gmail.com  or zhuosd@e.gzhu.edu.cn

