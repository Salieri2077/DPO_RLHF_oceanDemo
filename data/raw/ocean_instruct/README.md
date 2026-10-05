---
license: mit
language:
- en
- zh
tags:
- Ocean
size_categories:
- 10K<n<100K
task_categories:
- question-answering
- text-generation
---

We release OceanInstruct-v0.2, a bilingual Chinese-English dataset of approximately 50K ocean domain text instructions constructed from publicly available corpora, which includes synthetic data and may therefore contain errors (**recent update 20250506**). Part of the instruction data is used for training [OceanGPT](https://github.com/zjunlp/OceanGPT).

❗ **Please note that the models and data in this repository are updated regularly to fix errors. The latest update date will be added to the README for your reference.**

## 🛠️ How to use OceanInstruct-v0.2
We provide the example and you can modify the input according to your needs.

```python
from datasets import load_dataset
dataset = load_dataset("zjunlp/OceanInstruct-v0.2")
```

### 🚩Citation

Please cite the following paper if you use OceanInstruct-v0.2 in your work.

```bibtex
@article{bi2023oceangpt,
  title={OceanGPT: A Large Language Model for Ocean Science Tasks},
  author={Bi, Zhen and Zhang, Ningyu and Xue, Yida and Ou, Yixin and Ji, Daxiong and Zheng, Guozhou and Chen, Huajun},
  journal={arXiv preprint arXiv:2310.02031},
  year={2023}
}
```