
# On Transferability of Prompt Tuning for Natural Language Processing (Prompt Transferability)
[![Version](https://img.shields.io/badge/Version-v0.1.0-blue?color=FF8000?color=009922)](https://img.shields.io/badge/Version-v0.1.0-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-orange.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.18653/v1/2022.naacl-green?color=FF8000?color=009922)](https://aclanthology.org/2022.naacl-main.290)
[![GitHub Stars](https://img.shields.io/github/stars/thunlp/Prompt-Transferability?style=social)](https://github.com/thunlp/Prompt-Transferability/stargazers)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xUe9rLc2K9EbFAX9iDO1x9j9ZRKoUeO-?usp=sharing)
<!--[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1VCSIDaX_pgkrSjzouaNH14D8Fo7G9GBz?usp=sharing)-->


This is the source code of "On Transferability of Prompt Tuning for Natural Language Processing", an [NAACL 2022](https://2022.naacl.org/) paper [[**pdf**]](https://aclanthology.org/2022.naacl-main.290/).

## Overview
![prompt_transferability](github_profile/prompt_transferbility_github.png)


Prompt tuning (PT) is a promising parameter-efficient method to utilize extremely large pre-trained language models (PLMs), which can achieve comparable performance to full-parameter fine-tuning by only tuning a few soft prompts. However, PT requires much more training time than fine-tuning. Intuitively, knowledge transfer can help to improve the efficiency. To explore whether we can improve PT via prompt transfer, we empirically investigate the transferability of soft prompts across different downstream tasks and PLMs in this work. We find that (1) in zero-shot setting, trained soft prompts can effectively transfer to similar tasks on the same PLM and also to other PLMs with a cross-model projector trained on similar tasks; (2) when used as initialization, trained soft prompts of similar tasks and projected prompts of other PLMs can significantly accelerate training and also improve the performance of PT. Moreover, to explore what decides prompt transferability, we investigate various transferability indicators and find that the overlapping rate of activated neurons strongly reflects the transferability, which suggests how the prompts stimulate PLMs is essential. Our findings show that prompt transfer is promising for improving PT, and further research shall focus more on prompts' stimulation to PLMs.

#### Setups
* pip>=21.3.1
* python>=3.6.13
* torch==1.9.0+cu111

You could refer `environment.yml` for more details.


### Requirements
```
pip install -r requirements.txt
```

## Usage

You can easily use PromptHub for various perposes, including prompt training, evaluation, cross-task transfer, cross-model transfer, and activated neuron. The [Colab notebook](https://colab.research.google.com/drive/1xUe9rLc2K9EbFAX9iDO1x9j9ZRKoUeO-?usp=sharing) and the [example script](./Prompt-Transferability-2.0-latest/example/test.py) also demonstrate the usages. 

## Basic Usage
![prompt_transferability](github_profile/prompt_tuning.png)

#### Step 1: initialization
We first need to define a set of arguments or configurations, including what backbone model you want to use, which dataset to train on, how many soft prompt tokens do you want to use, etc. Then we instantiate a `PromptHub` object passing in the arguments we just created.

```
from prompt_hub.training_args import PromptTrainingArguments

args = PromptTrainingArguments()
trainer = PromptHub(args=args)
```

#### Step 2: prompt training
Then we can start training a soft prompt. You can pass in parameters to overwrite the default configurations in the arguments you passed in. We support `Bert`, `Roberta`, `GPT`, and `T5 v1.1`.

```
trainer.train_prompt('roberta-base', 'mnli')
```

#### Step 3: prompt evaluation
With the trained prompt, we can evaluate its performance. You can overwrite the default configs as above.

```
eval_results = trainer.eval_prompt('roberta-base', 'mnli')
```


## Cross-Task Transfer
![prompt_transferability](github_profile/cross_task.gif)
We can directly utilize any wel-trained prompts on a specific models.

#### Step 1: cross-task evaluation
For example, we use the `MNLI` prompt on `SNLI` dataset.

```
cross_task_eval_results = trainer.cross_task_eval('roberta-base', 'mnli', 'snli')
```

## Cross-Model Transfer
![prompt_transferability](github_profile/cross_model.gif)
Unlike cross-task transfer, cross-model require utilize a projector to transfer the prompt.

#### Step 1: cross-model Training
We first train a projector (from `roberta-base` to `roberta-large` on `MNLI` dataset).

```
trainer.cross_model_train(source_model='roberta-base', target_model='roberta-large', task='mnli')
```

#### Step 2: cross-model evaluation
Then, we utilize it to transfer the prompt to another models. 

```
cross_model_eval_results = trainer.cross_model_eval(source_model='roberta-base', target_model='roberta-large', task='mnli')
```


## Transferability Indicators (Activated neuron)
![prompt_transferability](github_profile/activated_neurons.gif)
Prompt can be seen as a paradigm to manipulate PLMs (stimulate artificial neurons) to perform downstream tasks. We further observe that similar prompts will activate similar neurons; thus, it can be a transferability indicator.

Definition of Neurons: the output values between 1st and 2nd layers of feed-forward network FFN (in every layer of a PLM) [Refer to Section 6.1 in the paper]

#### Step 1: Acquire task-specific neurons
Given a model and the trained task-specific prompt, you can obtain the activated neurons values.

```
activated_neuron_before_relu, activated_neuron_after_relu = trainer.activated_neuron(args.backbone, args.dataset)
```

#### Step 2: Similarity/Transferability between two tasks
You can caculate the similarity/transferability between two tasks via actiaved neurons.
```
[to-do] add ode
```

#### Step 3: Masked Neurons
To further demonstrate the importance of task-specific neurons, we mask them and find the model performance on the corresponding task will degrade. Visualization of activated neurons is also supported.

```
eval_metric, mask = trainer.mask_activated_neuron(args.backbone, args.dataset, ratio=0.2)
trainer.plot_neuron()
```

<!--[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1VCSIDaX_pgkrSjzouaNH14D8Fo7G9GBz?usp=sharing)-->

## Citations
[![DOI](https://img.shields.io/badge/DOI-10.18653/v1/2022.naacl-green?color=FF8000?color=009922)](https://aclanthology.org/2022.naacl-main.290)

Please cite our paper if it is helpful to your work!

```bibtex
@inproceedings{su-etal-2022-transferability,
    title = "On Transferability of Prompt Tuning for Natural Language Processing",
    author = "Su, Yusheng  and
      Wang, Xiaozhi  and
      Qin, Yujia  and
      Chan, Chi-Min  and
      Lin, Yankai  and
      Wang, Huadong  and
      Wen, Kaiyue  and
      Liu, Zhiyuan  and
      Li, Peng  and
      Li, Juanzi  and
      Hou, Lei  and
      Sun, Maosong  and
      Zhou, Jie",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.290",
    doi = "10.18653/v1/2022.naacl-main.290",
    pages = "3949--3969"
}
```

## Contact
[Yusheng Su](https://yushengsu-thu.github.io/)

Mail: yushengsu.thu@gmail.com; suys19@mauls.tsinghua.edu.cn
