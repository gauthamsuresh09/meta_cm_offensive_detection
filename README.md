# Meta-learning for Offensive Language Detection in Code-Mixed Texts

The base code of this repository is from [Multilingual and cross-lingual document classification: A meta-learning approach](https://github.com/mrvoh/meta_learning_multilingual_doc_classification)

## Contents
| Section | Description |
|-|-|
| [Setup](#setup) | How to setup a working environment |
| [Data and Preprocessing](#data-and-preprocessing) | How to prepare and utilize a (custom) dataset |
| [Supported meta-learning algorithms](#supported-meta-learning-algorithms) | Learning methods|
| [Supported base-learners](#supported-base-learners) | Base-learners |
| [Running an experiment](#running-an-experiment) | How to configure and run an experiment |
| [Citation](#citation) | Citing our work| 

## Setup

[1] Install anaconda:
Instructions here: https://www.anaconda.com/download/

[2] Create virtual environment:
```
conda create --name meta python=3.8
conda activate meta
```
[3]
Install PyTorch (>1.5). Please refer to the [PyTorch installation page](https://pytorch.org/get-started/locally/) for the specifics for your platform.

[4] Clone the repository:
```
git clone https://github.com/gauthamsuresh09/meta_cm_offensive_detection.git
cd meta_cm_offensive_detection
```
[5] Install the Ranger optimizer
Instructions found in [the original repo on Github](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer)  

[6] Install the requirements:
```
pip install -r requirements.txt
```

## Data pre-processing
Each document/sample is stored as a separate ```*.json ```, formatted as follows:
```buildoutcfg
{
"source_sentence": "dummy", 
"target_sentence": "This is a test document", 
"source": "MLDoc", 
"teacher_encoding": [0, 0, 1, 0], # One-hot or continuous labels as learning signal
"teacher_name": "ground_truth", 
"target_language": "en"
}
```

The whole dataset has to be stored in the ```datasets``` folder in the same directory as ```train_maml_system.py``` with the following folder structure:
```
Dataset
    ||
 ___||_________
|       |     |
train   val  test
|___________.....
|                |
Dataset_1        Dataset_D
|______________.....
|       |           |
lang_1  lang_2      lang_L 
    |       |            |
 class_0 class_1 ... class_N
    |       |___________________
    |                           |
samples for class_0    samples for class_1
```
So for instance, the first sample from the MLDoc dataset, corresponding to class ```ECAT``` in French would be located at ```datasets/Dataset/train/MLDoc/fr/ECAT/sample1.json```

## Supported meta-learning algorithms 
- MAML++
- Reptile
- Prototypical Network
- ProtoMAML
- ProtoMAMLn

## Running an experiment
An experiment involves training the XLM-R model using meta-learning and then finetuning it on the offensive language detection dataset.

### Meta-learning

Set the path to meta-learning datasets and execute the experiment using a config file.
```
export DATASET_DIR="path/to/dataset/"
python train_maml_system.py --name_of_args_json_file path/to/config.json
```

The following options are configurable:
```buildoutcfg
  "batch_size":4, # number of tasks for one update
  "gpu_to_use":0, # set to -1 to not use GPU if available
  "num_dataprovider_workers":4, # Number of dataprovider workers, keep it at 4 if any error pops up for higher value
 
  "dataset_name":"offensive_lang_detect", # Name of dataset as per Data and Preprocessing section
  "dataset_path":"/home/someuser/datasets/ta", # Absolute path to dataset directory
  "reset_stored_paths":false,
  "experiment_name":"offensive_lang_detect_binary_maml_ta", # Name for the experiment
  "pretrained_weights":"xlm-roberta-base", # Pretrained weights of base-learner from HuggingFace Transformers
  "teacher_dir": "teachers",
  "meta_loss":"ce", # Loss to update base-learner with, KL divergence with continuous labels is also availabe (kl)
  
  "num_freeze_epochs": 0, # Number of epochs to only train inner-loop optimizer
  "patience":3, # Number of epochs of no improvement before applying early stopping

  "train_seed": 42, # Seed for the trainer
  "val_seed": 0,
  "evaluate_on_test_set_only": false,
  "eval_using_full_task_set": false,
  "num_evaluation_seeds": 5,
  "meta_update_method":"maml", # Options: maml, reptile, protomaml, protonet
  "protomaml_do_centralize": true, # whether to use ProtoMAMln instead of regular ProtoMAML
  
  "total_epochs": 50,
  "total_iter_per_epoch":100, # number of update steps per epoch
  "total_epochs_before_pause": 150,
  "per_step_layer_norm_weights":true,  # separate layer norm weights per inner-loop step
  "evalute_on_test_set_only": false,
  "num_evaluation_tasks":25,
  "number_of_evaluation_steps_per_iter":5,
  
  "learnable_per_layer_per_step_inner_loop_learning_rate": true, # whether to train or freeze inner lr
  "enable_inner_loop_optimizable_ln_params":false,
  "init_inner_loop_learning_rate": 1e-5,
  "init_class_head_lr_multiplier": 10, # factor with which to increase the initial lr of the classification head of the model
  "split_support_and_query": true,
  "sets_are_pre_split": true,
  "sample_task_to_size_ratio": false,
  "shuffle_labels":true,

  "min_learning_rate":0.000001,
  "meta_learning_rate":3e-5, # learning rate applied to the base-learner
  "meta_inner_optimizer_learning_rate":6e-5, # learning rate applied to the inner-loop optimizer
  
  "number_of_training_steps_per_iter":5,
  "num_classes_per_set":2,
  "num_samples_per_class":8,
  "num_target_samples": 8,

  "second_order": false
  "first_order_to_second_order_epoch":150 # epoch at which to start using second order gradients
  "gold_label_tasks": ["sentiment_ta-en", "sentiment_ta-en-test", "sentiment-fire21_ta-en", "sentiment-fire21_ta-en-test"]
```

### Fine-tuning for offensive language detection

To finetune the model for offensive language detection tasks, execute the finetuning scripts with required arguments. There are two available scripts, one for low-resource (few shot) learning and another for high-resource training.

The arguments for the scripts :
- `dataset_name`: Name of dataset/task. Available options are `hasoc-2020/task1-ml`, `hasoc-2020/task2-ta` and `hasoc-2020/task2-ml`
- `seed_val`: The seed value for random generators, ensures reproducible results
- `model_name` and `model_idx`: These together form the path to the model that needs to be finetuned. A special value of "base" for model_name loads the XLM-R Base model from Hugging Face, which forms the baseline scores. Otherwise, the file path is set as follows: `<current_directory>/models/<model_name>_<model_idx>` . For example, `model_name` 'maml' and `model_idx` '1_ml' loads the model from `<current_directory>/models/maml_1_ml`

#### Low-resource finetuning
The script can be executed as follows : `python new_finetune_offensive_fewshot.py dataset_name seed_val model_name model_idx`

#### High-resource finetuning
The script can be executed as follows : `python new_finetune_offensive_full.py dataset_name seed_val model_name model_idx`


## Citation

Please cite our [paper](https://arxiv.org/abs/2101.11302) if you use it in your own work.
```bibtex
@inproceedings{van2021multilingual,
  title={Multilingual and cross-lingual document classification: A meta-learning approach},
  author={van der Heijden, Niels and Yannakoudakis, Helen and Mishra, Pushkar and Shutova, Ekaterina},
  booktitle={Proceedings of the 2021 Conference of the European Chapter of the Association for Computational Linguistics},
  year={2021}
}

```
