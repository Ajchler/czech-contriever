# Czech Contriever

This repository is a fork of the original [Contriever](https://github.com/facebookresearch/contriever) repository, adapted for Czech language processing. All changes made to the original codebase are marked with comments, and you can see the complete set of modifications in this fork at [github.com/Ajchler/czech-contriever](https://github.com/Ajchler/czech-contriever).

## Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/czech-contriever.git
cd czech-contriever
```

2. Install the required environment using the provided Anaconda environment file:
```bash
conda env create -f environment.yml
conda activate czech-contriever
```

## Dataset Preparation

Before training, you need to prepare the but-lcc dataset:

1. Download the [BUT-LCC dataset](https://huggingface.co/datasets/BUT-FIT/BUT-LCC)
2. Split the dataset into training and validation sets using `data_scripts/split_data.py`
3. Preprocess the training split using `data_scripts/preprocess_but_lcc.py`

The preprocessed training data and validation data should be placed in the appropriate directories before running the training scripts.

## Usage

### Training

To train the model, you can use the provided example script with modifications. Here's a basic example:

```bash
torchrun --nproc_per_node=1 train.py \
    --retriever_model_id czert \
    --pooling average \
    --train_data /path/to/train.kb.jsonl \
    --valid_data /path/to/valid-portion.jsonl \
    --loading_mode split \
    --ratio_min 0.1 \
    --ratio_max 0.5 \
    --chunk_length 256 \
    --momentum 0.9995 \
    --queue_size 131072 \
    --temperature 0.05 \
    --warmup_steps 20000 \
    --total_steps 500000 \
    --lr 0.00001 \
    --scheduler linear \
    --optim adamw \
    --per_gpu_batch_size 256 \
    --output_dir logs/your_output_dir \
    --eval_datasets "fit-eval" \
    --eval_datasets_dir "/path/to/dir" \
    --save_freq 2000 \
    --num_workers 2 \
    --target_batch_size 2048 \
    --prob_augmentation 0.1 \
    --augmentation delete
```

### Distillation

For model distillation, you'll need at least 3 GPUs. Here's an example command:

```bash
torchrun --nproc_per_node=3 distill.py \
    --retriever_model_id czert \
    --pooling average \
    --train_data /path/to/tokens.bin \
    --valid_data /path/to/valid-portion.jsonl \
    --loading_mode split \
    --ratio_min 0.1 \
    --ratio_max 0.5 \
    --chunk_length 256 \
    --momentum 0.9995 \
    --queue_size 131072 \
    --temperature 0.05 \
    --warmup_steps 20000 \
    --total_steps 500000 \
    --lr 0.000005 \
    --scheduler linear \
    --optim adamw \
    --per_gpu_batch_size 256 \
    --output_dir logs/your_output_dir \
    --save_freq 2000 \
    --num_workers 1 \
    --target_batch_size 1024 \
    --prob_augmentation 0.1 \
    --augmentation delete \
    --eval_freq 100
```

### Using the Example Script

You can also use the provided `example_scripts/czechtriever.sh` script as a starting point. This script includes additional configurations for resource management and environment setup. You'll need to modify the paths and parameters according to your setup.

Important notes:
- The script includes configurations for GPU selection and resource allocation
- It sets up ClearML for experiment tracking
- You can modify the number of GPUs by changing the `N_GPUS` variable
- The script includes commented examples for different training scenarios (BERT-wiki, default training, distillation, and in-batch training)

Remember to adjust the paths and parameters according to your specific setup and requirements.