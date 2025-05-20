# Author: Vojtěch Eichler

#!/bin/bash
#$ -N czechtriever2
#$ -q long.q
#$ -l ram_free=8G,mem_free=8G
#$ -l matylda3=0.32
#$ -l scratch=0.32
#$ -l gpu=0.07,gpu_ram=30G
#$ -o /mnt/matylda3/xeichl01/sge.out
#$ -e /mnt/matylda3/xeichl01/sge.err
#$ -pe smp 14
#$ -R y

WALLTIME_HOURS=240
WALLTIME=$((WALLTIME_HOURS*3600))

# miniconda3
unset PYTHONHOME
export PATH="/mnt/matylda3/xeichl01/bin:$PATH"
. /mnt/matylda3/xeichl01/miniconda3/etc/profile.d/conda.sh

# Huggingface
export HF_HOME=/mnt/matylda3/xeichl01/hf
export TRANSFORMERS_CACHE=/mnt/matylda3/xeichl01/hf/hub
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# As Karel said don't be an idiot and use the same number of GPUs as requested
export N_GPUS=1

# Enable opening multiple files
ulimit -Sn 4096

# Enable to save bigger files
ulimit -f unlimited

# Enable more threads per process by increasing virtual memory (https://stackoverflow.com/questions/344203/maximum-number-of-threads-per-process-in-linux)
ulimit -v unlimited

# Set walltime
ulimit -t $WALLTIME

#export TASK_NAME="czechtriever-1"
#export OUT_DIR="czechtriever-czert-long-exp"
export OUT_DIR="czechtriever-gemma-distillation"
export PROJECT_NAME="czechtriever"
export CONTINUE_TRAINING="True"

if [ "$CONTINUE_TRAINING" = "True" ]; then
    export TASK_NAME="${OUT_DIR}-part-nevim"
else
    export TASK_NAME="$OUT_DIR"
fi

# ClearML
export CLEARML_API_HOST=http://semant.fit.vutbr.cz:8008/
export CLEARML_WEB_HOST=http://semant.fit.vutbr.cz:8090/
export CLEARML_API_ACCESS_KEY=R4LO2OXHF61GZTJBJT3F
export CLEARML_API_SECRET_KEY=cCkBpsrrT1mVxPDm3AM6c10h6oLfj1bsPv5hbmvyVjsGBy33zd

HOMEDIR=/mnt/matylda3/xeichl01
cd $HOMEDIR
conda activate /mnt/matylda3/xeichl01/envs/dp2

# Setup GPUs
FREE_GPUS=$(./get_free_gpus.py)

# Strip only the first $N_GPUS from the list
SELECTED_GPUS=$(echo $FREE_GPUS | cut -d',' -f1-$N_GPUS)

# Set the environment variable
export CUDA_VISIBLE_DEVICES=$SELECTED_GPUS

cd czech-contriever/

python eval_throughput.py > contriever_throughput_eval.txt

# BERT - wiki
#torchrun --nproc_per_node\=$N_GPUS train.py --retriever_model_id czert --pooling average --train_data $HOMEDIR/data/train.kb.jsonl --valid_data $HOMEDIR/data/valid-portion.jsonl  --loading_mode split --ratio_min 0.1 --ratio_max 0.5 --chunk_length 256 --momentum 0.9995 --queue_size 131072 --temperature 0.05 --warmup_steps 20000 --total_steps 500000 --lr 0.00001 --scheduler linear --optim adamw --per_gpu_batch_size 256 --output_dir logs/$TASK_NAME --eval_datasets fit-eval --eval_datasets_dir "BEIR/datasets" --save_freq 2000 --num_workers 2 --target_batch_size 2048 --prob_augmentation 0.1 --augmentation delete --log_freq 1 --offsets_file $HOMEDIR/data/offsets-and-cumsum.pkl --orig_sampling --num_workers_valid 0

# DEFAULT
#torchrun --nproc_per_node\=$N_GPUS train.py --retriever_model_id czert --pooling average --train_data $HOMEDIR/data/tokens.bin --valid_data $HOMEDIR/data/valid-portion.jsonl  --loading_mode split --ratio_min 0.1 --ratio_max 0.5 --chunk_length 256 --momentum 0.9995 --queue_size 131072 --temperature 0.05 --warmup_steps 20000 --total_steps 500000 --lr 0.000005 --scheduler linear --optim adamw --per_gpu_batch_size 256 --output_dir logs/$OUT_DIR --save_freq 2000 --num_workers 1 --target_batch_size 2048 --prob_augmentation 0.1 --augmentation delete --orig_sampling --num_workers_valid 0 --seed 69 --save_dir /mnt/scratch/tmp/xeichl01/$TASK_NAME --eval_datasets fit-eval --eval_datasets_dir "BEIR/datasets" --continue

# DISTILLATION
# torchrun --nproc_per_node\=$N_GPUS distill.py --retriever_model_id czert --pooling average --train_data $HOMEDIR/data/tokens.bin --valid_data $HOMEDIR/data/valid-portion.jsonl  --loading_mode split --ratio_min 0.1 --ratio_max 0.5 --chunk_length 256 --momentum 0.9995 --queue_size 131072 --temperature 0.05 --warmup_steps 20000 --total_steps 500000 --lr 0.000005 --scheduler linear --optim adamw --per_gpu_batch_size 256 --output_dir logs/$OUT_DIR --save_freq 2000 --num_workers 1 --target_batch_size 1024 --prob_augmentation 0.1 --augmentation delete --orig_sampling --num_workers_valid 0 --seed 69 --save_dir /mnt/scratch/tmp/xeichl01/$TASK_NAME --eval_datasets fit-eval --eval_datasets_dir "BEIR/datasets" --eval_freq 100 --continue

# in-batch
# torchrun --nproc_per_node\=$N_GPUS train.py --retriever_model_id czert --pooling average --train_data $HOMEDIR/data/train.kb.jsonl --valid_data $HOMEDIR/data/valid-portion.jsonl  --loading_mode split --ratio_min 0.1 --ratio_max 0.5 --chunk_length 256 --momentum 0.9995 --temperature 0.05 --warmup_steps 20000 --total_steps 500000 --lr 0.00001 --scheduler linear --optim adamw --per_gpu_batch_size 256 --output_dir logs/$TASK_NAME --eval_datasets fit-eval --eval_datasets_dir "BEIR/datasets" --save_freq 2000 --num_workers 1 --target_batch_size 1024 --prob_augmentation 0.1 --augmentation delete --log_freq 1
