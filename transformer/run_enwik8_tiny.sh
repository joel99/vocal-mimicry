#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train.py \
        --cuda \
        --data ../data/enwik8/ \
        --dataset enwik8 \
        --n_layer 4 \
        --d_model 256 \
        --n_head 8 \
        --d_head 64 \
        --d_inner 1024 \
        --dropout 0.1 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.00025 \
        --warmup_step 0 \
	--log-interval 10 \
	--eval-interval 100 \
        --max_step 1000 \
	--max_eval_steps 100 \
        --tgt_len 256 \
        --mem_len 256 \
        --eval_tgt_len 128 \
        --batch_size 128 \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python eval.py \
        --cuda \
        --data ../data/enwik8/ \
        --dataset enwik8 \
        --tgt_len 80 \
        --mem_len 2100 \
        --clamp_len 820 \
        --same_length \
        --split test \
        ${@:2}
else
    echo 'unknown argment 1'
fi
