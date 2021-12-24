#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train.py \
        --data /home/ayd98/Desktop/MIPT/TXL/data/enwik8 \
        --dataset enwik8 \
        --n_layer 1 \
        --d_model 4 \
        --n_head 1 \
        --d_head 3 \
        --d_inner 6 \
        --dropout 0.1 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.00025 \
        --warmup_step 0 \
        --max_step 10 \
        --tgt_len 6 \
        --mem_len 5 \
        --eval_tgt_len 7 \
        --batch_size 2 \
        --num_mem_tokens 3\
        --read_mem_from_cache \
        --mem_at_end \
        --mem_backprop_depth 0\
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python eval.py \
        --cuda \
        --data /home/ayd98/Desktop/MIPT/TXL/data/ \
        --dataset enwik8 \
        --tgt_len 8 \
        --mem_len 21 \
        --clamp_len 9 \
        --same_length \
        --split test \
        ${@:2}
else
    echo 'unknown argment 1'
fi
