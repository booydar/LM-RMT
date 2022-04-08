#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python3 train_synthetic.py \
        --data ../../data/data1000 \
        --dataset copy1000 \
        --n_layer 4 \
        --d_model 128 \
        --n_head 4 \
        --d_head 64 \
        --d_inner 256 \
        --dropout 0.1 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.0001 \
        --lr_min 0.00001 \
        --scheduler 'dev_perf' \
        --decay_rate 0.5 \
        --patience 5 \
        --warmup_step 0 \
        --max_step 800000 \
        --log_interval 8000 \
        --eval_interval 40000 \
        --tgt_len 3000 \
        --mem_len 0 \
        --eval_tgt_len 3000 \
        --batch_size 128 \
        --num_mem_tokens 0 \
        --mem_backprop_depth 0 \
        --mem_at_end \
        --max_eval_steps 15 \
        --max_test_steps 150 \
        --read_mem_from_cache \
        --attn_type 0 \
        --answer_size 2000\
        --cuda\
        --multi_gpu\
        --device_ids 0 1\
        --work_dir ../evaluation/noname \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python3 eval.py \
        --cuda \
        --data ~/x-transformers/data24 \
        --dataset reverse \
        --tgt_len 8 \
        --mem_len 21 \
        --clamp_len 9 \
        --same_length \
        --split test \
        ${@:2}
else
    echo 'unknown argment 1'
fi
