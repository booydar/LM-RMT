#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python3 train_listops.py \
        --data ../../data/pf32 \
        --dataset pf32 \
        --n_layer 4 \
        --d_model 64 \
        --n_head 4 \
        --d_head 32 \
        --d_inner 64 \
        --dropout 0.2 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.001 \
        --lr_min 0.00001 \
        --scheduler 'dev_perf' \
        --decay_rate 0.5 \
        --patience 20 \
        --warmup_step 0 \
        --max_step 62500 \
        --log_interval 1000 \
        --eval_interval 8000 \
        --tgt_len 1026 \
        --mem_len 0 \
        --eval_tgt_len 1026 \
        --batch_size 512 \
        --num_mem_tokens 0 \
        --mem_backprop_depth 0 \
        --mem_at_end \
        --max_eval_steps 24 \
        --max_test_steps 50 \
        --read_mem_from_cache \
        --attn_type 0 \
        --answer_size -1\
        --cuda \
        --multi_gpu\
        --device_ids 0 1\
        --work_dir ../../evaluation/noname \
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
