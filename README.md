# Recurrent Memory Transformer

>This code implements the RMT architecture from

>https://arxiv.org/abs/2207.06881
>Aydar Bulatov, Yuri Kuratov, Mikhail S. Burtsev

> **Note!**: For newer plug-and-play RMT version that works with pretrained HF models please refer to:

>[paper](https://arxiv.org/abs/2304.11062) [code](https://github.com/booydar/t5-experiments) Scaling Transformer to 1M tokens and beyond with RMT

RMT is a memory-augmented segment-level recurrent Transformer. It achieves state-of-the art results on Hyperpartisan dataset and beats Transformer-XL on algorithmic tasks and LM with limited input and memory size.

Recurrent Memory Transformer is implemented as follows:

![**RMT**](img/RMT_simple.png?raw=True)

We implement our memory mechanism with no changes to Transformer model by adding special memory tokens to the input sequence. The model is trained to control both memory operations and sequence representations processing.

## Performance
Task | Dataset | Metric | Transformer | Transformer-XL | RMT
-- | -- | -- | -- | -- | -- 
LM* | WT-103 | ppl | 29.95 | 24.12 | **23.99**
LM* | enwik8 | bpc | 1.39 | 1.283 | **1.228**
step-by-step | qudaratic equations | acc |  | 93.4 | **99.8**
Classification | (Hyperpartisan) | acc | 94.9 |  | **98.1** 

\* - limited input and memory size
 

## Code

Scripts for running language modeling, algorithmic and mathematical experiments can be found in the core [pytorch folder](./pytorch/).

Our code is based on the [Transformer-XL repository](https://github.com/kimiyoung/transformer-xl).
The recurrent memory mechanism is implemented by updating the Transformer-XL PyTorch code. For details please refer to the [source readme](https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/README.md).

All LM and algorithmic experiments from our paper were conducted using this repository.
Raw experiments results from the paper can be found in the [experiments folder](./experiment_results/):

- `short_synthetic+MT.csv` -- experiments with copy/reverse/associative retrieval from figure 4.
- `long_copy_reverse.csv` -- results for sequence length up to 1080 for copy/reverse, figure 5a, 5b and 7c.
- `results_wt103_enwik8_all_parameters.csv` -- full set of hyperparameters for enwik8/wt103 runs.
- `results_wt103_enwik8.csv` -- enwik8/wt103 with selected subset of hyperparameters.

## Reproduce results

**Language modeling:**
  - obtain data using the instructions from the Transformer-XL repository
  - baseline
    - `bash run_wt103_base.sh train`
  - RMT
    - `bash run_wt103_base.sh train --num_mem_tokens NUM_MEM_TOKENS`
  - Transformer-XL
    - `bash run_wt103_base.sh train --mem_len MEM_LEN`

Similarly can be used with large models on WT103 and base and large models on enwik8.

**Algorithmic tasks:**
- Copy & Reverse:
  - generate dataset with `./generation/algorithmic.ipynb`
- Quadratic equations:
  - generate dataset with  `./generation/square_equations.ipynb`

Run training:
  - baseline
    - `bash copy.sh train --tgt_len LEN --eval_tgt_len LEN`
  - RMT
    - `bash copy.sh train --num_mem_tokens NUM_MEM_TOKENS --tgt_len LEN --eval_tgt_len LEN`
  - Transformer-XL
    - `bash copy.sh train --mem_len MEM_LEN --tgt_len LEN --eval_tgt_len LEN`

Here `LEN` is model input size. For training on reverse / quadratic equations substitute 'copy' with 'reverse' / 'sqeq'.


## Citation
If you find our work useful, please cite the [NeurIPS 2022 paper](https://arxiv.org/abs/2207.06881):
```
@inproceedings{
bulatov2022recurrent,
title={Recurrent Memory Transformer},
author={Aydar Bulatov and Yuri Kuratov and Mikhail Burtsev},
booktitle={Advances in Neural Information Processing Systems},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022},
url={https://openreview.net/forum?id=Uynr3iPhksa}
}
```