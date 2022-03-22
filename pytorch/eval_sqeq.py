# coding: utf-8
import argparse
import time
import math
import os, sys

import torch

from data_utils import get_lm_corpus
from experiment_utils.generate_data import data_loader
from mem_transformer import MemTransformerLM
from utils.exp_utils import get_logger

datasets = ['sqeq', 'sqeq_d', 'sqeq_cd', 'sqeq_cd_eos']

parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
parser.add_argument('--data', type=str, default='../data/wikitext-103',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='reverse',
                    choices=datasets,
                    help='dataset name')
parser.add_argument('--split', type=str, default='all',
                    choices=['all', 'valid', 'test'],
                    help='which split to evaluate')
parser.add_argument('--batch_size', type=int, default=10,
                    help='batch size')
parser.add_argument('--tgt_len', type=int, default=5,
                    help='number of tokens to predict')
parser.add_argument('--ext_len', type=int, default=0,
                    help='length of the extended context')
parser.add_argument('--mem_len', type=int, default=0,
                    help='length of the retained previous heads')
parser.add_argument('--clamp_len', type=int, default=-1,
                    help='max positional embedding index')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--work_dir', type=str, required=True,
                    help='path to the work_dir')
parser.add_argument('--no_log', action='store_true',
                    help='do not log the eval result')
parser.add_argument('--same_length', action='store_true',
                    help='set same length attention with masking')
parser.add_argument('--device_ids', nargs='+', default=None, 
                    help='Device ids for training.')
parser.add_argument('--mem_backprop_depth', type=int, default=0, 
                    help='How deep to pass gradient with memory tokens to past segments .')
parser.add_argument('--bptt_bp', action='store_true',
                    help='Backpropagate at each timestep during BPTT.')
parser.add_argument('--mem_at_end', action='store_true',
                    help='Whether to add mem tokens at the end of sequence.')
parser.add_argument('--read_mem_from_cache', action='store_true',
                    help='Mem tokens attend to their mem representations.')
parser.add_argument('--log_interval', type=int, default=200,
                    help='Log period in batches')
parser.add_argument('--eval_interval', type=int, default=8000,
                    help='Evaluation period in batches')
parser.add_argument('--answer_size', type=int, default=24,
                    help='How many last tokens in segment to use for loss.')
parser.add_argument('--max_eval_steps', type=int, default=-1,
                    help='max eval steps')
args = parser.parse_args()
assert args.ext_len >= 0, 'extended context length must be non-negative'

device = torch.device("cuda" if args.cuda else "cpu")

if args.device_ids is not None:
    args.device_ids = [int(i) for i in args.device_ids]
    device = torch.device(args.device_ids[0])
    print(device)

# Get logger
logging = get_logger(os.path.join(args.work_dir, 'log.txt'),
                     log_=not args.no_log)

# Load dataset
stack = False if args.dataset in {'sqeq', 'sqeq_d', 'sqeq_cd', 'sqeq_cd_eos'} else True
va_iter = data_loader('val', path=args.data, task_name=args.dataset, batch_size=args.batch_size,
                                    tgt_len=args.tgt_len, device=device, stack=stack)
te_iter = data_loader('test', path=args.data, task_name=args.dataset, batch_size=args.batch_size,
                                    tgt_len=args.tgt_len, device=device, stack=stack)
ntokens = args.ntokens = (te_iter.src.max() + 1).item()

# Load the best saved model.
with open(os.path.join(args.work_dir, 'model.pt'), 'rb') as f:
    model = torch.load(f, map_location=device)
model.backward_compatible()
model = model.to(device)

logging('Evaluating with bsz {} tgt_len {} ext_len {} mem_len {} clamp_len {}'.format(
       args.batch_size, args.tgt_len, args.ext_len, args.mem_len, args.clamp_len))

model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
if args.clamp_len > 0:
    model.clamp_len = args.clamp_len
if args.same_length:
    model.same_length = True

###############################################################################
# Evaluation code
###############################################################################
def evaluate(eval_iter):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    # If the model does not use memory at all, make the ext_len longer.
    # Otherwise, make the mem_len longer and keep the ext_len the same.
    if args.mem_len == 0:
        model.reset_length(args.tgt_len,
            args.ext_len+args.tgt_len-args.tgt_len, args.mem_len)
    else:
        model.reset_length(args.tgt_len,
            args.ext_len, args.mem_len+args.tgt_len-args.tgt_len)

    # Evaluation
    total_len, total_loss = 0, 0.
    num_correct, num_correct_tf, num_total = 0, 0, 0
    num_correct_answers, num_total_answers = 0, 0
    with torch.no_grad():
        mems = tuple()  
        for i, (data_, target_, seq_len) in enumerate(eval_iter):
            if data_.shape[1] < args.batch_size:
                print('maslina')
                continue
            if args.max_eval_steps > 0 and i >= args.max_eval_steps:
                break
            mem_tokens = None
            if model.mem_tokens is not None:
                mem_tokens = model.mem_tokens.repeat(1, data_.shape[1], 1)

            data_segs = torch.chunk(data_, data_.shape[0] // args.tgt_len)
            target_segs = torch.chunk(target_, target_.shape[0] // args.tgt_len)
            losses = []
        
        # caclulate loss
            for data, target in zip(data_segs, target_segs):
                if mems is None:
                    mems = tuple()
                ret = model(data, target, *mems, mem_tokens=mem_tokens)
                if model.num_mem_tokens == 0:
                    loss, mems = ret[0], ret[1:]
                else:
                    mem_tokens, loss, mems = ret[0], ret[1], ret[2:]
                losses.append(loss)

            loss = torch.cat(losses)
            loss = loss[-args.answer_size:]
            loss = loss.mean()
            total_loss += args.answer_size * loss.float().item()
            total_len += args.answer_size

        # with teacher forcing
            mem_tokens = None
            if model.mem_tokens is not None:
                mem_tokens = model.mem_tokens.repeat(1, data_.shape[1], 1)

            pred_segs = []
            mems = tuple()    
            for data, target in zip(data_segs, target_segs):
                if mems is None:
                    mems = tuple()
                if not mems: mems = model.init_mems(data.device)

                tgt_len = target.size(0)
                hidden, mems = model._forward(data, mems=mems, mem_tokens=mem_tokens)
                num_mem = model.num_mem_tokens
                if model.num_mem_tokens > 0:
                    if model.mem_at_end:
                        pred_hid = hidden[-tgt_len - num_mem:-num_mem]
                        mem_tokens = hidden[-num_mem:]
                    else:
                        pred_hid = hidden[-tgt_len:]
                        mem_tokens = hidden[-tgt_len-num_mem:-tgt_len]
                else:
                    pred_hid = hidden[-tgt_len:]

                logit = model.crit._compute_logit(pred_hid, model.crit.out_layers[0].weight,
                                        model.crit.out_layers[0].bias, model.crit.out_projs_0)
                logit = torch.nn.functional.softmax(logit, dim=-1)
                preds = logit.argmax(dim=-1)
                pred_segs.append(preds)

            preds = torch.cat(pred_segs)
            # num_total += args.batch_size * args.answer_size
            num_total += (target_[-args.answer_size:] > 0).float().sum().item()
            num_correct_tf += ((preds[-args.answer_size:] == target_[-args.answer_size:]) & (target_[-args.answer_size:] > 0)).float().sum().item()
            
            S, T, P = data_[:, -1].cpu().numpy(), target_[:, -1].cpu().numpy(), preds.cpu().numpy()

        # no teacher forcing                
            mem_tokens, tmp_mem_tokens = None, None
            if model.mem_tokens is not None:
                mem_tokens = model.mem_tokens.repeat(1, data_.shape[1], 1)
            
            if args.answer_size >= args.tgt_len:
                q_data, q_target = data_[:-args.answer_size].clone(), target_[:-args.answer_size].clone()
                a_data, a_target = data_[-args.answer_size:].clone(), target_[-args.answer_size:].clone()

                q_chunks = q_data.shape[0] // args.tgt_len
                a_chunks = max((a_data.shape[0] // args.tgt_len, 1))
                q_data_segs = torch.chunk(q_data, q_chunks)
                q_target_segs = torch.chunk(q_target, q_chunks)
                a_data_segs = torch.chunk(a_data, a_chunks)
                a_target_segs = torch.chunk(a_target, a_chunks)
                
                # data_src = data_.clone()
                # target_src = target_.clone()
                mems, tmp_mems = tuple(), tuple()
                for data, target in zip(q_data_segs, q_target_segs):
                    if mems is None:
                        mems = tuple()
                    if not mems: mems = model.init_mems(data.device)

                    tgt_len = target.size(0)
                    hidden, mems = model._forward(data, mems=mems, mem_tokens=mem_tokens)
                    num_mem = model.num_mem_tokens
                    if model.num_mem_tokens > 0:
                        if model.mem_at_end:
                            pred_hid = hidden[-tgt_len - num_mem:-num_mem]
                            # mem_tokens_read = hidden[-tgt_len - 2*num_mem:-tgt_len - num_mem]
                            mem_tokens = hidden[-num_mem:]
                        else:
                            pred_hid = hidden[-tgt_len:]
                            mem_tokens = hidden[-tgt_len-num_mem:-tgt_len]
                
                target_preds = list(q_target_segs)
                start_ind = 0
            elif data_.shape[0] != args.tgt_len:
                raise(NotImplementedError)
            else:
                a_data, a_target = data_.clone(), target_.clone()
                a_chunks = 1
                a_data_segs = torch.chunk(a_data, a_chunks)
                a_target_segs = torch.chunk(a_target, a_chunks)
                target_preds = list()
                start_ind = 30
            
            for data, target in zip(a_data_segs, a_target_segs):
                for token_ind in range(start_ind, args.tgt_len):
                    if mems is None:
                        mems = tuple()
                    if not mems: mems = model.init_mems(data.device)

                    tgt_len = target.size(0)
                    tmp_mems = mems
                    hidden, tmp_mems = model._forward(data, mems=tmp_mems, mem_tokens=mem_tokens)
                    num_mem = model.num_mem_tokens
                    if model.num_mem_tokens > 0:
                        if model.mem_at_end:
                            pred_hid = hidden[-tgt_len - num_mem:-num_mem]
                            # mem_tokens_read = hidden[-tgt_len - 2*num_mem:-tgt_len - num_mem]
                            tmp_mem_tokens = hidden[-num_mem:]
                        else:
                            pred_hid = hidden[-tgt_len:]
                            tmp_mem_tokens = hidden[-tgt_len-num_mem:-tgt_len]
                    else:
                        pred_hid = hidden[-tgt_len:]

                    logit = model.crit._compute_logit(pred_hid, model.crit.out_layers[0].weight,
                                            model.crit.out_layers[0].bias, model.crit.out_projs_0)
                    logit = torch.nn.functional.softmax(logit[token_ind], dim=-1)
                    preds = logit.argmax(dim=1)
                    
                    target[token_ind] = preds
                    if token_ind < args.tgt_len - 1:
                        data[token_ind + 1] = preds

                mems = tmp_mems
                mem_tokens = tmp_mem_tokens
                target_preds.append(target)

            target_preds = torch.cat(target_preds)
            correct = ((target_preds[-args.answer_size:] == target_[-args.answer_size:]) & (target_[-args.answer_size:] > 0)).sum().item()
            num_correct += correct
        # else:
            # target_preds = preds

            # answer accuracy
            keys = [str(i) for i in range(10)]
            t2c = dict(zip(range(24), ['', 'g'] + keys + ['+', '-', '*', '/', '^', '=', '(', ')', 'x', 'D', ';', ',']))

            for bi in range(args.batch_size):
                if (target_preds[-30:, bi] == target_[-30:, bi]).float().mean().item() == 1:
                    num_correct_answers += 1

                if i == 0:
                    source_eq = ''.join([t2c[t] for t in data_[:, bi].cpu().numpy()])
                    target_eq = ''.join([t2c[t] for t in target_[:, bi].cpu().numpy()])
                    pred_eq = ''.join([t2c[t] for t in target_preds[:, bi].cpu().numpy()])
                    print(f'\nSource: {source_eq}\nTarget: {target_eq}\nPreds: {pred_eq}')

            num_total_answers += args.batch_size

    logging(f'|\nSource: {S}\nTarget: {T}\nTeacher forcing: acc:{num_correct_tf/num_total}\nPreds:  {P[:, -1]}\n')
    # if args.answer_size >= args.tgt_len:
    logging(f'No teacher forcing: acc:{num_correct/num_total}\nPreds:  {target_preds[:, -1].cpu().numpy()}\n')
    accuracy = num_correct / num_total
    # else:
    #     accuracy = num_correct_tf / num_total
    
    print(num_correct_answers)
    logging(f'Answer acc:{num_correct_answers/num_total_answers}\n\n')
        
    # Switch back to the training mode
    model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
    model.train()
#     print('num_correct, num_total', num_correct, num_total)
    # print('\n\n\nParams:')
    # print(locals())
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                         key= lambda x: -x[1])[:10]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
    return total_loss / total_len, accuracy

# Run on test data.
if args.split == 'all':
    test_loss, test_acc = evaluate(te_iter)
    valid_loss, valid_acc = evaluate(va_iter)
elif args.split == 'valid':
    valid_loss, valid_acc = evaluate(va_iter)
    test_loss, test_acc = None, None
elif args.split == 'test':
    test_loss, test_acc = evaluate(te_iter)
    valid_loss, valid_acc = None, None

def format_log(loss, acc, split):
    if args.dataset in ['enwik8', 'text8']:
        log_str = '| {0} loss {1:5.2f} | {0} bpc {2:9.5f} | test acc {} '.format(
            split, loss, loss / math.log(2), acc)
    else:
        log_str = '| {0} loss {1:5.5f} | {0} ppl {2:9.5f} | {0} acc {3:1.5f} '.format(
            split, loss, math.exp(loss), acc)
    return log_str

log_str = ''
if valid_loss is not None:
    log_str += format_log(valid_loss, valid_acc, 'valid')
if test_loss is not None:
    log_str += format_log(test_loss, test_acc, 'test')

logging('=' * 100)
logging(log_str)
logging('=' * 100)

mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
# mem_gradients = sum([param.grad.nelement()*param.grad.element_size() for param in model.parameters()])
mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
# print(f'Model params size: {mem_params}\nGradients size: {mem_gradients}\nBuffers size: {mem_bufs}')
print(f'Model params size: {mem_params}\nBuffers size: {mem_bufs}')

for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),
                         key= lambda x: -x[1])[:10]:
    print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))