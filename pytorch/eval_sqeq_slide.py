import sys
def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

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

datasets = ['sqeq_slide']

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
stack = False
va_iter = data_loader('val', path=args.data, task_name=args.dataset, batch_size=args.batch_size,
                                    tgt_len=args.tgt_len, device=device, stack=stack,)
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

            # data_segs = torch.chunk(data_, data_.shape[0] // args.tgt_len)
            # target_segs = torch.chunk(target_, target_.shape[0] // args.tgt_len)

            # data_segs = [data_[i:i+args.tgt_len] for i in range(data_.shape[0]-args.tgt_len)]
            # target_segs = [target_[i:i+args.tgt_len] for i in range(target_.shape[0]-args.tgt_len)]
            # losses = []
        
        # caclulate loss
            # for data, target in zip(data_segs, target_segs):
            if mems is None:
                mems = tuple()
            ret = model(data_, target_, *mems, mem_tokens=mem_tokens)
            if model.num_mem_tokens == 0:
                loss, mems = ret[0], ret[1:]
            else:
                mem_tokens, loss, mems = ret[0], ret[1], ret[2:]
            
            # losses.append(loss)
            # loss = torch.cat(losses[:1] + [l[-1:] for l in losses[1:]])
            loss = loss[-args.answer_size:]
            loss = loss.mean()
            total_loss += args.answer_size * loss.float().item()
            total_len += args.answer_size

        # with teacher forcing

            data = data_
            target = target_
            mem_tokens = None
            if model.mem_tokens is not None:
                mem_tokens = model.mem_tokens.repeat(1, data_.shape[1], 1)

            # pred_segs = []
            mems = tuple()    
            # for data, target in zip(data_segs, target_segs):
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

            # pred_segs.append(preds)
            # preds = torch.cat(pred_segs[:1] + [p[-1:] for p in pred_segs[1:]])
            # num_total += args.batch_size * args.answer_size
            num_total += (target_[-args.answer_size:] > 0).float().sum().item()
            num_correct_tf += ((preds[-args.answer_size:] == target_[-args.answer_size:]) & (target_[-args.answer_size:] > 0)).float().sum().item()
            
            S, T, P = data_[:, -1].cpu().numpy(), target_[:, -1].cpu().numpy(), preds.cpu().numpy()

        # no teacher forcing
            mem_tokens = None
            if model.mem_tokens is not None:
                mem_tokens = model.mem_tokens.repeat(1, data_.shape[1], 1)

            # pred_segs = []
            mems = tuple()    
            # for data, target in zip(data_segs, target_segs):
            data_pred = data_.clone()
            target_pred = target_.clone()
            # for i in range(args.answer_size - args.tgt_len, data_.shape[0]-args.tgt_len):
            for i in range(-args.answer_size-1, -1, 1):

                if mems is None:
                    mems = tuple()
                if not mems: mems = model.init_mems(data_pred.device)

                tgt_len = target.size(0)
                hidden, mems = model._forward(data_pred, mems=mems, mem_tokens=mem_tokens)
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

                if i < -1:
                    data_pred[i+1] = preds[i]
                target_pred[i] = preds[i]

            # num_total += (target_[-args.answer_size:] > 0).float().sum().item()
            num_correct += ((target_pred[-args.answer_size:] == target_[-args.answer_size:]) & (target_[-args.answer_size:] > 0)).float().sum().item()
            
            S, T, P = data_[:, -1].cpu().numpy(), target_[:, -1].cpu().numpy(), preds.cpu().numpy()

            # answer accuracy
            keys = [str(i) for i in range(10)]
            t2c = dict(zip(range(24), ['', 'g'] + keys + ['+', '-', '*', '/', '^', '=', '(', ')', 'x', 'D', ';', ',']))
            for bi in range(args.batch_size):
                p = target_pred[:, bi]
                t = target_[:, bi]
                ans_start_ind = torch.where(t == 22)[0][0]
                if (t[ans_start_ind:] == p[ans_start_ind:]).float().mean().item() == 1:
                    num_correct_answers += 1
                
                if i == 0:
                    source_eq = ''.join([t2c[t] for t in data_[:, bi].cpu().numpy()])
                    target_eq = ''.join([t2c[t] for t in target_[:, bi].cpu().numpy()])
                    pred_eq = ''.join([t2c[t] for t in target_pred[:, bi].cpu().numpy()])
                    print(f'\nSource: {source_eq}\nTarget: {target_eq}\nPreds: {pred_eq}')

            num_total_answers += args.batch_size

    logging(f'|\nSource: {S}\nTarget: {T}\nTeacher forcing: acc:{num_correct_tf/num_total}\nPreds:  {P[:, -1]}\n')
    # if args.answer_size >= args.tgt_len:
    logging(f'No teacher forcing: acc:{num_correct/num_total}\nPreds:  {target_pred[:, -1].cpu().numpy()}\n')
    accuracy = num_correct / num_total
    # else:
    #     accuracy = num_correct_tf / num_total
    
    logging(f'Answer acc:{num_correct_answers/num_total_answers}\n\n')
        
    # Switch back to the training mode
    model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
    model.train()
#     print('num_correct, num_total', num_correct, num_total)
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