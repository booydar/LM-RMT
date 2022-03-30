from torch.utils import tensorboard
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import os
import argparse

def get_eval_values(row):
    step = int(row.split('step')[1].strip().split(' ')[0])
    loss = float(row.split('loss')[1].strip().split(' ')[0])
    metric = float(row.split(' ')[-1].strip())
    return step, loss, metric
    

def logs_to_tb(path, log_dir='logs_tb'):
    _, folders, names = next(os.walk(path))
    for folder in folders:
        folder_path = os.path.join(path, folder)
        logs_to_tb(folder_path, log_dir)
    
    names = [n for n in names if '.txt' in n]
    for name in names:
        filepath = os.path.join(path, name)
        print('Processing ', filepath)
        name = '_'.join(filepath.split('/')[-3:])
        log_path = os.path.join(log_dir, name)
        writer = SummaryWriter(log_dir=log_path)

      # valid metrics
        with open(filepath, 'r') as f:
            log = f.read()

        sep = '='*100
        if sep not in log:
            print('Not a log')
            continue
        rows = np.array(log.split('\n'))
        log_start_ind = np.where(rows == rows[0])[0][1] + 3
        rows = rows[log_start_ind:]
        eval_rows = [r for r in rows if 'Eval ' in r]
        eval_values = list(map(get_eval_values, eval_rows))

        for step, loss, metric in eval_values:
            writer.add_scalar('loss/valid', loss, step)
            writer.add_scalar('metric/valid', metric, step)

      # train metrics 
        df = pd.read_csv(filepath, error_bad_lines=False, skiprows=70, sep='|', names = ['1', 'ep_step', 'batches', 'lr', 'ms_batch', 'loss', 'metric'])
        df = df[df.columns[1:]].dropna()
        if df.shape[0] == 0:
            continue
        # print(df)
        df = df[df.ep_step.apply(lambda x: 'epoch' in x and 'step' in x)]
        df['epoch'] = df.ep_step.apply(lambda x: int(x.split('step')[0].split('epoch')[1].strip()))
        df['step'] = df.ep_step.apply(lambda x: int(x.split('step')[1].strip()))
        df = df.drop(['ep_step'], axis=1)
        df.batches = df.batches.apply(lambda x: x.strip().split(' ')[0]).astype(int)
        df.lr = df.lr.apply(lambda x: float(x[3:]))
        df.ms_batch = df.ms_batch.apply(lambda x: float(x[10:]))
        df.loss = df.loss.apply(lambda x: float(x[6:]))
        df.metric = df.metric.apply(lambda x: float(x[5:]))

        for i, row in df.iterrows():
            writer.add_scalar('loss/train', row.loss, row.step)
            writer.add_scalar('metric/train', row.metric, row.step)

        writer.flush()


## Sample usage:
# python logs_to_tb.py --path initial_logs --log_dir --tensorboard_logs
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='logs',
                        help='Location of initial logs')
    parser.add_argument('--log_dir', type=str, default='logs_tb',
                        help='where to write TB logs')

    args = parser.parse_args()    
    logs_to_tb(path=args.path, log_dir=args.log_dir)