import pandas as pd
import numpy as np
import os
import argparse

def extract_parameter(row):
    p = row.split(' - ')[1]
    key, value = p.split(' : ')
    return key, value


def parse_test(path, df_path='test_res.csv'):
    # result_df = pd.DataFrame()
    # paths = []

    _, folders, names = next(os.walk(path))
    
    for folder in folders:
        folder_path = os.path.join(path, folder)
        if 'fail' not in folder_path and '.ipynb_checkpoints' not in folder_path:
            parse_test(folder_path, df_path)
        # _, _, nms = next(os.walk(folder_path))
        # ns = [os.path.join(folder_path, n) for n in nms]
        # paths += ns

    # paths += [os.path.join(path, n) for n in names]
    names = [n for n in names if '.txt' in n]
    # print('Parsed paths: ', paths)
    if os.path.exists(df_path):
        result_df = pd.read_csv(df_path)
    else:
        result_df = pd.DataFrame()

    for name in names:
        filepath = os.path.join(path, name)
        print('Processing ', filepath)

        with open(filepath, 'r') as f:
            log = f.read()

        # parse parameters
        rows = np.array(log.split('\n'))
        sep = '=' * 100
        sep_inds = np.where(rows == sep)[0]
        param_rows = rows[sep_inds[0]+1 : sep_inds[1]]
        params = {k:v for k,v in map(extract_parameter, param_rows)}

        # parse results
        if 'End of training |' in log and 'Exiting from training early' not in log:
            results = log.split('End of training |')[-1].split('|')
            results = [r if '\n' not in r else r.split('\n')[0] for r in results]
            results = {' '.join(r.strip().split(' ')[:-1]).strip() : r.strip().split(' ')[-1] for r in results}

            params.update(results)
        
        if 'valid acc' in log:
            val_acc = float(log.split('valid acc')[-1].strip().split('\n')[0].strip())
            params['last_val_acc'] = val_acc

        result_df = result_df.append(params, ignore_index=True)
    
    if result_df.shape[0] > 0:
        result_df.to_csv(df_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='logs',
                        help='Location of logs')
    parser.add_argument('--df_path', type=str, default='logs',
                        help='Destination path')

    args = parser.parse_args()    
    parse_test(path=args.path, df_path=args.df_path)
