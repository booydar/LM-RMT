import argparse
import os

if __name__ == "__main__":

    models = ['base', 'txl', 'rmt']
    parser = argparse.ArgumentParser(description='script iterator')
    parser.add_argument('--model', type=str, default='base', choices=models)

    parser.add_argument('--seq_len', type=int, default=120)
    parser.add_argument('--tgt_len', nargs='+', type=int, default=[360])
    parser.add_argument('--mem_len', nargs='+', type=int, default=[0])
    parser.add_argument('--num_mem_tokens', nargs='+', type=int, default=[0])
    
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_step', type=int, default=600_000)
    parser.add_argument('--eval_interval', type=int, default=24000)
    parser.add_argument('--device_ids', nargs='+', default=0, help='Device ids for training.')
    parser.add_argument('--work_dir', type=str, default='../../long_synthetic/')
    # parser.add_argument('--data_path', type=str, default='../..//')

    parser.add_argument('--n_iter', type=int, default=2)
    
    args = parser.parse_args()

    template =  'bash run_copy{}.sh train --work_dir {}tgt{}xl{}mt{} --tgt_len {} --eval_tgt_len {} --batch_size {} --eval_interval {} --num_mem_tokens {} --mem_len {} --max_step {} --device_ids {} --seed {}';
    
    print(*args.device_ids)

    print('Commands list:')
    seed = 1111
    for it in range(args.n_iter):
        seed += it+1        
        for tgt_len in args.tgt_len: 
            for mem_len in args.mem_len:
                for num_mem_tokens in args.num_mem_tokens:
                    command = template.format(args.seq_len,
                                            args.work_dir,
                                            tgt_len,
                                            mem_len,
                                            num_mem_tokens,
                                            tgt_len,
                                            tgt_len,
                                            args.batch_size,
                                            args.eval_interval,
                                            num_mem_tokens,
                                            mem_len,
                                            args.max_step,
                                            *args.device_ids,
                                            seed)
                    print(command)

    print('\n')
    print('Start executing')
    cntr = 0
    n_launches = args.n_iter* len(args.tgt_len) * len(args.mem_len) * len(args.num_mem_tokens)
    seed = 1111
    for it in range(args.n_iter):
        print(f'Seed {seed}')
        seed += it+1        
        for tgt_len in args.tgt_len: 
            for mem_len in args.mem_len:
                for num_mem_tokens in args.num_mem_tokens:
                    cntr += 1
                    print(f'\n\nLaunch {cntr} of {n_launches}')
                    command = template.format(args.seq_len,
                                            args.work_dir,
                                            tgt_len,
                                            mem_len,
                                            num_mem_tokens,
                                            tgt_len,
                                            tgt_len,
                                            args.batch_size,
                                            args.eval_interval,
                                            num_mem_tokens,
                                            mem_len,
                                            args.max_step,
                                            *args.device_ids,
                                            seed)
                    print(f'Executing :\n{command}')
                    os.system(command)
